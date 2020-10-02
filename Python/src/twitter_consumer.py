import json
from collections import Counter
from string import punctuation

import torch
from kafka import KafkaConsumer
from torchtext import data
from torchtext import vocab
import boto3

from model import LstmModel

def get_vocab_counter():
    with open("../data/vocab_stoi.json", "r") as fp:
        vocab = json.load(fp)
    
    counter = Counter(vocab)
    return counter

def configure_field(vocab_counter):
    tokenizer = lambda s: s.split()
    text_field = data.Field(tokenize=tokenizer, include_lengths=True, batch_first=True)
    text_field.vocab = vocab.Vocab(vocab_counter)
    return text_field

def encode_string(text, field):
    # Preprocess the text (I don't think this does anything right now)
    text = field.preprocess(text)
    # Return a processed version of the text
    return field.process(text)

def process_string(msg):
    """
    Deserialize the message polled from kafka
    """
    # Convert the message to a string
    msg = str(msg)
    # The first element is going to be 'b' so just slice that off
    msg = msg[1:]
    # Make everything lowercase
    msg = msg.lower()
    # Remove all punctuation
    msg = "".join([ch for ch in msg if ch not in punctuation])
    return msg

def process_json(json):
    pass

def init_model(vocab_length):
    if torch.cuda.is_available() == False:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model = LstmModel(vocab_size=vocab_length,
                    emb_dim=50,
                    lstm_units=128,
                    num_hidden=256,
                    num_layers=3,
                    num_classes=3,
                    use_gpu=False)
    model.load_state_dict(torch.load("../data/sent140.pth", map_location=device))
    model.eval()
    return model

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        data = process_string(data)
        data = encode_string(data)

        msg, msg_lengths = data
        pred = model(msg, msg_lengths)
    return pred

topic = "twitter_topic"
group_id = "twitter_group"
bootstrap_servers = ["localhost:9092"]
json_deserializer = lambda m: json.loads(m.decode("utf-8"))
string_deserializer = process_string
tokenizer = lambda s: s.split()

def create_consumer(topic, group_id, bootstrap_servers, deserializer): 
    consumer = KafkaConsumer(topic, 
                            group_id=group_id, 
                            bootstrap_servers=bootstrap_servers, 
                            value_deserializer=deserializer,
                            auto_offset_reset='latest')
    return consumer

def run():
    consumer = create_consumer(topic, group_id, bootstrap_servers, string_deserializer)

    for msg in consumer:
        val = msg.value
        print(val)

def run_evaluation(message):
    vocab_counter = get_vocab_counter()
    text_field = configure_field(vocab_counter)
    sample_text = message
    model = init_model(len(vocab_counter))
    prediction = evaluate(model, sample_text)
    return prediction

if __name__ == '__main__':
    sample_text = "that's some spicy sample text"
    print(run_evaluation(sample_text))
    