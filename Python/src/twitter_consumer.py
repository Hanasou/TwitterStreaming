import json
from collections import Counter
from string import punctuation

import torch
from kafka import KafkaConsumer
from torchtext import data
from torchtext import vocab
from pymongo import MongoClient
from pymongo import errors
from bson.objectid import ObjectId
import boto3

from Model import LstmModel

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

def process_string(msg):
    # Convert the message to a string
    msg = str(msg)
    # Make everything lowercase
    msg = msg.lower()
    # Remove all punctuation
    msg = "".join([ch for ch in msg if ch not in punctuation])
    return msg

def encode_string(text, field):
    encoded_list = []
    for word in text.split():
        encoded_list.append(field.vocab.stoi[word])
    encoded = torch.LongTensor(encoded_list)
    length = torch.LongTensor([len(encoded_list)])
    reshaped = torch.reshape(encoded, (1,-1))
    return reshaped, length

def encode_string_old(text, field):
    # Preprocess the text (I don't think this does anything right now)
    text = field.preprocess(text)
    # Return a processed version of the text
    return field.process(text)

def process_json(message):
    message_dict = json.loads(message.decode("utf-8"))
    val_dict = dict()
    val_dict["id_str"] = message_dict["id_str"]
    val_dict["text"] = message_dict["text"]
    return val_dict

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
                    use_gpu=torch.cuda.is_available())
    model.load_state_dict(torch.load("../data/sent140.pth", map_location=device))
    model.eval()
    return model

def evaluate(model, data, field):
    model.eval()
    with torch.no_grad():
        data = process_string(data)
        data = encode_string(data, field)

        msg, msg_lengths = data
        msg_lengths = torch.sort(msg_lengths, descending=True)[0]
        pred = model(msg, msg_lengths)
    return pred.argmax().item()

topic = "twitter_topic_3"
group_id = "twitter_group_2"
bootstrap_servers = ["localhost:9092"]
json_deserializer = process_json
string_deserializer = process_string
tokenizer = lambda s: s.split()

def create_consumer(topic, group_id, bootstrap_servers, deserializer): 
    consumer = KafkaConsumer(topic, 
                            group_id=group_id, 
                            bootstrap_servers=bootstrap_servers, 
                            value_deserializer=deserializer,
                            auto_offset_reset='latest',
                            enable_auto_commit=False)
    return consumer

def get_credentials():
    credentials = {}
    # Open file
    with open("../data/dbcredentials.txt", "r") as creds:
        lines = creds.readlines()
    
    # Read lines
    for line in lines:
        cred = line.split()
        label = cred[0]
        value = cred[1]
        credentials[label] = value
    
    # Put into dictionary
    return credentials

def connect_client(creds):
    username = creds["username"]
    password = creds["password"]
    dbname = creds["dbname"]
    uri = "mongodb+srv://{}:{}@cluster0.zrbnr.mongodb.net/{}?retryWrites=true&w=majority".format(username, 
        password, dbname)
    client = MongoClient(uri)
    db = client.twitter_app
    return db

def run():
    print("Starting Consumer")
    consumer = create_consumer(topic, group_id, bootstrap_servers, json_deserializer)
    print("Polling")
    # Initialize vocab, field, and model
    vocab_counter = get_vocab_counter()
    text_field = configure_field(vocab_counter)
    model = init_model(len(vocab_counter))

    # Initialize database
    creds = get_credentials()
    db = connect_client(creds)
    collection = db.tweets

    # ObjectId needs to be 24 hex characters and our tweet_id is only 19
    # So append this to all of them 
    suffix = '12345'

    try:
        while True:
            try:
                msg_pack = consumer.poll(1000)
                for tp, messages in msg_pack.items():
                    record_count = len(messages)
                    documents = []
                    for msg in messages:
                        try:
                            # Unpack the message
                            val = msg.value
                            val_id = val["id_str"]
                            val_text = val["text"]
                            pred = evaluate(model, val_text, text_field)
                            val_sent = pred

                            # Append this document
                            document = {
                                '_id': ObjectId(val_id + suffix),
                                'text': val_text,
                                'sentiment': val_sent
                            }
                            documents.append(document)
                            print(val_text, val_sent)
                        except KeyError:
                            print("Bad Data")
                    if record_count > 0:
                        # Insert the documents
                        collection.insert_many(documents)
                        consumer.commit()
            except KeyError:
                print("Bad data")
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        print("Closing consumer")
        consumer.close()

def run_evaluation(message):
    vocab_counter = get_vocab_counter()
    text_field = configure_field(vocab_counter)
    model = init_model(len(vocab_counter))
    try:
        prediction = evaluate(model, message, text_field)
    except ValueError:
        print("Bad input")
        prediction = -1
    return prediction

if __name__ == '__main__':
    run()
    
    