import time
import json
import torch
import torch.nn as nn
from torchtext import data

from Model import LstmModel



# Train model
def train(epochs, model, iterator, criterion, optimizer):
    if model.use_gpu:
        model.cuda()
    start_time = time.time()
    print("Train Start")
    for i in range(epochs):
        for batch in iterator:
            # Reset gradient
            optimizer.zero_grad()

            # Initialize input data and labels
            # .squeeze() makes sure everything is in one dimension
            text, text_lengths = batch.text
            labels = batch.label.squeeze()
            if model.use_gpu:
                text, text_lengths = text.cuda(), text_lengths.cuda()
                labels = labels.cuda()

            # Make predictions
            preds = model(text, text_lengths)
            # Get loss
            loss = criterion(preds, labels)

            # Backpropagate
            loss.backward()
            # Clip gradient just in case to avoid exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            optimizer.step()
        print(f'Epoch {i} Loss {loss.item():10.8f}')
    duration = time.time() - start_time
    print(f'Train Duration: {duration:.0f}')
    # Save model
    torch.save(model.state_dict(), "sent140.net")

# Evaluate
def evaluate(model, iterator, criterion):
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            labels = batch.label.squeeze()
            if model.use_gpu:
                text, text_lengths = text.cuda(), text_lengths.cuda()
                labels = labels.cuda()
            
            preds = model(text, text_lengths)
            loss = criterion(preds, labels)
    print(f'CE Loss: {loss:.8f}')

# Encode string
def encode(text):
    encoded_list = []
    print("word list", text.split())
    for word in text.split():
        encoded_list.append(text_field.vocab.stoi[word])
    print("encoded list", encoded_list)
    return torch.FloatTensor(encoded_list), torch.FloatTensor([len(encoded_list)])

def train_eval():
    # Check if we're using cuda
    is_gpu = torch.cuda.is_available()

    # init tokenizer
    tokenizer = lambda s: s.split()

    # init text and label fields
    # Label is just a single number so no sequence, vocabulary, or padding
    # Data should already have been preprocessed in some other notebook
    text_field = data.Field(tokenize=tokenizer, include_lengths=True, batch_first=True)
    label_field = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

    # Initialize a TabularDataset
    # Create a TabularDataset with path to our csv file
    # Call split on it
    fields = [('label', label_field), ('text', text_field)]
    train_path = '../data/training_processed_3_short.csv'
    all_data = data.TabularDataset(path=train_path, format='CSV', skip_header=True, fields=fields)

    train_data, test_data = all_data.split(split_ratio=0.7)

    # Build the vocabulary
    # Access them with text_field.vocab and label_field.vocab
    text_field.build_vocab(train_data)
    label_field.build_vocab(train_data)

    # Initialize device
    device = torch.device('cuda' if is_gpu else 'cpu')
    batch_size = 100

    # Split data into batches. I think this also encodes my text data
    # Call batch.text to pass in text data or something
    # Each batch contains a tuple
    # The first is 100 batches of padded vectors
    # The second one contains the length of each example in the batch
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_key = lambda x: len(x.text),
        sort_within_batch=True,
        device=device
    )

    # Hyperparameters
    cpu_epochs = 5
    gpu_epochs = 10
    emb_dim = 50
    hidden_dim = 50
    num_classes = 3
    num_layers = 3
    lstm_units = 128
    num_hidden = 256

    vocab = text_field.vocab

    # Initialize model
    if is_gpu:
        model = LstmModel(vocab_size=len(vocab),
                    emb_dim=emb_dim,
                    lstm_units=lstm_units,
                    num_hidden=num_hidden,
                    num_layers=num_layers,
                    num_classes=num_classes,
                    use_gpu=True)
    else:
        model = LstmModel(vocab_size=len(vocab),
                        emb_dim=emb_dim,
                        lstm_units=lstm_units,
                        num_hidden=num_hidden,
                        num_layers=num_layers,
                        num_classes=num_classes)

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    train(cpu_epochs, model, train_iterator, criterion, optimizer)
    # Evaluate
    evaluate(model, test_iterator, criterion)
    
if __name__ == '__main__':
    # Check if we're using cuda
    is_gpu = torch.cuda.is_available()

    # init tokenizer
    tokenizer = lambda s: s.split()

    # init text and label fields
    # Label is just a single number so no sequence, vocabulary, or padding
    # Data should already have been preprocessed in some other notebook
    text_field = data.Field(tokenize=tokenizer, include_lengths=True, batch_first=True)
    label_field = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

    # Initialize a TabularDataset
    # Create a TabularDataset with path to our csv file
    # Call split on it
    fields = [('label', label_field), ('text', text_field)]
    train_path = '../data/training_processed_3_short.csv'
    all_data = data.TabularDataset(path=train_path, format='CSV', skip_header=True, fields=fields)

    train_data, test_data = all_data.split(split_ratio=0.7)

    text_field.build_vocab(train_data)
    label_field.build_vocab(train_data)

    device = torch.device('cuda' if is_gpu else 'cpu')
    batch_size = 100

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_key = lambda x: len(x.text),
        sort_within_batch=True,
        device=device
    )

    for batch in train_iterator:
        break

    print(encode("this is a little bit of text")[1])