# Script to train model on Amazon Sagemaker
import time
import json
import argparse
import os
import torch
import torch.nn as nn
from torchtext import data

from Model import LstmModel

# Trains and saves model
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
    with open(os.path.join(args.model_dir, 'sent140.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

# Evaluate
def evaluate(model, iterator, criterion):
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            labels = batch.labels.squeeze()
            if model.use_gpu:
                text, text_lengths = text.cuda(), text_lengths.cuda()
                labels = labels.cuda()
            
            preds = model(text, text_lengths)
            loss = criterion(preds, labels)
    print(f'CE Loss: {loss:.8f}')

# Encode string
def encode(text):
    text = text_field.preprocess(text)
    return text_field.process(text)

def run_train(is_gpu):
    if is_gpu:
        train(gpu_epochs, model, train_iterator, criterion, optimizer)
    else:
        train(cpu_epochs, model, train_iterator, criterion, optimizer)

def model_fn(model_dir):
    model = LstmModel()
    with open(os.path.join(model_dir, 'model.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # Pass in hyperparams as arguments for production
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--emb_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lstm_units', type=int, default=128)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_cuda', type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()

    # Hyperparameters
    lr = args.lr
    batch_size = args.batch_size
    cpu_epochs = 1
    gpu_epochs = args.epochs
    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    num_classes = args.num_classes
    num_layers = args.num_layers
    lstm_units = args.lstm_units
    num_hidden = args.num_hidden

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
    all_data = data.TabularDataset(path=args.train, format='CSV', skip_header=True, fields=fields)

    train_data, test_data = all_data.split(split_ratio=0.7)

    # Build the vocabulary
    # Access them with text_field.vocab and label_field.vocab
    text_field.build_vocab(train_data)
    with open(os.path.join(args.model_dir, 'vocab.json'), 'w') as fp:
        json.dump(text_field.vocab, fp)
    label_field.build_vocab(train_data)

    # Initialize device
    device = torch.device('cuda' if is_gpu else 'cpu')
    
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

    # Initialize vocab
    with open(os.path.join(args.model_dir, 'vocab.json'), 'r') as fp:
        vocab = json.load(fp)
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

    # Train the model and save it
    run_train(is_gpu)