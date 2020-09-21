# Make sure everything works when running inside Docker container
import torch
import torch.nn as nn
from torchtext import data
import argparse
import os
import boto3

from Model import LstmModel

is_gpu = torch.cuda.is_available()
print("Cuda", is_gpu)



# Hyperparams
# Pass in hyperparams as arguments for production
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=50)
parser.add_argument('--num_classes', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--lstm_units', type=int, default=128)
parser.add_argument('--num_hidden', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)

# Data, model, and output directories
parser.add_argument('--output_data_dir', type=str, default=os.environ['OUTPUT_DATA_DIR'])
parser.add_argument('--model_dir', type=str, default=os.environ['MODEL_DIR'])
parser.add_argument('--train_path', type=str, default=os.environ['TRAIN_FILE_PATH'])
parser.add_argument('--bucket_name', type=str, default=os.environ['BUCKET_NAME'])
parser.add_argument('--train_key', type=str, default=os.environ['TRAIN_FILE'])
args, _ = parser.parse_known_args()

# Print environment variables
print("Output directory", os.environ['OUTPUT_DATA_DIR'])
print("Model directory", os.environ['MODEL_DIR'])
print("Train file", os.environ['TRAIN_FILE_PATH'])

# Torchtext things
tokenizer = lambda s: s.split()

# init text and label fields
# Label is just a single number so no sequence, vocabulary, or padding
# Data should already have been preprocessed in some other notebook
s3_resource = boto3.resource('s3')
train_object = s3_resource.Object(bucket_name=args.bucket_name, key=args.train_key)
train_object.download_file('train.csv')

text_field = data.Field(tokenize=tokenizer, include_lengths=True, batch_first=True)
label_field = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
fields = [('label', label_field), ('text', text_field)]
all_data = data.TabularDataset(path='train.csv', format='CSV', skip_header=True, fields=fields)
print("Tabular Dataset", all_data)

# Initialize model
if is_gpu:
    model = LstmModel(vocab_size=50,
                emb_dim=args.emb_dim,
                lstm_units=args.lstm_units,
                num_hidden=args.num_hidden,
                num_layers=args.num_layers,
                num_classes=args.num_classes,
                use_gpu=True)
else:
    model = LstmModel(vocab_size=50,
                emb_dim=args.emb_dim,
                lstm_units=args.lstm_units,
                num_hidden=args.num_hidden,
                num_layers=args.num_layers,
                num_classes=args.num_classes)

print("Model", model)

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Criterion", criterion)
print("Optimizer", optimizer)