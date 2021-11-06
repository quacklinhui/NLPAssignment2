#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim

import data
import model

# getting args for model
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 FNN Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--context_size', type=int, default=8, help='size of ngram model')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

# def batchify(data, bsz):
#     # Work out how cleanly we can divide the dataset into bsz parts.
#     nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     return data.to(device)

def batchify(data, bsz):
    value=[]
    data = data.numpy()
    for i,word in enumerate(data):
        if i+bsz>= len(data):
            # sentence boundary reached
            # ignoring sentence less than 8 words
            break
        # convert word to id
        value1 = []
        for j in range(bsz+1):
            value1.append(data[i+j])
        value.append(value1)
    value = torch.LongTensor(value)
    return value.to(device)

eval_batch_size = 8
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model (ii)
###############################################################################

ntokens = len(corpus.dictionary)
model = model.FNNModel(ntokens, args.emsize, args.nhid, args.context_size, args.tied).to(device)

# using negative log likelihood
criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. 

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len, 0:args.context_size]
    target = source[i+1:i+1+seq_len, args.context_size-1:args.context_size]
    target = target.narrow(1,0,1).contiguous().view(-1)
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    # using ADAM optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        data, targets = data.to(device), targets.to(device)
        model.zero_grad() # zero out the gradients from the old instance
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

###############################################################################
# Main code --> (iii) - (v)
###############################################################################

print('-' * 89)
print('--- Part iii ---')
print('-'*89)

# Loop over epochs.
lr = args.lr
best_val_loss = None
best_perplexity = 999999999999999

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        
        # Save the model if the perplexity is the best seen
        perplexity = math.exp(val_loss)
        if perplexity < best_perplexity:
            print(perplexity)
            with open(args.save, 'wb') as f:
                torch.save(model, f)
                
            best_perplexity = perplexity
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
print('=' * 89)

###############################################################################
# Main code --> (iii) - (v)
###############################################################################
import model

print('-'*89)
print('--- Part vi ---')
print('-'*89)

###########################################################################
# Build the model
###########################################################################

ntokens = len(corpus.dictionary)
model_vi = model.FNNModel(ntokens, args.emsize, args.emsize, args.context_size, args.tied).to(device)

# using negative log likelihood
criterion = nn.NLLLoss()

 # Training the model
def train_vi():
    # Turn on training mode which enables dropout.
    model_vi.train()
    # using ADAM optimizer
    optimizer = optim.Adam(model_vi.parameters(), lr = args.lr)
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        data, targets = data.to(device), targets.to(device)
        model_vi.zero_grad() # zero out the gradients from the old instance
        output = model_vi(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

def evaluate_vi(data_source):
    # Turn on evaluation mode which disables dropout.
    model_vi.eval()
    total_loss = 0.
    #ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            data, targets = data.to(device), targets.to(device)
            output = model_vi(data)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) -1)


# Loop over epochs.
lr = args.lr
best_val_loss = None
best_perplexity = 999999999999999

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_vi()
        val_loss = evaluate_vi(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        
        # Save the model if the perplexity is the best seen
        perplexity = math.exp(val_loss)
        if perplexity < best_perplexity:
            print(perplexity)
            with open('model_vi.pt', 'wb') as f:
                torch.save(model_vi, f)
                
            best_perplexity = perplexity
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open('model_vi.pt', 'rb') as f:
    model_vi = torch.load(f)

# Run on test data.
test_loss = evaluate_vi(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
print('=' * 89)