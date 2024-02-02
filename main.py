import torch
import random
import numpy as np

from torch.utils.data import DataLoader 
from Dataset import Dataset, collate_fn
from Model import Model
from Trainer import Trainer, compute_accuracy

import time

# Reproducability
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
    
epochs = 10
early_stopping = 4
learning_rate = 1e-3
batch_size = 32

# Load the weights for the WeightedRandomSampler (one weight per training sample)
with open('weights.txt', 'r') as f:
    weights = [float(w) for w in f.read().split(', ')]

train = Dataset(dataset = 'train')

# Use a WeightedRandomSampler to combat the category imbalance
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
dataloader_train = DataLoader(train, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

validation = Dataset(dataset = 'validation')
dataloader_validation = DataLoader(validation, batch_size=batch_size, collate_fn=collate_fn)

# Initialize the model
model = Model(number_of_dimensions=101, number_of_categories=8)

# Initialize the trainer
trainer = Trainer(epochs, early_stopping, learning_rate, model, dataloader_train, dataloader_validation)

print(f"Start training ...")
print()

start_training = time.time()
trainer.train()
stop_training = time.time()

print()
print(f"Done training! Time used: {stop_training-start_training}")

test = Dataset(dataset = 'test')
dataloader_test = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn)

print()
print("Test the trained model ...")

micro, macro = compute_accuracy(model, dataloader_test)

print()
print(f"Micro F1-Score: {micro:.3f}, Macro F1-Score: {macro:.3f}")

