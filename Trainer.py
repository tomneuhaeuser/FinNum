import torch
import random
import numpy as np

import collections
import sklearn

import time

# Reproducability
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Compute the Micro and Macro F1-Scores for a given model and given data
def compute_accuracy(model, dataloader):
    micro = 0.0
    macro = 0.0
        
    model.eval()
        
    with torch.no_grad():
        for X,Y in dataloader:
            prediction = model(X)
              
            # Consider target numerals only
            prediction = prediction[Y>0][:,1:]
            Y = Y[Y>0]-1
                
            micro += sklearn.metrics.f1_score(torch.flatten(torch.argmax(prediction, 1)), torch.flatten(Y), average='micro')
            macro += sklearn.metrics.f1_score(torch.flatten(torch.argmax(prediction, 1)), torch.flatten(Y), average='macro')

    model.train()

    micro /= len(dataloader)
    macro /= len(dataloader)
    
    return micro, macro

class Trainer:
    def __init__(self, epochs, early_stopping, learning_rate, model, dataloader_train, dataloader_validation):
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        
        self.model = model
        
        self.dataloader_train = dataloader_train
        self.dataloader_validation = dataloader_validation
        
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        
        self.validation_losses = collections.OrderedDict()
        
        self.steps_per_validation = int(len(dataloader_train)/2)
        self.global_step = 0
    
    def __compute_loss_and_accuracy__(self):
        loss = 0.0
        
        micro = 0.0
        macro = 0.0
            
        self.model.eval()
            
        with torch.no_grad():
            for X,Y in self.dataloader_validation:
                prediction = self.model(X)
                
                # Consider target numerals only
                prediction = prediction[Y>0][:,1:]
                Y = Y[Y>0]-1
                    
                loss += self.loss_function(prediction, Y)
                    
                micro += sklearn.metrics.f1_score(torch.flatten(torch.argmax(prediction, 1)), torch.flatten(Y), average='micro')
                macro += sklearn.metrics.f1_score(torch.flatten(torch.argmax(prediction, 1)), torch.flatten(Y), average='macro')

        self.model.train()

        loss /= len(self.dataloader_validation)
        
        micro /= len(self.dataloader_validation)
        macro /= len(self.dataloader_validation)

        return loss, (micro, macro)
    
    # Every so often, validate the model
    def __validation_step__(self):
        validation_loss, validation_accuracy = self.__compute_loss_and_accuracy__()
        
        self.validation_losses[self.global_step] = validation_loss
        
        print(f"Epoch: {self.epoch}", f"Global step: {self.global_step}", f"Validation Loss: {validation_loss:.3f}", f"Micro F1-Score: {validation_accuracy[0]:.3f}, Macro F1-Score: {validation_accuracy[1]:.3f}", sep=", ")

    def __train_step__(self, X, Y):
        prediction = self.model(X)
        
        # Consider target numerals only
        prediction = prediction[Y>0][:,1:]
        Y = Y[Y>0]-1
        
        loss = self.loss_function(prediction, Y)
        
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss
    
    def train(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            
            for (X,Y) in self.dataloader_train:
                loss = self.__train_step__(X, Y)
                
                self.global_step += 1
                
                if self.global_step % self.steps_per_validation == 0:
                    self.__validation_step__()
                    
                    # If the validation losses stop to improve, stop early
                    if len(self.validation_losses) >= self.early_stopping:
                        latest_losses = list(self.validation_losses.values())[-self.early_stopping:]
                        
                        if latest_losses[0] == min(latest_losses):
                            return # early stopping
        