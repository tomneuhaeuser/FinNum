import torch
import random
import numpy as np

import time

# Reproducability
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class Model(torch.nn.Module):
    def __init__(self, number_of_dimensions, number_of_categories):

        super().__init__()
        
        self.number_of_dimensions = number_of_dimensions
        self.number_of_categories = number_of_categories
        
        # The first convolution with a kernel of size 3
        self.convolution_3 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.number_of_dimensions,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            torch.nn.ReLU()
        )
        
        # The second convolution with a kernel of size 5
        self.convolution_5 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.number_of_dimensions,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),
            torch.nn.ReLU()
        )
        
        # The third convolution with a kernel of size 7
        self.convolution_7 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.number_of_dimensions,
                out_channels=128,
                kernel_size=7,
                padding=3
            ),
            torch.nn.ReLU()
        )
        
        # The classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(3*128, self.number_of_categories)
        )
        

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
    
        output_convolution_3 = self.convolution_3(x)
        output_convolution_3 = torch.transpose(output_convolution_3, 1, 2)
        
        output_convolution_5 = self.convolution_5(x)
        output_convolution_5 = torch.transpose(output_convolution_5, 1, 2)
        
        output_convolution_7 = self.convolution_7(x)
        output_convolution_7 = torch.transpose(output_convolution_7, 1, 2)
        
        # Concatenate the outputs of the three convolutions
        output_convolution = torch.cat((output_convolution_3, output_convolution_5, output_convolution_7), 2)
        
        output_classifier = self.classifier(output_convolution)
        
        return output_classifier
        
        