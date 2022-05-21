
# coding: utf-8

# In[ ]:


# Developing an AI application


# In[1]:


# Imports here
 
#  Matplotlib Information - Jupyter

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


#Importing torch essentials - Libraries

import torch
from torch import nn, optim
import torch.nn as nn 
import torch.nn.functional as F # or Function
import torchvision 
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.autograd import Variable

from PIL import Image

import time 
import numpy as np
import seaborn as sns
import json

import argparse

import os
import sys

print(sys.version)


# In[3]:


def args_paser():
    paser = argparse.ArgumentParser(description='trainer file')
    paser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    paser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    paser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    paser.add_argument('--arch', type=str, default='densenet201', help='architecture')
    paser.add_argument('--hidden_layer', type=int, default=[25088, 102], help='hidden layer for layer')
    paser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')
    args = paser.parse_args()

    return args


# In[ ]:





# In[55]:


#data_dir = 'flowers'
#train_dir = data_dir + '/train'
#valid_dir = data_dir + '/valid'
#test_dir = data_dir + '/test'


# In[4]:


# TODO: Define your transforms for the training, validation, and testing sets
#data_transforms = 
def procedure_data(train_dir, valid_dir, test_dir):
    
# TODO: Load the datasets with ImageFolder
#image_datasets = 

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 


    train_transforms = transforms.Compose([
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(225),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])
                                      ])
    valid_transforms = transforms.Compose([transforms.Resize(299),
                                       transforms.CenterCrop(255),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])
                                      ])

    test_transforms = transforms.Compose([transforms.Resize(299),
                                       transforms.CenterCrop(255),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,00.456,.406],
                                                           [0.229,0.224,0.225])
                                      ])

    
    
    
    
    
    # TODO: Load the datasets with ImageFolder
#image_datasets = 

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 
#download and load the training data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

#download and load the test data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

#download and load the valid data
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    
    return trainloader, testloader, validloader

    print('dataloaders')   
    
    #image_datasets = [train_data, valid_data, test_data]
    #dataloaders = [trainloader, validloader, testloader]
    return trainloader, testloader, validloader, train_data

    


# In[5]:


def what_model(arch):
   # Load pretrained_network
    print('Please only select densenet201 or alexnet as pretrained model. By default densenet201 is selected!')
    if arch == None or arch == 'densenet201':
        load_model = models.densenet201(pretrained = True)
#load_model.name = 'densenent201'
        print('Current model: densenet201 loaded')
    else:
        load_model = models.alexnet (pretrained = True)
        print('Current model: alexnet loaded')
#model.name = alexnet

    return load_model
 


# In[6]:


def tech_classifier(model, hidden_layer):
    if hidden_layer == None:
        hidden_layer = 25088
    input = model.classifier[0].in_features

    
    
    #Create classifier    
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256, 102),
                                nn.LogSoftmax(dim=1))
#train the model
    print('Running Model')


    return model


# In[ ]:





# In[9]:


def train_model(epochs, trainloader, validloader, gpu, model, optimizer, criterion):
    accuracy = 0
    running_loss = 0
    
    if type(epochs) == type(None):
        epochs = 10
    print("Epochs = 10")


    if gpu==True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')  #cpu
    steps = 0
        
       
    
    model.to(device)
        
    for epoch in range(epochs): 
        
        
        for inputs, labels in trainloader: ## check while, for, else
        
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            #Gradients to zero

            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        else:
            valid_loss = 0
            
            
            
        with torch.no_grad():
             for inputs, labels in validloader:
                    
                inputs, labels = inputs.to(device), labels.to(device)  #GPU
                    
                    
                logps = model.forward(inputs)
                    
                valid_loss += criterion(logps, labels).item()
                    
                 # Accuracy
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    
                    
        print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/len(['trainloader']):.3f}.. "
                f"Validation loss: {valid_loss/len(['validloader']):.3f}.. "
                f"Accuracy: {accuracy/len(['validloader']):.3f}.. "
                 )
            
            
        running_loss = 0
            
        model.train()  #chaning back
        
        return model
            


# In[11]:


def valid_model(model, testloader, gpu):

    if gpu==True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accuracy = 0
    model.eval()

    for inputs, labels in testloader: ## check 
                     
        inputs, labels = inputs.to(device), labels.to(device)
        
        # forward pass with test data
        logps = model.forward(inputs)

        # accuracy
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim = 1)
        equality = top_class == labels.view(*top_class.shape)
        test_accuracy += equality.type(torch.FloatTensor).mean()

    print("Test Accuracy: {:.3f}".format(100 * (test_accuracy/len(testloader))))
    


# In[ ]:





# In[14]:


def s_check(model, train_dataset, arch, hidden_layer, epochs, optimizer, save_dir):   
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': arch,
                  'hidden_layer': 25088,
                  'output': 512,
                  'mapping': model.class_to_idx,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }
   
    torch.save(checkpoint, 'checkpoint.pth')


# In[ ]:


def main():

    args = args_paser()

    # Definition of image data directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define dataloaders
    trainloader, testloader, validloader, train_dataset = process_data(train_dir, test_dir, valid_dir)

    # Load pretrained model according to selection
    model = what_model(args.arch)

    # Freeze parameters of pretrained model to not backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier for the model
    model = tech_classifier(model, args.hidden_layers)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    #print(device)

    # Loss: negative log likelihood
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate=args.lr)
    print('Criterion and Optimizer !')

    # Call of function to train the model
    print('Model:')
    #trained_model = train_model(args.epochs, trainloader, validloader, args.gpu, model, optimizer, criterion)

    # Call of fucntion to validate the model
    print('Validation - wait:')
    #valid_model(trained_model, testloader, args.gpu)
    #save_checkpoint(trained_model, train_dataset, args.arch, args.hidden_units, args.epochs, optimizer, args.save_dir)
    print('Kaput!')

if __name__ == '__main__': main()

    

