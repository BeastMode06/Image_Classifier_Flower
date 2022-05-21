
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict	
from PIL import Image
import seaborn as sb


import argparse
import os


# In[2]:


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


# In[3]:


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


# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_value(file, arch):

    # Load checkpoint according to filename/filepath
    checkpoint = torch.load(file)

    # Load model corresponding to arch - type
    model = what_model(arch)

    # Freeze parameters	
    for param in model.parameters():
        param.requires_grad = False	

  
        model.classifier =  checkpoint['classifier']
        optimizer.load_state_dict(checkpoint['optimizer'])

     
         
                               
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
    
    return model, optimizer


 

#model = loading_value('checkpoint.pth')
 


# In[ ]:





# In[7]:


def preprocess_image(image_path):
    
    # Open image from the image_path
    im = Image.open(image_path)
    
    # Preprocess using a the approach from the previous transformations
    
    preprocess = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    fin_im = preprocess(im)

    return np.array(fin_im)


# In[8]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[9]:


#!ls flowers/test/81


# In[ ]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''#model: pre-trained model
    # image path
    
    # TODO: Implement the code to predict the class from an image file
     # Checking if the 'GPU' is available to pass it for the device variable, and if it's not, pass the 'CPU'
    device = torch.device("cuda:0")
    # Move model to the device
    model.to(device)
    # Model in inference mode, dropout is off
    model.eval()
    
    image = process_image(image_path)
    #print(image.size()) >>> torch.Size([3, 244, 244])
    image.unsqueeze_(0) 
    #print(image.size()) >>> torch.Size([1, 3, 244, 244])
    
    # Move image tensors to the device.
    image = image.to(device)
    
    # Turn off gradients for testing saves memory and computations, so will speed up inference.
    with torch.no_grad():
        # Forward pass through the network to get the outputs.
        prediction = model.forward(image)
    # Take exponential to get the probabilities from log softmax output.
    ps = torch.exp(prediction)
    # The most likely (topk) predicted prbabilities with their indices.
    probs, top_k_indices = ps.topk(topk)
    
    # Extracting the classes from the indices.
    classes = []
    for indice in top_k_indices.gpu()[0]:  #cpu
        classes.append(list(model.class_to_idx)[indice.numpy()]) # Here we take the class from the index.
    
    return probs.gpu()[0].numpy(), classes    

    


# In[ ]:


def main():
    args = args_paser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
        
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    print('Checkpoint')
    
    optimizer, load_model = load_checkpoint('checkpoint.pth', args.arch)
    print("Predicting Image Probability and Class...")
    probs, classes = predict(device, img_path, model, top_k)
    print(probs)
    print(classes)
    print("WoW!")
    
if __name__ == "__main__":
    main()
    
     
    
    
   

