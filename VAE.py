# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:35:35 2020

@author: tingtinghou
"""

from IPython import display
import datetime
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
from util import Logger
DATA_FOLDER = './torch_data/VAE/MNIST'
starttime = datetime.datetime.now()
bs = 100
# MNIST Dataset
out_dir = '{}/dataset'.format(DATA_FOLDER)
train_dataset = datasets.MNIST(root=out_dir, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=out_dir, train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
# Num batches
num_batches = len(train_loader)


def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 =  nn.Sequential( 
            nn.Linear(x_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
            )
        self.fc2 =  nn.Sequential( 
            nn.Linear(1024, h_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
            )
        self.fc3 =nn.Sequential(  
            nn.Linear(h_dim1, h_dim2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.fc41 = nn.Sequential(  
            nn.Linear(h_dim2, z_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
            )
        self.fc42 = nn.Sequential(  
            nn.Linear(h_dim2, z_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
            )
        
        # decoder part
        self.fc5 =  nn.Sequential( 
            nn.Linear(z_dim, h_dim2),
            nn.LeakyReLU(0.2),
            )
        self.fc6 = nn.Sequential(
            nn.Linear(h_dim2, h_dim1),
            nn.LeakyReLU(0.2)
            ) 
        self.fc7 =  nn.Sequential(
            nn.Linear(h_dim1, 1024),
            nn.LeakyReLU(0.2)
            )
        self.fc8 =  nn.Sequential(
            nn.Linear(1024, x_dim),
            nn.LeakyReLU(0.2))
        
    def encoder(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        return self.fc41(h), self.fc42(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = self.fc5(z)
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)
        return F.sigmoid(h) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)  
       
v_optimizer = optim.Adam(vae.parameters(), lr=0.0002)

# Loss function
#loss = nn.BCELoss()

num_epochs = 2000

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train_VAE(optimizer, data):
    
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    recon_batch, mu, log_var = vae(data)
    # Calculate error and backpropagate
    error = loss_function(recon_batch, data, mu, log_var)
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


logger = Logger(model_name='VAE', data_name='MNIST')
num_test_samples = 16

for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for n_batch, (real_batch,_) in enumerate(train_loader):
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        loss = train_VAE(v_optimizer,real_data)
        train_loss += loss.item()
        
        
        if n_batch % 100 == 0:
            display.clear_output(True)
            # Display Images
            z = torch.randn(num_test_samples, 2)
            sample = vae.decoder(z)
            test_images = vectors_to_images(sample).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, n_batch * len(real_batch), len(train_loader.dataset),
                    100. * n_batch / len(train_loader), loss.item() / len(real_batch)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data
            real_data = Variable(images_to_vectors(real_batch))
            recon, mu, log_var = vae(real_data)
            
            # sum up batch loss
            test_loss += loss_function(recon, real_data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

endtime = datetime.datetime.now()
print (endtime - starttime).seconds

































