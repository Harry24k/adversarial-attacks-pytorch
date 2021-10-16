import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,5), # 16*24*24
            nn.ReLU(),
            nn.Conv2d(16,32,5), # 32*20*20
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32*10*10
            nn.Conv2d(32,64,5), # 64*6*6
            nn.ReLU(),
            nn.MaxPool2d(2,2) #64*3*3
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )       
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(-1,64*3*3)
        out = self.fc_layer(out)

        return out

class Holdout(nn.Module):
    def __init__(self):
        super(Holdout, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 5 * 5, 100),
            nn.ReLU(),
            nn.Linear(100, 10)              
        )
        
    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(-1, 64*5*5)
        out = self.fc_layer(out)
        
        return out
    
class Target(nn.Module):
    def __init__(self):
        super(Target, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3,96,3), # 96*30*30
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Dropout2d(0.2),
            
            nn.Conv2d(96, 96, 3), # 96*28*28
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Conv2d(96, 96, 3), # 96*26*26
            nn.GroupNorm(32, 96),
            nn.ELU(),
            
            nn.Dropout2d(0.5),
            
            nn.Conv2d(96, 192, 3), # 192*24*24
            nn.GroupNorm(32, 192),
            nn.ELU(),
            
            nn.Conv2d(192, 192, 3), # 192*22*22
            nn.GroupNorm(32, 192),
            nn.ELU(),
           
            nn.Dropout2d(0.5),
            
            nn.Conv2d(192, 256, 3), # 256*20*20
            nn.GroupNorm(32, 256),
            nn.ELU(),
            
            nn.Conv2d(256, 256, 1), # 256*20*20
            nn.GroupNorm(32, 256),
            nn.ELU(),
            
            nn.Conv2d(256, 10, 1), # 10*20*20
            nn.AvgPool2d(20) # 10*1*1
        )

    def forward(self,x):
        out = self.conv_layer(x)
        out = out.view(-1,10)

        return out