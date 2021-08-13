import torch.nn as nn
import torchvision
from torchvision import models

class Segmentation_Network(nn.Module):
    """
    

    Parameters
    ----------
    Network_Output_Channel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    def __init__(self, Network_Output_Channel):
        super(Segmentation_Network, self).__init__()       
        self.seq= nn.Sequential(
            
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1),       
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1),       
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1,padding=0),       
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(in_channels=64, out_channels=Network_Output_Channel, kernel_size=1, stride=1,padding=0),       
        nn.BatchNorm2d(Network_Output_Channel),
        nn.ReLU(), 
        

        )
        self.initialize_weights()
                       
    def forward(self, x):       
        return self.seq(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

class Classification_Network(nn.Module):
    def __init__(self , Network_Output_Channel):       
        super(Classification_Network, self).__init__()       
        self.conv1 = nn.Conv2d(in_channels=Network_Output_Channel, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.batchnorm2 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool  = nn.MaxPool2d(kernel_size=2)
        
        self.batchnorm3 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.batchnorm4 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout2d(0.2)
        self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.batchnorm2(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.pool(x)
        
        x= self.batchnorm3(x)
        x = self.conv3(x)
        x = self.relu3(x)

        x = self.batchnorm4(x)
        x = self.conv4(x)
        x = self.relu4(x)

        x= self.dropout(x)
        x = x.view(-1, 16 * 16 * 24)

        x = self.fc(x)
        return x
# =============================================================================
#                             STL10 Classification
# =============================================================================
class Segmentation_Network_STL(nn.Module):
    """
    

    Parameters
    ----------
    Network_Output_Channel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    def __init__(self, Network_Output_Channel):
        super(Segmentation_Network_STL, self).__init__()       
        self.seq= nn.Sequential(
            
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1),       
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1),       
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        # nn.Dropout2d(0.2),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=1),       
        nn.BatchNorm2d(256),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=256, out_channels=Network_Output_Channel, kernel_size=1, stride=1,padding=0),       
        nn.BatchNorm2d(Network_Output_Channel),
        nn.ReLU(), 
        

        )
        self.initialize_weights()
                       
    def forward(self, x):       
        return self.seq(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
 
    
class Classification_Network_STL(nn.Module):
    def __init__(self , Network_Output_Channel):       
        super(Classification_Network_STL, self).__init__()       
        self.conv1 = nn.Conv2d(in_channels=Network_Output_Channel, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.batchnorm2 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool  = nn.MaxPool2d(kernel_size=2)
        
        self.batchnorm3 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.batchnorm4 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout2d(0.2)
        self.fc = nn.Linear(in_features=48 * 48 * 24, out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.batchnorm2(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.pool(x)
        
        x= self.batchnorm3(x)
        x = self.conv3(x)
        x = self.relu3(x)

        x = self.batchnorm4(x)
        x = self.conv4(x)
        x = self.relu4(x)

        x= self.dropout(x)
        x = x.view(-1, 48 * 48 * 24)

        x = self.fc(x)
        return x
class Classification_Network2_STL(nn.Module):
    """CNN."""

    def __init__(self,Network_Output_Channel):
        """CNN Builder."""
        super(Classification_Network2_STL, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=Network_Output_Channel, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Dropout2d(p=0.05),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            
            nn.Dropout(p=0.1),
            nn.Linear(36864, 20),
    
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x
# =============================================================================
#                               PascalVOC Claffification
# =============================================================================
class Segmentation_Network_Pascal(nn.Module):
    """
    

    Parameters
    ----------
    Network_Output_Channel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    def __init__(self, Network_Output_Channel):
        super(Segmentation_Network_Pascal, self).__init__()       
        self.seq= nn.Sequential(
            
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1),       
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        # nn.Dropout2d(0.05),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1),       
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        # nn.Dropout2d(0.05),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=1),       
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.Conv2d(in_channels=256, out_channels=Network_Output_Channel, kernel_size=1, stride=1,padding=0),       
        nn.BatchNorm2d(Network_Output_Channel),
        nn.ReLU(), 
        
        )
        self.initialize_weights()
                       
    def forward(self, x):       
        return self.seq(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

class Classification_Network_Pascal(nn.Module):
    def __init__(self , Network_Output_Channel):       
        super(Classification_Network_Pascal, self).__init__()       
        self.conv1 = nn.Conv2d(in_channels=Network_Output_Channel, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.batchnorm2 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool  = nn.MaxPool2d(kernel_size=2)
        
        self.batchnorm3 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.batchnorm4 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout2d(0.2)
        self.fc = nn.Linear(in_features=64*64 *24, out_features=20)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.batchnorm2(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.pool(x)
        
        x= self.batchnorm3(x)
        x = self.conv3(x)
        x = self.relu3(x)

        x = self.batchnorm4(x)
        x = self.conv4(x)
        x = self.relu4(x)

        x= self.dropout(x)
        x = x.view(-1, 64*64*24 )

        x = self.fc(x)
        return x
    
class Classification_Network2_Pascal(nn.Module):
    def __init__(self , Network_Output_Channel):       
        super(Classification_Network2_Pascal, self).__init__()  
        
        self.conv1 = nn.Conv2d(in_channels=Network_Output_Channel, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.dropout1 = nn.Dropout2d(0.05)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        self.dropout2 = nn.Dropout2d(0.05)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(3)
        self.relu3 = nn.ReLU()
        
        self.dropout3 = nn.Dropout2d(0.05)
        
        
        # calling resnet50
        self.model=models.resnet50(pretrained=True)
        self.num_ftrs=self.model.fc.in_features        
        self.model.fc=nn.Linear(in_features=self.num_ftrs, out_features=20)

        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu1(x)
        
        x = self.dropout3(x)
   
        x = self.model(x)

        return x


# =============================================================================
# 
# =============================================================================

class Autoencoder_NetworkFCN(nn.Module):
    def __init__(self, Network_Output_Channel):
        super(Autoencoder_NetworkFCN, self).__init__()       
        self.seq1= nn.Sequential(
            
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),       
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout2d(0.25),
        
        
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),       
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=64, out_channels=Network_Output_Channel, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(Network_Output_Channel),
        nn.ReLU(),
          ) 
        
        #------#
        self.seq2= nn.Sequential(                    
        # nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),       
        # nn.BatchNorm2d(3),
        # nn.ReLU(),
        # # # #     
        nn.Conv2d(in_channels=32, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(96),
        nn.ReLU(),
             
             
        # # # # 
        nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1),       
        # nn.BatchNorm2d(3),
        nn.ReLU(),
        )
        self.initialize_weights()
                       
    def forward(self, x):
        
        x=self.seq1(x)
        x1 = x
        x = self.seq2(x)        
        return x1,x 
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)        
                    
            elif isinstance(m, nn.BatchNorm2d):                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                             
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

# =============================================================================
# 
# =============================================================================


    
                

                
# =============================================================================
#              AN  FCN    
# =============================================================================
class Autoencoder_NetworkFCN2(nn.Module):
    def __init__(self, Network_Output_Channel):
        super(Autoencoder_NetworkFCN2, self).__init__() 
        
        self.seq1= nn.Sequential(
            
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),       
        nn.BatchNorm2d(64),
        nn.ReLU(),
        # nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),       
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        # nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(64),
        nn.ReLU(),
        # nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=64, out_channels=Network_Output_Channel, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(Network_Output_Channel),
        # nn.ReLU(),
          ) 

        #------#
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)       
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=Network_Output_Channel, kernel_size=1, stride=1, padding=0)       
        self.batchnorm3 = nn.BatchNorm2d(Network_Output_Channel)
        self.relu3 = nn.ReLU()
        
        #------#
        self.seq2= nn.Sequential(          
        
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(3),
        nn.ReLU(),
        # # # # encoder    
        nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(96),
        nn.ReLU(),
             
             
        # # # # decoder
        nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1),       
        # nn.BatchNorm2d(3),
        nn.ReLU(),
        )

        self.initialize_weights()
                       
    def forward(self, x):
        
        x=self.seq1(x)
        x1 = x
        # x1 = self.conv2(x1)
        # x1 = self.batchnorm2(x1)
        # x1 = self.relu2(x1)
        
        # x1 = self.conv3(x1)
        # x1 = self.batchnorm3(x1)
        # x1 = self.relu3(x1)
        
        x = self.seq2(x)        
        return x1,x 
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)        
                    
            elif isinstance(m, nn.BatchNorm2d):                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                             
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
                
# =============================================================================
#                 Sigmoid  nn.Sigmoid()
# =============================================================================
    
class Autoencoder_NetworkFCN_CIFAR(nn.Module):
    def __init__(self, Network_Output_Channel, activation_function= nn.ReLU):
        super(Autoencoder_NetworkFCN_CIFAR, self).__init__()       
        self.seq1= nn.Sequential(
            
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),       
        nn.BatchNorm2d(64),
        activation_function(),
        
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(32),
        activation_function(),

        
        
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),       
        nn.BatchNorm2d(64),
        activation_function(),
        
        nn.Conv2d(in_channels=64, out_channels=Network_Output_Channel, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(Network_Output_Channel),
        activation_function(),
          ) 
        
        #------#
        self.seq2= nn.Sequential(                    
        # nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),       
        # nn.BatchNorm2d(3),
        # nn.ReLU(),
        # # # #     
        nn.Conv2d(in_channels=32, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        activation_function(),
        
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        activation_function(),
        
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        activation_function(),
        
        nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(96),
        activation_function(),
             
             
        # # # # 
        nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        activation_function(),
        
        nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        activation_function(),
        
        nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        activation_function(),
        
        nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1),       
        # nn.BatchNorm2d(3),
        nn.Sigmoid(),
        )
        self.initialize_weights_kaiming()
                       
    def forward(self, x):
        
        x=self.seq1(x)
        x1 = x
        x = self.seq2(x)        
        return x1,x 
    
    def initialize_weights_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)        
                    
            elif isinstance(m, nn.BatchNorm2d):                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                             
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
        
    def initialize_weights_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)        
                    
            elif isinstance(m, nn.BatchNorm2d):                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                             
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias,0)
                
# =============================================================================
#               STL10                 
# =============================================================================
class Autoencoder_NetworkFCN_STL10(nn.Module):
    def __init__(self, Network_Output_Channel):
        super(Autoencoder_NetworkFCN_STL10, self).__init__()       
        self.seq1= nn.Sequential(
            
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),       
        nn.BatchNorm2d(64),
        nn.ReLU(),
   
        
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(32),
        nn.ReLU(),
       
          ) 
        
        #------#
        self.seq2= nn.Sequential(                    
    
        nn.Conv2d(in_channels=32, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(96),
        nn.ReLU(),
             
             
        # # # # 
        nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1),       
        # nn.BatchNorm2d(3),
        nn.Sigmoid(),
        )
        self.initialize_weights()
                       
    def forward(self, x):
        
        x=self.seq1(x)
        x1 = x
        x = self.seq2(x)          
        return x1,x 
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)        
                    
            elif isinstance(m, nn.BatchNorm2d):                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                             
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
                

# =============================================================================
#                 PascalVOC
# =============================================================================
class Autoencoder_NetworkFCN_PascalVOC(nn.Module):
    def __init__(self, Network_Output_Channel):
        super(Autoencoder_NetworkFCN_PascalVOC, self).__init__()       
        self.seq1= nn.Sequential(
            
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  #kernel_size=3, stride=1, padding=1),     
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # kernel_size=3, stride=1, padding=1),    
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),    #kernel_size=3, stride=1, padding=1),   
        nn.BatchNorm2d(256),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=256, out_channels=96, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(96),
        nn.ReLU(),
        
        # nn.Conv2d(in_channels=128, out_channels=96, kernel_size=1, stride=1, padding=0),       
        # nn.BatchNorm2d(96),
        # nn.ReLU(),
        # nn.Dropout2d(0.25),
        
        
        # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),       
        # nn.BatchNorm2d(64),
        # nn.ReLU(),
        
        # nn.Conv2d(in_channels=64, out_channels=Network_Output_Channel, kernel_size=1, stride=1, padding=0),       
        # nn.BatchNorm2d(Network_Output_Channel),
        # nn.ReLU(),
          ) 
        
        #------#
        self.seq2= nn.Sequential(                    
        # nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),       
        # nn.BatchNorm2d(3),
        # nn.ReLU(),
        # # # #     
        nn.Conv2d(in_channels=96, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(96),
        nn.ReLU(),
             
             
        # # # # 
        nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(24),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1),       
        nn.BatchNorm2d(12),
        nn.ReLU(),
        
        nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1),       
        # nn.BatchNorm2d(3),
        nn.Sigmoid(),
        )
        self.initialize_weights()
                       
    def forward(self, x):
        
        x=self.seq1(x)
        x1 = x
        x = self.seq2(x)        
        return x1,x 
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)        
                    
            elif isinstance(m, nn.BatchNorm2d):                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                             
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
                
# =============================================================================
#             Pascal with Skip Connection
# =============================================================================
class Autoencoder_NetworkFCN_PascalVOC_with_skip(nn.Module):
    def __init__(self, Network_Output_Channel):
        super(Autoencoder_NetworkFCN_PascalVOC_with_skip, self).__init__()       
        self.seq1= nn.Sequential(
         
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  #kernel_size=3, stride=1, padding=1),     
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # kernel_size=3, stride=1, padding=1),    
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),    #kernel_size=3, stride=1, padding=1),   
        nn.BatchNorm2d(256),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=256, out_channels=96, kernel_size=1, stride=1, padding=0),       
        nn.BatchNorm2d(96),
        nn.ReLU(),
        
        # nn.Conv2d(in_channels=128, out_channels=96, kernel_size=1, stride=1, padding=0),       
        # nn.BatchNorm2d(96),
        # nn.ReLU(),
        # nn.Dropout2d(0.25),
        
        
        # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),       
        # nn.BatchNorm2d(64),
        # nn.ReLU(),
        
        # nn.Conv2d(in_channels=64, out_channels=Network_Output_Channel, kernel_size=1, stride=1, padding=0),       
        # nn.BatchNorm2d(Network_Output_Channel),
        # nn.ReLU(),
          )  
        #------------------
        # self.convS = nn.Conv2d(in_channels=96, out_channels=3, kernel_size=1, stride=1, padding=0)       
        # self.batchnormS = nn.BatchNorm2d(3)
        # self.reluS = nn.ReLU()
        
        #------#
                      
        # nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0),       
        # nn.BatchNorm2d(3),
        # nn.ReLU(),
        # # # #     
        self.conv1a  = nn.Conv2d(in_channels=96, out_channels=12, kernel_size=4, stride=2, padding=1)       
        self.batchnorm1a  = nn.BatchNorm2d(12)
        self.relu1a  = nn.ReLU() #1a
        
        self.conv2a  = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1)       
        self.batchnorm2a  = nn.BatchNorm2d(24)
        self.relu2a  = nn.ReLU() #2a
        
        self.conv3a  = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1)       
        self.batchnorm3a  = nn.BatchNorm2d(48)
        self.relu3a  = nn.ReLU() #3a
        
        self.conv4a  =     nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=1)       
        self.batchnorm4a  = nn.BatchNorm2d(96)
        self.relu4a  = nn.ReLU() #4a
             
             
        # # # # 
        self.conv4b  = nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1)       
        self.batchnorm4b  = nn.BatchNorm2d(48)
        self.relu4b  = nn.ReLU()  #4b
        
        self.conv3b  = nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1)       
        self.batchnorm3b  = nn.BatchNorm2d(24)
        self.relu3b  = nn.ReLU()  #3b
        
        self.conv2b  = nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1)       
        self.batchnorm2b = nn.BatchNorm2d(12)
        self.relu2b  = nn.ReLU()#2b
        
        self.conv1b  = nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1)       
        # nn.BatchNorm2d(3),
        self.sig1b  = nn.Sigmoid() #1b
        
        self.initialize_weights()
                       
    def forward(self, x):
        
        x=self.seq1(x)
        
        x1=self.conv1a(x)
        x1=self.batchnorm1a(x1)
        x1=self.relu1a(x1)
        
        
        x2=self.conv2a(x1)
        x2=self.batchnorm2a(x2)
        x2=self.relu2a(x2)
        
        x3=self.conv3a(x2)
        x3=self.batchnorm3a(x3)
        x3=self.relu3a(x3)
        
        x4=self.conv4a(x3)
        x4=self.batchnorm4a(x4)
        x4=self.relu4a(x4)
        
        x5=self.conv4b(x4)
        x5=self.batchnorm4b(x5)
        x5=self.relu4b(x5)
        
        # skip connection 1
        xsk1= x3 + x5 
        xsk1=self.batchnorm4b(xsk1)
        xsk1=self.relu4b(xsk1)
        
        x6=self.conv3b(xsk1)
        x6=self.batchnorm3b(x6)
        x6=self.relu3b(x6)
        
        # skip connection 2
        xsk2= x2+ x6 
        xsk2=self.batchnorm3b(xsk2)
        xsk2=self.relu3b(xsk2)
    
        x7=self.conv2b(xsk2)
        x7=self.batchnorm2b(x7)
        x7=self.relu2b(x7)
           
        # skip connection 3
        xsk3= x1+x7
        xsk3=self.batchnorm1a(xsk3)
        xsk3=self.relu1a(xsk3)
     
        x8 = self.conv1b(xsk3)
        x8=self.sig1b(x8)
    

        return x,x8 
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)       
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)        
                    
            elif isinstance(m, nn.BatchNorm2d):                
                 nn.init.constant_(m.weight,1)
                 nn.init.constant_(m.bias,0)
                             
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
                
                
