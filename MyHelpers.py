# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:40:20 2020

@author: Sajol
"""
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import sys
import HelperFunctions as HelpFunc
import pandas as pd
import PIL
from itertools import product
import torch.utils.data
import torchvision
from sklearn.metrics import jaccard_score


#%%  
    
""" 1.    Configure Default Packages      """    
def find_current_directory():    
    print(" # Finding Courrent Directory")
    torch.backends.cudnn.benchmark=True
    plt.rc('font',family='Times New Roman')
    plt.rcParams.update({'figure.max_open_warning': 0})
    SMALL_SIZE=13
    SMALL_MEDIUM=14
    MEDIUM_SIZE=16
    BIG_MEDIUM=18
    BIG_SIZE=20
    plt.rc('font', size=SMALL_MEDIUM)          # controls default text sizes
    plt.rc('axes', titlesize=BIG_MEDIUM)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_MEDIUM)    # legend fontsize
    plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title
    
    # Create Training-Filepath
    global _ROOTFOLDER, _TRAINFOLDER, _RUNFOLDER
    _ROOTFOLDER=os.getcwd()
    print("_ROOTFOLDER-->  ", _ROOTFOLDER)
    
    # Create str
    if sys.platform=="win32":
        time_str=time.strftime("%d_%m_%y__%H_%M_%S\\")
        _TRAINFOLDER=_ROOTFOLDER+ "\\TrainResults\\"
    
    elif sys.platform=="linux":
        time_str=time.strftime("%y_%m_%d__%H_%M_%S/")
        _TRAINFOLDER=_ROOTFOLDER+"/TrainResults/"
        
    # Create Folder
    _RUNFOLDER=_TRAINFOLDER+time_str
    if not os.path.exists(_RUNFOLDER):
        os.makedirs(_RUNFOLDER)
    print("\n_RUNFOLDER-->  ", _RUNFOLDER)
    return _RUNFOLDER,time_str
    
    #####################################################
    ################## YOUR CODE BELOW ##################
    #####################################################
    
    # USE _RUNFOLDER TO STORE ALL DATA
    #####################################################
    


def imshow(img):
    print(" # Defining imshow function")
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
        
def get_device():
    print(" # Defining get_device function")
    if torch.cuda.is_available():
        _device = 'cuda'
    else:
        _device = 'cpu'
    return _device
        
def PlotList(x_list, y_list, plt_title_name,label_name, window_length=1, window_mul=5):
    """
    
    

    Parameters
    ----------
    x_list : list
    y_list : list
    plt_title_name : str
    label_name : str
    window_length : int, optional
        DESCRIPTION. The default is 1.
    window_mul : TYPEint optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    fig : figure of the list
    """

    fig=plt.figure(figsize=(8,4))
    plt.plot(x_list[:-window_length+1], HelpFunc.MovingAverage(y_list, window=window_length), label=label_name)   # main graph
    plt.plot(x_list[:-window_length*window_mul+1], HelpFunc.MovingAverage(y_list, window=window_length*window_mul), label='Moving Average' )
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")  #
    plt.grid(b=True, which="both", axis="both")
    xmax=x_list[:-window_length*window_mul+1][-1]
    plt.xlim(0, xmax)

    plt.title(plt_title_name)
    plt.tight_layout(rect=[0, 0.03, 1 , 0.95])
    return fig

def MinMaxScaler(x, minval=0, maxval=1):
    return (maxval-minval)*(x-x.min())/(x.max()-x.min()) + minval

def save_list_as_csv_file(x_list, y_list,_SUB_RUNFOLDER, file_name):
    
    #Zip the lists to create a list of tuples
    zippedList =  list(zip(x_list, y_list))
    data_frame = pd.DataFrame(zippedList, columns = [ 'Epoch', file_name]) 
    suffix=".csv"    
    path=os.path.join(_SUB_RUNFOLDER,file_name+suffix)
    data_frame.to_csv(path, index = False, header=True)
    
def save_all_dictonary_parameteres_in_a_text_file(_RUNFOLDER,parameters_dict,param_values):
    combination_sequence=0
    
    # save parameter dictionary  as a txt file
    completeName = os.path.join(_RUNFOLDER, "Parameters_Dictionary.txt")  
    if not os.path.isdir(_RUNFOLDER):
        os.makedirs(_RUNFOLDER)
    f = open(completeName, "a+") 
    f.write( str(parameters_dict) )
    f.write("\n\n")
    
    # saving all the combinations
    for Learning_Rate, Batch_Size,Optimizer in product(*param_values):
        combination_sequence+=1
        comb_list = ["Combination"+str(combination_sequence),Learning_Rate,Batch_Size,Optimizer]
        f.write( str(comb_list)[1:-1])
        f.write("\n")       
    f.close()
    
# =============================================================================
#               STL10 DataLoader
# =============================================================================
def GetDatasetPath(folder_name = "Datasets", max_depth = 10):
    """


    Parameters
    ----------
    folder_name : str, optional
        Folder name to look for, first occurence is returned. The default is "Datasets".
    max_depth : int, optional
        Maximum number of parental directories to search. The default is 10.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    Depth = 0
    CurrentDir = os.getcwd()

    while Depth < max_depth:
        if folder_name in os.listdir(CurrentDir):
            break
        CurrentDir = os.path.dirname(CurrentDir)
        Depth += 1
    return os.path.join(CurrentDir, folder_name)

def LoadSTL10(path="Datasets", transforms_train = False, transforms_test = False, minibatch=32, worker=0, normalization = "mean", resize=96):
    """
    Return training- and testing-dataloaders for the STL10 data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\\': will use the specified location to store the dataset.\n
        The default is "Datasets".
    transforms_train : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for training data set. Will use RandomCrop, RandomHorizontalFlop and Normalize if False.
    transforms_test : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for testing data set. Will use Normalize if False.
    minibatch : int, optional
        Number of Images per Minibatch. The default is 32.
    worker : int, optional
        Number of worker processes. The default is 0.
    normalization : str, optional
        Which normalization function to use.
        \t 'mean':\t\t standardize to zero mean and unit std.\n
        \t '-11':\t\t normalize to the range -1...1.\n
        \t otherwise:\t normalize to the range 0...1.\n
        The default is "mean".

    Returns
    -------
    train_loader :
        DataLoader for training data set.
    test_loader :
        DataLoader for testing data set.

    """
    if normalization == "mean":
        Normalize = torchvision.transforms.Normalize((0.4384, 0.4314, 0.3989), (0.2647, 0.2609, 0.2741))
    elif normalization == "-11":
        Normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        Normalize = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
    
    # setting up default tranformers
    DefaultTransformsTrain = torchvision.transforms.Compose([
                                               # torchvision.transforms.RandomCrop(96, padding=0),
                                               torchvision.transforms.Resize((resize,resize)),
                                               torchvision.transforms.RandomHorizontalFlip(),
                                               torchvision.transforms.RandomRotation(5),
                                               torchvision.transforms.ToTensor(),
                                               Normalize])
    DefaultTransformsTest = torchvision.transforms.Compose([
                                               torchvision.transforms.Resize((resize,resize)),
                                               torchvision.transforms.ToTensor(),
                                               Normalize])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "Datasets", "STL10")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "STL10")
        else:
            path = os.path.join(GetDatasetPath(path), "STL10")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))


    # use given transformers or default ones
    TransformsTrain = transforms_train if transforms_train else DefaultTransformsTrain
    TransformsTest = transforms_train if transforms_train else DefaultTransformsTest

    trainset = torchvision.datasets.STL10(root = path,  split="train",
                                            download=True, transform=TransformsTrain)
    testset = torchvision.datasets.STL10(root = path, split="test",
                                           download=True, transform=TransformsTest)

    # load data sets and create data loaders
    train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)

    test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)
    return train_loader, test_loader,trainset,testset

# =============================================================================
#                          Pascal VOC Segmentation Dataloader
# =============================================================================


def LoadPascalVOC_Segmentation(path="Datasets", transforms_train = False, transforms_test = False, minibatch=32, worker=0, normalization = "mean", resize=128):
    """
    Return training- and testing-dataloaders for the PascalVOC data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\\': will use the specified location to store the dataset.\n
        The default is "Datasets".
    transforms_train : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for training data set. Will use RandomCrop, RandomHorizontalFlop and Normalize if False.
    transforms_test : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for testing data set. Will use Normalize if False.
    minibatch : int, optional
        Number of Images per Minibatch. The default is 32.
    worker : int, optional
        Number of worker processes. The default is 0.
    normalization : str, optional
        Which normalization function to use.
        \t 'mean':\t\t standardize to zero mean and unit std.\n
        \t '-11':\t\t normalize to the range -1...1.\n
        \t otherwise:\t normalize to the range 0...1.\n
        The default is "mean".

    Returns
    -------
    train_loader :
        DataLoader for training data set.
    test_loader :
        DataLoader for testing data set.

    """
    if normalization == "mean":
        Normalize_3d = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        Normalize_1d = torchvision.transforms.Normalize((0.485), (0.229))
    elif normalization == "-11":
        Normalize_3d = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        Normalize_1d = torchvision.transforms.Normalize((0.5), (0.5))
    else:
        Normalize_3d = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
        Normalize_1d = torchvision.transforms.Normalize((0), (1))
    
    # setting up default tranformers
    DefaultTransformsTrain_3d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomRotation(5),
                                                # torchvision.transforms.ColorJitter(brightness=0.25),
                                                torchvision.transforms.ToTensor(),
                                                Normalize_3d,
                                               ])
    DefaultTransformsTest_3d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1 ),
                                                torchvision.transforms.ToTensor(),                                         
                                                Normalize_3d,
                                               ])
    DefaultTransformsTrain_1d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomRotation(5),
                                                # torchvision.transforms.ColorJitter(brightness=0.25),
                                                torchvision.transforms.ToTensor(),
                                                Normalize_1d,
                                               ])
    DefaultTransformsTest_1d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1 ),
                                                torchvision.transforms.ToTensor(),                                         
                                                Normalize_1d,
                                               ])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "Datasets", "PascalVOC")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "PascalVOC")
        else:
            path = os.path.join(GetDatasetPath(path), "PascalVOC")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))

    # use given transformers or default ones
    TransformsTrain_3d = transforms_train if transforms_train else DefaultTransformsTrain_3d
    TransformsTest_3d = transforms_train if transforms_train else DefaultTransformsTest_3d
    
    TransformsTrain_1d = transforms_train if transforms_train else DefaultTransformsTrain_1d
    TransformsTest_1d = transforms_train if transforms_train else DefaultTransformsTest_1d

    trainset = torchvision.datasets.VOCSegmentation(root = path,  year='2012',image_set="train", 
                                            download=True, transform=TransformsTrain_3d, target_transform=TransformsTrain_1d)
    testset = torchvision.datasets.VOCSegmentation(root = path, year='2012',image_set="val", 
                                            download=True, transform=TransformsTest_3d, target_transform=TransformsTest_1d)
    

    # load data sets and create data loaders  
    train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)

    test_loader = torch.utils.data.DataLoader(
           testset,
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)
    
    # del trainset1,testset1,x,input_train,input_test
    return train_loader, test_loader,trainset,testset    
    
   
# =============================================================================
#                  Pascal VOC Classification Datalaoder 2
# =============================================================================
def LoadPascalVOC_Classification2(path="Datasets", transforms_train = False, transforms_test = False, minibatch=32, worker=0, normalization = "mean", resize=128):
    """
    Return training- and testing-dataloaders for the PascalVOC data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\\': will use the specified location to store the dataset.\n
        The default is "Datasets".
    transforms_train : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for training data set. Will use RandomCrop, RandomHorizontalFlop and Normalize if False.
    transforms_test : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for testing data set. Will use Normalize if False.
    minibatch : int, optional
        Number of Images per Minibatch. The default is 32.
    worker : int, optional
        Number of worker processes. The default is 0.
    normalization : str, optional
        Which normalization function to use.
        \t 'mean':\t\t standardize to zero mean and unit std.\n
        \t '-11':\t\t normalize to the range -1...1.\n
        \t otherwise:\t normalize to the range 0...1.\n
        The default is "mean".

    Returns
    -------
    train_loader :
        DataLoader for training data set.
    test_loader :
        DataLoader for testing data set.

    """
    if normalization == "mean":
        Normalize_3d = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        Normalize_1d = torchvision.transforms.Normalize((0.485), (0.229))
    elif normalization == "-11":
        Normalize_3d = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        Normalize_1d = torchvision.transforms.Normalize((0.5), (0.5))
    else:
        Normalize_3d = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
        Normalize_1d = torchvision.transforms.Normalize((0), (1))
    
    # setting up default tranformers
    DefaultTransformsTrain_3d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                 # torchvision.transforms.ColorJitter(brightness=0.25), # Change brightness of image
                                                torchvision.transforms.RandomRotation(5),
                                                torchvision.transforms.ToTensor(),
                                                Normalize_3d,
                                               ])
    DefaultTransformsTest_3d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1 ),
                                                torchvision.transforms.ToTensor(),                                         
                                                Normalize_3d,
                                               ])
    DefaultTransformsTrain_1d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                 # torchvision.transforms.ColorJitter(brightness=0.25), # Change brightness of image
                                                torchvision.transforms.RandomRotation(5),
                                                torchvision.transforms.ToTensor(),
                                                Normalize_1d,
                                               ])
    DefaultTransformsTest_1d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1 ),
                                                torchvision.transforms.ToTensor(),                                         
                                                Normalize_1d,
                                               ])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "Datasets", "PascalVOC")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "PascalVOC")
        else:
            path = os.path.join(GetDatasetPath(path), "PascalVOC")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))

    # use given transformers or default ones
    TransformsTrain_3d = transforms_train if transforms_train else DefaultTransformsTrain_3d
    TransformsTest_3d = transforms_train if transforms_train else DefaultTransformsTest_3d
    
    TransformsTrain_1d = transforms_train if transforms_train else DefaultTransformsTrain_1d
    TransformsTest_1d = transforms_train if transforms_train else DefaultTransformsTest_1d

    trainset = torchvision.datasets.VOCDetection(root = path,  year='2012',image_set="train", 
                                            download=True, transform=TransformsTrain_3d, target_transform=None,transforms=None)#TransformsTrain_3d
    testset = torchvision.datasets.VOCDetection(root = path, year='2012',image_set="val", 
                                            download=True, transform=TransformsTest_3d, target_transform=None,transforms=None)


    # load data sets and create data loaders  
    train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker,collate_fn=lambda x:x, drop_last = True)  #,drop_last = True,collate_fn(minibatch)

    test_loader = torch.utils.data.DataLoader(
           testset,
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker,collate_fn=lambda x:x,drop_last = True)

    
    return train_loader, test_loader,trainset,testset    


   
# =============================================================================
#                            BSDS500 Dataloader
# =============================================================================


def LoadBSDS500(path="Datasets", transforms_train = False, transforms_test = False, minibatch=32, worker=0, normalization = "mean", resize=128):
    """
    Return training- and testing-dataloaders for the PascalVOC data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\\': will use the specified location to store the dataset.\n
        The default is "Datasets".
    transforms_train : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for training data set. Will use RandomCrop, RandomHorizontalFlop and Normalize if False.
    transforms_test : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for testing data set. Will use Normalize if False.
    minibatch : int, optional
        Number of Images per Minibatch. The default is 32.
    worker : int, optional
        Number of worker processes. The default is 0.
    normalization : str, optional
        Which normalization function to use.
        \t 'mean':\t\t standardize to zero mean and unit std.\n
        \t '-11':\t\t normalize to the range -1...1.\n
        \t otherwise:\t normalize to the range 0...1.\n
        The default is "mean".

    Returns
    -------
    train_loader :
        DataLoader for training data set.
    test_loader :
        DataLoader for testing data set.

    """
    if normalization == "mean":
        Normalize_3d = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        Normalize_1d = torchvision.transforms.Normalize((0.485), (0.229))
    elif normalization == "-11":
        Normalize_3d = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        Normalize_1d = torchvision.transforms.Normalize((0.5), (0.5))
    else:
        Normalize_3d = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
        Normalize_1d = torchvision.transforms.Normalize((0), (1))
    
    # setting up default tranformers
    DefaultTransformsTrain_3d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomRotation(5),
                                                # torchvision.transforms.ColorJitter(brightness=0.25),
                                                torchvision.transforms.ToTensor(),
                                                Normalize_3d,
                                               ])
    DefaultTransformsTest_3d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1 ),
                                                torchvision.transforms.ToTensor(),                                         
                                                Normalize_3d,
                                               ])
    DefaultTransformsTrain_1d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomRotation(5),
                                                # torchvision.transforms.ColorJitter(brightness=0.25),
                                                torchvision.transforms.ToTensor(),
                                                Normalize_1d,
                                               ])
    DefaultTransformsTest_1d = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((resize, resize)),
                                                # torchvision.transforms.CenterCrop(resize),
                                                # torchvision.transforms.RandomCrop((resize,resize), padding=0,pad_if_needed=1 ),
                                                torchvision.transforms.ToTensor(),                                         
                                                Normalize_1d,
                                               ])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "Datasets","BSR_bsds500")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "BSR_bsds500")
        else:
            path = os.path.join(GetDatasetPath(path), "BSR_bsds500")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))

    # use given transformers or default ones
    TransformsTrain_3d = transforms_train if transforms_train else DefaultTransformsTrain_3d
    TransformsTest_3d = transforms_train if transforms_train else DefaultTransformsTest_3d
    
    TransformsTrain_1d = transforms_train if transforms_train else DefaultTransformsTrain_1d
    TransformsTest_1d = transforms_train if transforms_train else DefaultTransformsTest_1d
    
    trainset = torchvision.datasets.ImageFolder(root =path, transform=TransformsTrain_3d)
                                            
    testset = torchvision.datasets.ImageFolder(root = path,  transform=TransformsTest_3d)
                                         
    

        
    # load data sets and create data loaders  
    train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)

    test_loader = torch.utils.data.DataLoader(
           testset,
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)
    
    # del trainset1,testset1,x,input_train,input_test
    return train_loader, test_loader,trainset,testset 
# =============================================================================
#    
# =============================================================================
def get_images_and_labels_from_voc_detection_loader(data):
    """
    This function extracts class labels from Pascal VOC Detection dataloader

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    images : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.

    """
    Names = ['dog', 'person','boat', 'bottle','diningtable','car','sheep','motorbike','bus',
     'bird','bicycle','horse','chair','train', 'aeroplane','cat','pottedplant','cow','sofa',  
     'tvmonitor']
    image_list=[]
    for i in range(len(data)):
            image_p=data[i][0]
            # trainlabel_p=traindata[i][1]['annotation']['object'][0]['name']
            image_list.append(image_p)

    all_class_names_in_a_batch = [data[i][1]["annotation"]["object"][0]["name"] for i in range(len(data))]
    labels_of_those_images=[Names.index(i) for i in all_class_names_in_a_batch]   # Names.index('dog')
    
    labels = torch.FloatTensor(labels_of_those_images)
    images = torch.stack(image_list)
    
    return images,labels


    
    
    
    
    
    
    
    
    
    
    
