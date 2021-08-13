"""This is the main code file to do segmentation with classication extension """
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import os.path
import dill
import gc
import HelperFunctions as HelpFunc
import MyHelpers as MyHelpFunc
import Classes as ClassLib
from itertools import product
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from skimage.segmentation import slic
from torchvision.utils import make_grid
from pathlib import Path
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

#%% Global variables

HelpFunc.ConfigurePlt()
current_filename=Path(__file__)
# plt.ioff()

#Default_Dataset available "s"=> STL10 dataset OR "c"=> CIFAR10 dataset OR "p"=> PascalVOC dataset OR None
Default_Dataset="c"    # type "s", "c" , "p" or None    
Num_Epochs = 30   # set minimum 20
Normalization= "minmax" # "-11 " 
Resize_PascalVoc_Image= 224  #128  # size should be a multiplier of 16
Resize_STL10_Image=96

Network_Output_Channel=96 # good also at  128, select 96 for STL10/ Pascal dataset, and 128 for CIFAR10 

Activation_Function=nn.ReLU

Ploting_Window_Length=5
Ploting_Window_Multiplier=20
Number_of_Images_To_Save = 10


if  Default_Dataset == "c":
    Testing_Dataset_Every_Batches=500
    Show_Segmentation_Masks_Every= 250
elif Default_Dataset =="p" or Default_Dataset == "s":
    Testing_Dataset_Every_Batches=200
    Show_Segmentation_Masks_Every= 250
        
##----SLIC Parameters----##
Number_Of_Segments = 7   # should be 6 to 11
Compactness=.0001   # should be less
Max_Iter=100
Sigma=1.1#3.5
s1=1.1
s2=1.1
s3=1.1
Spacing=(s1,s2,s3)

Multichannel=False
Convert2lab=False
Enforce_Connectivity=False # imp should be always false

Min_Size_Factor=0.25 #.25   # should be less
Max_Size_Factor=3  #should be  3 /4,
Slic_Zero=False # imp should be always false
Start_Label=1  # imp
Mask=None
##--------##
Training_Images= "Training Set"
Testing_Images= "Testing Set"


#%% Functions and Classes defintion

def get_device():
    """
    This Function returns the usable cuda device if available.

    Returns
    -------
    _device : str
        default device is cpu but if gpu is available then use cuda.

    """
    if torch.cuda.is_available():
        _device = 'cuda'
    else:
        _device = 'cpu'
    return _device
Device = get_device() 

def get_sample_segmentation_masks(imgs, net, batch, epoch, images_type,combination,Learning_Rate,Batch_Size,optimizer):
    """ This functions takes some sample images to observe the performance of segmentation.
    , images are fed into the network, then it reconstructs the images and apply argmax for segmentation,
    finally it implies SLIC refinement on argmax output. 
    the output 
    

    Parameters
    ----------
    imgs :  torch.Tensor
        some sample images to observe segmentation.
    net :  Classes.Any_Network_Used        
    batch : int        
    epoch : int
    images_type: string
    combination : int
    Learning_Rate: float
    Batch Size : int
    Optimizer: string
    
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        a figure of original image, reconstructed image, segmented mast and SLIC output.

    """
    net.eval()  
    Mask = net(imgs).argmax(dim=1)
    Slic_Mask= slic_refinement_of_every_batchimages(Mask.cpu(), Number_Of_Segments )
    net.train()
    
    fig=plt.figure(figsize=(imgs.shape[0]*3*0.5,10*0.6))   #  (imgs.shape[0]*3*0.5,10*0.6))  
    plt.suptitle(" %s : Original Image and Segmentation Mask and Slic Output \n for Running Batch=%i, Epoch=%i,Combi=%i, Lr=%f, Bs=%i, Op=%s" %(images_type,
                                                               batch,epoch,combination,Learning_Rate,Batch_Size,optimizer),  fontname='Times New Roman',fontsize=20)
    
    for col in range(imgs.shape[0]):
        for row in range(3):
            
            ax=plt.subplot(4, imgs.shape[0], row*imgs.shape[0] + col +1)
            
            if row==0:
                img = MyHelpFunc.MinMaxScaler(imgs[col].permute(1,2,0).detach().cpu())
                title = "original"
            elif row==1:
                img = Mask[col].detach().cpu()
                title = "argmax output"
            elif row==2:
                img = Slic_Mask[col].detach().cpu()
                title = "slic output"
                       
            plt.imshow(img.squeeze())
            ax.set_title(title,fontsize=12)
    plt.tight_layout(rect=[0, 0, 1 , 0.85],pad=0,h_pad=0, w_pad=0)
    plt.show()
    del net,Mask,Slic_Mask,
    return fig

def slic_refinement_of_every_batchimages(batch_images,
                                        N_Segments = 10,):
    """
    

    Parameters
    ----------
    batch_images : TYPE
        Input image, which can be 2D or 3D,.
    N_Segments : int, optional
        The (approximate) number of labels in the segmented output image
        . The default is 100.
    Compactness : float, optional
        Balances color proximity and space proximity. . The default is 10.
    Max_Iter : int, optional
        should be less than something else for good performance. The default is 10.
    Sigma : float, optional
        Width of Gaussian smoothing kernel for pre-processing for each dimension of the image. The default is 0.
    Spacing : tuple, optional
        DESCRIThe voxel spacing along each image dimension.PTION. The default is None.
    Multichannel : bool, optional
        DESCRIPTWhether the last axis of the image is to be interpreted as multiple channels or another spatial dimension.ION. The default is True.
    Convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to segmentation.. The default is None.
    Enforce_Connectivity : bool, optional
        Whether the generated segments are connected or not. The default is True.
    Min_Size_Factor : float, optional
        DESCRIPProportion of the minimum segment size to be removed with respect to the supposed segment size `depth*width*height/n_segments`TION. The default is 0,5.
    Max_Size_Factor : int, optional
        Proportion of the maximum connected segment size. A value of 3 works in most of the cases.. The default is 3
    Slic_Zero : bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. 2. The default is False.
    Start_Label : int, optional
        DESCRIThe labels' index start. Should be 0 or 1.PTION. The default is None.
    Mask : Tensor, optional
        If provided, superpixels are computed only where mask is True. The default is None.

    Returns
    -------
    image tensor.

    """
    batch_images.cpu()
    segmented_batch_list=[]  
    for eachimage in batch_images:
        input_image_slic = (eachimage.unsqueeze(2)*1.0).numpy() 
        segmented_eachimage=slic(input_image_slic, 
                                  n_segments=Number_Of_Segments, 
                                  compactness=Compactness,   
                                  max_iter=Max_Iter, 
                                  sigma=Sigma,
                                  spacing=Spacing,
                                  multichannel=Multichannel,
                                  convert2lab=Convert2lab,
                                  enforce_connectivity=Enforce_Connectivity, 
                                  min_size_factor=Min_Size_Factor, 
                                  max_size_factor=Max_Size_Factor,   
                                  slic_zero=Slic_Zero,
                                  start_label=Start_Label,  
                                  mask=Mask)       
        segmented_batch_list.append(segmented_eachimage)   
    segmented_batch= np.array(segmented_batch_list)
    segmented_batch_tensor= torch.from_numpy(segmented_batch).to(get_device())
    del input_image_slic,segmented_eachimage,segmented_batch,batch_images,segmented_batch_list
    return segmented_batch_tensor


#%% Data Preprocessing

def data_loader(called_dataset,Batch_Size=32):
    """
    Three datasets can be loaded with this dataloader. You have to chose any one 
    at a time.
    

    Parameters
    ----------
    called_dataset : string
        DESCRIPTION.
    Batch_Size : int, optional
        DESCRIPTION. The default is 32.

    Returns
    -------
    trainloader : class 'torch.utils.data.dataloader.DataLoader' 
        DESCRIPTION: loads the traindata on pytorch
    testloader : class 'torch.utils.data.dataloader.DataLoader'
        DESCRIPTION: loads the testdata on pytorch
    trainset : <class 'torchvision.datasets.cifar.CIFAR10'>
        DESCRIPTION : training dataset
    testset : <class 'torchvision.datasets.cifar.CIFAR10'>
        DESCRIPTION: testing dataset

    """


    if called_dataset=="PascalVOC":
        trainloader, testloader,trainset,testset= MyHelpFunc.LoadPascalVOC_Classification2(path="Datasets",
                                                                minibatch=Batch_Size ,
                                                                normalization = Normalization,
                                                                resize=Resize_PascalVoc_Image)
        print("\n\nPascalVOC Dataset loaded...")

    elif called_dataset=="STL10":
        trainloader, testloader,trainset,testset = MyHelpFunc.LoadSTL10(path="Datasets",
                                                                minibatch=Batch_Size ,
                                                                normalization = Normalization,
                                                                resize=Resize_STL10_Image)
        print("\n\nSTL10 Dataset loaded...")
    else:
        trainloader, testloader,trainset,testset = HelpFunc.LoadCifar10(path="Datasets",
                                                                minibatch=Batch_Size ,
                                                                normalization = Normalization)
        print("\n\nCIFAR10 Dataset loaded...")
           
    return trainloader, testloader,trainset,testset 



#%% Criterion and Optimizer

def select_some_specific_images_from_dataloader(dataloader,called_dataset):
    
    """Some specific Images that we want to save to observe segmentation 
    over several batches
    
    Returns
    -------
    images_to_show_n_save : Tensor
        A set of specific images for observation.

    """
    if called_dataset=="PascalVOC":
        image_size=Resize_PascalVoc_Image
        images_to_show_n_save = torch.zeros((0,3,image_size,image_size), device=Device) 
        for _, sample_data in enumerate(dataloader): 
            Data,Label=MyHelpFunc.get_images_and_labels_from_voc_detection_loader(sample_data)
        
            Data = Data.to(Device)   
            images_to_show_n_save = torch.cat((images_to_show_n_save, Data[0].unsqueeze(0)))   # len(Data[0])=3, 
            if images_to_show_n_save.shape[0] == Number_of_Images_To_Save:
                break

    elif called_dataset=="STL10" or called_dataset=="CIFAR10" :
        image_size=Resize_STL10_Image if called_dataset=="STL10" else 32
        images_to_show_n_save = torch.zeros(0,3,image_size,image_size, device=Device) 
        for _, (Data, Label) in enumerate(dataloader):
            Data = Data.to(Device)   
            images_to_show_n_save = torch.cat((images_to_show_n_save, Data[0].unsqueeze(0)))   # len(Data[0])=3, 
            if images_to_show_n_save.shape[0] == Number_of_Images_To_Save:
                break
                  
    del Data
    return images_to_show_n_save

#%% Saving Parameters
def save_all_parameters_as_txt_file(Learning_Rate,Batch_Size, _SUB_RUNFOLDER,combination,time_str,seg_network,cls_network,optimizer_s,optimizer_c,criterion,scheduler_c,scheduler_s,called_dataset):
    """
    This function saves all the important parameters in a text file.

    Parameters
    ----------
    Learning_Rate : float
    Batch_Size  : int   
    _SUB_RUNFOLDER : str   
    combination : int  
    time_str : str
    seg_network : segmentation network class        
    cls_network : classification network class   
    optimizer_s : class torch.optimizer     
    optimizer_c : class torch.optimizer   
    criterion   : class torch.nn.modules.loss  
    scheduler_c : class torch.optim.lr_scheduler   
    scheduler_s : class torch.optim.lr_scheduler      
    called_dataset : str
   

    Returns
    -------
    A text file that saves all the important information related to this run

    """
    ParamFile=HelpFunc.ParameterStorage(_SUB_RUNFOLDER)   # giving directory to write files
    
    ParamFile.WriteTab("time_str ",time_str)
    ParamFile.WriteTab("combination ",combination)
    ParamFile.WriteTab("current py filename_location: ",current_filename)
    ParamFile.WriteTab("DataSet used in this run = ",called_dataset)
    ParamFile.Write("\n\nHyper Parameters:\n")
    ParamFile.WriteTab("Network_Output_Channel",Network_Output_Channel)
    ParamFile.WriteTab("Learning_Rate", Learning_Rate)
    ParamFile.WriteTab("batch_size", Batch_Size )
    ParamFile.WriteTab("epochs", Num_Epochs)   
    ParamFile.WriteTab("optimizer_c:", optimizer_c)
    ParamFile.WriteTab("optimizer_s:", optimizer_s)
    ParamFile.WriteTab("criterion :", criterion)

    ParamFile.WriteTab("Normalization :", Normalization)
    ParamFile.WriteTab("Resize_PascalVoc_Image:", Resize_PascalVoc_Image)
    ParamFile.WriteTab("Resize_STL10_Image :", Resize_STL10_Image)

    ParamFile.Write("Initializer =xavier_uniform_")
    ParamFile.Write("scheduler_c = StepLR(optimizer_c,step_size=30,gamma=0.1)")
    ParamFile.Write("scheduler_s = StepLR(optimizer_s,step_size=30,gamma=0.1)")
    # ParamFile.Write("scheduler = ReduceLROnPlateau(optimizer,patience=5)")
    ParamFile.WriteTab("\n\seg_network :   \n \n", seg_network)
    ParamFile.WriteTab("\n\cls_network :   \n \n", cls_network)

    ParamFile.Write("#-------------##-------------# \n\n")
    ParamFile.Write("Slic Parameters:\n")
    ParamFile.WriteTab("Number_Of_Segments",Number_Of_Segments)
    ParamFile.WriteTab("Compactness",Compactness)
    ParamFile.WriteTab("Max_Iter",Max_Iter)
    ParamFile.WriteTab("Sigma",Sigma)
    ParamFile.Write("Spacing")
    ParamFile.WriteTab("s1",s1)
    ParamFile.WriteTab("s2",s2)
    ParamFile.WriteTab("s3",s3)
    ParamFile.WriteTab("Multichannel",Multichannel)
    ParamFile.WriteTab("Convert2lab",Convert2lab)
    ParamFile.WriteTab("Enforce_Connectivity",Enforce_Connectivity)
    ParamFile.WriteTab("Min_Size_Factor",Min_Size_Factor)
    ParamFile.WriteTab("Max_Size_Factor",Max_Size_Factor)
    ParamFile.WriteTab("Slic_Zero", Slic_Zero)
    ParamFile.WriteTab("Start_Label", Start_Label)
    ParamFile.WriteTab("Mask",Mask)
    
    ParamFile.Write("#-------------##-------------# \n\n")
    ParamFile.WriteTab("Ploting Window Length", Ploting_Window_Length)
    ParamFile.WriteTab("Ploting Window Multiplier", Ploting_Window_Multiplier)

    
#%% Ploting and Saving Figures

def plot_and_save_loss_accuracy_list(Figure_Storage_Instance,Dynamic_Reporter,_SUB_RUNFOLDER,combination,time_str,epoch): 
    """
    Saving PlotTrainLoss & PlotTestLoss Figures

    Parameters
    ----------
    Figure_Storage_Instance : str
        Directory to store figures.
    Dynamic_Reporter : list
        Reporter to show values.
    _SUB_RUNFOLDER : str
    combination : int
    time_str : str
    epoch :str

    Returns
    -------
    Store figures

    """
    # creating directory to save graphs in _SUB_RUNFOLDER 
    Graphs_Directory=os.path.join(_SUB_RUNFOLDER, "All_Graphs")          
    if not os.path.exists(Graphs_Directory):
            os.makedirs(Graphs_Directory)
            
    # creating sub directory to save graphs files in Graphs_Directory 
    Graphs_Directory_for_Epochwise_Subfolders=os.path.join(Graphs_Directory, "Graphs_till_epoch"+str(epoch).zfill(3))        
    if not os.path.exists(Graphs_Directory_for_Epochwise_Subfolders):
            os.makedirs(Graphs_Directory_for_Epochwise_Subfolders)
            
    # creating directory to save CSV files in _SUB_RUNFOLDER         
    CSV_files_Directory=os.path.join(_SUB_RUNFOLDER, "All_CSV_files")          
    if not os.path.exists(CSV_files_Directory):
            os.makedirs(CSV_files_Directory)
    
    # creating sub directory to save CSV  files in CSV_files_Directory 
    CSV_files_directory_for_epochwise_subfolders=os.path.join(CSV_files_Directory, "CSV_files_till_epoch"+str(epoch).zfill(3))        
    if not os.path.exists(CSV_files_directory_for_epochwise_subfolders):
            os.makedirs(CSV_files_directory_for_epochwise_subfolders)
            
    # Classification Losses and Accuracy
    PlotTrainLoss_C = MyHelpFunc.PlotList(x_list=Dynamic_Reporter.StoredValues["Epoch"], y_list=Dynamic_Reporter.StoredValues["TrainLoss_C"],plt_title_name="Train Loss of Classification in batches,  epochs={}".format(epoch),label_name="Train Loss Classification " , window_length=Ploting_Window_Length, window_mul=Ploting_Window_Multiplier)
    PlotTrainAcc_C = MyHelpFunc.PlotList(x_list=Dynamic_Reporter.StoredValues["Epoch"], y_list=Dynamic_Reporter.StoredValues["TrainAcc_C"],plt_title_name="Train Accuracy of Classification in batches, epochs={} ".format(epoch),label_name="Train Accuracy Classification" , window_length=Ploting_Window_Length, window_mul=Ploting_Window_Multiplier)
    PlotTestLoss_C = MyHelpFunc.PlotList(x_list=Dynamic_Reporter.StoredValues["Epoch"], y_list=Dynamic_Reporter.StoredValues["TestLoss_C"],plt_title_name=" Test Loss of Classification , epochs={}".format(epoch),label_name="Test Loss Classification" , window_length=Ploting_Window_Length, window_mul=Ploting_Window_Multiplier)
    PlotTestAcc_C = MyHelpFunc.PlotList(x_list=Dynamic_Reporter.StoredValues["Epoch"], y_list=Dynamic_Reporter.StoredValues["TestAcc_C"],plt_title_name="Test Accuracy of Classification, epochs={}".format(epoch), label_name= "Test Accuracy Classification", window_length=Ploting_Window_Length, window_mul=Ploting_Window_Multiplier)
        
    # Segmentation Losses
    PlotTrainLoss_S=MyHelpFunc.PlotList(Dynamic_Reporter.StoredValues["Epoch"],
                           y_list=Dynamic_Reporter.StoredValues["TrainLoss_S"],
                           plt_title_name="Train Loss of Segmentation in batches,  epochs=%i"%(epoch),
                           label_name="Train Loss of Segmentation",
                           window_length=Ploting_Window_Length,
                           window_mul=Ploting_Window_Multiplier)
    
    PlotTestLoss_S=MyHelpFunc.PlotList(Dynamic_Reporter.StoredValues["Epoch"],
                          y_list=Dynamic_Reporter.StoredValues["TestLoss_S"],
                          plt_title_name=" Test Loss of Segmentation, epochs={}".format(epoch),
                          label_name="Test Loss of Segmentation", 
                          window_length=Ploting_Window_Length,
                          window_mul=Ploting_Window_Multiplier)
    
    
    Figure_Storage_Instance = HelpFunc.FigureStorage(Graphs_Directory_for_Epochwise_Subfolders, dpi = 400, autosave = True) 
    
    # Saving Classification Figures
    Figure_Storage_Instance.Store(fig=PlotTrainLoss_C, name="_Train Loss Classification com{}_time_{}".format(combination,time_str))
    Figure_Storage_Instance.Store(fig=PlotTrainAcc_C, name="_Train Accuracy Classification com{}_time_{}".format(combination,time_str))
    Figure_Storage_Instance.Store(fig=PlotTestLoss_C, name="_Test Loss Classification com{}_time_{}".format(combination,time_str))
    Figure_Storage_Instance.Store(fig=PlotTestAcc_C, name="_Test Accuracy Classification com{}_time_{}".format(combination,time_str))
    # Saving Segmentation Figures
    Figure_Storage_Instance.Store(fig=PlotTrainLoss_S, name="_Train Loss of Segmentation com{}_time_{} ".format(combination,time_str))
    Figure_Storage_Instance.Store(fig=PlotTestLoss_S, name="_Test Loss of Segmentation com{}_time_{}".format(combination,time_str))
        
   
    # Saving All List as csv File
    MyHelpFunc.save_list_as_csv_file(x_list=Dynamic_Reporter.StoredValues["Epoch"] ,
                                     y_list=Dynamic_Reporter.StoredValues["TrainLoss_C"], 
                                     _SUB_RUNFOLDER=CSV_files_directory_for_epochwise_subfolders, 
                                     file_name = "TrainLoss_C__Train Loss Classification")
    MyHelpFunc.save_list_as_csv_file(x_list=Dynamic_Reporter.StoredValues["Epoch"] ,
                                     y_list=Dynamic_Reporter.StoredValues["TrainAcc_C"], 
                                     _SUB_RUNFOLDER=CSV_files_directory_for_epochwise_subfolders, 
                                     file_name = "TrainAcc_C__Train Accuracy Classification")
    MyHelpFunc.save_list_as_csv_file(x_list=Dynamic_Reporter.StoredValues["Epoch"] ,
                                     y_list=Dynamic_Reporter.StoredValues["TestLoss_C"], 
                                     _SUB_RUNFOLDER=CSV_files_directory_for_epochwise_subfolders, 
                                     file_name = "TestLoss_C__Test Loss Classification")
    MyHelpFunc.save_list_as_csv_file(x_list=Dynamic_Reporter.StoredValues["Epoch"] ,
                                     y_list=Dynamic_Reporter.StoredValues["TestAcc_C"], 
                                     _SUB_RUNFOLDER=CSV_files_directory_for_epochwise_subfolders, 
                                     file_name = "TestAcc_C__Test Accuracy Classification")
    MyHelpFunc.save_list_as_csv_file(x_list=Dynamic_Reporter.StoredValues["Epoch"] ,
                                     y_list=Dynamic_Reporter.StoredValues["TrainLoss_S"], 
                                     _SUB_RUNFOLDER=CSV_files_directory_for_epochwise_subfolders, 
                                     file_name = "TrainLoss_S__Train Loss Segmentation")
    MyHelpFunc.save_list_as_csv_file(x_list=Dynamic_Reporter.StoredValues["Epoch"] ,
                                     y_list=Dynamic_Reporter.StoredValues["TestLoss_S"], 
                                     _SUB_RUNFOLDER=CSV_files_directory_for_epochwise_subfolders, 
                                     file_name = "TestLoss_S__Test Loss Segmentation")

    del PlotTrainLoss_C,PlotTrainAcc_C,PlotTestLoss_C,PlotTestAcc_C
    del PlotTrainLoss_S,PlotTestLoss_S

#%%   Training loop
def train_and_test_loop(Optimizer,Batch_Size, Learning_Rate, _SUB_RUNFOLDER,combination,time_str,_RUNFOLDER, called_dataset):
    """
    Starts training when this function is called by the main function. 
    During training it also tests the network at every certain consecutive batches. 
    

    Parameters
    ----------
    Optimizer : str    
    Batch_Size : int   
    Learning_Rate : float      
    _SUB_RUNFOLDER : str      
    combination : int
    time_str : str
    _RUNFOLDER : str
    called_dataset : str
        

    Returns
    -------
    None.

    """
    start_time = datetime.now()   
    ReporterNames=["Epoch","Batch", "TrainAcc_C", "TrainLoss_C", "TestAcc_C", "TestLoss_C", "TrainLoss_S", "TestLoss_S"]

    #DataStorage() stores training data (ReporterNames) while also offering customizable prints of the data 
    # Creating instance of class-DataStorage()
    Dynamic_Reporter=HelpFunc.DataStorage(names=ReporterNames,
                                    precision=3,
                                    average_window=300,
                                    show=100,
                                    line=2000,
                                    header=1000)
    
    # creating instance of class-FigureStorage, giving directory to store figures
    Figure_Storage_Instance = HelpFunc.FigureStorage(_SUB_RUNFOLDER, dpi = 400, autosave = True)        
    trainloader, testloader,trainset,testset = data_loader(called_dataset,Batch_Size )
        
    train_images_to_show_n_save=select_some_specific_images_from_dataloader(trainloader,called_dataset) 
    test_images_to_show_n_save = select_some_specific_images_from_dataloader(testloader,called_dataset)
    
    # network declaration
    if called_dataset=="PascalVOC":
        seg_network = ClassLib.Segmentation_Network_Pascal(Network_Output_Channel)
        cls_network=ClassLib.Classification_Network2_Pascal(Network_Output_Channel)
    elif called_dataset=="STL10":
        seg_network = ClassLib.Segmentation_Network_STL(Network_Output_Channel)
        cls_network= ClassLib.Classification_Network2_STL(Network_Output_Channel)
    elif called_dataset=="CIFAR10":        
        seg_network = ClassLib.Segmentation_Network(Network_Output_Channel)
        cls_network=ClassLib.Classification_Network(Network_Output_Channel)
         
    criterion =  nn.CrossEntropyLoss() #Loss

    if Optimizer== "SGD":
        optimizer_s = optim.SGD(seg_network.parameters(), lr=Learning_Rate, momentum=0.9)
        optimizer_c = optim.SGD(cls_network.parameters(), lr=Learning_Rate, momentum=0.9)
        
    elif Optimizer== "Adam":
        optimizer_s = optim.Adam(seg_network.parameters(),lr=Learning_Rate)
        optimizer_c = optim.Adam(cls_network.parameters(),lr=Learning_Rate)
    elif Optimizer== "RMSprop":
        optimizer_s = optim.RMSprop(seg_network.parameters(),lr=Learning_Rate)
        optimizer_c = optim.RMSprop(cls_network.parameters(),lr=Learning_Rate)
    elif Optimizer== "Adadelta":   
        optimizer_s = optim.Adadelta(seg_network.parameters(),lr=Learning_Rate)
        optimizer_c = optim.Adadelta(cls_network.parameters(),lr=Learning_Rate)
    elif Optimizer== "Adagrad":   
        optimizer_s = optim.Adagrad(seg_network.parameters(),lr=Learning_Rate)
        optimizer_c = optim.Adagrad(cls_network.parameters(),lr=Learning_Rate)
    
    scheduler_s = StepLR(optimizer_s,step_size=100,gamma=0.1)
    scheduler_c = StepLR(optimizer_c,step_size=100,gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer,patience=5)
    
    save_all_parameters_as_txt_file(Learning_Rate,Batch_Size, _SUB_RUNFOLDER,combination,time_str,seg_network,cls_network,optimizer_s,optimizer_c,criterion,scheduler_c,scheduler_s,called_dataset)  
    print(' \n# Starting Training of combination {}'.format(combination))  
    Batch=0
    
    for epoch in range(Num_Epochs): 

        for _, traindata in enumerate(trainloader):

            if Batch % Testing_Dataset_Every_Batches==0:
                             
                total_tests_correct_c=0
                cls_network.eval()
                seg_network.eval()
                cls_network.to(Device)
                seg_network.to(Device)
                with torch.no_grad():                   
                    for _, test_data in enumerate(testloader):
                        
                        if called_dataset=="PascalVOC":
                            test_images,test_labels=MyHelpFunc.get_images_and_labels_from_voc_detection_loader(test_data)
                        elif called_dataset=="STL10" or called_dataset=="CIFAR10" : 
                           
                            test_images, test_labels = test_data

                        test_images = test_images.to(Device)
                        test_labels  = test_labels.to(Device)
                        test_outputs_s = seg_network(test_images) 
                        test_outputs_c = cls_network(test_outputs_s)
                        
                        argmax_test_outputs_s = torch.argmax(test_outputs_s, dim=1)
                        segmented_batch_tensor_test = slic_refinement_of_every_batchimages(batch_images=argmax_test_outputs_s.cpu())        
                        segmented_batch_tensor_test=segmented_batch_tensor_test.to(Device)                         
                        test_loss_c = criterion(test_outputs_c,test_labels.long().squeeze())
                        test_loss_s = criterion(test_outputs_s,segmented_batch_tensor_test.squeeze())                    
                        tests_correct_c=(test_outputs_c.argmax(dim=1)==test_labels).float().sum()       
                        total_tests_correct_c+=tests_correct_c.item()                
                        test_accuracy_c=total_tests_correct_c/len(testset) *100

                cls_network.train()
                seg_network.train()
                
            # get the inputs
            if called_dataset=="PascalVOC":
                train_images, train_labels = MyHelpFunc.get_images_and_labels_from_voc_detection_loader(traindata)
            elif called_dataset=="STL10" or called_dataset=="CIFAR10":
                train_images, train_labels = traindata 
                           
            train_images = train_images.to(Device)
            train_labels = train_labels.to(Device)
            optimizer_s.zero_grad()
            optimizer_c.zero_grad()  
            
            train_outputs_s = seg_network(train_images) 
            train_outputs_c = cls_network(train_outputs_s) 
            argmax_train_outputs_s = torch.argmax(train_outputs_s, dim=1)

            # SLIC Refinement
            segmented_batch_tensor=slic_refinement_of_every_batchimages(batch_images=argmax_train_outputs_s.cpu())
            segmented_batch_tensor=segmented_batch_tensor.to(Device)  
            
            train_loss_c = criterion(train_outputs_c, train_labels.long().squeeze())
            train_loss_s = criterion(train_outputs_s, segmented_batch_tensor.squeeze())
            
            
            # Saving Segmented Images in every "x=500"  batches            
            if Batch % Show_Segmentation_Masks_Every == 0:  
                
                #From Training Set
                FigMasks = get_sample_segmentation_masks((train_images_to_show_n_save+1)/2, seg_network, Batch, epoch, Training_Images,combination,Learning_Rate,Batch_Size,Optimizer)
                saved_image_name= "_"+ time_str+ "_Training_Combination"+str(combination) + "_MasksBatch_"+str(Batch).zfill(5)
                Figure_Storage_Instance.Store(fig=FigMasks,
                                 name=os.path.join("Segmentation_Masks_of_Training_Set", saved_image_name ))                
                
            if Batch % (1*Show_Segmentation_Masks_Every) == 0:
                
                #From Testing Set
                FigMasks_test = get_sample_segmentation_masks((test_images_to_show_n_save+1)/2, seg_network, Batch, epoch, Testing_Images,combination,Learning_Rate,Batch_Size,Optimizer)
                saved_testimage_name= "_"+ time_str+ "_Testing_Combination"+str(combination) + "_MasksBatch_"+str(Batch).zfill(5)
                Figure_Storage_Instance.Store(fig=FigMasks_test,
                                      name=os.path.join("Segmentation_Masks_of_Testing_Set", saved_testimage_name) )
                             
            
            train_loss =( train_loss_c + train_loss_s)  
            train_accuracy_c = (train_outputs_c.argmax(dim=1) == train_labels).float().sum()*100.0 / train_images.shape[0]
            
            Running_Epoch_Fraction=(Batch_Size/len(trainset))*Batch  
            
            # assigning values in every batch to  Dynamic_Reporter   
            # ReporterNames=["Epoch",Batch", "TrainAcc_C", "TrainLoss_C", "TestAcc_C", "TestLoss_C", "TrainLoss_S", "TestLoss_S"]
            Dynamic_Reporter.Store([Running_Epoch_Fraction,
                                  Batch,
                                  train_accuracy_c.item(),
                                  train_loss_c.item(),
                                  test_accuracy_c,
                                  test_loss_c.item(), 
                                  train_loss_s.item(),
                                  test_loss_s.item(),
                                  ])
   
            train_loss.backward()
            optimizer_s.step()
            optimizer_c.step()         
            Batch += 1    
            
        # scheduler.step(Test_Loss)
        scheduler_s.step()
        scheduler_c.step()
        
        
        # creating directory to save models
        saved_model_directory=os.path.join(_SUB_RUNFOLDER, "All_Saved_Models")          
        if not os.path.exists(saved_model_directory):               
            os.makedirs(saved_model_directory)

               # saving model in every "x" epoch after epoch>49
        if epoch > 19 and (epoch+1)% 10 == 0:

            
            saved_seg_network_filename="Combination_"+str(combination).zfill(3) +"_"+"Run"+"_"+str(time_str)+"_"+ "till_epoch"+str((epoch+1))+"_"+"seg_network.pth"
            saved_cls_network_filename="Combination_"+str(combination).zfill(3) +"_"+"Run"+"_"+str(time_str)+"_"+ "till_epoch"+str((epoch+1))+"_"+"cls_network.pth"
            saved_reporter_filename="Combination_"+str(combination).zfill(3) +"_"+"Run"+"_"+str(time_str)+ "_"+"till_epoch"+str((epoch+1))+"_"+"reporter.pt"
            
            saved_seg_network_path=os.path.join (_SUB_RUNFOLDER,"All_Saved_Models",saved_seg_network_filename )
            saved_cls_network_path=os.path.join (_SUB_RUNFOLDER,"All_Saved_Models",saved_cls_network_filename )
            saved_reporter_path=os.path.join (_SUB_RUNFOLDER,"All_Saved_Models",saved_reporter_filename)
            
            # saving models in every "x" epoch
            torch.save(seg_network.state_dict(),saved_seg_network_path )
            torch.save(cls_network.state_dict(),saved_cls_network_path )
            torch.save(Dynamic_Reporter, saved_reporter_path, pickle_module=dill) 
            print("...Model Saved.......")
            
            # saving plots and list in every "x" epoch
            plot_and_save_loss_accuracy_list(Figure_Storage_Instance,Dynamic_Reporter,_SUB_RUNFOLDER,combination,time_str,(epoch+1))
    
    # saving complete model 
    saved_full_seg_network_filename="Combination_"+str(combination).zfill(3) +"_"+"Run"+"_"+str(time_str)+"_"+ "till_epoch"+str((epoch+1))+"_"+"Full_Seg_Network.pth"
    saved_full_cls_network_filename="Combination_"+str(combination).zfill(3) +"_"+"Run"+"_"+str(time_str)+"_"+ "till_epoch"+str((epoch+1))+"_"+"Full_Cls_Network.pth"
    saved_fullreporter_filename="Combination_"+str(combination).zfill(3) +"_"+"Run"+"_"+str(time_str)+ "_"+"till_epoch"+str((epoch+1))+"_"+"Full_Reporter.pt"
    
    saved_full_seg_network_path=os.path.join (_SUB_RUNFOLDER,"All_Saved_Models",saved_full_seg_network_filename )
    saved_full_cls_network_path=os.path.join (_SUB_RUNFOLDER,"All_Saved_Models",saved_full_cls_network_filename )
    saved_reporter_path=os.path.join (_SUB_RUNFOLDER,"All_Saved_Models",saved_fullreporter_filename)
                 
    torch.save(seg_network.state_dict(), saved_full_seg_network_path)
    torch.save(cls_network.state_dict(), saved_full_cls_network_path)
    torch.save(Dynamic_Reporter, saved_reporter_path, pickle_module=dill) 
    
    plot_and_save_loss_accuracy_list(Figure_Storage_Instance,Dynamic_Reporter,_SUB_RUNFOLDER,combination,time_str,(epoch+1))
    
    # calculate and save runtime
    end_time = datetime.now()
    time_taken = end_time - start_time
    print('Time: ',time_taken) 
    ParamFile=HelpFunc.ParameterStorage(_SUB_RUNFOLDER)
    ParamFile.WriteTab("Time Taken for this run",time_taken)
    
    # clear memory
    ReporterNames.clear()
    Dynamic_Reporter.StoredValues.clear()
    del Dynamic_Reporter
    del Figure_Storage_Instance
    del  cls_network, seg_network               
    del trainloader, testloader,trainset,testset,criterion
    del train_images,test_images
    del train_loss,train_loss_c,train_loss_s ,test_loss_c,test_loss_s
    del segmented_batch_tensor,train_outputs_s,train_outputs_c,argmax_train_outputs_s
    del FigMasks,FigMasks_test
    torch.cuda.empty_cache()
    gc.collect()


    print('\n # Finished Training of combination {}'.format(combination))
    print('\n You can check:--> %s  <-- folder for current running results'%(_RUNFOLDER))
            
#%% Main Loop
def main_function():  
    """
    This is the main function that initiates training by training all other function. The batch size, learning rate, and optimizer
    can be put by changing the values of the parameters_dict dictionary 

    Returns
    -------
    None.

    """

    if  Default_Dataset == "c" or Default_Dataset =="p" or Default_Dataset == "s":
        typed_string=Default_Dataset
    else:
        print(" Which Dataset do you want to use?\n Press button c or s or p for CIFAR10 , STL10 and PascalVOC dataset respectively.")
        typed_string = input ("Press button c or s or p (lower case only) + Enter :")
        
        while True:
            if typed_string=="c" or typed_string=="s" or typed_string=="p":
                break
            else:  
                print("\n Sorry,you pressed wrong button. \n Try again with pressing c or s or p ")
                typed_string = input ("Press button c or s or p (lower case only) + Enter :")
                  
    # assign dataset              
    if typed_string== "c":       
        called_dataset="CIFAR10"
    elif typed_string=="s":
        called_dataset="STL10"
    elif typed_string=="p":
        called_dataset="PascalVOC"

    parameters_dict= { "Batch_Size":[16,50] ,   # 16,32,50, or any other value
                        "Learning_Rate":[ 0.001,0.01], #,0.0005,0.01,0.005,0.0001,0.1 or any other value
                        "Optimizer":["Adam","SGD"]}      #Available optimizers "Adam","SGD", "RMSprop","Adadelta","Adagrad" 
                  
    param_values= [params for params in parameters_dict.values()]
    combination=0

    # check current _RUNFOLDER()/directory
    _RUNFOLDER, time_str=HelpFunc.GenerateTrainFolder()       
    print(" # _RUNFOLDER --> ",_RUNFOLDER)
 
    # save Dictionary parameters
    MyHelpFunc.save_all_dictonary_parameteres_in_a_text_file(_RUNFOLDER,parameters_dict,param_values)
    
    # mainloop for Run
    for Batch_Size,Learning_Rate,Optimizer in product(*param_values):
        combination+=1
        
         # creating _SUB_RUNFOLDER for each combination of Run
        _SUB_RUNFOLDER=os.path.join(_RUNFOLDER, "Combination_"+str(combination).zfill(3))          
        if not os.path.exists(_SUB_RUNFOLDER):
            os.makedirs(_SUB_RUNFOLDER)
                                           
        train_and_test_loop(Optimizer,Batch_Size, Learning_Rate,_SUB_RUNFOLDER, combination,time_str,_RUNFOLDER, called_dataset)
    

    print('\nCheck %s folder for results'%(_RUNFOLDER))

main_function()













