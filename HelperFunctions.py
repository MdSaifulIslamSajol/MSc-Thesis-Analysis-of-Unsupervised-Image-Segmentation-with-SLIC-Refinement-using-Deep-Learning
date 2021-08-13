import math as m
import torch
import torch.utils.data
import torchvision
import numpy as np
import os
import sys
import time

################################################
################################################
################################################

def GenerateTrainFolder(location = False):
    """
    Returns the name of the training folder.

    Parameters
    ----------
    location : str, optional
        Alternative location of the folder. 01_Training-folder will be\n
        automatically created at this location. The default is false (using cwd).

    Returns
    -------
    str
        Train folder.

    """
    time_str=time.strftime("%y_%m_%d__%H_%M_%S")
    
    if not location:
        location = os.getcwd()
    return os.path.join(location, "TrainResults", time_str),time_str

################################################
################################################
################################################


def ConfigurePlt(check_latex = True):
    """
    Sets Font sizes for plt plots.


    Parameters
    ----------
    check_latex : bool, optional
        Use LaTex-mode (if available). The default is True.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    if check_latex:
        
        from distutils.spawn import find_executable

        if find_executable('latex'):
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
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

################################################
################################################
################################################

def GetDatasetPath(folder_name = "00_Datasets", max_depth = 10):
    """


    Parameters
    ----------
    folder_name : str, optional
        Folder name to look for, first occurence is returned. The default is "00_Datasets".
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

################################################
################################################
################################################

def LoadMnist(path="00_Datasets", transforms_train = False, transforms_test = False, minibatch=32, worker=0, normalization = "mean"):
    """
    Return training- and testing-dataloaders for the MNIST data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\': will use the specified location to store the dataset.\n
        The default is "00_Datasets".
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
        Normalize = torchvision.transforms.Normalize((0.13063), (0.30811))
    elif normalization == "-11":
        Normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        Normalize = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
    
    # setting up default tranformers
    DefaultTransformsTrain = torchvision.transforms.Compose([
                                               torchvision.transforms.RandomCrop(28, padding=0),
                                               torchvision.transforms.RandomHorizontalFlip(),
                                               torchvision.transforms.ToTensor(),
                                               Normalize])
    DefaultTransformsTest = torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               Normalize])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "00_Datasets", "Mnist")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "Mnist")
        else:
            path = os.path.join(GetDatasetPath(path), "Mnist")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))


    # use given transformers or default ones
    TransformsTrain = transforms_train if transforms_train else DefaultTransformsTrain
    TransformsTest = transforms_train if transforms_train else DefaultTransformsTest

    # load data sets and create data loaders
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=path, train=True, download=True, transform=TransformsTrain),
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=path, train=False, download=True, transform=TransformsTest),
            batch_size=minibatch, shuffle=False, pin_memory=True, num_workers=worker)
    return train_loader, test_loader

################################################
################################################
################################################

def LoadCifar10(path="00_Datasets", transforms_train = False, transforms_test = False, minibatch=32, worker=0, normalization = "mean"):
    """
    Return training- and testing-dataloaders for the CIFAR10 data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\': will use the specified location to store the dataset.\n
        The default is "00_Datasets".
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
    if normalization == "mean":   # standardization
        Normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    elif normalization == "-11":
        Normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif normalization == "minmax":
        Normalize = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
    
    # setting up default tranformers
    DefaultTransformsTrain = torchvision.transforms.Compose([
                                               torchvision.transforms.RandomCrop(32, padding=0),
                                               torchvision.transforms.RandomHorizontalFlip(),
                                               torchvision.transforms.ToTensor(),
                                               Normalize])
    DefaultTransformsTest = torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               Normalize])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "00_Datasets", "Cifar10")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "Cifar10")
        else:
            path = os.path.join(GetDatasetPath(path), "Cifar10")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))


    # use given transformers or default ones
    TransformsTrain = transforms_train if transforms_train else DefaultTransformsTrain
    TransformsTest = transforms_train if transforms_train else DefaultTransformsTest
    #####
    trainset = torchvision.datasets.CIFAR10(root = path, train=True,
                                            download=True, transform=TransformsTrain)
    testset = torchvision.datasets.CIFAR10(root = path, train=False,
                                           download=True, transform=TransformsTest)
    ####

    # load data sets and create data loaders
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=TransformsTrain),
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=TransformsTest),
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)
    return train_loader, test_loader,trainset,testset

################################################
################################################
################################################

def LoadCifar100(path="00_Datasets", transforms_train = False, transforms_test = False, minibatch=32, worker=0, normalization = "mean"):
    """
    Return training- and testing-dataloaders for the CIFAR100 data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\': will use the specified location to store the dataset.\n
    The default is "00_Datasets".
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
        Normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    elif normalization == "-11":
        Normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        Normalize = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))

    # setting up default tranformers
    DefaultTransformsTrain = torchvision.transforms.Compose([
                                               torchvision.transforms.RandomCrop(32, padding=0),
                                               torchvision.transforms.RandomHorizontalFlip(),
                                               torchvision.transforms.ToTensor(),
                                               Normalize])
    DefaultTransformsTest = torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               Normalize])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "00_Datasets", "Cifar100")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "Cifar100")
        else:
            path = os.path.join(GetDatasetPath(path), "Cifar100")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))

    # use given transformers or default ones
    TransformsTrain = transforms_train if transforms_train else DefaultTransformsTrain
    TransformsTest = transforms_train if transforms_train else DefaultTransformsTest

    # load data sets and create data loaders
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=TransformsTrain),
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=TransformsTest),
            batch_size=minibatch, shuffle=False, pin_memory=True, num_workers=worker)
    return train_loader, test_loader

################################################
################################################
################################################

def MovingAverage(data, window=3):
    """
    Calculates the moving average over the data-vector with the given window size.

    Parameters
    ----------
    data : iterable
        Data to calculate the average over. Best used with 1-D iterables, i.e. 1-D torch.tensors.
    window : int, optional
        Length of the moving average window. Each datapoint is equally weighed.\n
        The default is 3. Exception raised for window <= 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if window==0:
        raise AttributeError("Window length has to be >= 1.")
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


################################################
################################################
################################################


def Histogram(matrix, minval, maxval, segments=10):
    """
    Calculates the Histogram values for given inputs.

    Parameters
    ----------
    matrix : n-D torch.tensor
        Values to calculate the Histogram from. Matrix will be flattened and only one\n
        Histogram will be created.
    minval : float
        Lower bound of the Histogram.
    maxval : float
        Upper bound of the Histogram.
    segments : int, optional
        Number of segments to be created uniformly between minval and maxval. The default is 10.

    Returns
    -------
    xs : torch.tensor
        Center x-value for all segments.
    heights : torch.tensor
        Relative number of points for all segments.
    width : float
        Width of each segment.

    """
    width=(maxval-minval)/segments
    xs=width/2 + torch.arange(minval, maxval, width)
    heights=torch.zeros(segments)

    vector=matrix.view(-1,1).squeeze()
    vector=vector.sort(dim=0, descending=False)[0]

    bar=0
    idx=0
    num=vector.shape[0]
    while idx<vector.shape[0]:
        if m.isnan(vector[idx]):
            num-=1
            idx+=1
            continue
        if vector[idx]<=xs[bar]+width/2:
            heights[bar]+=1
            idx+=1
        else:
            bar+=1
    heights/=num

    return xs, heights, width


################################################
################################################
################################################


def StrToDict(text):
    """
    Converts a text with a given structure into a dictionary. Best used with ArgumentParsers.

    Parameters
    ----------
    text : str
        Text to convert into a dict. Values have to be convertible to floats.\n
        Format: {'key1':vals,'key2':vals}

    Returns
    -------
    NewDict : dict
        DESCRIPTION.

    """
    NewDict={}
    if "," in text:
        Elements=text.split(",")
    else:
        Elements=[text]
    idx=0
    for element in Elements:
        if len(Elements)==1:
            key=element.split(":")[0][1:]
            val=float(element.split(":")[1][:-1])
        else:
            if idx==0:
                key=element.split(":")[0][1:]
                val=float(element.split(":")[1])
            elif idx==len(Elements)-1:
                key=element.split(":")[0]
                val=float(element.split(":")[1][:-1])
            else:
                key=element.split(":")[0]
                val=float(element.split(":")[1])
        NewDict[key]=val
        idx+=1
    return NewDict


################################################
################################################
################################################

def SaltAndPepper(img, prob):
    """
    Distorts an image with salt-and-pepper noise.

    Parameters
    ----------
    img : torch.tensor
        C x W x H Tensor to add Salt and Pepper noise to.

    prob : float
        Probability of Salt and Pepper noise.

    Returns
    -------
    img : torch.tensor
        Noisy Image.

    """
    retval=img*1.0
    noise = torch.rand(img.shape[1], img.shape[2]).unsqueeze(0).repeat(img.shape[0],1,1)

    salt=torch.where(noise<prob)
    pepper=torch.where(noise>(1-prob))

    retval[salt]=img.min().item()
    retval[pepper]=img.max().item()
    return retval

################################################
################################################
################################################

class ParameterStorage():
    """
    Offers easy access to save used parameters.

    Usage
    ----------
    Create an instance.\n
    Call Write or WriteTab functions to save Parameters.

    Parameters
    ----------
    train_folder : str
        Folder name of the folder which contains all training results. Parameterfile will be saved here.
    file_name : str, optional
        Filename of the parameterfile. The default is "ParameterStorage.txt".
    column_width : int, optional
        Maximum number of characters in the first column. The default is 30.

    Returns
    -------
    None.

    """

    def __init__(self, train_folder, file_name = "ParameterStorage.txt", column_width = 30):
        self.TrainFolder = train_folder
        self.Location = os.path.join(train_folder, file_name)
        self.ColumnWidth = column_width
        self._Create()

    def _Create(self):
        if not os.path.exists(self.TrainFolder):
            os.makedirs(self.TrainFolder)

    def Write(self, txt):
        """
        Stores input unformated in one line.

        Parameters
        ----------
        txt : str
            Text to store.

        Returns
        -------
        None.

        """
        with open(self.Location, "a") as file:
            file.write(txt+"\n")

    def WriteTab(self, col1, col2):
        """
        Stores values in a pre-formated (tabular) way.

        Parameters
        ----------
        col1 : str
            Text shown in the first column (i.e. the name of the parameter).
        col2 : any
            Number(s) to display. Supports int, float, list of int or, list of floats.\n
            Other datasets are saved as str.

        Returns
        -------
        None.

        """
        with open(self.Location, "a") as file:
            file.write(col1[:self.ColumnWidth])

            if type(col2) == list:
                c = 0
                for i in col2:
                    if c == 0:
                        file.write(" "*(self.ColumnWidth-len(col1)))
                    else:
                        file.write(" "*self.ColumnWidth)

                    if type(i) == int:
                        file.write("%i\n"%i)
                    elif type(i) == float:
                        file.write("%f\n"%i)
                    else:
                        file.write("%s\n"%i)
                    c+=1
            elif type(col2) == dict:
                c = 0
                for i in col2:
                    val=col2[i]
                    if c == 0:
                        file.write(" "*(self.ColumnWidth-len(col1))+"\n")
                        c+=1
                        continue
                    else:
                        file.write("%s"%i+" "*(self.ColumnWidth-len(i)))

                    if type(val) == int:
                        file.write("%i\n"%val)
                    elif type(i) == float:
                        file.write("%f\n"%val)
                    else:
                        file.write("%s\n"%val)
                    c+=1
            elif type(col2) == int:
                file.write(" "*(self.ColumnWidth-len(col1))+"%i\n"%col2)
            elif type(col2) == float:
                file.write(" "*(self.ColumnWidth-len(col1))+"%f\n"%col2)
            else:
                file.write(" "*(self.ColumnWidth-len(col1))+"%s\n"%col2)
            file.write("\n")

    def DashSigns(self):
        """
        Writes one line of dash signs.

        Returns
        -------
        None.

        """
        with open(self.Location, "a") as file:
            file.write("---------------------------\n")

    def EqualSigns(self):
        """
        Writes on line of equal signs.

        Returns
        -------
        None.

        """
        with open(self.Location, "a") as file:
            file.write("===========================\n")

################################################
################################################
################################################

class DataStorage():
    """
    Stores training data while also offering customizable prints of the data.

    Usage
    ----------
    Create an instance.\n
    Call the Store function in every Batch with a given list of the values to store.

    Parameters
    ----------
    names : list of str
        List of str of values to store. Automatically creates and computed a moving average\n
        for the names 'Loss' and 'Acc' if those are given in this list.
    average : int, optional
        Window size (in Batches) for the moving average calculation. The default is 100.
    show : int, optional
        Number of Batches between each new print. The default is 25.
    line : int, optional
        Number of Batches to show values in a new line. The default is 500.
    header : int, optional
        Number of Batches to reprint the names for the columns. The default is 5000.
    step : int, optional
        Step size of data storage in Batches. Data gets stored every step Batches.\n
        The default is 1. step = 2 reduces memory consumption by 50\%.
    precision : int, optional
        Number of decimal digits shown.
    auto_show : bool, optional
        Enable/Disable automatic value display. The default is True.

    Returns
    -------
    None.

    """
    def __init__(self, names, average_window=100, show=25, line=500, header=5000, step=1, precision=3):
        self.Names = ["Time"]
        for name in names:
            self.Names.append(name)
        self.AverageWindow = average_window
        self.Show = show
        self.Line = line
        self.Header = header
        self.Step = step
        self.Precision = precision
        self.Batch = 0

        if "Loss" in self.Names:
            self.Names.append("avg. Loss")
        if "Acc" in self.Names:
            self.Names.append("avg. Acc")

        self.Lens = [len(self.Names[idx])+5 for idx in range(len(self.Names))]
        self.StoredValues = {}

        for name in self.Names:
            self.StoredValues[name] = []

        self.Columns = len(self.Names)
        self.DumpValues = {}


    def Store(self, vals):
        """
        Stores data in internal StoredValues-dictionary.

        Parameters
        ----------
        vals : list of values
            List of values to be stored in the internal 'StoredValues'-dictionary.\n
            Order has to be the same as given during initialization. Best used with \n
            int, float or torch.tensor.

        Returns
        -------
        None.

        """
        # save time when first storing
        if self.Batch == 0:
            self.DumpValues["TimeStart"] = time.time()
        if self.Batch%self.Step == 0:
            if len(self.StoredValues["Time"]) == 0:
                self.StoredValues["Time"] = [(time.time() - self.DumpValues["TimeStart"])/60]
            else:
                self.StoredValues["Time"].append((time.time() - self.DumpValues["TimeStart"])/60.0)
            for col in range(1,self.Columns):
                name = self.Names[col]
                if name == "avg. Loss":
                    self.StoredValues[name].append(torch.sum(torch.tensor(self.StoredValues["Loss"][-self.AverageWindow:]))/self.AverageWindow)
                elif name == "avg. Acc":
                    self.StoredValues[name].append(torch.sum(torch.tensor(self.StoredValues["Acc"][-self.AverageWindow:]))/self.AverageWindow)
                else:
                    self.StoredValues[name].append(vals[col-1])

            if self.Batch == 0:
                self._GetHead()
                self._Display()
                print("")
            else:
                if self.Batch%self.Show == 0:
                    self._Display()
                if self.Batch%self.Line == 0:
                    print("")
                if self.Batch%self.Header == 0:
                    self._GetHead()
        self.Batch+=1

    def _Display(self):
        outstr = "\r"
        args = []
        for col in range(self.Columns):
            val = self.StoredValues[self.Names[col]][-1]
            outstr += "{:s}"

            if type(val) == float:
                val = str(round(val, self.Precision))
            elif type(val) == torch.Tensor:
                val = str(round(val.item(), self.Precision))
            else:
                val = str(val)
            args.append(val+(self.Lens[col]-len(val))*" ")
        print(outstr.format(*args), end="")

    def _GetHead(self):
        print("")
        string = ""
        for col in range(self.Columns):
            name = self.Names[col]
            string += name+(self.Lens[col]-len(name))*" "
        print(string)

################################################
################################################
################################################

class FigureStorage():
    """
    Automatically store and save matplotlib figures to .png and .svg files. Folderstructure\n
    relative to train_folder has to be given in individual filenames.

    Usage
    ----------
    Create an instance.\n
    Call 'Store' to store Figure in this object. If 'AutoSave' disabled, call 'SaveAll' once when code is done.


    Parameters
    ----------
    train_folder : str
        Folder name of the folder which contains all training results. Parameterfile will be saved here.
    dpi : int, optional
        Global DPI value to use. Can be overwritten for individual images in 'Store'. The default is 200.
    autosave : bool, optional
        Enable or disable autosave. Can be overwritten for individual images in 'Store'. The default is False.
    printing : bool, optional
        Enable or disable console printing the filename when saving an image. The default is False.

    Returns
    -------
    None.

    """
    def __init__(self, train_folder, dpi=200, autosave=False, printing=False):
        self.Figures = []
        self.Names = []
        self.Dpis = []
        self.TrainFolder = train_folder
        self.Dpi = dpi
        self.AutoSave = autosave
        self.Printing = printing

        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
            os.makedirs(os.path.join(train_folder, "png"))
            os.makedirs(os.path.join(train_folder, "svg"))

    def Store(self, fig, name = False, dpi = False, save = False):
        """
        Store one image in this object. Saving with an individual DPI value is possible.

        Parameters
        ----------
        fig : matplotlib figure
            Figure to store.
        name : str, optional
            Filename of the figure. Only give filename without datatype, given folderstructure\n
            will be used relative to 'train_folder'. Class will automatically save both the\n
            .png and the. The default is False.
        dpi : int, optional
            Individual DPI to overwrite the default DPI for this object. The default is False.
        save : bool, optional
            Individual saving option if 'AutoSave' is disabled. The default is False.

        Raises
        ------
        AttributeError
            AttributeError when 'name' is not a string.

        Returns
        -------
        None.

        """
        if str(type(fig)) == "<class 'matplotlib.figure.Figure'>":
            if type(name) == str:
                self.Names.append(name)
            else:
                raise AttributeError("No figure name given.")
            self.Figures.append(fig)
            self.Dpis.append(dpi if dpi else self.Dpi)
            if save or self.AutoSave:
                self._SaveOne(fig = fig, name = name, dpi = self.Dpis[-1], printing = self.Printing)
        else:
            print("No Figure given, skipping.")

    def SaveAll(self, dpi = False, printing = False):
        """
        Save all stored images at their corresponding filepaths. Not necessary when using 'AutoSave'.

        Parameters
        ----------
        dpi : int, optional
            Individual DPI to overwrite the default DPI for this object. The default is False.
        printing : bool, optional
            Enable or disable console printing the filename when saving an image. The default is False.

        Returns
        -------
        None.

        """
        for fignum in range(len(self.Figures)):
            dpi = dpi if dpi else self.Dpis[fignum]
            fig = self.Figures[fignum]
            name = self.Names[fignum]
            if fig and name:
                if not os.path.exists(os.path.dirname(os.path.join(self.TrainFolder, "png", name))):
                    os.makedirs(os.path.dirname(os.path.join(self.TrainFolder, "png", name)))
                if not os.path.exists(os.path.dirname(os.path.join(self.TrainFolder, "svg", name))):
                    os.makedirs(os.path.dirname(os.path.join(self.TrainFolder, "svg", name)))
                fig.savefig(os.path.join(self.TrainFolder, "png", name+".png"), Dpi = dpi)
                fig.savefig(os.path.join(self.TrainFolder, "svg", name+".svg"), Dpi = dpi)
                if printing or self.Printing:
                    print("%s saved."%(name))

    def _SaveOne(self, fig, name, dpi = False, printing = False):
        if fig and name:
            dpi = dpi if dpi else self.Dpi
            
            if not os.path.exists(os.path.dirname(os.path.join(self.TrainFolder, "png", name))):
                os.makedirs(os.path.dirname(os.path.join(self.TrainFolder, "png", name)))
            if not os.path.exists(os.path.dirname(os.path.join(self.TrainFolder, "svg", name))):
                os.makedirs(os.path.dirname(os.path.join(self.TrainFolder, "svg", name)))
            fig.savefig(os.path.join(self.TrainFolder, "png", name+".png"), dpi = dpi)
            fig.savefig(os.path.join(self.TrainFolder, "svg", name+".svg"), dpi = dpi)
            if printing:
                print("%s saved."%(name))











