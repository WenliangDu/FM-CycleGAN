import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
# segmentation
import scipy.io as sio  # for reading .mat files
import numpy as np
#from util import util

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        self.transform_Seg_A = get_transform(self.opt, grayscale=True)
        self.transform_Seg_B = get_transform(self.opt, grayscale=True)

        ### load segmentation
        if ((not self.opt.no_segmentation if self.opt.isTrain else bool(0)) or (not self.opt.no_seg_input)):
            self.dir_A_seg = os.path.join(opt.dataroot, opt.phase + '_seg_A')
            self.dir_B_seg = os.path.join(opt.dataroot, opt.phase + '_seg_B')
            self.A_seg_paths = sorted(make_dataset(self.dir_A_seg))
            self.B_seg_paths = sorted(make_dataset(self.dir_B_seg))

            A_centers_path = os.path.join(opt.dataroot, opt.phase + '_seg_A/Centers.mat')
            B_centers_path = os.path.join(opt.dataroot, opt.phase + '_seg_B/Centers.mat')
            Centers_A = sio.loadmat(A_centers_path)
            Centers_B = sio.loadmat(B_centers_path)

            opt.A_centers = Centers_A['C']
            opt.B_centers = Centers_B['C']
        else:
            opt.A_centers = 0
            opt.B_centers = 0

        #if not self.opt.no_ganFeat_loss if self.opt.isTrain else bool(0):
            #self.opt.netD = 'feat'

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size # make sure index is within then range
        A_path = self.A_paths[index_A]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            while index_B == index_A: # make sure that even the images in the dataset are aligned, the images input into the cycle model are unaligned.
                index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # apply the same transform to A_seg_map and B_seg_map (segmentation)
        A_seg_map = B_seg_map = 0
        A_seg_map_load = B_seg_map_load = 0
        if (not self.opt.no_segmentation if self.opt.isTrain else bool(0)) or (not self.opt.no_seg_input):
            A_seg_map_path = self.A_seg_paths[index_A]
            A_seg_map_load = Image.open(A_seg_map_path).convert('RGB')
            A_seg_map = self.transform_Seg_A(A_seg_map_load)
            A_seg_map_load = np.array(A_seg_map_load)

            B_seg_map_path = self.B_seg_paths[index_B]  # unaligned
            B_seg_map_load = Image.open(B_seg_map_path).convert('RGB')
            B_seg_map = self.transform_Seg_B(B_seg_map_load)
            B_seg_map_load = np.array(B_seg_map_load)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_seg_map': A_seg_map, 'B_seg_map': B_seg_map, 'A_seg_map_load': A_seg_map_load, 'B_seg_map_load': B_seg_map_load}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
