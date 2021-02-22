import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

from .segmentation import getCrossEntropyInput, getFakeSegMap
from util import util
from data.base_dataset import get_transform
import PIL
import numpy as np
from torch.autograd import Variable

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        return parser

        """
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser
        """

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        if self.isTrain:
            if not self.opt.no_segmentation:
                self.loss_names = self.loss_names + ['G_Seg_A', 'G_Seg_B']
            if not self.opt.no_ganFeat_loss:
                self.loss_names = self.loss_names + ['G_Feat_A', 'G_Feat_B']
        #self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # determine netG_input_nc
        netG_input_nc = opt.input_nc
        if not opt.no_seg_input:
            netG_input_nc += opt.seg_nc

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators (add feature matching loss option)
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.num_D,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, not opt.no_ganFeat_loss, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.num_D,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, not opt.no_ganFeat_loss, self.gpu_ids) # input=>output

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLossHD(use_lsgan=1, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # define segmentation loss
            self.criterionSeg = torch.nn.CrossEntropyLoss()
            # dfine feature matching loss
            self.criterionFeat = torch.nn.L1Loss()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # load segmentaions for inputs of generator
        self.A_seg_map = input['A_seg_map' if AtoB else 'B_seg_map'].to(self.device)
        self.B_seg_map = input['B_seg_map' if AtoB else 'A_seg_map'].to(self.device)
        # load segmentations for calculating segmentation losses
        self.A_seg_map_load = input['A_seg_map_load' if AtoB else 'B_seg_map_load']
        self.B_seg_map_load = input['B_seg_map_load' if AtoB else 'A_seg_map_load']

        # if segmentation, encode the input
        if not self.opt.no_seg_input:
            self.input_A = self.catImgAndSeg(self.real_A, self.A_seg_map)
            self.input_B = self.catImgAndSeg(self.real_B, self.B_seg_map)
        else:
            self.input_A = self.real_A
            self.input_B = self.real_B

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.input_A)  # G_A(A)
        self.fake_A = self.netG_B(self.input_B)  # G_B(B)

        if not self.opt.no_seg_input:
            fakeSegMap_A = self.getSegMap(self.fake_A, self.opt.A_centers)
            fakeSegMap_B = self.getSegMap(self.fake_B, self.opt.B_centers)

            self.input_fake_A = self.catImgAndSeg(self.fake_A, fakeSegMap_A)
            self.input_fake_B = self.catImgAndSeg(self.fake_B, fakeSegMap_B)
        else:
            self.input_fake_A = self.fake_A
            self.input_fake_B = self.fake_B

        self.rec_A = self.netG_B(self.input_fake_B)  # G_B(G_A(A))
        self.rec_B = self.netG_A(self.input_fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.input_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.input_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        #pred_fake_B = self.netD_A(self.fake_B)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        #loss_G_A_Test = self.criterionGANTest(pred_fake_B, True)
        # GAN loss D_B(G_B(B))
        #pred_fake_A = self.netD_B(self.fake_A)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        #loss_G_B_Test = self.criterionGANTest(pred_fake_A, True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # add segmentation loss
        self.loss_G_Seg_A = 0
        self.loss_G_Seg_B = 0
        if not self.opt.no_segmentation:
            # A
            RecSegTensorA, RealSegTensorA = getCrossEntropyInput(self.rec_A.data, self.opt.A_centers, self.A_seg_map_load.data[0])
            self.loss_G_Seg_A = self.criterionSeg(RecSegTensorA, RealSegTensorA) * self.opt.lambda_Seg_A
            # B
            self.fakeSegTensorB, self.realSegTensorB = getCrossEntropyInput(self.rec_B, self.opt.B_centers, self.B_seg_map_load.data[0])
            self.loss_G_Seg_B = self.criterionSeg(self.fakeSegTensorB, self.realSegTensorB) * self.opt.lambda_Seg_B

        # add feature matching loss
        pred_rec_A = self.netD_B(self.rec_A)
        pred_real_A = self.netD_B(self.real_A)

        pred_rec_B = self.netD_A(self.rec_B)
        pred_real_B = self.netD_A(self.real_B)
        self.loss_G_Feat_A = self.loss_G_Feat_B = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                # A
                for j in range(len(pred_rec_A[i]) - 1):
                    self.loss_G_Feat_A += D_weights * feat_weights * \
                                          self.criterionFeat(pred_rec_A[i][j],pred_real_A[i][j].detach()) * self.opt.lambda_feat_A
                # B
                for j in range(len(pred_rec_B[i]) - 1):
                    self.loss_G_Feat_B += D_weights * feat_weights * \
                                          self.criterionFeat(pred_rec_B[i][j],pred_real_B[i][j].detach()) * self.opt.lambda_feat_B


        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_Seg_A + self.loss_G_Seg_B + self.loss_G_Feat_A + self.loss_G_Feat_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def getSegMap(self, Img, centers):
        Img_I = util.tensor2im(Img)
        if len(Img_I.shape) > 2:
            Img_I = Img_I[:, :, 0]
        SegMap = getFakeSegMap(Img_I, centers)
        # PIL=>uint8=>Tensor=>cuda
        SegMap = SegMap.astype(np.uint8)
        SegMap = torch.from_numpy(SegMap)
        SegMap = Variable(SegMap.data.cuda())# cpu=>gpu
        return SegMap

    def catImgAndSeg(self, Img, Seg):
        size = Seg.size()
        while len(size) < 4:
            Seg = torch.unsqueeze(Seg, 0)
            size = Seg.size()
        oneHot_size = (size[0], self.opt.seg_nc, size[2], size[3])
        input_seg = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_img_seg = input_seg.scatter_(1, Seg.data.long().cuda(), 1.0)
        catImgSeg = torch.cat((Img, input_img_seg), dim=1)
        return catImgSeg