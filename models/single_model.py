import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
# import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
import sys
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter
from .ssim import ssim, ms_ssim, SSIM, MS_SSIM


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)
        self.input_A_I = self.Tensor(nb, 1, size, size)

        self.max = networks.max_operation()
        self.edge = networks.edge_operation()

        self.radiux = [2, 4, 8, 16, 32]
        self.eps_list = [0.001, 0.0001]
        
        #PerceptualLoss
        if opt.vgg > 0:#nonono
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        #SemanticLoss
        elif opt.fcn > 0:#noonono
            self.fcn_loss = networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = networks.load_fcn("./model")
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks
        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)
 
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if self.opt.use_max:
                self.load_network(self.max, 'max', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            if opt.use_wgan:#0
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                #print("useganloss")
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:#0
                #print("usemseloss")
                self.criterionCycle = torch.nn.MSELoss()
            else:
                #print("useLIloss")
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))


        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)

        if opt.isTrain:
            self.netG_A.train()
            # self.netG_B.train()
        else:
            self.netG_A.eval()
            # self.netG_B.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        #print("set_input")
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        input_A_I = input['A_I']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_A_I.resize_(input_A_I.size()).copy_(input_A_I)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def test(self):
        #print("testself")
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)


        self.real_B = Variable(self.input_B, volatile=True)

    def predict(self):
        #print("defpredict")
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        self.real_A_gray_o = self.real_A_gray
        self.real_img = Variable(self.input_img)
        if self.opt.use_max:
            self.real_A_gray = self.max(self.real_A_gray)
        if self.opt.use_edge:#,,,
            self.edge_out = self.edge(self.real_A_gray_o)
            self.real_A_gray = torch.cat([self.edge_out, self.real_A_gray], 1)
        if self.opt.img_decomposition:#.......................
            lf, hf = self.decomposition(self.real_img)
            self.real_A_gray = torch.cat([hf, self.real_A_gray], 1)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))

        if self.opt.skip == 1:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_img)

        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)


        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([ ('fake_B', fake_B)]) 

    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def forward(self):#............
        #print("defforward")
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_A_gray_o = self.real_A_gray
        #self.real_A_I = Variable(self.input_A_I)
        self.real_img = Variable(self.input_img)
        if self.opt.use_max:
            self.real_A_gray = self.max(self.real_A_gray)
        if self.opt.use_edge:
            self.edge_out = self.edge(self.real_A_gray_o)
            self.real_A_gray = torch.cat([self.edge_out, self.real_A_gray], 1)
        if self.opt.img_decomposition:
            lf, hf = self.decomposition(self.real_img)
            self.real_A_gray = torch.cat([hf, self.real_A_gray], 1)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.fake_B= self.netG_A.forward(self.real_img, self.real_img)
            
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        


    def backward_G(self, epoch):#...............
        #print("backward_G")
        mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        l1_fn = torch.nn.L1Loss(size_average=True)
        self.loss_G_A = mse_fn(self.fake_B, self.real_B)
        self.real_smooth_loss = F.smooth_l1_loss(self.fake_B, self.real_B)
        self.loss_l1_fn = l1_fn(self.fake_B, self.real_B)
        self.ms_ssim_loss = 1 - ms_ssim( self.fake_B, self.real_B, data_range=255, size_average=True )
        
        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                             self.fake_B,
                                                             self.real_B) * self.opt.vgg if self.opt.vgg > 0 else 0

            self.loss_G = self.loss_G_A + self.loss_vgg_b * vgg_w

        else:
            #self.loss_G = self.loss_G_A
            #self.loss_G = self.loss_G_A+self.real_smooth_loss
            self.loss_G=0.8*self.ms_ssim_loss+0.2*self.loss_G_A
        print(self.loss_G)
        # self.loss_G = self.L1_AB + self.L1_BA
        self.loss_G.backward() 


    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A

    def get_current_errors(self, epoch):
        G_A = self.loss_G_A.item()
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.item() / self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict([('G_A', G_A), ("vgg", vgg)])
        else:
            return OrderedDict([('G_A', G_A),("vgg", 0)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)


        return OrderedDict([('real_A', real_A), ('fake_B', fake_B),('real_B', real_B)])
        

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)

    def update_learning_rate(self):

        if self.opt.new_lr:
            lr = self.old_lr / 2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def decomposition(self, x):
            #print("decomposition")
            LF_list = []
            HF_list = []
            res = self.get_residue(x)
            res = res.repeat(1, 3, 1, 1)
            for radius in self.radiux:
                for eps in self.eps_list:
                    self.gf = GuidedFilter(radius, eps)
                    LF = self.gf(res, x)
                    LF_list.append(LF)
                    HF_list.append(x - LF)
            LF = torch.cat(LF_list, dim=1)
            HF = torch.cat(HF_list, dim=1)
            return LF, HF

    def get_residue(self,tensor):
        max_channel = torch.max(tensor, dim=1, keepdim=True)
        min_channel = torch.min(tensor, dim=1, keepdim=True)
        res_channel = max_channel[0] - min_channel[0]
        return res_channel
        