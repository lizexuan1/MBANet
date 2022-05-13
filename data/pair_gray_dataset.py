import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch
from pdb import set_trace as st

import numpy as np

class Pair_Gray_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        transform_list = []
        '''
        transform_list += [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        '''
        transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('RGB')
        #B_img = Image.open(A_path.replace("low", "normal").replace("A", "B")).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        '''
        BB = np.asarray(B_img)#.max()
        B_img.save('/data/hzh/experiment/RetinexGAN-edge/checkpoints/derain_YCrCb/web/BBBB1.JPG')
        print(BB)
        print('*******\n%s\n%s\n*******\n' % (A_path,B_path))
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        B_img = B_img.cpu().float().numpy()
        B_img = np.transpose(B_img, (1, 2, 0)) * 255.0
        B_img = np.maximum(B_img, 0)
        B_img = np.minimum(B_img, 255)
        B_img = B_img.astype(np.uint8)
        #B_img = Image.fromarray(np.uint8(B_img))
        B_img = Image.fromarray(B_img)
        BBB=B_img#.max()
        print(BBB)
        B_img.save('/data/hzh/experiment/RetinexGAN-edge/checkpoints/derain_YCrCb/web/BBBB2.JPG')
        exit()
        
        A = transforms.ToPILImage()(A_img)
        B = transforms.ToPILImage()(B_img)
        A.save('/data/hzh/experiment/RetinexGAN-edge/checkpoints/derain_YCrCb/a.jpg')
        B.save('/data/hzh/experiment/RetinexGAN-edge/checkpoints/derain_YCrCb/b.jpg')
        exit()
        
        A = A_img.float().numpy()
        A2 = A_img2.float().numpy()
        print(A)
        print('#############################')
        print(A2)
        exit()
        A.mode = 'YCbCr'
        A.save('/data/hzh/experiment/RetinexGAN-edge/checkpoints/derain_YCrCb/web/11.JPG')
        A = A.convert('RGB')
        print(A)
        A.save('/data/hzh/experiment/RetinexGAN-edge/checkpoints/derain_YCrCb/web/12.JPG')
        A.mode = 'YCbCr'
        print(A)
        A.save('/data/hzh/experiment/RetinexGAN-edge/checkpoints/derain_YCrCb/web/11.JPG')
        exit()
        '''
        w = A_img.size(2)
        h = A_img.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A_img = A_img[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B_img = B_img[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]


        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            #r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            #A_gray = (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            #A_gray = input_img[0,:,:]
            #A_CrCb = input_img[1:3,:,:]
            #B_gray = B_img[0,:,:]
            #B_CrCb = B_img[1:3,:,:]
            #A_gray = np.expand_dims(A_gray,axis = 1)
            A_gray = torch.unsqueeze(A_gray, 0)
            B_gray = torch.unsqueeze(B_gray, 0)
            #print(B_gray.shape)
            
            
            #B_gray = torch.cat((B_gray, B_gray, B_gray), 0)
            #B_gray = B_gray.cpu().float().numpy()
            #B_gray = (np.transpose(B_gray, (1, 2, 0))) * 255.0
            #B_gray = B_gray/(B_gray.max()/255.0)
            #B_gray.astype(np.uint8)
            #print(B_gray.type)
            #exit()
            #B_gray = B_gray[:,:,0]
           # B_gray = Image.fromarray(np.uint8(np.array(B_gray)))
            #B_gray.save('/data/hzh/experiment/RetinexGAN-edge/checkpoints/derain_YCrCb/web/gray.JPG')
            #exit()
        return {'A': A_img, 'B': B_gray, 'A_gray': A_gray, 'input_img':A_gray, 'A_I':A_gray,
                'A_paths': A_path, 'B_paths': B_path, 'A_CrCb':A_CrCb, 'B_CrCb':B_CrCb}
        #return {'A': A_gray, 'B': B_img, 'A_gray': A_gray,'input_img':input_img， 'A_I':A_gray,
                #'A_paths': A_path, 'B_paths': B_path}
                
    def __len__(self):
        return self.A_size

    def name(self):
        return 'Pair_Gray_Dataset'
