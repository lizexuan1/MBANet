# from __future__ import print_function
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import torch
import os
import collections
from torch.optim import lr_scheduler
import torch.nn.init as init
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import torch.nn.functional as F

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def illufeature2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = np.expand_dims(image_numpy[:,:,16], 2)
    image_numpy = image_numpy * 255.0
    image_numpy = np.concatenate([image_numpy,image_numpy,image_numpy],2)
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def lowfeature2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = np.expand_dims(image_numpy[:,:,16], 2)
    image_numpy = image_numpy * 255.0
    image_numpy = np.concatenate([image_numpy, image_numpy, image_numpy], 2)
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def illu2img(image_numpy, imtype=np.uint8):
    image_numpy = image_numpy * 255.0
    image_numpy = np.concatenate([image_numpy, image_numpy, image_numpy], 2)
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def low2img(image_numpy, imtype=np.uint8):
    image_numpy = image_numpy * 255.0
    image_numpy = np.concatenate([image_numpy, image_numpy, image_numpy], 2)
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


# def visualizedfeature(illu_list, original_list, divided_list, conv_list):
#     data_root = 'D:/code/RetinexGAN-edge/ablation/nolocal/test_200/images'
#
#     if len(illu_list) == len(original_list) == len(divided_list) == len(conv_list):
#         for i in range(len(illu_list)):
#             illu = np.transpose(illu_list[i][0].cpu().float().numpy(), (1, 2, 0))
#             orig = np.transpose(original_list[i][0].cpu().float().numpy(), (1, 2, 0))
#             divi = np.transpose(divided_list[i][0].cpu().float().numpy(), (1, 2, 0))
#             conv = np.transpose(conv_list[i][0].cpu().float().numpy(), (1, 2, 0))
#             channels = illu.shape[2]
#             print(channels)
#
#             for j in range(channels):
#
#                 illu_map = illu2img(np.expand_dims(illu[:, :, j], 2))
#                 orig_map = low2img(np.expand_dims(orig[:, :, j], 2))
#                 divi_map = low2img(np.expand_dims(divi[:, :, j], 2))
#                 conv_map = low2img(np.expand_dims(conv[:, :, j], 2))
#
#                 illu_root = data_root + '/' + 'illu_' + str(i + 1) #+ '/' + str(j + 1) + '.png'
#                 orig_root = data_root + '/' + 'orig_' + str(i + 1) #+ '/' + str(j + 1) + '.png'
#                 divi_root = data_root + '/' + 'divi_' + str(i + 1) #+ '/' + str(j + 1) + '.png'
#                 conv_root = data_root + '/' + 'conv_' + str(i + 1)
#
#                 if not os.path.exists(illu_root):
#                     os.makedirs(illu_root)
#
#                 if not os.path.exists(orig_root):
#                     os.makedirs(orig_root)
#
#                 if not os.path.exists(divi_root):
#                     os.makedirs(divi_root)
#
#                 if not os.path.exists(conv_root):
#                     os.makedirs(conv_root)
#
#                 illu_path = illu_root + '/' + str(j + 1) + '.png'
#                 orig_path = orig_root + '/' + str(j + 1) + '.png'
#                 divi_path = divi_root + '/' + str(j + 1) + '.png'
#                 conv_path = conv_root + '/' + str(j + 1) + '.png'
#
#                 save_image(illu_map, illu_path)
#                 save_image(orig_map, orig_path)
#                 save_image(divi_map, divi_path)
#                 save_image(conv_map, conv_path)
#
#     return


def visualizedfeature(upfeatures, copyfeatures, illu_features, upfeatures_before, copyfeatures_before, gray_5, L_b_5, L_5):
    data_root = 'D:/code/RetinexGAN-edge/ablation/graye+gg1+s/test_200/images'
    g_5 = np.transpose(gray_5[0].cpu().float().numpy(), (1, 2, 0))
    l_b_5 = np.transpose(L_b_5[0].cpu().float().numpy(), (1, 2, 0))
    l_5 = np.transpose(L_5[0].cpu().float().numpy(), (1, 2, 0))

    channels = g_5.shape[2]
    print(channels)

    for j in range(channels):

        g5_map = low2img(np.expand_dims(g_5[:, :, j], 2))
        l_b_5_map = low2img(np.expand_dims(l_b_5[:, :, j], 2))
        l_5_map = low2img(np.expand_dims(l_5[:, :, j], 2))


        illu_0_root = data_root + '/' + 'illu_0'  # + '/' + str(j + 1) + '.png'
        l_b_0_root = data_root + '/' + 'l_b_0'   # + '/' + str(j + 1) + '.png'
        l_0_root = data_root + '/' + 'l_0'  # + '/' + str(j + 1) + '.png'

        if not os.path.exists(illu_0_root):
            os.makedirs(illu_0_root)
        if not os.path.exists(l_b_0_root):
            os.makedirs(l_b_0_root)
        if not os.path.exists(l_0_root):
            os.makedirs(l_0_root)

        illu_0_path = illu_0_root + '/' + str(j + 1) + '.png'
        l_b_0_path = l_b_0_root + '/' + str(j + 1) + '.png'
        l_0_path = l_0_root + '/' + str(j + 1) + '.png'

        save_image(g5_map, illu_0_path)
        save_image(l_b_5_map, l_b_0_path)
        save_image(l_5_map, l_0_path)

    if len(upfeatures) == len(copyfeatures):
        for i in range(len(copyfeatures)):
            up = np.transpose(upfeatures[i][0].cpu().float().numpy(), (1, 2, 0))
            copy = np.transpose(copyfeatures[i][0].cpu().float().numpy(), (1, 2, 0))
            illu = np.transpose(illu_features[i][0].cpu().float().numpy(), (1, 2, 0))
            up_b = np.transpose(upfeatures_before[i][0].cpu().float().numpy(), (1, 2, 0))
            copy_b = np.transpose(copyfeatures_before[i][0].cpu().float().numpy(), (1, 2, 0))

            channels = up.shape[2]
            print(channels)

            for j in range(channels):

                up_map = illu2img(np.expand_dims(up[:, :, j], 2))
                copy_map = low2img(np.expand_dims(copy[:, :, j], 2))
                illu_map = low2img(np.expand_dims(illu[:, :, j], 2))
                up_b_map = low2img(np.expand_dims(up_b[:, :, j], 2))
                copy_b_map = low2img(np.expand_dims(copy_b[:, :, j], 2))

                up_root = data_root + '/' + 'up_' + str(i + 1) #+ '/' + str(j + 1) + '.png'
                copy_root = data_root + '/' + 'copy_' + str(i + 1) #+ '/' + str(j + 1) + '.png'
                illu_root = data_root + '/' + 'illu_' + str(i + 1)  # + '/' + str(j + 1) + '.png'
                up_b_root = data_root + '/' + 'up_b_' + str(i + 1)  # + '/' + str(j + 1) + '.png'
                copy_b_root = data_root + '/' + 'copy_b_' + str(i + 1)  # + '/' + str(j + 1) + '.png'

                if not os.path.exists(up_root):
                    os.makedirs(up_root)
                if not os.path.exists(copy_root):
                    os.makedirs(copy_root)
                if not os.path.exists(illu_root):
                    os.makedirs(illu_root)
                if not os.path.exists(up_b_root):
                    os.makedirs(up_b_root)
                if not os.path.exists(copy_b_root):
                    os.makedirs(copy_b_root)

                up_path = up_root + '/' + str(j + 1) + '.png'
                copy_path = copy_root + '/' + str(j + 1) + '.png'
                illu_path = illu_root + '/' + str(j + 1) + '.png'
                up_b_path = up_b_root + '/' + str(j + 1) + '.png'
                copy_b_path = copy_b_root + '/' + str(j + 1) + '.png'

                save_image(up_map, up_path)
                save_image(copy_map, copy_path)
                save_image(illu_map, illu_path)
                save_image(up_b_map, up_b_path)
                save_image(copy_b_map, copy_b_path)

    return

def adjust(input, adj_percent):
    minn = np.min(input)
    input = input - minn
    maxx = np.max(input)
    input = input / maxx
    return input



def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.where(image_numpy > 1, 1, image_numpy)
    image_numpy = np.where(image_numpy < 0, 0, image_numpy)
    image_numpy = np.power(image_numpy, 1 / 1)
    image_numpy = adjust(image_numpy,[0.01,0.93])
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def atten2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor[0]
    #image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy/(image_numpy.max()/255.0)
    return image_numpy.astype(imtype)

def latent2im(image_tensor, imtype=np.uint8):
    # image_tensor = (image_tensor - torch.min(image_tensor))/(torch.max(image_tensor)-torch.min(image_tensor))
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def getvision(low_tensor, high_tensor, illu, imtype=np.uint8):
    low = low_tensor[0].cpu().float().numpy()
    #low = (np.transpose(low, (1, 2, 0)) + 1) / 2.0 * 255.0
    low = (np.transpose(low, (1, 2, 0))) * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    low = np.maximum(low, 0)
    low = np.minimum(low, 255)

    high = high_tensor[0].cpu().float().numpy()
    #high = (np.transpose(high, (1, 2, 0)) + 1) / 2.0 * 255.0
    high = (np.transpose(high, (1, 2, 0))) * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    high = np.maximum(high, 0)
    high = np.minimum(high, 255)

    latent = high - low

    illu = illu[0]
    illu = illu.cpu().float().numpy()
    illu = np.transpose(illu, (1, 2, 0))
    f_L = ((latent / 255.0) * illu) * 255.0

    f_L = np.maximum(f_L, 0)
    f_L = np.minimum(f_L, 255)

    return latent.astype(imtype), f_L.astype(imtype)


def getvision(low_tensor, high_tensor, illu, imtype=np.uint8):
    low = low_tensor[0].cpu().float().numpy()
    #low = (np.transpose(low, (1, 2, 0)) + 1) / 2.0 * 255.0
    low = (np.transpose(low, (1, 2, 0))) * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    low = np.maximum(low, 0)
    low = np.minimum(low, 255)

    high = high_tensor[0].cpu().float().numpy()
    #high = (np.transpose(high, (1, 2, 0)) + 1) / 2.0 * 255.0
    high = (np.transpose(high, (1, 2, 0))) * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    high = np.maximum(high, 0)
    high = np.minimum(high, 255)

    latent = high - low

    illu = illu[0]
    illu = illu.cpu().float().numpy()
    illu = np.transpose(illu, (1, 2, 0))
    f_L = ((latent / 255.0) * illu) * 255.0

    f_L = np.maximum(f_L, 0)
    f_L = np.minimum(f_L, 255)

    return latent.astype(imtype), f_L.astype(imtype)


def max2im(image_1, image_2, imtype=np.uint8):
    image_1 = image_1[0].cpu().float().numpy()
    image_2 = image_2[0].cpu().float().numpy()
    #image_1 = (np.transpose(image_1, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_1 = (np.transpose(image_1, (1, 2, 0))) * 255.0
    image_2 = (np.transpose(image_2, (1, 2, 0))) * 255.0
    output = np.maximum(image_1, image_2)
    output = np.maximum(output, 0)
    output = np.minimum(output, 255)
    return output.astype(imtype)

def variable2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].data.cpu().float().numpy()
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    e_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = scipy.misc.toimage(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)

    return init_fun
    
def color_constency_loss(enhances):
    plane_avg = enhances.mean((2, 3))
    col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                          + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                          + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
    return col_loss

#def color_constency_loss(enhances, originals):
    #enh_cols = enhances.mean((2, 3))
    #ori_cols = originals.mean((2, 3))
    #rg_ratio = (enh_cols[:, 0] / enh_cols[:, 1] - ori_cols[:, 0] / ori_cols[:, 1]).abs()
    #gb_ratio = (enh_cols[:, 1] / enh_cols[:, 2] - ori_cols[:, 1] / ori_cols[:, 2]).abs()
    #br_ratio = (enh_cols[:, 2] / enh_cols[:, 0] - ori_cols[:, 2] / ori_cols[:, 0]).abs()
    #col_loss = (rg_ratio + gb_ratio + br_ratio).mean()
    #return col_loss
    
def alpha_total_variation(A):
    '''
    Links: https://remi.flamary.com/demos/proxtv.html
           https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
    '''
    delta_h = A[:, :, 1:, :] - A[:, :, :-1, :]
    delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]

    # TV used here: L-1 norm, sum R,G,B independently
    # Other variation of TV loss can be found by google search
    tv = delta_h.abs().mean((2, 3)) + delta_w.abs().mean((2, 3))
    loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
    return loss

#def rgb2gray(rgb):
 
    #r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
 
    #return gray 
def rgb_to_gray(x):
    R = x[:, 0:1, :, :]
    G = x[:, 1:2, :, :]
    B = x[:, 2:3, :, :]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    #gray = torch.max(R,1,True) + torch.max(G,1,True) + torch.max(B,1,True)
    return gray

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()