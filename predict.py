from time import time
import os
import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
t1 = time()
opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join("./results/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))
with torch.no_grad():
    for i, data in enumerate(dataset):
        model.set_input(data)
        visuals = model.predict()
        img_path = model.get_image_paths()
        #print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
t2 = time()
t = t2 - t1
print('time=',t)
