#!/usr/bin/env python
# coding: utf-8
import torch
from torch.utils import data
try:
    from IPython.display import clear_output
except:
    pass
import matplotlib.pyplot as plt
import numpy as np
import copy
from pathlib import Path
import pickle
from .data_generator import get_data_generator
from .torch_ssim import ssim
from .plot_utils import plot_obs_pred
from .utils import binarize

def SSIM(output, target):
    '''
    Input:
        should be 5D (samples, channels, frames, heigth, width).
    Return:
        the structural similarity between two matrices. 
        (using only first dimension of channels, so it should have 1 channel to make it work)
    '''
    return ssim(output[:,0,...], target[:,0,...], size_average=1, data_range=1)


class Torch_Trainer():

    def __init__(self, train_data, val_data, get_model, get_loss_fx):
        '''
        Handles a pytorch models. 
         - If used for training give all arguments. 
         - If used for making predictions only the model is important -> Torch_Trainer(None, None, get_model, None)
        Inputs:
            train_data: training dataset path
            val_data: validation dataset path
            get_model: function that returns a model
                TODO: the arguments that this functions take are hardcoded, so it would be nice to make it parametric
            get_loss_fx: function that takes two matrices and returns the error between those
        '''
        self.train_data = train_data
        self.val_data = val_data
        self.get_model = get_model
        self.get_loss_fx = get_loss_fx
        self.model = None
        
    def train(self, params):
        '''
        * page references are from the thesis "Machine Learning models applied to Stom Nowcasting"
        
        Trains a pytorch model.
            Verbose is done through "show" key in 'params'.
            The optimizer is Adam.
            Uses a data_generator to fetch data chunk by chunk.
        Inputs:
            params: dictionary with all the parameters. If some keys are missing, the default values are use.
                  weight_decay   : 0,               # Parameter of the optimizer
                  eps            : 1e-08,           # Parameter of the optimizer
                  beta1          : 0.9,             # Parameter of the optimizer
                  beta2          : 0.999,           # Parameter of the optimizer
                  lr             : 0.0001,          # Learning rate
                  lr_steps       : 3,               # Steps evenly distributed across the max_epochs to reduce by 0.7 the learning rate
                  max_epochs     : 100,             # Maximum number of epochs to run the training
                  batch          : 4,               # Number of samples processed at a time
                  log_interval   : 10,              # Every how many epochs the verbose is shown
                  th             : 0                # Use only with 'mode=bin' to binarize the targets.
                  show           : True,            # Verbose flag
                  fix            : False,           # How to handle events longer than needed. False: random section is used on each epoch. True: mid point is fixed, same every epoch. (*pag. 33)
                  save_filename  : 'unnamed.pth',   # Name of the checkpoint, suggestion "architecture_framesIn_framesOut_loss.pth"
                  # model params
                  mode           : 'absolute',      # absolute, bin: targets a Z above 'th', diff:relative to the last context frame (*pag. 22)
                  in_frames      : 16,              # Number of input frames (context)
                  out_frames     : 16,              # Number of output frames (prediction)
                  step           : 1,               # Number of skipped frames (*pag. 34)
                  dropout        : 0.0,             # Parameter of dropout layers. TODO: implemented on some models only.
                  n_filter       : 128              # Initial number of filters/channels/features of convolutional layers.
                  
                 
        '''

          
        weight_decay    = params['weight_decay']  if 'weight_decay' in params else 0.0
        eps             = params['eps']           if 'eps'          in params else 1e-08
        beta1           = params['beta1']         if 'beta1'        in params else 0.9      
        beta2           = params['beta2']         if 'beta2'        in params else 0.999
        lr              = params['lr']            if 'lr'           in params else 0.0001
        lr_steps        = params['lr_steps']      if 'lr_steps'     in params else 3
        max_epochs      = params['max_epochs']    if 'max_epochs'   in params else 100
        batch           = params['batch']         if 'batch'        in params else 4
        log_interval    = params['log_interval']  if 'log_interval' in params else 10
        th              = params['th']/70         if 'th'           in params else 0
        show            = params['show']          if 'show'         in params else True        
        fix             = params['fix']           if 'fix'          in params else False   
        mode            = params['mode']  if 'mode' in params else 'absolute'  # options = absolute, diff, bin
        in_frames       = params['in_frames']     if 'in_frames'    in params else 16
        out_frames      = params['out_frames']    if 'out_frames'   in params else 16
        step            = params['step']          if 'step'         in params else 1
        dropout         = params['dropout']       if 'dropout'      in params else 0.0
        n_filter        = params['n_filter']      if 'n_filter'     in params else 128

        self.save_filename  = params['save_filename']  if 'save_filename' in params else 'unnamed.pth'
        self.params = params
        
        self.out_frames = out_frames



        # Generators
        training_set   = get_data_generator('torch',self.train_data,batch_size=batch,
                                         in_frame=in_frames, out_frame=out_frames, step=step, fix=fix)
        validation_set = get_data_generator('torch',self.val_data,batch_size=batch,
                                         in_frame=in_frames, out_frame=out_frames, step=step, fix=fix)

        training_generator = data.DataLoader(training_set, batch_size=1,shuffle=False, num_workers=0)
        validation_generator = data.DataLoader(validation_set, shuffle=False, num_workers=0)

        self.model = self.get_model(n_filter, dropout).float()
        
         # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True  # automatically detects best algorithm for the running machine     
        
        # Multi-GPU
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print("Using",n_gpus, "GPUs")
            model = nn.DataParallel(model)
            
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
        LR_step_size = int(max_epochs/lr_steps)
        gamma = 0.7
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

        self.model.cuda()
        train_loss_values = []
        val_loss_values = []
        val_ssim_values = []
        lossF = self.get_loss_fx()
        lr_acum = []
        best_vloss = 1000
        best_vssim = 0
        
        # Load Checkpoint to resume training if available
        start_epoch = 0
        best_accuracy = torch.tensor(0).float()
        try:
            checkpoint = torch.load(self.save_filename)
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(self.save_filename, checkpoint['epoch']))
        except: pass

        # Train
        for epoch in range(start_epoch,max_epochs):
            self.epoch = epoch
            # Training
            running_loss = 0.0
            self.model.train()
            for batch_idx, (context, target) in enumerate(training_generator):
                context, target = context[0,...], target[0,...] #to adjust batch size already given in the training set
             
                if mode=='diff':
                    base = context[:,-1:,...]
                    target -= base
                    base = base.to(device).float()
                elif mode=='bin':
                    target = torch.tensor(binarize(target,th))

                context, target = context.to(device).float(), target.to(device).float()
 
                # Model computations
                optimizer.zero_grad()
                output = self.model(context)
                loss = lossF(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 100 == 0 and False:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]'.
                        format(epoch, batch_idx * len(context), 10000*0.9, batch_idx * len(context)/ 10000/0.9*100))
            epoch_tloss = running_loss / len(training_generator)
            lr_acum.append(lr_scheduler.get_lr())
            lr_scheduler.step()
            
            if epoch > 0:
            	train_loss_values.append(epoch_tloss)
            self.model.eval()
            
            
            # Validation
            running_loss = 0.0
            acum_ssim = 0.0
            
            with torch.set_grad_enabled(False):
                for val_context, val_target in validation_generator:
                   # Transfer to GPU
                    val_context, val_target = val_context[0,...], val_target[0,...] 
                    
                    if mode=='diff':
                        val_base = val_context[:,-1:,...]
                        val_target -= val_base
                        val_base = val_base.to(device).float()
                    elif mode=='bin':
                        val_target = torch.tensor(binarize(val_target,th))

                    val_context, val_target = val_context.to(device).float(), val_target.to(device).float()
                    
                    # Model computations
                    val_output = self.model(val_context)
                    val_loss = lossF(val_output, val_target)
                    running_loss += val_loss.item()
                    
                    val_ssim = SSIM(val_output, val_target)
                    acum_ssim += val_ssim.item()
                epoch_vloss = running_loss / len(validation_generator)
                epoch_vssim = acum_ssim / len(validation_generator)
                self.val_loss = epoch_vloss
                if epoch > 0:
                	val_loss_values.append(epoch_vloss)
                	val_ssim_values.append(epoch_vssim)

                if epoch_vloss < best_vloss:
                    best_vloss = epoch_vloss
                    state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_model_loss = {'epoch': self.epoch + 1,'state_dict': state_dict,'best_accuracy': best_vloss}

                # To save a model checkpoint based on the SSIM
                #if epoch_vssim > best_vssim:
                #    best_vssim = epoch_vssim
                #    state_dict = copy.deepcopy(self.model.state_dict())
                #    self.best_model_ssim = {'epoch': self.epoch + 1,'state_dict': state_dict,'best_accuracy': epoch_vloss}
 
            # Verbose
            if epoch % log_interval == 0 and show==True and epoch > 0:
                try:
                    clear_output()
                except:
                    pass
                print('Train Epoch: {} Loss: {:.6f} Val_Loss: {:.6f}'.format(epoch, epoch_tloss, epoch_vloss))
                val_output = val_output[0,0].detach().cpu().numpy()
                val_target = val_target[0,0].detach().cpu().numpy()
                plot_obs_pred(val_target,val_output)
                if mode=='diff':
                    val_base = val_base[0,0].detach().cpu().numpy()
                    plot_obs_pred(val_target+val_base, val_output+val_base)

                plt.figure(figsize=(15,4))
                plt.subplot(1,3,1)
                plt.plot(train_loss_values)
                plt.plot(val_loss_values)
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend(['train_loss','val_loss'])
                plt.subplot(1,3,2)
                plt.plot(val_ssim_values)
                plt.xlabel('Epochs')
                plt.ylabel('MS-SSIM')
                plt.legend(['val'])
                plt.subplot(1,3,3)
                plt.plot(lr_acum)
                plt.xlabel('Epochs')
                plt.ylabel('Learning Rate')
                plt.legend(['lr'])
                plt.tight_layout()
                plt.show()

        try:
            return epoch_vloss
        except:
            print("No training happend. \n Make sure there is not a checkpoint trained for the same epochs you are trying to train.")


    def save_checkpoint(self, path=None):
        '''
        Save checkpoints.
        Inputs:
            path: optional directoy where to save the checkpoint. Default location: mlnowcasting/data/checkpoints/
        
        TODO: as shown in *pag. 39, stopping training when validation loss is at the minimum should be the best possible checkpoint. 
        So, implement stopping when validation loss starts increasing (use histeresis to missinterpretate noise as minimum)
        You can comment/uncomment the 3 saving in the fx as what suits you best. They are just to reproduce results.
        '''
        if path==None:
            base_dir = str(Path(__file__).parent.parent.parent)
            data_dir = '/data/checkpoints/'
            path = base_dir + data_dir

        state = {'epoch': self.epoch + 1,'state_dict': self.model.state_dict(),'best_accuracy': self.val_loss}
        torch.save(state, path + self.save_filename)
        
        try:
            torch.save(self.best_model_loss, path + 'bestLOSS_' + self.save_filename)
            #torch.save(self.best_model_ssim, path + 'bestSSIM_' + self.save_filename)
        except:
            pass
        filename = self.params['save_filename'].replace('pth','param')
        with open(path + filename,'wb') as my_file_obj:
                pickle.dump(self.params,my_file_obj) 


    def load_checkpoint(self, params, ckpoint_path=None , cuda=True, verbose=True):
        '''
        Load a checkpoint, to make predictions or to resume training.
        Inputs:
            params: see Class help
            ckpoint_path: checkpoint path. 
                If not given it looks for the checkpoint in mlnowcasting/data/checkpoints/ with the name of the params['save_filename']
            cuda: if True try to uses GPU
            verbose: print information
        '''
        if not self.model:
            self.model = self.get_model(params['n_filter'], params['dropout']).float()
            
        if not ckpoint_path:
            base_dir = str(Path(__file__).parent.parent.parent)
            data_dir = '/data/checkpoints/'
            path = base_dir + data_dir
            ckpoint_path = path + params['save_filename']
        
        use_gpu = torch.cuda.is_available()
        if cuda and use_gpu:
            self.model.cuda()
            checkpoint = torch.load(ckpoint_path)
        else:
            checkpoint = torch.load(ckpoint_path, map_location=torch.device('cpu') )
        
        self.model.eval()
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        self.model.load_state_dict(checkpoint['state_dict'])
        if verbose:
            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(params['save_filename'], checkpoint['epoch']))
