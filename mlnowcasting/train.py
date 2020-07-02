import sys

from .utils.torch_trainer import Torch_Trainer
from .utils.utils import any2reflectivity, any2normalize
import os, sys
import pickle
import re
import torch
import numpy as np


def _is_path(name):
    return '/' in name

def _get_params(params_name):
    '''
    params_name: 'params_name'
    '''
    exec("from {m}.{s} import {f}".format(m='.models.params', s=params_name,  f='params'), globals())
    return params   
                                                               
def _get_model_fx(model_name):
    '''
    model_name: can be only 'model_name'
    '''
    # replace any '_binXX' to import the generic model
    model_file = re.sub(r'_bin\d\d', '', model_name)
    exec("from {m}.{s} import {f}".format(m='.models.architectures', s=model_file,  f='get_model'), globals())
    return get_model
 
def _get_loss_fx(loss_name):
    '''
    model_name: can be only 'model_name'
    '''
    exec("from {m}.{s} import {f}".format(m='.models.losses', s=loss_name,  f='get_loss_fx'), globals())
    return get_loss_fx                                                             
    
def train(taining_dataset, validation_dataset, model_name, loss_name, params_module=None, save_dir=None):
    '''
    Parameters:
        taining_dataset/validation_dataset: path of dataset or dataset name if in default directory; 
            e.g. for path '../data/datasets/example_dataset.npy' or 'example_dataset'
        model_name: name of the model; 
            e.g. 'resConv_16_16'
        loss_name: name of the loss file;
            e.g. 'example_loss'
        params_module: path of the params .py file (optional). Otherwise, use 'model_name' to find a params file in the default directory;
            e.g. './dir/resConv_16_16.py'
    Returns:
        model: train model
    '''
    # params can be retrieve from explicit full path, or model name
    if params_module is None:
        params_module = model_name
        
    if not _is_path(taining_dataset): # if not path (just filename) use default location
        taining_dataset = '../data/datasets/'+ taining_dataset.replace('.npy','') + '.npy'
    if not _is_path(validation_dataset): # if not path (just filename) use default location
        validation_dataset = '../data/datasets/'+ validation_dataset.replace('.npy','') + '.npy'
                                              
    # import all files                                                         
    params       = _get_params(params_module)
    get_model    = _get_model_fx(model_name)    
    get_loss_fx  = _get_loss_fx(loss_name)

    model = Torch_Trainer(taining_dataset, validation_dataset, get_model, get_loss_fx)
    model.train(params)
    model.save_checkpoint(path=save_dir)
    print("Training finished")
    return model
