
from .utils.torch_trainer import Torch_Trainer
from .utils.utils import any2reflectivity, any2normalize
import os, sys
import pickle
import re
import torch
import numpy as np

def _is_path(name):
    return '/' in name

def get_params(filename):
    '''
    filename: can be './dir/model_name.param', './dir/model_name.pth', or 'model_name'
    '''
    if _is_path(filename) :
        if '.param' in filename:
            params = pickle.load(open(filename,'rb'));
        else: #means that the extension is '.pth'
            path = os.path.dirname(os.path.abspath(filename)) + '/'
            params_file = os.path.basename(filename).replace('pth','param')
            param_filename = path + params_file
            params = pickle.load(open(param_filename,'rb'));
    else: # use default directory
        param_filename = '../data/checkpoints/' + filename + '.param'
        params = pickle.load(open(param_filename,'rb'));
                                                                                                
    return params   
                                                               
def get_model_fx(model_name):
    '''
    model_name: can be only 'model_name'
    '''
    # replace any '_binXX' to import the generic model
    model_file = re.sub(r'_bin\d\d', '', model_name)
    exec("from {m}.{s} import {f}".format(m='.models.architectures', s=model_file,  f='get_model'), globals())
    return get_model
                                                               
    
def predict(dataset, model_name, checkpoint_filename=None, params_filename=None, N=None):
    '''
    Parameters:
        dataset: path of dataset or numpy array; 
            e.g. for path '../data/datasets/example_dataset.npy' or 'example_dataset'
        model_name: name of the model; 
            e.g. 'resConv_16_16'
        checkpoint_filename: path of the checkpoint (optional). Use use 'model_name' if not given;
            e.g. './dir/resConv_16_16.pth'
        params_filename: path of the params (optional). Use 'checkpoint_filename' if given, use 'model_name' if none is given);
            e.g. './dir/resConv_16_16.param'
        N: specific event of the dataset to make the prediction for (optional). N < len(dataset)
    Returns:
        predictions: numpy array with the predictions
        target:  numpy array with the observations (only if the dataset has those frames, otherwise None)
    '''
    # params can be retrieve from explicit full path, path of checkpoint, model name
    if params_filename is None and checkpoint_filename is None:
        params_filename = model_name
    elif params_filename is None and checkpoint_filename is not None:
        params_filename = checkpoint_filename
                  
    if isinstance(dataset, np.ndarray):
        data = dataset
    else:
        if not _is_path(dataset): # if not path (just filename) use default location
            dataset = '../data/datasets/'+ dataset.replace('.npy','') + '.npy'
        data = np.load(dataset)
    
                                                             
    # import all files                                                         
    params     = get_params(params_filename)
    get_model  = get_model_fx(model_name)
                                                            
    # define the observation/s to use
    if N is None:
        n0 = 0
        n1 = len(data)
    else:
        n0 = N
        n1 = N+1

    # defines in what units the outputs are going to be.
    conversion = any2normalize if params['mode'] == 'bin' else any2reflectivity
    
    use_gpu = torch.cuda.is_available()
    with torch.no_grad():
        T = Torch_Trainer(None, None, get_model, None)
        T.load_checkpoint(params, ckpoint_path=checkpoint_filename , cuda=use_gpu)
        device = torch.device("cuda:0" if use_gpu else "cpu")
        fi = params['in_frames']
        fo = params['out_frames']
        context = np.expand_dims(data[n0:n1, :fi],1)/255
        if data.shape[1] >= fi+fo:
            target = np.expand_dims(data[n0:n1, fi:fi+fo],1)/255
            target = conversion(target.squeeze(1))
        else:
            target = None
        prediction_device = T.model(torch.tensor(context).to(device).float())
    
    if params['mode'] == 'diff':
        prediction_device += torch.tensor(context[:,-1:]).to(device).float()
    prediction = conversion(prediction_device)
    if use_gpu:
        # release CUDA memory
        T = prediction_device = None
        torch.cuda.empty_cache()
    return prediction.squeeze(1), target
   
