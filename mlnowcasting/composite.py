import context
import sys
import torch
import numpy as np
import pickle
import re

from mlnowcasting.models.architectures.example_model import get_model
from mlnowcasting.models.params.example_params import params
from mlnowcasting.utils.torch_trainer import Torch_Trainer
from mlnowcasting.utils.utils import any2reflectivity
from .predict import predict



def composite_prediction(dataset, base_model_file, N=None):

    ths = [0,20,25,30,35,40,45,50]
    for i in range(len(ths)):
        th = ths[i]
        if th==0:            
            base_layer, target = predict(dataset, base_model_file, N=N) 
            samples, frames, H, W = base_layer.shape
            layers = np.zeros((len(ths),samples, frames, H, W))
            layers[i] = base_layer
        else:
            th_layer_file = str(base_model_file) + '_bin' + str(th)
            try:
                layers[i], target = predict(dataset, th_layer_file, N=N) 
            except Exception as e: 
                print(e)
                pass

    first = True
    for i in range(len(ths)):
        if first:
            composite = layers[i] # base layer
            first = False
        else:
            layer_binzarize = np.where(layers[i]>0.5, ths[i], 0) 
            composite = np.maximum(composite,layer_binzarize)
    return composite, target, layers

def main():
    model_file    = sys.argv[1] # location of the model
    params_file   = sys.argv[2] # location of the param file (which has the weights' file location)
    ths           = sys.argv[3]
    dataset       = sys.argv[4] # location of the npy file with the events you want to get predictions for 
    save_filepath = sys.argv[5] # directory + filename for the npy file to be saved with the predictions 
    
    predict, _, _ = composite_prediction(model_file, params_file, ths)
    np.save(save_filepath, predict)

if __name__ == "__main__":
    main()
