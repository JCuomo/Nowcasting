import os, sys
import warnings
warnings.filterwarnings("ignore")
import gc 

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
        
        
def rainnet(context):

    with HiddenPrints():
        from keras.models import load_model
        from tensorflow.keras.backend import clear_session

        import copy
        import numpy as np
        model = load_model('../data/checkpoints/RainNet.h5')       

        obs = copy.copy(context)
        obs = np.moveaxis(obs,1,3)
        for i in range(16):
            pred = model.predict(obs)
            obs[:,:,:,:-1] = obs[:,:,:,1:]
            obs[:,:,:,-1:] = pred

        prediction = np.moveaxis(obs,3,1)
        #free mem in cuda
        
        clear_session()
        for i in range(20):
        	gc.collect()


        del model
        del obs
        del pred
        return prediction
