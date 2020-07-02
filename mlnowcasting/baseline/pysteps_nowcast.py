
from pysteps import motion, nowcasts
import numpy as np
import os, sys

class pysteps_nowcast:
    ###########################################################################################
    def any2Z(self, data):
        if np.max(data) < 3:
            data = data*70
        elif np.max(data) > 70:
            data = data/255*70
        return data
    ###########################################################################################
    def getETS(self, target, prediction, th_dbz):
        

        
        t = np.where(self.any2Z(target)>th_dbz, 1, 0)
        p = np.where(self.any2Z(prediction)>th_dbz, 1, 0)

        N = t.shape[0]

        hits = np.zeros(N)
        misses = np.zeros(N)
        false_alarms = np.zeros(N)
        T = t.shape[1] * t.shape[2]

        combined = t*2 + p # factor 2 is to distinguish the contribution in the SUM from each term
        hits = (combined==3).sum()
        misses = (combined==2).sum()
        false_alarms = (combined==1).sum()

        C = (hits+false_alarms)*(hits+misses)/T
        ETS = (hits-C)/(hits+false_alarms+misses-C)

        return ETS
    ###########################################################################################
    def getV(self, history, n_frame, block_size, win_size, decl_scale):
        oflow_method = motion.get_method("lucaskanade")

        fd_kwargs = {}
        fd_kwargs["max_corners"] = 1000
        fd_kwargs["quality_level"] = 0.01
        fd_kwargs["min_distance"] = 2
        fd_kwargs["block_size"] = block_size  # 4-8

        lk_kwargs = {}
        lk_kwargs["winsize"] = (win_size, win_size) # 5-10

        oflow_kwargs = {}
        oflow_kwargs["fd_kwargs"] = fd_kwargs
        oflow_kwargs["lk_kwargs"] = lk_kwargs
        oflow_kwargs["decl_scale"] = decl_scale # 2-4

        V = oflow_method(history[-n_frame:], **oflow_kwargs) # 3 to 5 frames
        return V
    
    
    
    ###########################################################################################
    def STEPS(self, history, observation, th_dbz, n_leadtimes, 
                   n_frames=4, 
                   block_sizes=6, 
                   win_sizes=6, 
                   decl_scales=3, 
                   n_cascade_levels=3,
                   ar_window_radius=None):

        V = self.getV(history, n_frames, block_sizes, win_sizes, decl_scales)
        if observation is not None:
            n_ens_members=20
        else:
            n_ens_members=100
            
        th = th_dbz/70*255
        steps = nowcasts.get_method("steps");
        prediction_steps = steps(
            history, # last 3
            V,
            n_leadtimes,
            n_ens_members=20,
            n_cascade_levels=n_cascade_levels, # 3-4
            R_thr=th, # Z th
            kmperpixel=4.6875, # pixel resolution in km 64/300=4.6875
            timestep=5,
            decomp_method="fft",
            bandpass_filter_method="gaussian",
            noise_method="nonparametric",
            vel_pert_method="bps",
            mask_method="incremental",
            seed=24,
        );

        #Two options: get the best prediction or do the average (for the avg use 50 or 100 samples)
        if observation is not None:
            best_ETS = -2
            for p in prediction_steps:
                ETS = self.getETS(observation, p, th_dbz=th_dbz)
                if ETS > best_ETS:
                    best_ETS = ETS
                    best_prediction = p
        else:
            best_prediction = prediction_steps.mean(axis=0)
        return best_prediction


    ###########################################################################################
    def S_PROG(self, history, observation, th_dbz, n_leadtimes, 
                   n_frames=3, 
                   block_sizes=7, 
                   win_sizes=8, 
                   decl_scales=3, 
                   n_cascade_levels=3):

        V = self.getV(history, n_frames, block_sizes, win_sizes, decl_scales)
        
        th = th_dbz/70*255
        sprog = nowcasts.get_method("sprog")
        prediction_sprog = sprog(
            history, # it takes last three
            V,
            n_leadtimes,
            n_cascade_levels=n_cascade_levels, # 3-4
            R_thr=th, # reflectivity th
            decomp_method="fft",
            bandpass_filter_method="gaussian",
            probmatching_method="mean",
        )
        return prediction_sprog
    
    ###########################################################################################
    def extrapolation(self, history, observation, th_dbz, n_leadtimes, 
                   n_frames=3, 
                   block_sizes=4, 
                   win_sizes=8, 
                   decl_scales=2):

        V = self.getV(history, n_frames, block_sizes, win_sizes, decl_scales)
        
        extrapolate = nowcasts.get_method("extrapolation")
        prediction_extrapolate = extrapolate(history[-1], V, n_leadtimes)
        return prediction_extrapolate
    
    ###########################################################################################
    def DARTS(self, history, observation, th_dbz, n_leadtimes, 
                   n_frames=3, 
                   block_sizes=4, 
                   win_sizes=8, 
                   decl_scales=2):
    
        oflow_method = motion.get_method("DARTS")
        V = oflow_method(history)

        extrapolate = nowcasts.get_method("extrapolation")
        prediction_extrapolate = extrapolate(history[-1], V, n_leadtimes)
        return prediction_extrapolate
    
    ###########################################################################################
    def __init__(self, model_name):
            
        if model_name == 'ANVIL':
            self.model = self.ANVIL
        elif model_name == 'STEPS':
            self.model = self.STEPS
        elif model_name == 'S-PROG':
            self.model = self.S_PROG
        elif model_name == 'DARTS':
            self.model = self.DARTS
        else:
            self.model = self.extrapolation
            
            
    def predict(self, history, observation=None, th_dbz=None, n_leadtimes=16):
        with HiddenPrints():
            S,F,H,W = history.shape
            predictions = np.zeros((S,n_leadtimes,H,W))
            for n in range(len(history)):
                if observation is not None: 
                    predictions[n] = self.model(history[n], observation[n], th_dbz, n_leadtimes)
                else:
                    predictions[n] = self.model(history[n], None, n_leadtimes)

        return predictions
###########################################################################################
    def ANVIL(self, history, observation, th_dbz, n_leadtimes, 
                   n_frames=3, 
                   block_sizes=6, 
                   win_sizes=9, 
                   decl_scales=3,
                   n_cascade_levels=23, 
                   ar_window_radius=3):
        
        V = self.getV(history, n_frames, block_sizes, win_sizes, decl_scales);
        
        anvil = nowcasts.get_method("anvil");
        prediction_anvil = anvil(history[-3:], 
                     V, 
                     n_leadtimes, 
                     n_cascade_levels = n_cascade_levels,
                     ar_window_radius = ar_window_radius, # half of domain 15-40
                     ar_order = 1);

        return prediction_anvil
        
def reflectivity2rainfall(data, a=300, b=1.4):
    '''
    data: should be in dbz
    dBZ = 10.log(a) + b.10.log(R)         unit: dbz
    R = 10^[(dBZ - 10.log(a))/(b.10)]     unit: mm/h
    a,b values for some radars from NWS: https://www.weather.gov/lmrfc/experimental_ZR_relationships
    '''
    R = 10**((data - 10*np.log10(a))/(b*10))
    return R

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
