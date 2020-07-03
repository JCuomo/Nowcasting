import numpy as np
import torch
from scipy.signal import correlate

def any2normalize(data, verbose=False): 
    '''
    Converts data to 0-1 scale.
    Possible inputs:
        - reflectivity
        - normalized to 0-1 
        - pixel values 0-255
    If it is a torch tensor it returns a numpy array.
    '''
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu().detach().numpy()
   
    if np.max(data) > 70:
        if verbose: print("pixel [0-255]")
        data /= 255
    elif np.max(data) > 1:
        if verbose: print("Z [0-70]")
        data /= 70
    else:
        if verbose: print("pixel [0-1]")
    return data
    
def any2reflectivity(data, verbose=False): 
    '''
    Converts data to reflectivity values.
    Possible inputs:
        - reflectivity (does nothing)
        - normalized to 0-1 (normally output of a model)
        - pixel values 0-255
    If it is a torch tensor it returns a numpy array.
    '''
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu().detach().numpy()
        else:
            data = data.numpy()
   
    if np.max(data) < 3:
        if verbose: print("pixel [0-1]")
        data = data*255
    elif np.max(data) > 70:
        if verbose: print("pixel [0-255]")
    else:
        if verbose: print("Z [0-70]")
        return data
    return pixel2reflectivity(data)
    
def pixel2reflectivity(x):
    '''
    data should be between 0 and 255. Currently, 0s=70dBz and 255s=0dBz
    70dbz came from processNEXRAD.ipynb where that is the clip value
    '''
    x = x.astype('float32')   # to get decimal point in the dbZ values
    return x*70/255              # training data was clip to 0-70 and rescaled it from 0-255 to 0-1
     

def reflectivity2rainfall(data, a=300, b=1.4):
    '''
    data: should be in dbz
    dBZ = 10.log(a) + b.10.log(R)         unit: dbz
    R = 10^[(dBZ - 10.log(a))/(b.10)]     unit: mm/h
    a,b values for some radars from NWS: https://www.weather.gov/lmrfc/experimental_ZR_relationships
    '''
    R = 10**((data - 10*np.log10(a))/(b*10))
    return R

def rainfall2pixel(r, a=300, b=1.4):
    '''
    Does the inverse of the Z-R equation.
    Return range = 0-255
    '''
    z_ = np.log10(r)*b*10+10*np.log10(a)
    return np.round(np.array(z_)/70*255).astype('uint8')

def binarize(x, th):
    '''
    Convert continuous-value matrix into binary-value matrix by using a threshold.
    'x' and 'th' should be in the same scale
    '''
    return np.where(x <= th, 0, 1)
    
def is_rainy_frame(x, th, k=None, return_score=False):
    '''
    x: (Height,Width) single frame
    th: threshold for binarize in same units as input (dBZ or 0-1)
    k: kernel size to convolve on frame and compute rain average (think of it as how big the cloud show be in pixel size to be considered)
    return_score: flag to return the score
    
    Returns:
        bool: True is the frame contains at least on significant 'cloud'.
        score: % of pixels from with value above the th. The higher score the rainier the frame is.
    '''
    if k is None:
        k = int(len(x)/8)
    l = k*k # normalizing factor
    kernel = np.ones((k,k))
    # binarize the image using a th 
    b = binarize(x,th)
    # convolute de binary image with a kernel of 1's to make averages
    c = correlate(b, kernel, mode="valid")/l 
    if th > 1: # only if the th was given in dBZ
        th = th/70 #output of correlate is normalize to 0-1, so I normalize th (dbz) too
    # binarize again
    b = binarize(c,th) 
    # and return True if any value was below the threshold
    if return_score: # it could also returns the score, which is how many pixels have are part of an acceptable size 'cloud'
        return ((b==1).any(), np.sum(b)/np.size(b)*100) 
    else: 
        return (b==1).any()

def is_rainy_event(s, th, k=None, n_frame_th=None, return_score=False):
    ''' 
    s: sequence of frames (frames, hight, width)
    th: threshold for binarize in same units as input (dBZ or 0-1)
    k: kernel size to convolve on frame and compute rain average (think of it as how big the cloud show be in pixel size to be considered)
    n_frame_th: number of frames with rain content within the sequence (s) to consider it as a rainy sequence. If None, half of the frames should be rainy to be considered as a rainy sequence.
    return_score: flag to return the score

    Returns:
        bool: True is the number of frames containing at least a significant 'cloud' is higher than 'n_frame_th'
        score: (% of pixels with value above the th, % of frames with rain)
    '''
    if n_frame_th==None:
        n_frame_th = 0.5*len(s)
    rainy_frames = 0
    acumm_score = 0
    for f in range(len(s)):
        is_rainy,score = is_rainy_frame(s[f], th, k, return_score=True)
        if is_rainy:
            rainy_frames += 1
        acumm_score += score
    if return_score: 
        return (rainy_frames > n_frame_th, (acumm_score/len(s),rainy_frames/len(s)*100)) 
    else: 
        return rainy_frames > n_frame_th
