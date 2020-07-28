import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from .utils import binarize, is_rainy_event, any2reflectivity, is_rainy_event
from .plot_utils import plot_obs_pred
from skimage.measure import compare_ssim

np.seterr(divide='ignore', invalid='ignore') #to ignore some division by zero in the metrics


def correlation(target,prediction):
    '''Correlation defined in https://arxiv.org/pdf/1506.04214.pdf page 7'''
    fip = np.multiply(target,prediction).sum()     # Frobenius inner product 
    g_norm = np.linalg.norm(target, ord='fro')     # Frobenius norm
    p_norm = np.linalg.norm(prediction, ord='fro') # Frobenius norm
    e = 1e-9
    return fip/(g_norm*p_norm+e)  

def _event_metrics(target, prediction, th):
    '''
    Returns the metrics for a given prediction and ground thruth in a certain threshold.
    Metrics returned are MSE, CSI, FAR, POD, ETS, ACC, MAE, SSIM, where MSE, MAE, CORR, and SSIM are not threshold dependent.
    target/prediction = [<frames>, width, height] 2 or 3 dimesions, output is always 3.
    All inputs should be in the same units (recommended dBZ).
    '''
    #if passing only 1 image add the frame dimension
    if target.ndim == 2:
        target = np.expand_dims(target,0)
        prediction = np.expand_dims(prediction,0)
       
    # apply the threshold
    target_binary     = binarize(target,th)
    prediction_binary = binarize(prediction,th)
    
    N = target.shape[0]
    CORR = np.zeros(N)
    MSE = np.zeros(N)
    MAE = np.zeros(N)
    SSIM = np.zeros(N)
    hits = np.zeros(N)
    misses = np.zeros(N)
    false_alarms = np.zeros(N)
    correct_reject = np.zeros(N)
    T = target.shape[1] * target.shape[2]
    for frame in range(N):
        combined = target_binary[frame]*2 + prediction_binary[frame] 
        # factor 2 is to distinguish the contribution in the SUM from each term
        hits[frame] = (combined==3).sum()
        misses[frame] = (combined==2).sum()
        false_alarms[frame] = (combined==1).sum()
        correct_reject[frame] = (combined==0).sum()
       
        CORR[frame] = correlation(target[frame], prediction[frame])
        MSE[frame] = np.mean((target[frame] - prediction[frame])**2)
        MAE[frame] = np.mean(np.abs(target[frame] - prediction[frame]))
        SSIM[frame] = compare_ssim(target[frame], prediction[frame], data_range=1)
    CSI = (hits/(hits+misses+false_alarms))
    FAR = (false_alarms/(hits+false_alarms))
    POD = (hits/(hits+misses))
    C = (hits+false_alarms)*(hits+misses)/T
    ETS = (hits-C)/(hits+false_alarms+misses-C)
    ACC = (hits+correct_reject)/T
    FBS = (hits+false_alarms)/(hits+misses)
    
    # if changed, update _get_metric_names() to reflect same metrics in the same order
    return MSE, CSI, FAR, POD, ETS, ACC, MAE, SSIM


def _squeeze_generic(a, axes_to_keep):
    ''' all single dimension except axis 1 (events)'''
    out_s = [s for i,s in enumerate(a.shape) if i in axes_to_keep or s!=1]
    return a.reshape(out_s)


def _get_metric_names():
    ''' list of metric names'''
    return ['MSE', 'CSI', 'FAR', 'POD', 'ETS', 'ACC', 'MAE', 'SSIM']

def events_metrics(target, prediction, th_dbz=20, plot=False, printT=True, N=None):
    '''
    target/prediction should be in dBZ for best use. It must be in same dimensions as th_dbz.
    target/prediction should be (samples, frames, height, width) and could any other dimension (like "channels") but with size 1.
    th_dbz: threshold in dBZ
    plot: flag to plot both target/prediction pre-/post-binarization for debugging purposes mainly
    printT: print tables with metrics
    N: only used when 'plot' is True, to plot the N sample, if not specified last samples is plotted.
    
    Returns:
        metric_stat: all samples metrics, in the same order as list returned by _get_metric_names()
    '''
    target = any2reflectivity(target)
    prediction = any2reflectivity(prediction) 

    th = th_dbz
    
    # if there is no samples cancel execution
    if target.shape[0]==0:
        return None, None

    # remove unwanted dimensions
    target     = _squeeze_generic(target,     axes_to_keep=[0])
    prediction = _squeeze_generic(prediction, axes_to_keep=[0])

    n_samples = target.shape[0]
    n_frames  = target.shape[1]

    
    labels = _get_metric_names()
    labels = [l+' (th='+str(th_dbz)+' dBZ)' if l in ['CSI', 'FAR', 'POD', 'ETS', 'ACC'] else l for l in labels]
    
    metric_stat = np.zeros(shape=(len(labels),n_frames,n_samples))

    # get metrics for each sample
    for n in range(n_samples):
        M = _event_metrics(target[n], prediction[n], th_dbz)
        metric_stat[:,:,n] = np.array(M)           

    # plot for debugging, to make sure the binazarization is correct
    if plot: 
        if not N:
            N = n
        print('Original')
        plot_obs_pred(target[N],prediction[N], cmap='darts')
        print('Converted to binary')
        plot_obs_pred(binarize(target[N],th),binarize(prediction[N],th), cmap='binary')
        
    np.seterr(divide='ignore', invalid='ignore')

    # compute the means
    batch_mean = np.nanmean(metric_stat,axis=2)
    frame_mean = np.nanmean(metric_stat,axis=1)
    total_mean = np.nanmean(metric_stat,axis=(1,2))

    # print tables
    if printT:
        df_batch_mean = pd.DataFrame(data=batch_mean, index=labels, columns=list(range(n_frames)))  
        print("Mean across batches => columns are frames")
        display(df_batch_mean)
        print()
        print()
        print()

        df_frame_mean = pd.DataFrame(data=frame_mean, index=labels, columns=list(range(n_samples)))  
        print("Mean across frames => columns are bacthes")
        display(df_frame_mean)
        print()
        print()
        print()

        df_total_mean = pd.DataFrame(data=total_mean, index=labels, columns=[1])  
        print("Mean across frames and batches")
        display(df_total_mean)

    return metric_stat

def multiple_th_metrics(target, prediction, channel_pos=None, plot=False, N=None):
    '''
    Return metrics for multiple thresholds
    obsolete: recommend to use metric_plots() instead
    '''
    target = any2reflectivity(target)
    prediction = any2reflectivity(prediction) 
        
    if target.ndim == 3:
        target = np.expand_dims(target, axis=0)
    if prediction.ndim == 3:
        prediction = np.expand_dims(prediction, axis=0)
        
    th20 = 20 #rainfall2pixel(1)
    th30 = 30 #rainfall2pixel(5)
    th40 = 40 #rainfall2pixel(10)
    th50 = 50 #rainfall2pixel(30)
    
    idx0 = idx1 = idx5 = idx10 = idx30 = []
    
    if channel_pos:
        target = np.squeeze(target, axis=channel_pos)
        prediction = np.squeeze(prediction, axis=channel_pos)
        
    for i in range(len(target)):
        if is_rainy_event(target[i], th=th20, n_frame_th=1):
            idx1 = np.append(idx1,int(i))
        else: 
            idx0 = np.append(idx0,i)
            continue
        if is_rainy_event(target[i], th=th30, n_frame_th=1):
            idx5 = np.append(idx5,i)
        else: 
            continue
        if is_rainy_event(target[i], th=th40, n_frame_th=1):
            idx10 = np.append(idx10,i)   
        else: 
            continue
        if is_rainy_event(target[i], th=th50, n_frame_th=1):
            idx30 = np.append(idx30,i)   

    #reconvert to list of integers..idk why append converts the indeces into double
    idx0,idx1,idx5,idx10,idx30 = list(map(int,idx0)), list(map(int,idx1)), list(map(int,idx5)), list(map(int,idx10)), list(map(int,idx30))
    #events_metrics(target[idx0],  prediction[idx0],  th_dbz=1,  channel_pos=None, plot=plot)   
    events_metrics(target[idx1],  prediction[idx1],  th_dbz=th20,  channel_pos=None, plot=plot, N=N);   
    events_metrics(target[idx5],  prediction[idx5],  th_dbz=th30,  channel_pos=None, plot=plot, N=N);   
    events_metrics(target[idx10], prediction[idx10], th_dbz=th40, channel_pos=None, plot=plot, N=N);
    events_metrics(target[idx30], prediction[idx30], th_dbz=th50, channel_pos=None, plot=plot, N=N);
        
def metric_plots(target, prediction, time_res=5, th_dbz=20, std_lines=True, fig=None, model=None):
    '''
    target and prediction can be in dBZ, 0-255, or 0-1 range values
    target and prediction can have 3 or 4 dimensions (<samples>, frames, height, width)
    time_res: spacing in minutes between frames
    th_dbz: threshold in dBZ
    std_lines: flag to plot standard deviation lines
    fig, axs: if passed the metrics are drawn on top. Use this to compare different models.
    model: name of the model used to make the predictions
    
    Return:
        metric_stat: all samples metrics, in the same order as list returned by _get_metric_names()
        fig, axs: for keep ploting other models metrics
    '''
    target     = any2reflectivity(target)
    prediction = any2reflectivity(prediction)
    
    labels = _get_metric_names()
    
    if target.ndim == 3:
        target = np.expand_dims(target, axis=0)
    if prediction.ndim == 3:
        prediction = np.expand_dims(prediction, axis=0)
    #rr  = rainfall2pixel(th_rr)

    axs = fig.axes if fig is not None else None        
    idx = []
    
    # remove unwanted dimensions
    target     = _squeeze_generic(target,     axes_to_keep=[0])
    prediction = _squeeze_generic(prediction, axes_to_keep=[0])
        
    # only use samples with reflectivy content on the threshold range, otherwise metrics will wrong
    for i in range(len(target)):
        if is_rainy_event(target[i], th=th_dbz, n_frame_th=1):
            idx = np.append(idx,int(i))

    #reconvert to list of integers..idk why append converts the indexes into double
    idx = list(map(int,idx))
    print("Samples used:", len(idx)) # shows how many samples were used to computed the metrics
    
    # compute the metrics
    metric_stat = events_metrics(target[idx],  prediction[idx],  th_dbz=th_dbz, plot=0, printT=0);   
    batch_mean = np.nanmean(metric_stat,axis=2)
    batch_var  = np.nanvar(metric_stat,axis=2)
    
    n_metrics = metric_stat.shape[0]
    n_frames  = metric_stat.shape[1]
    rows = 3
    cols = int(np.ceil(n_metrics/rows))+1
    # ranges for each metric. Should match _get_metric_names() order
    ranges = [None, (0,1), (0,1), (0,1), (-0.33,1), (0,1), None, (0,1)]
    # defines the time axis array
    time = np.linspace(1*time_res, n_frames*time_res, n_frames)
    # if no fig or axs were give, create them
    if not fig and not axs :
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15,rows*3.5));
        axs = list(axs.reshape(-1))
    # for each metric plot it on a different subplot
    for m in range(n_metrics):
        axs[m].locator_params(nbins=4)
        axs[m].set_title(labels[m])
        axs[m].set_xlabel('lead time [min]')
        p = axs[m].plot(time, batch_mean[m], label=model)
        if std_lines:
            axs[m].plot(time, batch_mean[m]+batch_var[m],'--', color=p[0].get_color(), alpha=0.7)
            axs[m].plot(time, batch_mean[m]-batch_var[m],'--', color=p[0].get_color(), alpha=0.7)
        axs[m].set_ylim(ranges[m]) #uncomment if you want absolute ranges on plots
        axs[m].grid()
    for m in range(n_metrics,rows*cols):    
        axs[m].axis('off')
    # show the labels not for every subplot but only once
    handles, labels = axs[n_metrics-1].get_legend_handles_labels()
    pos = axs[-1].get_position() #bbox_to_anchor=(pos.x0, pos.y0), 
    fig.legend(handles, labels, loc='lower right', fancybox=True, shadow=True)
    plt.tight_layout()
    plt.close()
    return metric_stat, fig

def events_specific_metric(target, prediction, metric_name='CSI', th_dbz=20):
    '''
    Returns a single value of a specific metric.
    It is design to be used for hyper-parameter optimization.
    '''
    # get all metrics
    metric_stats =  events_metrics(target, prediction, th_dbz=th_dbz, plot=False, printT=False, N=None)
    # compute the mean across samples
    batch_mean = np.nanmean(metric_stats,axis=2)
    # get the index of the desired metric
    metric_index = _get_metric_names().index(metric_name)
    # return the last value of the desired metric
    return batch_mean[metric_index,-1]
