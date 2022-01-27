import datetime
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import warnings
import numpy as np
try:
	import cartopy.feature as cfeature
	import cartopy.crs as ccrs
	from cartopy.io import img_tiles
	cartopy_available = True
except:
	cartopy_available = False


warnings.filterwarnings("ignore")

def plot_obs_pred(target, prediction, N=5, cmap='binary', title_files=None):
    '''
    Simple plotting of some frames of the target and prediction on a figure with two rows, one for each.
    N: how many frames of each to plot
    cmap: colormap. Recommended to use 'darts' or 'binary'.
    title_files: list of each frame original file to use as titles.
    '''
    # define cmap and ranges
    if cmap == 'darts':
        cmap = get_cmap()

    max_ = np.max(target)
    vmax = 1
    if max_ > 2:
        vmax = 70
    elif max_ > 70:
        vmax = 255
    else:
        vmax = 1
        
    n_frames = target.shape[0]
    skip = int(n_frames/N) # how many frames to skip to plot only N frames
    fig, axes = plt.subplots(nrows=2, ncols=N, figsize=(15,8))
    i = 0
    plot_sequence = target # auxiliary variable to use for plotting
    for ax in axes.flat:
        # once the target has been plotted, switch to prediction
        if i == N:
            plot_sequence = prediction
            i = 0
        im = ax.imshow(plot_sequence[i*skip,...],vmin=0, vmax=vmax, cmap=cmap)
        if title_files is not None:
            ax.set_title(get_title(title_files, i*skip))
        #ax.axis('off')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        i += 1
        
    axes[0,0].set_ylabel('OBS')
    axes[1,0].set_ylabel('PRED')
    # show the colorbar at the bottom horizontally
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='bottom', aspect=100)
    cbar.set_label('dbZ', rotation=0)
    fig.show()
    plt.show()
    #fig.tight_layout()
    return fig

def plot_pred(frames, N_frames=5, cmap='darts', label='OBS', 
              title_files=None, scan_time=5, title_1st=None,
              plot_map=False, 
              plot_colorbar=True, bottom_label=False):
    '''
    Plotting of some frames of an event with coordinates and map in the background.
    N_frames: how many frames of each to plot
    cmap: colormap. Recommended to use 'darts' or 'binary'.
    label: label of the plot. e.g. name of the model use for the prediction or 'Observation' if the ground truth.
    title_files: list of each frame original file to use as titles.
    scan_time: time between each frame. Only used if no 'title_files' are given
    title_1st: if given, use it for the first frame (normally to show the radar used and the initial time)
    plot_map: if True plot the coordinates and base map
    plot_colorbar: if True plot colorbar
    bottom_label: if True plot X-label
    '''    
    if plot_map and not cartopy_available:
    	plot_map = False
    	print('Cartopy could not be imported, so no map is plotted')
    
    # define cmap and ranges
    if cmap == 'darts':
        cmap = get_cmap()
    max_ = np.max(frames)
    vmax = 1
    if max_ > 2:
        masked_th = 20
        vmax = 70
    elif max_ > 70:
        masked_th = 72
        vmax = 255
    else:
        masked_th=0.5
        vmax = 1
        
    n_frames = frames.shape[0]
    skip = int(n_frames/N_frames)# how many frames to skip to plot only N frames
    
    # if the base map is being used, the projection needs to be defined.
    if plot_map:
        fig, axes = plt.subplots(nrows=1, ncols=N_frames, figsize=(15,4), subplot_kw={'projection': ccrs.Mercator()})
    else:
        fig, axes = plt.subplots(nrows=1, ncols=N_frames, figsize=(15,4))
        
    i = 0
    # arguments for the imshow()
    imshow_kwargs = {"vmin" : 0, "vmax" : vmax, "cmap" : cmap}
    for ax in axes.flat:       
        if plot_map: #if base map is wanted
            left_label = (i==0)
            kwargs = get_map(ax, left_label=left_label, bottom_label=bottom_label)
            # base map is shown in the background and it is visible where the reflectivity values are below 20 dBZ
            frames = np.ma.masked_where(frames < masked_th, frames)
            imshow_kwargs = {**imshow_kwargs, **kwargs}
        im = ax.imshow(frames[i*skip,...], **imshow_kwargs)
        if title_files is not None:
            ax.set_title(get_title(title_files, i*skip))
        else:
            ax.set_title('+ ' + str(i*skip*scan_time) + ' min')
        #ax.axis('off')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        i += 1
        
    # draw the name of the figure on the right side
    #plt.subplots_adjust(wspace=0.3)
    axes[0].text(-0.25,0.55,label, va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=axes[0].transAxes)
    #axes[0].set_ylabel(label)
    if title_1st is not None:
        axes[0].set_title(get_1st_title(title_1st))
    if plot_colorbar:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='bottom', aspect=100)
        cbar.set_label('dBZ', rotation=0)
    fig.show()
    plt.show()
    #fig.tight_layout()
    return fig



def get_map(ax,corners=[-98.900450,-95.705828,31.220247,33.931325], 
       center=[-97.30314636230469,32.573001861572266], 
       radius=[1,100,200],
       alpha=0.6, 
       left_label=True, 
       bottom_label=True):

    ''' 
    Gets the map in the given location using Cartopy
    ax: axes
    corners: box coordinates -> long low, long up, lat low, lat up
    center = long, lat
        default is Dallas Forth Worth NEXRAD radar with 150km to each direction
    radius: radius of concentric circles in km to draw on the map
    alpha: transparency factor (0-1)
    left_label: flag to draw label on Y-axis
    bootom_label: flag to draw label on X-axis
    
    many available drawing option are commented in the code:
        #ax.coastlines()                         draw the coastlines
        #ax.stock_img()                          draw a low resolution land 
        #ax.add_feature(land_50m)                draw a land with a bit more resolution 
        #ax.add_feature(cfeature.RIVERS)         draw the rivers
        ax.add_feature(cfeature.STATES)          draw the states (US)
        #ax.add_image(tiles_sat, 8, alpha=alpha) draw the google satellite images
    '''

    #ax.set_global();
    #ax.coastlines();
    ax.set_extent(corners); # fix the map to coordinates
    #ax.stock_img()
    gl = ax.gridlines(xlocs=range(-99,-95,1), ylocs=range(29,35,1), draw_labels=True, 
                      color='black', linestyle='-.'); # draw the coordinates axes
    gl.right_labels, gl.xlabels_top = None, None # undraw coordinates on the top and right
    if not left_label:
        gl.left_labels = None # undraw coordinates on the left
    if not bottom_label:
        gl.xlabels_bottom = None # undraw coordinates on the bottom
    #ax.add_feature(land_50m)
    #ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.STATES, linestyle='dotted', alpha=0.8)

    #tiles_sat = img_tiles.GoogleTiles(style='satellite', desired_tile_form='RGB')
    #web_merc = img_tiles.GoogleTiles().crs
    #ax.add_image(tiles_sat, 8, alpha=alpha)

    # draw concentric circles
    for radio in radius:
        ax.tissot(rad_km=radio, lons = center[0], lats = center[1], 
                  n_samples = 36, facecolor = 'none', edgecolor = 'black', linewidth = 1, alpha = 1, linestyle='-.')
                                        
    # returns the kwargs for more drawings outside this function, to have the coordinates
    kwargs = {"origin" : "upper", "extent" : corners, "transform" : ccrs.PlateCarree(), "zorder": 0.5}
    return kwargs

def get_cmap():
    '''
    return custom colormap use in the reflectivity plots
    '''
    colors = [(255,255,255), (164, 140, 177), (83,2,125), (49,0,199), (0,0,255), (5,101,134), (10,182,18), (105,170,18), (255,255,0), (255,213,0), (253,169,2), (255,84,0), (255,0,0)]  # R -> G -> B
    colors = [[e/255 for e in c] for c in colors]
    cmap_name = 'darts'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=13)

def get_delta(sec):
    '''given N seconds it converts it to a string 
    "+ [hours] hr [minutes] min [seconds] sec" 
    and does not include non-zero values
    '''
    sign = ' -' if sec < 0 else ' +'
    
    hours, remainder = divmod(abs(int(sec)), 3600)
    minutes, seconds = divmod(remainder, 60)
    delta = sign
    if int(hours) != 0:
        delta += ' ' + str(int(hours)) + ' hr'
    if int(minutes) != 0:
        delta += ' ' + str(int(minutes)) + ' min'
    if int(seconds) != 0:
        delta += ' ' + str(int(seconds)) + ' sec'
    # if there is no time difference...
    if int(seconds) == 0 and int(minutes) == 0 and int(hours) == 0:
        delta += ' 0 min'

    #print('{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))  
    return delta

def get_1st_title(n):
    '''
    Given a NEXRAD file, such as 'KFWS20190529_110900_v06.ar2v' it returns the name formatted like 'KFWS 2019-05-29 11:09:00'
    '''
    return n[0:4]+' '+n[4:8]+'-'+n[8:10]+'-'+n[10:12]+' '+n[13:15]+':'+n[15:17]+':'+n[17:19]

    
def get_title(filenames, n_steps=None, n_time=None):
    '''
    filename = the file that you want the title for
    n_steps = minutes from the reference (filename_1st) if the title is from a prediction
    e.g.:
    get_title('20170603_000300', '20170603_000200') #'2017-06-03 00:03:00 + 1 min'
    get_title(None, '20170603_000200')              #'2017-06-03 00:02:00 + 0 min'
    get_title('20170603_000200', None)              #'2017-06-03 00:02:00 + 0 min'
    get_title(None, '20170603_000100',3)            #'2017-06-03 00:01:00 + 3 min'
    '''
    
    n1 = filenames[0].split('.')[0][4:-4]                                            
    n  = filenames[n_steps].split('.')[0][4:-4]  if n_steps else n1   
    date_time_obj1 = datetime.datetime.strptime(n1, '%Y%m%d_%H%M%S')
    date_time_obj  = datetime.datetime.strptime(n,  '%Y%m%d_%H%M%S')
    delta = date_time_obj - date_time_obj1                             # delta in seconds
    seconds = datetime.timedelta(delta.days, delta.seconds).total_seconds() #correction for negative numbers  
    delta = get_delta(seconds)                                    # delta in str of hr min sec
    date_time = n[0:4]+'-'+n[4:6]+'-'+n[6:8]+' '+n[9:11]+':'+n[11:13]+':'+n[13:15]
    if n_steps==0:
        return date_time + delta
    else:
        return  delta


from math import sin, cos, sqrt, atan2, radians

def coordinates2km(lat1,lon1,lat2,lon2):
    '''
    gives the distance in km between two point
    '''
    R = 6373.0           # approximate radius of earth in km

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def equi(m, centerlon, centerlat, radius, ax, *args, **kwargs):
    '''
    Draw radius on fig when using Basemap
    '''
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])
 
    #~ m.plot(X,Y,**kwargs) #Should work, but doesn't...
    X,Y = m(X,Y)
    ax.plot(X,Y,'--k', **kwargs)
    
def plotCASAqpe(qpe,f,f1, ax=None):
    '''
    OBSOLETE
    using basemap
    '''
    #f=filename, f1=filename_1st of the sequence
    if ax is None:
        ax = plt.gca()
        
    bbox = [30, 33, -98, -95]
    m = Basemap(projection='merc',
                 llcrnrlat=bbox[0],
                 urcrnrlat=bbox[1],
                 llcrnrlon=bbox[2],
                 urcrnrlon=bbox[3],
                 lat_ts=10,
                 resolution='i',
                 ax = ax)

    m.shadedrelief()
    m.drawparallels(np.arange(int(bbox[0]),int(bbox[1]),1),labels=[1,0,0,0])
    m.drawmeridians(np.arange(int(bbox[2]),int(bbox[3]),1),labels=[0,0,0,1])

    qpe = np.ma.masked_where(qpe < 20, qpe)

    m.imshow(qpe, cmap='jet', alpha=0.9, ax = ax)
    #color_bar = plt.colorbar( ax = ax)                            
    #color_bar.set_label('Label name')
    ax.set_title(get_title(f,f1))
    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')
    
    radii = [100,200]

    centerlon = (bbox[2]+bbox[3])/2
    centerlat = (bbox[0]+bbox[1])/2
    for radius in radii:
        equi(m, centerlon, centerlat, radius, ax,lw=1)
