#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

with suppress_stdout():
    import socket
    import sys
    import pyart 
    import imageio
    import cv2
    import os 
    import numpy as np
    import time
    
print("Node name: " + socket.gethostname())



def splitImagesInEvents(file_list, start_line, number_lines, i_fmt='ar2v', o_fmt='png', size=(64,64)):
    i_event = 0                                         
    dir_used = False
    first_time = True
    start = time.time()
    start_line = int(start_line)
    number_lines = int(number_lines)
    file = open(file_list)
    for i, line in enumerate(file):
        if first_time:
            first_time = False
            base_dir = o_path = os.path.dirname(line) + '/PNG/'
            try:
                print("creating dir:",base_dir)
                os.mkdir(base_dir)
            except:
                pass

        if i >= start_line and i < start_line+number_lines:
            line = line.strip('\n')
            line = line.strip('\t')

            if not line.endswith(i_fmt): continue
            filename = line.split('/')[-1].split('.')[0]                    #get filename without whole path and extension
            outfile = o_path + filename + '.' + o_fmt
            try:
                radar = pyart.io.read_nexrad_archive(line)                      # open the ar2v file
            except Exception as e:
                print(outfile," failed")
                print(e)
                continue
            grids = pyart.map.grid_from_radars(                             # convert to cartesian
                radar, 
                grid_shape = (1, size[0], size[1]),                         # define shape of frames (first dim is Z)
                grid_limits = ((radar.altitude['data'][0], 20000.0),(-150000, 150000), (-150000, 150000)),
                fields=['reflectivity'], gridding_algo= 'map_gates_to_grid', 
                weighting_function='BARNES2')
            img_mtx = grids.fields['reflectivity']['data'][0,:,:]           # grab the reflectivity data (2D image)
            img_mtx = np.clip(img_mtx,0,70)                                    # truncate the dbz to lower and upper bounds to produce same colored images when normalizing 
            if isRainy(img_mtx):                                            # if it's a rainy frame..
                img_mtx = np.ma.filled(img_mtx, fill_value=0)
                img_mtx = np.rint(img_mtx/70*255)
                img_mtx = img_mtx.astype('uint8')
                #img_mtx = np.invert(img_mtx)
                imageio.imwrite(outfile, img_mtx)
                print(outfile)
                dir_used = True                          
            elif dir_used:                                                  # frames are divided in folder of continues frames
                dir_used = False                                            # flag to change dir to a new one
                i_event += 1
                o_path = base_dir + 'event_' + str(i_event) + '/'
                os.mkdir(o_path)
            
        elif i >= start_line+number_lines:
            break
    file.close()  
        
    print(time.time()-start)
    
    
        
    
def isRainy(mtx, value=False):
    return True
    l=mtx.shape[0]*mtx.shape[1]
    x=(mtx > 30/4).sum()/mtx.shape[0]
    if value:
        return x
    else:
        return x > 0



splitImagesInEvents(sys.argv[1], sys.argv[2], sys.argv[3], size=(64,64))
print("DONE python script")

 
