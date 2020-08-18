#!/usr/bin/env python3

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
    import nexradaws
    import time
    import numpy as np
    from datetime import datetime
    from os.path import dirname, abspath
    
print("Node name: " + socket.gethostname())



def get_data(file_list, startline):
    
    #start = time.time()
    start_line = int(startline)
    number_lines = 1
    file = open(file_list)
    start_date = [] 
    end_date = []
    for i, line in enumerate(file):
        if i == start_line:
            start_date = [int(c) for c in line.split(" ")]
            print("start date is ",start_date)
        if i == start_line+number_lines:
            end_date = [int(c) for c in line.split(" ")]
            print("end date is ",end_date)
        elif i > start_line+number_lines:
            break
       
    path = dirname(dirname(dirname(abspath(__file__))))
    save_dir = path + '/data/auxiliary_dataset_folder/'
    radar = 'KFWS'                                 # from which radar
    
    try:
        os.mkdir(save_dir)
    except:
        pass
    conn = nexradaws.NexradAwsInterface()
    for year in range(start_date[0],end_date[0]+1):
        try:
            start = datetime(year, start_date[1], start_date[2], 0, 0)
            end = datetime(year, end_date[1], end_date[2], 0, 0)
            scans=conn.get_avail_scans_in_range(start, end, radar)
            localfiles = conn.download(scans, save_dir);
            print('##############',year, len(scans))
        except:
            print('##############',year, 'failed')

    #delete files finishing in _MDM
    for filename in os.listdir(save_dir):
        if '_MDM' in filename:
            os.remove(save_dir+'/'+filename)
    
    #print(time.time()-start)



get_data(sys.argv[1], sys.argv[2])
print("DONE python script")