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
    import os 
    import numpy as np
    from sh import gunzip
    import time

print("Node name: " + socket.gethostname())

def unzip(file_list, start_line, number_lines, o_fmt='.ar2v'):
	start = time.time()
	start_line = int(start_line)
	number_lines = int(number_lines)
	file = open(file_list)
	for i, line in enumerate(file):
		if i >= start_line and i < start_line+number_lines:
			line = line.strip('\n')
			line = line.strip('\t')
			if '.' in line: continue		
			try:
				os.rename(line,line+o_fmt) 
			except:
				pass
		elif i >= start_line+number_lines:
		    break
	file.close()  
		
	print(time.time()-start)


unzip(sys.argv[1], sys.argv[2], sys.argv[3])
print("DONE python script")
 
