#!/usr/bin/env python
# coding: utf-8

import numpy as np

# generic data generator supporting PyTorch

def get_data_generator(framework, dataset, batch_size=4, in_frame=10, out_frame=10, step=1, fix=False):
    '''
    wrapper function to return Data_Generator for keras or pytorch. See Data_Generator help for arguments.
    '''
    if framework=='torch':
        from torch.utils.data import Dataset
        from torch import tensor
        base_class = Dataset
    elif framework=='keras':
        from keras.utils import Sequence
        base_class = Sequence
        
    class Data_Generator(base_class):
        'Generates data for PyTorch training'
        def __init__(self, dataset, batch_size=4, in_frame=10, out_frame=10, step=1, fix=False, framework='torch'):
            '''
            dataset: path to npy file with shape (samples, frames, height, width)
            batch_size: how many samples are being fetched
            in_frame: how many frames are used as input
            out_frame: how many frames are used as output
                in_frame + out_frame = total number of fecthed frames
            step: every how many frames are used. e.g. step=1 doesn't skip frames, step=2 skips every other frame
            fix: makes deterministic every epoch. It fixes the used portion of the dataset. For example, if the dataset has 40 samples and only 10 are required, with fix=True only the first 10 of those 30 are going to be used each epoch. If fix=False there are 20 diferent sequences that randomly are being chosen on each epoch.
            '''

            #Initialization
            self.fix       = fix 
            self.step      = step
            self.in_frame  = in_frame
            self.out_frame = out_frame
            self.dataset   = dataset
            self.sequences = np.load(dataset, mmap_mode='r').shape[0]
            self.framework = framework

            # if batch_size is greater than the number of samples use half of the samples as batch size
            if batch_size > self.sequences:
                self.batch_size = int(np.ceil(self.sequences/2))
                print("Using batch_size {} instead of {}".format(self.batch_size,batch_size))
            else:
                self.batch_size = batch_size


        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(self.sequences/self.batch_size)

        def __getitem__(self, index):
            'Generate one batch of data'
            start_index = index * self.batch_size
            #start_index = np.random.randint(0,self.sequences-self.batch_size,1)[0] #random
            self.data = self._read_npy_chunk(self.dataset, start_index, self.batch_size) #ordered
            return self.__data_generation()

        #def on_epoch_end(self):
        #    print("gc.collect()",gc.collect()) # if it's done something you should see a number being outputted

        def __data_generation(self):
            '''
            Creates the input and target data from the dataset
            Uses a middle point to split the frames into in_frame and out_frames to minimize variance when using different lengths models and 'fix' is set to True.
            '''
            step = self.step
            fi = self.in_frame
            fo = self.out_frame
            length = self.data.shape[1]

            # set middle frame
            if self.fix: #deterministic
                mid = int(length/2)
            else: #random
                mid = np.random.randint(fi*step,length-fo*step+1,1)[0]

            X = np.expand_dims(self.data[:1, mid-fi*step:mid:step],1)/255
            Y = np.expand_dims(self.data[:1, mid:mid+fo*step:step],1)/255     

            self.data = []

            if self.framework == 'torch':
                X,Y = tensor(X), tensor(Y)

            return X,Y


        def _read_npy_chunk(self, filename, start_row, num_rows):
            '''
            Reads chunk of a npy file
            '''
            assert start_row >= 0 and num_rows > 0
            with open(filename, 'rb') as fhandle:
                major, minor = np.lib.format.read_magic(fhandle)
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fhandle)
                assert not fortran, "Fortran order arrays not supported"
                # Make sure the offsets aren't invalid.
                assert start_row < shape[0], (
                    'start_row is beyond end of file'
                )
                assert start_row + num_rows <= shape[0], (
                    'start_row + num_rows > shape[0]'
                )
                # Get the number of elements in one 'row' by taking
                # a product over all other dimensions.
                row_size = np.prod(shape[1:])
                start_byte = start_row * row_size * dtype.itemsize
                fhandle.seek(start_byte, 1)
                n_items = row_size * num_rows
                flat = np.fromfile(fhandle, count=n_items, dtype=dtype)
                return flat.reshape((-1,) + shape[1:])


    return Data_Generator(dataset, batch_size, in_frame, out_frame, step, fix)



