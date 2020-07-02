params = {# training params
          'weight_decay'   : 0, 
          'eps'            : 1e-08, 
          'beta1'          : 0.9, 
          'beta2'          : 0.999, 
          'lr'             : 0.0001, 
          'lr_steps'       : 3, 
          'max_epochs'     : 20, 
          'batch'          : 4,
          'log_interval'   : 10, 
          'th'             : 20,
          'show'           : True,
          'fix'            : True,
          'save_filename'  : 'trajGRU_16_16.pth', 
          # model params
          'mode'           : 'absolute',
          'in_frames'      : 16,
          'out_frames'     : 16, 
          'dropout'        : 0.0,  
          'n_filter'       : 32
           }   
