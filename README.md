# Nowcasting
This repository has all the necessary to start doing video prediction specially focus in radar weather data from the NEXRAD network.

# Usage
In the "examples" directory there are plenty of jupyter notebook examples to start on that.

# Installation  
*GPU is necessary for training models*  
Create a separate conde environment (optiona):  
`conda create -n mlnowcasting`  
`source activate mlnowcasting`  
Install the minimum requisites  
`conda config --append channels conda-forge`  
`conda install numpy`  
if GPU: `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch` (with the proper cuda version)  
if CPU: `conda install pytorch-cpu torchvision-cpu -c pytorch`  
Install the package  
`pip install .`  

At this point you should be able to run prediction, like in "examples/How to make predictions"  
The following packages will allow you to use every functionality, if you don't want to go for all read above what each package is used for.  
`pip install nexradaws`  
`conda install cartopy arm_pyart IPython pysteps hyperopt sh pillow imageio opencv scikit-learn`  
if GPU: `conda install tensorflow-gpu keras`  
if CPU: `conda install tensorflow keras `   
