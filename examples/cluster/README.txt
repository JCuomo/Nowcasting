The scripts in this folder are used to manipulate data in cluster mode.
Each of them basicly works like this:
    1_ Define a list of available machines in the network (with all the necesary software)
    2_ List all files to be processed
    3_ Divide the files between the machines evenly (files remain in same storage, which has to be available to all machines)
    4_ Each machine computes whatever there is in a given numpy file
    5_ Local machine does the same for the remaining files (if the number of total files is not multiple of number of available machines)


To run the scripts:


# download NEXRAD data
bash download_data.sh


# unzip files 
(if using AWS they are already unzipped)
bash unzip.sh <dirextory_containing_gz_files>


# add extension "ar2v" to files without any extension 
(if using AWS it's easier to skip this step as any has the extension, but nr2png.py has to be change to skip the verification of the extension)
bash addextenstion.sh <dirextory_containing_ar2v_files>


# extract from NEXRAD files (ar2v) the reflectivity and save it as PNG after gridding it.
bash grid_reflectivity.sh <dirextory_containing_ar2v_files>


# kill all process of the given user in all the listed machines
# **** IMPORTANT: be careful not to have other processes on those machines ****
bash _killall.sh <myusername>

