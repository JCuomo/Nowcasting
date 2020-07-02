#!/bin/bash

# extracts from NEXRAD data (ar2v) the reflectivity and save it as PNG after gridding it.
# mode of use: grid_reflectivity.sh <directory where ar2v files are>
# it uses an auxiliary python file "gridNexRad.py" which you may need to put the correct path in the last loop of this script


# list of machines used for parallel processing of the files
declare -a all_machines=(
"linux1" 
"linux2" 
"linux3"
"linux4"
"linux5"
"linux6"
"linux7" 
"linux8" 
"linux9"
"linux10"
"linux11" 
"linux12"
"linux13"
"linux14" 
"linux15"
)

# keep only machines that are reachable
machines=()
for m in "${all_machines[@]}"
do
   conex_code=$(ssh -q $m exit; echo $?)
   if [ "0" -eq "$conex_code" ]
        then
            machines+=("$m")
    fi
done

echo "machines available:"
echo "${machines[@]}"

log_dir='log_nr2png'
rm -rf $log_dir
mkdir $log_dir

list=$log_dir'/list_of_files.txt'
#get all files to process
find $1 -type f -name "*.ar2v*" -print > $list
echo 'files saved to file'

n_files=($(wc -l $list))

n_machines=${#machines[@]}
files_per_machine=$(( $n_files/$n_machines ))

echo "files per machine: $n_files / $n_machines =  $files_per_machine"

i=0;
for m in "${machines[@]}"
do
	echo "machine: $m"
    ssh $m "nohup python cluster/nr2png.py cluster/$list $i $files_per_machine > cluster/$log_dir/log_$m" &
    #ssh $m "nohup python cluster/nr2png.py cluster/$list $i $files_per_machine"
	i=$((i+files_per_machine));
done
rest_of_files=$(( $n_files-$files_per_machine*$n_machines ))
echo $rest_of_files
python nr2png.py $list $i $rest_of_files > $log_dir'/log_local' &
exit
