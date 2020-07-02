#!/bin/bash

# unzip files.
# mode of use: unzip.sh <directory where gz files are>



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
find $1 -type f -name '*.gz' -print > $list
echo 'files saved to file'

n_files=($(wc -l $list))

n_machines=${#machines[@]}
files_per_machine=$(( $n_files/$n_machines ))

echo "files per machine: $n_files / $n_machines =  $files_per_machine"

i=0;
for m in "${machines[@]}"
do
	echo "machine: $m"
    ssh $m "nohup python cluster/unzip.py cluster/$list $i $files_per_machine > cluster/log_$m" &
	i=$((i+files_per_machine));
done
rest_of_files=$(( $n_files-$files_per_machine*$n_machines ))
echo $rest_of_files
python unzip.py $list $i $rest_of_files > 'log_local' &
exit

