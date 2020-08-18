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
#"linux6"
#"linux7" 
#"linux8" 
#"linux9"
"linux10"
"linux11" 
"linux12"
#"linux13"
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

log_dir='log_download_data'
rm -rf $log_dir
mkdir $log_dir

list=$log_dir'/days_input_on_machine.txt'

echo "Enter startdate ex)20141030"
read startdate
echo "Enter enddate ex)20141120"
read enddate

n_start=${startdate//[^[:digit:]]/}
n_end=${enddate//[^[:digit:]]/}

if [[ ${#n_start} -ne 8 || ${#n_end} -ne 8 ]] ; then 
    echo "[error] : date doesn't match the format"
    exit
elif [ $startdate -ge $enddate ] ; then
    echo "[error] : $startdate is bigger than $enddate"
    exit
fi


number_of_days=0;
start_date="$(date --date="$startdate" +'%s')"
end_date="$(date --date="$enddate" +'%s')"

number_of_days=$(( ($end_date - $start_date) / (60 * 60 * 24) +1))
echo "number of day is $number_of_days"

n_machines=${#machines[@]}
files_per_machine=$(( $number_of_days/$n_machines ))

echo "days per machine: $number_of_days / $n_machines =  $files_per_machine"

date="$startdate"
startdate_per_machine=$date
enddate_per_machine="$(date --date="$date + $files_per_machine days" +'%Y%m%d')"
echo $startdate_per_machine
echo $enddate_per_machine

date="$startdate"
for m in "${machines[@]}"
do
    startdate_per_machine=$date
    startdate_day="$(date --date="$startdate_per_machine" +%d)"
    startdate_month="$(date --date="$startdate_per_machine" +%m)"
    startdate_year="$(date --date="$startdate_per_machine" +%Y)"
    echo $startdate_year $startdate_month $startdate_day >> $list
    
    enddate_per_machine="$(date --date="$date + $files_per_machine days" +'%Y%m%d')"
    enddate_day="$(date --date="$enddate_per_machine" +%d)"
    enddate_month="$(date --date="$enddate_per_machine" +%m)"
    enddate_year="$(date --date="$enddate_per_machine" +%Y)"
    echo $enddate_year $enddate_month $enddate_day >> $list
    date="$(date --date="$date + $files_per_machine days" +'%Y%m%d')"
    echo "input of startdate of machine $m is $startdate_year $startdate_month $startdate_day and enddate is $enddate_year $enddate_month $enddate_day"
done
i=0;
j=0;
for m in "${machines[@]}"
do
    echo "machine: $m"
    ssh $m "nohup python3 /top/students/GRAD/ECE/pachung/home/Desktop/Nowcasting/Nowcasting-master/examples/cluster/download_data.py /top/students/GRAD/ECE/pachung/home/Desktop/Nowcasting/Nowcasting-master/examples/cluster/$list $j > /top/students/GRAD/ECE/pachung/home/Desktop/Nowcasting/Nowcasting-master/examples/cluster/$log_dir/log_$m" &
    j=$(($j+2));
	i=$((i+files_per_machine));
done
rest_of_files=$(( $number_of_days-$files_per_machine*$n_machines ))
echo $rest_of_files

startdate_per_machine=$date
startdate_day="$(date --date="$startdate_per_machine" +'%d')"
startdate_month="$(date --date="$startdate_per_machine" +'%m')"
startdate_year="$(date --date="$startdate_per_machine" +'%Y')"
echo $startdate_year $startdate_month $startdate_day >> $list

enddate_per_machine="$(date --date="$date + $rest_of_files days" +'%Y%m%d')"
enddate_day="$(date --date="$enddate_per_machine" +'%d')"
enddate_month="$(date --date="$enddate_per_machine" +'%m')"
enddate_year="$(date --date="$enddate_per_machine" +'%Y')"
date="$(date --date="$date + $rest_of_files days" +'%Y%m%d')"
echo $enddate_year $enddate_month $enddate_day >> $list
echo "input of startdate of local machine is $startdate_year $startdate_month $startdate_day and enddate is $enddate_year $enddate_month $enddate_day"
python download_data.py $list $j > $log_dir'/log_local' &
exit