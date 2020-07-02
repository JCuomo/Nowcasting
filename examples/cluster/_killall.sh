#!/bin/bash

# kill al process of the given user in all the listed machines
# bash _killall.sh <myusername>
declare -a machines=(
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



for m in "${machines[@]}"
do
	ssh $m "pkill -9 -u $1" &
done
echo "DONE"
