#! /bin/bash

if [[ $1 = '--mdp'  && $3 = '--algorithm' ]]
then
    mdp=$2;
    algorithm=$4;
else
    mdp=$4;
    algorithm=$2;    
fi

python3 planner.py $mdp $algorithm