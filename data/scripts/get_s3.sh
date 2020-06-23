#!/bin/bash

awsdir=$1
locdir=${2:-.}

cd $locdir
mkdir $awsdir
aws s3 sync s3://continualrl-bucket/doodad-rlkit/$awsdir $awsdir --include "*/validation_stats.csv" --include "*/variant.json" --include "*/visit_*.npy" --exclude "*.log" --exclude "*/itr*.pkl"
cd -
