#!/bin/bash
set -x

pip install missinglink

#mali data clone 5685154290860032 --query '@version:f6f90ef73a86ab88f5dfec00433038555ad9a508 @sample:0.1 @seed:1337' --dest-folder ./data/mldx2
#mali data clone 5685154290860032 --query '@version:f6f90ef73a86ab88f5dfec00433038555ad9a508 @sample:0.05 @seed:1337' --destFolder './data/mldx2/$dir' --destFile '$name'
#mali data clone 5685154290860032 --query '@version:f6f90ef73a86ab88f5dfec00433038555ad9a508 @sample:1 @seed:1337' \
#  --destFolder './data/mldx2/$dir' --destFile '$name'

#ls ./data/mldx2

python /code/script.py



