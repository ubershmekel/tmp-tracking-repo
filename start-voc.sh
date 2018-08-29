#!/bin/bash
set -x
#  --data-volume 5685154290860032
#ml run xp --org yuval --project 5686084218388480 --git-repo git@github.com:ubershmekel/tmp-please-ignore.git --command '/code/remote-script.sh' --gpu
#$ mali data clone 5729313835974656 --query '@version:34de10b59fba578321b915eae1f984e82b324f89 @seed:1337' --dest-folder ./


ml run xp --org yuval --project 5686084218388480 \
    --command '/code/remote-script.sh' \
    --gpu \
    --data-volume 5685154290860032 \
    --data-query '@version:f6f90ef73a86ab88f5dfec00433038555ad9a508 @sample:0.01 @seed:1337' \
    --data-dest '/data/voc1/$dir' \
    --env DATA_ROOT=/data \
    --env EPOCHS=10
    #--git-repo git@github.com:ubershmekel/tmp-please-ignore.git \


