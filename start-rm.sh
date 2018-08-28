#!/bin/bash
set -x
#  --data-volume 5685154290860032
#ml run xp --org yuval --project 5686084218388480 --git-repo git@github.com:ubershmekel/tmp-please-ignore.git --command '/code/remote-script.sh' --gpu


ml run xp --org yuval --project 5686084218388480 \
    --command '/code/remote-script.sh' \
    --gpu \
    --data-volume 5685154290860032 \
    --data-query '@version:ecc29bb3531f39963c16f8578f58c0466a4d7143 @sample:0.01 @seed:1337' \
    --data-dest '/data/$phase/$name' \
    --env DATA_ROOT=/data \
    --env EPOCHS=10
    #--git-repo git@github.com:ubershmekel/tmp-please-ignore.git \


