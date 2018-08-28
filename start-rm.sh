#!/bin/bash
set -x
#  --data-volume 5685154290860032
#ml run xp --org yuval --project 5686084218388480 --git-repo git@github.com:ubershmekel/tmp-please-ignore.git --command '/code/remote-script.sh' --gpu

# full = 50,000
# 0.1 = 5,000
# 0.2 * 0.1 = 1,000
# 0.05 * 0.1 = 250

ml run xp --org yuval --project 5686084218388480 \
    --command 'EPOCHS=10 DATA_ROOT=/data /code/remote-script.sh' \
    --gpu \
    --data-volume 5685154290860032 \
    --data-query '@version:ecc29bb3531f39963c16f8578f58c0466a4d7143 @seed:1337 @split:0.1:0.8:0.1 NOT class:test-multiple_fruits' \
    --data-dest '/data/$phase/$class'

#    --env EPOCHS=10 \
#    --env DATA_ROOT=/data


    #--git-repo git@github.com:ubershmekel/tmp-please-ignore.git \


