#!/bin/bash
set -x
#  --data-volume 5685154290860032
#ml run xp --org yuval --project 5686084218388480 --git-repo git@github.com:ubershmekel/tmp-please-ignore.git --command '/code/remote-script.sh' --gpu


ml run xp --org yuval --project 5686084218388480 \
    --git-repo git@github.com:ubershmekel/tmp-please-ignore.git \
    --command '/code/remote-script.sh' \
    --gpu \
    --data-volume 5685154290860032 \
    --data-query '@version:f6f90ef73a86ab88f5dfec00433038555ad9a508 @sample:0.05 @seed:1337' \
    --data-dest '/data/mldx2/$dir'


