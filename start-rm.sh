#!/bin/bash
set -x
#  --data-volume 5685154290860032
ml run xp --org yuval --project 5686084218388480 --git-repo git@github.com:ubershmekel/tmp-please-ignore.git --command '/code/remote-script.sh' --gpu
