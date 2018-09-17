#!/bin/bash

# print commands before executing
set -x

ml run xp \
    --env EXPERIMENT_NAME resnet50-20e-sgd-10-50-lr-0.001 \
    --env MODEL resnet50 \
    --env EPOCHS 20 \
    --env SIMPLE_LAYER_COUNT 10 \
    --env SIMPLE_LAYER_DIMENSIONALITY 50 \
    --env OPTIMIZER sgd \
    --env LEARNING_RATE 0.001 \
    --env QUERY '@version:aca1a37a00aa7cc4ac10d876f5331ea94300ed06 @seed:1337 @split:0.2:0.2:0.6 yummy:True' \
    --env CLASS_COUNT 11

ml run xp \
    --env EXPERIMENT_NAME resnet50-20e-sgd-10-50-lr-0.01 \
    --env MODEL resnet50 \
    --env EPOCHS 20 \
    --env SIMPLE_LAYER_COUNT 10 \
    --env SIMPLE_LAYER_DIMENSIONALITY 50 \
    --env OPTIMIZER sgd \
    --env LEARNING_RATE 0.01 \
    --env QUERY '@version:aca1a37a00aa7cc4ac10d876f5331ea94300ed06 @seed:1337 @split:0.2:0.2:0.6 yummy:True' \
    --env CLASS_COUNT 11

ml run xp \
    --env EXPERIMENT_NAME resnet50-20e-sgd-10-50-lr-0.1 \
    --env MODEL resnet50 \
    --env EPOCHS 20 \
    --env SIMPLE_LAYER_COUNT 10 \
    --env SIMPLE_LAYER_DIMENSIONALITY 50 \
    --env OPTIMIZER sgd \
    --env LEARNING_RATE 0.1 \
    --env QUERY '@version:aca1a37a00aa7cc4ac10d876f5331ea94300ed06 @seed:1337 @split:0.2:0.2:0.6 yummy:True' \
    --env CLASS_COUNT 11

ml run xp \
    --env EXPERIMENT_NAME overfitting-simple-50e-sgd-0-0-lr-0.01 \
    --env MODEL simple \
    --env EPOCHS 50 \
    --env SIMPLE_LAYER_DIMENSIONALITY 0 \
    --env SIMPLE_LAYER_COUNT 0 \
    --env OPTIMIZER sgd \
    --env LEARNING_RATE 0.01 \
    --env QUERY '@version:aca1a37a00aa7cc4ac10d876f5331ea94300ed06 @seed:1337 @split:0.1:0.2:0.7 NOT class:test-multiple_fruits' \
    --env CLASS_COUNT 75

ml run xp \
    --env EXPERIMENT_NAME simple-20e-sgd-10-50-lr-0.01 \
    --env MODEL simple \
    --env EPOCHS 20 \
    --env SIMPLE_LAYER_COUNT 10 \
    --env SIMPLE_LAYER_DIMENSIONALITY 50 \
    --env OPTIMIZER sgd \
    --env LEARNING_RATE 0.01 \
    --env QUERY '@version:aca1a37a00aa7cc4ac10d876f5331ea94300ed06 @seed:1337 @split:0.2:0.2:0.6 yummy:True' \
    --env CLASS_COUNT 11

ml run xp \
    --env EXPERIMENT_NAME simple-20e-adam-10-50-lr-0.01 \
    --env MODEL simple \
    --env EPOCHS 20 \
    --env SIMPLE_LAYER_COUNT 10 \
    --env SIMPLE_LAYER_DIMENSIONALITY 50 \
    --env OPTIMIZER adam \
    --env LEARNING_RATE 0.01 \
    --env QUERY '@version:aca1a37a00aa7cc4ac10d876f5331ea94300ed06 @seed:1337 @split:0.2:0.2:0.6 yummy:True' \
    --env CLASS_COUNT 11

ml run xp \
    --env EXPERIMENT_NAME simple-20e-adam-10-50-lr-0.05 \
    --env MODEL simple \
    --env EPOCHS 20 \
    --env SIMPLE_LAYER_COUNT 10 \
    --env SIMPLE_LAYER_DIMENSIONALITY 50 \
    --env OPTIMIZER adam \
    --env LEARNING_RATE 0.05 \
    --env QUERY '@version:aca1a37a00aa7cc4ac10d876f5331ea94300ed06 @seed:1337 @split:0.2:0.2:0.6 yummy:True' \
    --env CLASS_COUNT 11