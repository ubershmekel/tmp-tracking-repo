"""
Get data from https://github.com/missinglinkai/Fruit-Images-Dataset/archive/master.zip
"""

from sys import platform
import os

# Make neural network initialization deterministic
import random
import numpy.random
from tensorflow import set_random_seed
random.seed(1983)
numpy.random.seed(1997)
set_random_seed(42)

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend
import numpy as np

# MissingLink snippet
import missinglink

# ops vars
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "")
EXPERIMENT_NOTE = os.environ.get("EXPERIMENT_NOTE", "")
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.expanduser('~/sra/data/mldx-small'))
OWNER_ID ='5cbb4c75-b52a-4386-af35-ce9ba735a4bb'
PROJECT_TOKEN ='ejHztrwUToiIucAA'

# Hyper paramaters
EPOCHS = int(os.environ.get("EPOCHS", "4"))
MODEL = os.environ.get("MODEL", "simple")
SIMPLE_LAYER_DIMENSIONALITY = int(os.environ.get("SIMPLE_LAYER_DIMENSIONALITY", "64"))
SIMPLE_LAYER_COUNT = int(os.environ.get("SIMPLE_LAYER_COUNT", "0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
OPTIMIZER = os.environ.get("OPTIMIZER", "sgd")
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))

missinglink_callback = missinglink.KerasCallback(
    owner_id=OWNER_ID,
    project_token=PROJECT_TOKEN)
missinglink_callback.set_properties(
    display_name=EXPERIMENT_NAME,
    description=EXPERIMENT_NOTE)

missinglink_callback.set_hyperparams(
    MODEL=MODEL,
    SIMPLE_LAYER_DIMENSIONALITY=SIMPLE_LAYER_DIMENSIONALITY,
    SIMPLE_LAYER_COUNT=SIMPLE_LAYER_COUNT,
    BATCH_SIZE=BATCH_SIZE,
    OPTIMIZER=OPTIMIZER,
    LEARNING_RATE=LEARNING_RATE,
)

# Note Adam defaults lr to 0.001
#       SGD defaults lr to 0.01
if OPTIMIZER == "adam":
    optimizer = optimizers.Adam(lr=LEARNING_RATE)
elif OPTIMIZER == "sgd":
    optimizer = optimizers.SGD(lr=LEARNING_RATE, momentum=0.9)
else:
    raise Exception("Invalid optimizer '{}'".format(OPTIMIZER))

train_data_dir = DATA_ROOT + '/train'
validation_data_dir = DATA_ROOT + '/validation'
test_data_dir = DATA_ROOT + '/test'

# Dimensions of images need to match the models we're transfer-learning from.
# The input shape for ResNet-50 is 224 by 224 by 3 with values from 0 to 1.0
img_width, img_height = 224, 224
input_channels = 3
image_shape = (img_height, img_width)
input_shape = (img_height, img_width, input_channels)


# Convert RGB [0, 255] to [0, 1.0]
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True
)

print("Train:")
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_shape,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

print("Test:")
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_shape,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

print("Validation:")
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=image_shape,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

class_names_list = list(train_generator.class_indices.keys())
class_count = len(class_names_list)

# Make sure class names are the same accross datasets.
assert train_generator.class_indices == test_generator.class_indices == validation_generator.class_indices

# invert key-value to value-key for MissingLink class name reporting.
index_to_name = {v: k for k, v in train_generator.class_indices.items()}
missinglink_callback.set_properties(class_mapping=index_to_name)

def get_transfer_model():
    # import inception with pre-trained weights. do not include fully #connected layers
    if MODEL == "resnet50":
        base_model = applications.ResNet50(weights='imagenet', include_top=False)
    elif MODEL == "mobilenet":
        base_model = applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise Exception("Invalid model: '{}'".format(MODEL))


    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add a fully-connected layer
    x = Dense(512, activation='relu')(x)

    # and a fully connected output/classification layer
    predictions = Dense(class_count, activation='softmax')(x)

    # create the full network so we can train on it
    transfer_learning_model = Model(inputs=base_model.input, outputs=predictions)

    # create the full network so we can train on it
    transfer_learning_model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return transfer_learning_model


def get_simple_model():
    model = Sequential()
    model.add(Dense(input_channels, input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=4, kernel_size=(4, 4), strides=8, input_shape=input_shape, activation = 'relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Conv2D(16, (3, 3), input_shape=input_shape, activation = 'relu'))
    # model.add(MaxPooling2D())
    #model.add(Dense(2, activation='relu'))
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())

    for _ in range(SIMPLE_LAYER_COUNT):
        model.add(Dense(SIMPLE_LAYER_DIMENSIONALITY, activation='relu'))

    model.add(Dense(class_count, activation='softmax'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.compile(loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])    
    return model

def evaluate(model):
    print("Beginnging model evaluation of {} batches".format(len(test_generator)))
    with missinglink_callback.test(model):
        score = model.evaluate_generator(test_generator, steps=len(test_generator))
        for name, score in zip(model.metrics_names, score):
            print("Metric '{}': {}".format(name, score))


    input_instances, labels = next(test_generator)
    preds = model.predict(input_instances)
    def max_indices(predictions_seq):
        return (np.argmax(probs) for probs in predictions_seq)
    print("Example (prediction, label) pairs:")
    import pprint
    pprint.pprint(list(zip(max_indices(preds), max_indices(labels))))

if MODEL == "simple":
    model = get_simple_model()
else:
    model = get_transfer_model()
model.summary()

#evaluate(model)
history_pretrained = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    shuffle=True,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[missinglink_callback])

evaluate(model)

