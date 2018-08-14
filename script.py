
# Install dependencies
#!pip install keras
#!pip install plotly


# export to py
#!jupyter nbconvert --to script config_template.ipynb


import missinglink

OWNER_ID ='5cbb4c75-b52a-4386-af35-ce9ba735a4bb'
PROJECT_TOKEN ='ejHztrwUToiIucAA'
missinglink_callback = missinglink.KerasCallback(
    owner_id=OWNER_ID, project_token=PROJECT_TOKEN)
missinglink_callback.set_properties(
    display_name='Keras convolutional neural network',
    description='Two dimensional convolutional neural network')

import os
from os import listdir, makedirs
from os.path import join, exists, expanduser

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend
import tensorflow as tf
# Any results you write to the current directory are saved as output.

# This cell is only relevant to kaggle notebooks

#cache_dir = expanduser(join('~', '.keras'))
#if not exists(cache_dir):
#    makedirs(cache_dir)
#models_dir = join(cache_dir, 'models')
#if not exists(models_dir):
#    makedirs(models_dir)

#!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
#!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
#!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
#print("Available Pretrained Models:\n")
#!ls ~/.keras/models

# dimensions of our images.
# We set the img_width and img_height according to the pretrained models we are
# going to use. The input shape for ResNet-50 is 224 by 224 by 3 with values from 0 to 1.0
img_width, img_height = 224, 224

train_data_dir = './fruits-360/Training/'
validation_data_dir = './fruits-360/Test/'
nb_train_samples = 31688
nb_validation_samples = 10657
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# RGB 255
test_datagen = ImageDataGenerator(rescale=1. / 255)

# TODO: How do I get 10% of the data in `flow_from_directory`?
#           Stackoverflow question?
#           Blog post?
#           Hash or random?
# TODO: read about overfitting early for understanding if my architecture works.

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

import pandas as pd
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

training_data = pd.DataFrame(train_generator.classes, columns=['classes'])
testing_data = pd.DataFrame(validation_generator.classes, columns=['classes'])

def create_stack_bar_data(col, df):
    aggregated = df[col].value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values

x1, y1 = create_stack_bar_data('classes', training_data)
#x1 = list(train_generator.class_indices.keys())
# TODO: is class_names_list in the right order?
class_names_list = list(train_generator.class_indices.keys())

trace1 = go.Bar(x=class_names_list, y=y1, opacity=0.75, name="Class Count")
layout = dict(height=400, width=1200, title='Class Distribution in Training Data', legend=dict(orientation="h"), 
                yaxis = dict(title = 'Class Count'))
fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)

x1, y1 = create_stack_bar_data('classes', testing_data)
train_class_names = list(validation_generator.class_indices.keys())

trace1 = go.Bar(x=train_class_names, y=y1, opacity=0.75, name="Class Count")
layout = dict(height=400, width=1100, title='Class Distribution in Validation Data', legend=dict(orientation="h"), 
                yaxis = dict(title = 'Class Count'))
fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)

#import inception with pre-trained weights. do not include fully #connected layers
inception_base = applications.ResNet50(weights='imagenet', include_top=False)

print("!@#~#~!@!@#$#$@#$@#$@#$@$%^&*()(*&^%$#@#$%^&*(*&^%$#@")
print("Inception")
print("!@#~#~!@!@#$#$@#$@#$@#$@$%^&*()(*&^%$#@#$%^&*(*&^%$#@")
print(inception_base.summary())
print("!@#~#~!@!@#$#$@#$@#$@#$@$%^&*()(*&^%$#@#$%^&*(*&^%$#@")

# TODO: Put the fruits images in data management
# TODO: Look at print(inception_base.summary())
# TODO: Maybe put the resnet50 model (mali data clone)??? Maybe too advanced.

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a fully connected output/classification layer
predictions = Dense(len(class_names_list), activation='softmax')(x)
# create the full network so we can train on it
inception_transfer = Model(inputs=inception_base.input, outputs=predictions)

#import inception with pre-trained weights. do not include fully #connected layers
inception_base_vanilla = applications.ResNet50(weights=None, include_top=False)

# add a global spatial average pooling layer
x = inception_base_vanilla.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a fully connected output/classification layer
predictions = Dense(len(class_names_list), activation='softmax')(x)
# create the full network so we can train on it
inception_transfer_vanilla = Model(inputs=inception_base_vanilla.input, outputs=predictions)

inception_transfer.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

inception_transfer_vanilla.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

from tensorflow.python.client import device_lib

print('----------- devices ------------')
print(device_lib.list_local_devices())
print('----------- devices ------------')

import tensorflow as tf

#model.fit(
#    x_train, y_train, batch_size=BATCH_SIZE,
#    nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT,
#    callbacks=[missinglink_callback])

with tf.device("/device:GPU:0"):
    history_pretrained = inception_transfer.fit_generator(
    train_generator,
    epochs=5, shuffle = True, verbose = 1, validation_data = validation_generator,
    callbacks=[missinglink_callback])

with tf.device("/device:GPU:0"):
    history_vanilla = inception_transfer_vanilla.fit_generator(
    train_generator,
    epochs=5, shuffle = True, verbose = 1, validation_data = validation_generator,
    callbacks=[missinglink_callback])

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_pretrained.history['val_acc'])
plt.plot(history_vanilla.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_pretrained.history['val_loss'])
plt.plot(history_vanilla.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()

from IPython.lib import kernel
print(dir(kernel))
print(kernel.get_connection_info())
print(kernel.get_connection_file())

