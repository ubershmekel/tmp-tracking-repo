"""
Get data from https://github.com/missinglinkai/Fruit-Images-Dataset/archive/master.zip
"""

from sys import platform
import os

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend

# MissingLink snippet
import missinglink

EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "Sample")
EXPERIMENT_NOTE = os.environ.get("EXPERIMENT_NOTE", "")
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.expanduser('~/sra/data/mldx9'))
EPOCHS = int(os.environ.get("EPOCHS", "5"))

OWNER_ID ='5cbb4c75-b52a-4386-af35-ce9ba735a4bb'
PROJECT_TOKEN ='ejHztrwUToiIucAA'
missinglink_callback = missinglink.KerasCallback(
    owner_id=OWNER_ID, project_token=PROJECT_TOKEN)
missinglink_callback.set_properties(
    display_name=EXPERIMENT_NAME,
    description=EXPERIMENT_NOTE)

train_data_dir = DATA_ROOT + '/train'
validation_data_dir = DATA_ROOT + '/validation'
test_data_dir = DATA_ROOT + '/test'

# Dimensions of images need to match the models we're transfer-learning from.
# The input shape for ResNet-50 is 224 by 224 by 3 with values from 0 to 1.0
img_width, img_height = 224, 224

#batch_size = 16
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Convert RGB [0, 255] to [0, 1.0]
test_validate_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_validate_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_validate_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

class_names_list = list(train_generator.class_indices.keys())
test_class_names_list = list(test_generator.class_indices.keys())
validation_class_names_list = list(validation_generator.class_indices.keys())

# Make sure class names are the same accross datasets.
assert train_generator.class_indices == test_generator.class_indices == validation_generator.class_indices
print(validation_generator.class_indices)

# invert key-value to value-key for MissingLink class name reporting.
index_to_name = {v: k for k, v in train_generator.class_indices.items()}
missinglink_callback.set_properties(class_mapping=index_to_name)

def get_model():
    # import inception with pre-trained weights. do not include fully #connected layers
    #base_model = applications.ResNet50(weights='imagenet', include_top=False)
    base_model = applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add a fully-connected layer
    x = Dense(512, activation='relu')(x)

    # and a fully connected output/classification layer
    predictions = Dense(len(class_names_list), activation='softmax')(x)

    # create the full network so we can train on it
    inception_transfer_model = Model(inputs=base_model.input, outputs=predictions)

    # create the full network so we can train on it
    inception_transfer_model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])
    return inception_transfer_model

model = get_model()

history_pretrained = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    shuffle=True,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[missinglink_callback])

print("Beginnging Model Evaluation")
with missinglink_callback.test(model):
    score = model.evaluate_generator(test_generator, steps=len(test_generator))
    for name, score in zip(model.metrics_names, score):
        print("Metric '{}': {}".format(name, score))

