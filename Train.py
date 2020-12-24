from os.path import dirname, abspath
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import time
# from setup import CURRENT_DATASET
# from setup import TRAIN_FOLDER
# from setup import TEST_FOLDER
# from setup import CHECKPOINT_FILE
# from setup import BATCH_SIZE_
# from setup import PATIENCE_
# from setup import VAL_STEPS_

SETUP_PATH = dirname(dirname(dirname(abspath(__file__)))) + '\\' + 'setup.txt'


def get_conf(filename, conf):
    with open(filename) as f:
        for line in f:
            raw = line
            if raw.startswith(conf):
                return raw.strip(conf).strip("\n").strip("'")


CURRENT_DATASET = str(get_conf(SETUP_PATH, "CURRENT_DATASET = "))
# DATASET_PATH = dirname(dirname(dirname(abspath(__file__)))) + '\\' + 'Dataset\\' + str(CURRENT_DATASET) + '\\'

path = dirname(dirname(dirname(abspath(__file__)))) + '\\Dataset' + '\\' + CURRENT_DATASET + '\\'
TRAINFOLDER = str(get_conf(SETUP_PATH, "TRAIN_FOLDER = "))
TESTFOLDER = str(get_conf(SETUP_PATH, "TEST_FOLDER = "))
CHECKPOINTFILE = str(get_conf(SETUP_PATH, "CHECKPOINT_FILE = "))
""" Modificare BATCH_SIZE con il numero di frame presenti in ogni video di un dataset """
BATCH_SIZE = int(get_conf(SETUP_PATH, "BATCH_SIZE_ = "))
""" PATIENCE definisce il numero di epoch senza miglioramente dopo la quale la fase di training verrà stoppata """
PATIENCE = int(get_conf(SETUP_PATH, "PATIENCE_ = "))
""" Inserire il numero di video presenti nella cartella di test, sia Violence che NonViolence """
VAL_STEPS = int(get_conf(SETUP_PATH, "VAL_STEPS_ = "))

# Helper: Save the min val_loss model in each epoch.
checkpointer = ModelCheckpoint(
    monitor='val_binary_accuracy',
    filepath='./checkpoints/' + CHECKPOINTFILE + '.{epoch:03d}-{val_binary_accuracy:.3f}.h5',
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)

# Helper: Stop when we stop learning.
# patience: number of epochs with no improvement after which training will be stopped.
early_stopper = EarlyStopping(patience=PATIENCE)

csv_logger = CSVLogger('training.log', separator=',', append=False)


def get_generators():
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=20,
                                       shear_range=0.2,
                                       zoom_range=0.4,
                                       horizontal_flip=True
                                       )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        path + '/' + TRAINFOLDER + '/',
        target_size=(150, 150),
        batch_size=8,
        shuffle=True,
        classes=['Violence', 'NonViolence'],
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        path + '/' + TESTFOLDER + '/',
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        shuffle=False,
        classes=['Violence', 'NonViolence'],
        class_mode='binary')
    label_map = (train_generator.class_indices)
    print(label_map)

    return train_generator, validation_generator


def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, input_shape=(150, 150, 3), include_top=False)

    # add a global spatial average pooling layer
    x = Sequential()(base_model.output)
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)

    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model


def train_model(model, nb_epoch, val_steps, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=val_steps,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model


def main(weights_file):
    generators = get_generators()
    model = get_model()
    print("Model loaded.")
    start = time.perf_counter()
    model = train_model(model, 1000, VAL_STEPS, generators, [checkpointer, early_stopper, csv_logger])
    print('Training phase execution time > ', time.perf_counter() - start)

weights_file = None
main(weights_file)