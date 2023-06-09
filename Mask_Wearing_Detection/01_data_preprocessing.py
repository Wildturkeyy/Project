import tensorflow
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from imgaug import parameters as iap
import imgaug as ia
import imgaug.augmenters as iaa
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications import DenseNet121, DenseNet169, MobileNet, MobileNetV2
from keras.optimizers import Adam
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm
from keras.callbacks import Callback
from keras import applications
import inspect
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from imgaug.augmenters import size
from imgaug.augmenters.arithmetic import Invert
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

TRAIN_DIR = './mask_dataset/train/'
TEST_DIR = './mask_dataset/test/'

IM_HEIGHT, IM_WIDTH = 224, 224

def sometimes(aug): return iaa.Sometimes(0.1, aug)

seq_train = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        sometimes(iaa.Crop(percent=(0, 0.1))),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),
        iaa.SomeOf((0, 5),
                   [
            sometimes(
                iaa.Superpixels(
                    p_replace=(0, 1.0),
                    n_segments=(20, 200)
                )
            ),
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 11)),
            ]),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            sometimes(iaa.OneOf([
                iaa.EdgeDetect(alpha=(0, 0.7)),
                iaa.DirectedEdgeDetect(
                    alpha=(0, 0.7), direction=(0.0, 1.0)
                ),
            ])),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05*255), per_channel=0.5
            ),
            iaa.OneOf([
                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                iaa.CoarseDropout(
                    (0.03, 0.15), size_percent=(0.02, 0.05),
                    per_channel=0.2
                ),
            ]),
            iaa.Invert(0.05, per_channel=True),
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            sometimes(
                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
            ),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
        ],
            random_order=True
        )
    ],
    random_order=True
)

def img_preprocessing_train(img):
    img = img.reshape(1, img.shape[0], img.shape[1], 3)
    img = img.astype(np.uint8)
    generate_img = seq_train(images=img)
    generate_img = generate_img/255.
    return generate_img.reshape(IM_HEIGHT, IM_WIDTH, 3).astype(np.float32)

train_datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.2],
    zoom_range=[0.8, 1.2],
    preprocessing_function=img_preprocessing_train
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IM_HEIGHT, IM_WIDTH),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

test_datagen = ImageDataGenerator(rescale=1/255.)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IM_HEIGHT, IM_WIDTH),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

print(train_generator.n, train_generator.batch_size)
print(test_generator.n)

img, label = train_generator.next()
label_num = np.argmax(label, 1)
print(label_num)


fig = plt.figure(figsize=(30, 20))

for i in range(30):

    subplot = fig.add_subplot(6, 5, i + 1)

    subplot.set_xticks([])

    subplot.set_yticks([])

    subplot.set_title('label: %d' % label_num[i])

    subplot.imshow(img[i].reshape((IM_WIDTH, IM_HEIGHT, 3)))

plt.show()


class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


timing_callback = TimingCallback()


model_dictionary = {
    m[0]: m[1] for m in inspect.getmembers(applications, inspect.isfunction)
}

keys_to_remove = [
    key
    for key in model_dictionary
    if key.startswith("EfficientNet")
]

for key in keys_to_remove:
    model_dictionary.pop(key, None)


for key in model_dictionary:
    print(key)


model_benchmarks = {
    "model_name": [],
    "num_model_params": [],
    "val_loss": [],
    "val_accuracy": [],
    "avg_train_time": [],
}

for model_name, model in tqdm(model_dictionary.items()):

    if "NASNetLarge" in model_name:
        input_shape = (331, 331, 3)
    elif "NASNetMobile" in model_name:
        input_shape = (224, 224, 3)

    else:
        input_shape = (224, 224, 3)

    pre_trained_model = model(include_top=False, input_shape=input_shape)
    pre_trained_model.trainable = False

    clf_model = Model(
        inputs=pre_trained_model.input,
        outputs=Dense(4, activation="softmax")(
            GlobalAveragePooling2D()(
                Dense(128, activation="relu")(
                    Dropout(0.5)(pre_trained_model.output))
            )
        ),
    )

    clf_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = clf_model.fit(
        train_generator,
        epochs=3,
        callbacks=[timing_callback],
        validation_data=test_generator,
    )

    model_benchmarks["model_name"].append(model_name)
    model_benchmarks["num_model_params"].append(
        pre_trained_model.count_params())
    model_benchmarks["val_loss"].append(history.history["val_loss"][-1])
    model_benchmarks["val_accuracy"].append(
        history.history["val_accuracy"][-1])
    model_benchmarks["avg_train_time"].append(sum(timing_callback.logs) / 3)


benchmark_df = pd.DataFrame(model_benchmarks)

benchmark_df.to_csv(r'C:\Work\pretrained_model_benchmarks.csv')

bm_params_df = benchmark_df.sort_values('val_accuracy', ascending=False)
bm_params_df.head(10)


cb_checkpoint = ModelCheckpoint(filepath=r"C:\Work\DenseNet121B",

                                monitor='val_acc',
                                vervose=1,
                                save_best_only=True
                                )
