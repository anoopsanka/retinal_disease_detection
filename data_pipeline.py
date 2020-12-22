import tensorflow as tf
import pandas as pd


def __get_data(csv_file):
    return pd.read_csv(csv_file)

def __pre_process(data):
    LABELS = list(data.columns[1:])

    def build_label(row):
        return [LABELS[idx] for idx, val in enumerate(row[1:]) if val == 1][0]
    
    data["label"] = data.apply(lambda x: build_label(x), axis=1)

    return data


def get_data_generators(data_dir, image_size, batch_size, validation_split=.30, csv_file='./Data/train/train-filtered.csv'):

    data = __get_data(csv_file)
    data = __pre_process(data)


    datagen_kwargs = dict(rescale=1./255, validation_split=validation_split)
    dataflow_kwargs = dict(target_size=image_size, batch_size=batch_size, interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

    valid_generator = valid_datagen.flow_from_dataframe(dataframe=data,directory=data_dir,
                        x_col="filename",
                        y_col="label",
                        subset="validation",
                        shuffle=False,
                        class_mode="categorical", **dataflow_kwargs)
    do_data_augmentation = False
    if do_data_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2,
            **datagen_kwargs)
    else:
        train_datagen = valid_datagen

    train_generator = train_datagen.flow_from_dataframe(dataframe=data,directory="./Data/train/train/",
                        x_col="filename",
                        y_col="label",
                        subset="training",
                        shuffle=True,
                        class_mode="categorical", **dataflow_kwargs)

    return (train_generator, valid_generator)
    


