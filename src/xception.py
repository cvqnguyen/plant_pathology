from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, SpatialDropout2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.applications.xception import Xception

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image
from datetime import datetime
ts = str(datetime.now().timestamp())
np.random.seed(4666)  # for reproducibility

def append_ext(n):
    return n+".jpg"

def load_and_featurize_data():

    # Read in data
    train_df = pd.read_csv('data/df_class.csv')
    # Train test split
    # train_df["image_id"] = train_df["image_id"].apply(append_ext)
    train_df, val_df = train_test_split(train_df, test_size=0.30, random_state=4666)
    val_df, test_df = train_test_split(val_df, test_size = .50, random_state=4666)
    return train_df, val_df, test_df
    

def generators():
    ## Reduce overfit by shearing, zoom, flip
    train_datagen = ImageDataGenerator(rescale=1./255., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255.)
    columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
    # Create generators for train, test, val to save memory
    train_generator=train_datagen.flow_from_dataframe(dataframe=train_df, directory="./data/", 
                        x_col="image_id", y_col='class', subset='training', batch_size=batch_size, seed=4666, 
                        shuffle=True, class_mode='categorical', target_size=(img_rows, img_cols))
    
    val_generator=test_datagen.flow_from_dataframe(dataframe=val_df, directory="./data/", x_col="image_id", 
                        y_col='class', batch_size=batch_size, seed=4666, shuffle=True, class_mode="categorical", 
                        target_size=(img_rows, img_cols))

    test_generator=test_datagen.flow_from_dataframe(dataframe=test_df, directory="./data/", x_col="image_id", 
                        y_col='class', batch_size=batch_size, seed=4666, shuffle=False, class_mode="categorical", 
                        target_size=(img_rows, img_cols))
    
    return train_generator, val_generator, test_generator



def define_model(nb_filters, kernel_size, input_shape, pool_size):
    #nb_filters, kernel_size, input_shape, pool_size
    base_model = Xception(include_top=False, input_shape=(300, 300, 3), weights='imagenet')  # model is a linear stack of layers (don't change)
	
    # add new classifier layers
    model = base_model.output
    model = GlobalAveragePooling2D()(model)
    denseout = Dense(nb_classes, activation='softmax')(model)
    # define new model
    model = Model(inputs=base_model.inputs, outputs=denseout)


    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # and KEEP metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    
    return image

def log_confusion_matrix(epoch, logs):
    
    # Use the model to predict the values from the test_images.
    test_pred_raw = model.predict(test_generator)
    
    test_pred = np.argmax(test_pred_raw, axis=1)
    
    # Calculate the confusion matrix using sklearn.metrics
    cm = confusion_matrix(test_labels, test_pred)
    
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)
    
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

if __name__ == '__main__':
    # important inputs to the model: don't changes the ones marked KEEP
    batch_size = 16  # number of training samples used at a time to update the weights
    nb_classes = 4  # number of output possibilities: [0 - 9] KEEP
    nb_epoch = 2       # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 300, 300   # the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, 3)   # 1 channel image input (grayscale) KEEP
    nb_filters = 32    # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (3, 3)  # convolutional kernel size, slides over image to learn features
    # strides = (1, 1)
    train_df, val_df, test_df = load_and_featurize_data()
    train_generator, val_generator, test_generator = generators()

    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    steps_per_epoch = int(train_df.shape[0] / batch_size)


    

    # model.summary()
    # checkpoint = ModelCheckpoint(filepath='../temp/'+ts+'.hdf5', verbose=1, save_best_only=True)
    # tensorboard = TensorBoard(log_dir = 'log/', histogram_freq = 0, write_graph=True)
    cm_callback = LambdaCallback(on_epoch_end=log_confusion_matrix)

    model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = nb_epoch, verbose = 1, validation_data=val_generator, validation_steps=val_df.shape[0]//batch_size)
    #callbacks=[checkpoint, tensorboard]
    #Call plot function
    # plot_hist(hist)

    # during fit process watch train and test error simultaneously
    score = model.evaluate(test_generator, verbose=1)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  # this is the one we care about

    
    # logdir = "log/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    

    # file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')