from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, SpatialDropout2D, BatchNormalization, AveragePooling1D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.xception import Xception

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
np.random.seed(4666)  # for reproducibility

def append_ext(n):
    return n+".jpg"

def load_and_featurize_data():

    # Read in data
    train_df = pd.read_csv('data/train.csv')
    # Train test split
    train_df["image_id"] = train_df["image_id"].apply(append_ext)
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
                        x_col="image_id", y_col=columns, subset='training', batch_size=batch_size, seed=4666, 
                        shuffle=True, class_mode="raw", target_size=(img_rows, img_cols))
    
    val_generator=test_datagen.flow_from_dataframe(dataframe=val_df, directory="./data/", x_col="image_id", 
                        y_col=columns, batch_size=batch_size, seed=4666, shuffle=True, class_mode="raw", 
                        target_size=(img_rows, img_cols))

    test_generator=test_datagen.flow_from_dataframe(dataframe=test_df, directory="./data/", x_col="image_id", 
                        y_col=columns, batch_size=batch_size, seed=4666, shuffle=False, class_mode="raw", 
                        target_size=(img_rows, img_cols))
    
    return train_generator, val_generator, test_generator



def define_model(nb_filters, kernel_size, input_shape, pool_size):
    #nb_filters, kernel_size, input_shape, pool_size
    base_model = Xception(include_top=False, input_shape=(300, 300, 3), weights='imagenet')  # model is a linear stack of layers (don't change)
	
    # add new classifier layers
    model = base_model.output
    model = GlobalAveragePooling2D
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

def confusion_matrix(genarator):
    text_X = generator[0][0]
    test_y = generator.classes

    probs = model.predict(test_X)
    indices = probs.argsort(axis = 1)
    top_prediction = np.flip(indices, 1)[:, 0]
    top_prediction.reshape(1, -1)

    class_names = ['healthy', 'rust','multiple diseases', 'rust', 'scab']
    cnf_matrix = confusion_matrix(test_y, top_prediction)
    np.set_printoptions(precision=2)

    print(cnf_matrix)
    #Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # plt.savefig('./images/'+ts+'confusion_matrix.png')

    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    # plt.savefig('./images/'+ts+'normalized_confusion_matrix.png')

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
    
    model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = nb_epoch, verbose = 1, validation_data=val_generator, validation_steps=val_df.shape[0]//batch_size)
    
    #Call plot function
    # plot_hist(hist)

    # during fit process watch train and test error simultaneously
    score = model.evaluate(test_generator, verbose=1)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  # this is the one we care about

    # checkpoint = ModelCheckpoint(filepath='./temp/weights.hdf5', verbose=1, save_best_only=True)
    