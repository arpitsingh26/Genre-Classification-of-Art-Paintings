import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import optimizers
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
#from baseline import *

mergedDatapath = '../Pandora_18k_merged'
img_width,img_height = 224,224

def createCustomModel(num_classes=18):
    VGG_model_path = "../VGG16_original.h5"
    #model = VGG16(input_shape = (img_width, img_height, 3))
    model = load_model(VGG_model_path)
    """
    image = load_img('239.jpg', target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))
    """

    #model.layers.pop()
    x = model.layers[-5].output # dis regard the fc layers after this
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # creating the final model 
    model_final = Model(input = model.input, output = predictions)
        #Total 24 layers,16 with weights
    """
    [<keras.engine.topology.InputLayer at 0x7f0eb73f48d0>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb73f4a90>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb73f4ad0>,
     <keras.layers.pooling.MaxPooling2D at 0x7f0eb73f4b50>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb73f4e90>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb73f4ed0>,
     <keras.layers.pooling.MaxPooling2D at 0x7f0eb73f4fd0>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e150>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e190>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e290>,
     <keras.layers.pooling.MaxPooling2D at 0x7f0eb639e390>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e4d0>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e510>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e610>,
     <keras.layers.pooling.MaxPooling2D at 0x7f0eb639e710>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e850>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e890>,
     <keras.layers.convolutional.Conv2D at 0x7f0eb639e990>,
     <keras.layers.pooling.MaxPooling2D at 0x7f0eb639ea90>,  #19 till here freezing
     <keras.layers.core.Flatten at 0x7f0eb4e3c3d0>,
     <keras.layers.core.Dense at 0x7f0eb4e3c2d0>,
     <keras.layers.core.Dropout at 0x7f0eb4e3c050>,
     <keras.layers.core.Dense at 0x7f0eb4e64410>,
     <keras.layers.core.Dense at 0x7f0eb4e64390>]
    """
    for layer in model.layers[:19]:
        layer.trainable = False
    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


    print model_final.summary()
    model_final.save("../CustomVGG.h5")




def getDataset(perClassCount = -1, size=(500,500)):
    
    pkfilename = mergedDatapath + '/' + 'feature_images_keras.sav'
    
    d = mergedDatapath
    dirsNames = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    dirs = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

    np.random.seed(0)
    X,Y= [],[]
    for i in range(len(dirs)):
        d = dirs[i]
        filenames = glob.glob(d+'/*.jpg')
        l = len(filenames)

        if perClassCount == -1 : # take all data
            X += filenames
            Y += [i for j in xrange(len(filenames))]
        else:
            f = list(np.random.choice(filenames,replace=False,size=perClassCount))
            X += f
            Y += [i for j in xrange(perClassCount)] # label of these images
        

    print("Reading images...")
    images = []
    j = 0
    for i in range(len(X)):
        images.append(img_to_array(load_img(X[i], target_size=size)))
        if (i+1)%500==0:
            print "Read ",i+1," images."
    images=np.array(images) 
    images=preprocess_input(images)
    X = images
    #print("Dumping the features for later use..")
    
    #pickle.dump((X,Y), open(pkfilename, 'wb'))

    return X,keras.utils.to_categorical(np.array(Y),18)
    

if __name__ == '__main__':
    #createCustomModel()
    model = load_model("../CustomVGG.h5")
    print model.summary()
    
    X,Y = getDataset(perClassCount = 350,size=(img_width, img_height))
    trainx,testx, trainy ,testy = train_test_split(X,Y,test_size=0.2)
    print trainx.shape, testx.shape
    print "Traing the custom model"
    model.fit(trainx, trainy,batch_size=20,nb_epoch=20,shuffle=True,verbose=1,validation_data=(testx, testy))
    model.save_weights("../try1_weights.h5")
    