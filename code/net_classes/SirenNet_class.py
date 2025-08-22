# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: file containing the class that will contain the various versions of CNN made for KWS problem

description: network description at the end of the file in the note
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate

# ------------------------------------ start: utility methods ------------------------------------

# utility function to implement the inception module in which 1×1, 3×3, 5×5 convolution and 3×3 max pooling are executed in parallel and their output is merged.
# in_net: is the input , fil_1x1: is the number of filters of conv 1x1 layer, the same for other similar fil
# fil_1x1_3x3: is the number of filters of the 1x1 reduction convolutionary layer before conv 3x3 and so on for others similar fil
# fil_m_pool: is the number of filter of the 1x1 convolutionary layer after max pooling 
def inception_mod(in_net, fil_1x1, fil_1x1_3x3, fil_3x3, fil_1x1_5x5, fil_5x5, fil_m_pool):
    # four parallel path
    
    path1 = layers.Conv2D(filters=fil_1x1, kernel_size=(1, 1), padding='same', activation='relu')(in_net)       # conv 1x1
    
    path2 = layers.Conv2D(filters=fil_1x1_3x3, kernel_size=(1, 1), padding='same', activation='relu')(in_net)   # conv 1x1 to reduce
    path2 = layers.Conv2D(filters=fil_3x3, kernel_size=(1, 1), padding='same', activation='relu')(path2)        # conv 3x3
    
    path3 = layers.Conv2D(filters=fil_1x1_5x5, kernel_size=(1, 1), padding='same', activation='relu')(in_net)   # conv 1x1 to reduce
    path3 = layers.Conv2D(filters=fil_5x5, kernel_size=(1, 1), padding='same', activation='relu')(path3)        # conv 5x5
    
    path4 = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(in_net)                          # max pool
    path4 = layers.Conv2D(filters=fil_m_pool, kernel_size=(1, 1), padding='same', activation='relu')(path4)     # conv 1x1 to reduce
    
    output = Concatenate(axis=3)([path1, path2, path3, path4])      # merge of the different path
    return output

# ------------------------------------ end: utility methods ------------------------------------

# class that implement the IfriNet models
class SirenNet:
    # constructor
    # constructor with image size of the input layer
    def __init__(self,class_number,img_width = 224,img_height = 224,img_channel = 3):
        self.model = None                       # var that will contain the model of the CNN AlexNet
        self.num_classes = class_number         # var that will contain the number of the classes of the problem (in our case is 2 (fire, no_fire))
        # var for the image dimension
        self.img_height = img_height            # height of the images in input to CNN
        self.img_width = img_width              # width of the images in input to CNN
        self.img_channel = img_channel          # channel of the images in input to CNN (RGB)

    # method for make the models of the CNN. 'version_model' indicate the version of the CNN Ifrit
    def make_model(self,version_model):
        
        if version_model == 0:                      # first verstion, for more information see Note 1 at the end of the file
            inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))                       # input
            
            net = layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(2,2), padding='same', activation='relu')(inp)      # first conv layer
            net = layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2))(net)                                                 # max pool
            
            net = inception_mod(net, fil_1x1=16, fil_1x1_3x3=8, fil_3x3=32, fil_1x1_5x5=16, fil_5x5=64, fil_m_pool=32)      # first inception layer
            net = layers.MaxPooling2D(3, strides=1)(net)                                                                    # max pool 
            
            net = inception_mod(net, fil_1x1=16, fil_1x1_3x3=8, fil_3x3=32, fil_1x1_5x5=16, fil_5x5=64, fil_m_pool=32)      # second inception layer
            net = inception_mod(net, fil_1x1=32, fil_1x1_3x3=16, fil_3x3=64, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=64)     # third inception layer
            net = layers.MaxPooling2D(3, strides=2)(net)                                                                    # max pool
            
            net = inception_mod(net, fil_1x1=64, fil_1x1_3x3=16, fil_3x3=128, fil_1x1_5x5=8, fil_5x5=16, fil_m_pool=16)     # fourth inception layer
            net = layers.GlobalAveragePooling2D()(net)                                                                      # avg pool
            
            net = layers.Dense(64, activation='relu')(net)                              # fully connect 
            net = layers.Dropout(0.3)(net)                                              # dropout
            out = layers.Dense(self.num_classes, activation='softmax')(net)             # output layer
            
            self.model = Model(inputs = inp, outputs = out)             # assign the CNN in model
            
        elif version_model == 2:                    # second verstion, for more information see Note 2 at the end of the file
            self.model = models.Sequential()
            # 1st Conv layer
            self.model.add(layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd Conv layer
            self.model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 3rd Conv layer
            self.model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 4th Conv layer
            self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            self.model.add(layers.Flatten())
            # 1th dense layer
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd dense layer
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            # Output layer
            self.model.add(layers.Dense(self.num_classes, activation='softmax'))
            
        elif version_model == 3:                    # third verstion, for more information see Note 3 at the end of the file
            self.model = models.Sequential()
            # 1st Conv layer
            self.model.add(layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd Conv layer
            self.model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 3rd Conv layer
            self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 4th Conv layer
            self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            self.model.add(layers.Flatten())
            # 1th dense layer
            self.model.add(layers.Dense(64, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd dense layer
            self.model.add(layers.Dense(64, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            # Output layer
            self.model.add(layers.Dense(self.num_classes, activation='softmax'))
            
        elif version_model == 4:                    # fourth verstion, for more information see Note 4 at the end of the file
            self.model = models.Sequential()                                   # rete del modello
            # 1st Conv layer
            self.model.add(layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd Conv layer
            self.model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 3rd Conv layer
            self.model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), padding='valid', activation='relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            self.model.add(layers.Flatten())
            # 1th dense layer
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd dense layer
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            # Output layer
            self.model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model.summary()
            
    # method for compile the model
    def compile_model(self):
        # compile Adam
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
    # method for return the model
    def return_model(self):
        return self.model  

"""
-------- Notes --------
-- Note 0 --
    CNN that takes inspiration from the GoogLeNet model. This network consists of a convolutional layer, with max pool, 
    followed by four inception modules, a fully connect layer, with dropout, and the output layer.
    In particular, in the four inception modules there is maxpool after the first and third, and at the end there is 
    global average pooling. In these modules we tried to give more emphasis from first to filters with larger kenrel and then, 
    as the depth of the network increases and the input size decreases give more emphasis to filters with smaller kernels.
    Is a net of my creation which had excellent results in image recognition.
    
-- Note 2 --
    CNN inspired by the AlexNet model. Network consisting of 4 convolutional layers, 2 fully connect layers, with dropout, 
    and the output layer. As in AlexNet the max pool is present only after the first and fourth convolutional layers. 
    Normalization is present after all of the layers except the penultimate one. 

-- Note 3 --
    Model inspired by a "lite" version of the second model. In this network we tried to go to reduce the number of training 
    weights in order to have a network similar to the architecture proposed before but much lighter and faster. 

-- Note 4 --
    Simple CNN consisting of 3 convolutional layers, cascaded with max pool and batch normalization, 
    followed by 2 fully connect layers before the output layer. 
"""