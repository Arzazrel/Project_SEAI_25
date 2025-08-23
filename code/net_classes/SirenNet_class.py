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
        
        if version_model == 0:                      # first verstion, for more information see Note 0 at the end of the file
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
            
        elif version_model == 1:                    # second verstion, for more information see Note 1 at the end of the file
            # calculate dimensions for rectangular filters
            k_s_freq = self.img_width // 10     # size for the filter that operate for each frequency band on 'k_s_freq' frames time -> (1, k_s_freq)
            k_s_time = self.img_height // 10    # size for the filter that operate for each frames time on 'k_s_time' frequency bands -> (k_s_freq, 1)
                        
            inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))       # input
            # 1st Conv layer 
            # - double branch to get both temporal and spectral features.
            branch_1 = layers.Conv2D(16, (1, k_s_freq), padding="same", activation="relu")(inp) # frequency
            branch_2 = layers.Conv2D(16, (k_s_time, 1), padding="same", activation="relu")(inp) # temporal
            branches = layers.Concatenate(axis=-1)([branch_1, branch_2])                        # concatenate
            # - conv
            net  = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(branches)       # fusion
            net  = layers.MaxPooling2D(pool_size=(2, 1))(net)                                   # pool on time

            # 2nd Conv layer 
            k_s_freq = k_s_freq * 2  
            k_s_time = k_s_time * 2
            # - double branch to get both temporal and spectral features.
            branch_1 = layers.Conv2D(32, (1, k_s_freq), padding="same", activation="relu")(net) # frequency
            branch_2 = layers.Conv2D(32, (k_s_time, 1), padding="same", activation="relu")(net) # temporal
            branches = layers.Concatenate(axis=-1)([branch_1, branch_2])
            # - conv
            net = layers.Conv2D(64, (3,3), padding="same", activation="relu")(branches)
            # -- maxpool
            pool_time = layers.MaxPooling2D(pool_size=(2,1))(net)                               # Pooling on time
            pool_freq = layers.MaxPooling2D(pool_size=(1,2))(net)                               # Pooling on frequency
            pools = layers.Concatenate(axis=-1)([pool_time, pool_freq])                         # concatenate
            
            # 3rd layer - inception modules
            net = inception_mod(pools, fil_1x1=16, fil_1x1_3x3=8, fil_3x3=32, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=32)    # first inception layer
            net = inception_mod(net, fil_1x1=32, fil_1x1_3x3=16, fil_3x3=64, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=64)     # second inception layer
            net = layers.MaxPooling2D(3, strides=2)(net)                                                                    # max pool (quadratic)
            
            # 4th layer - inception modules
            net = inception_mod(net, fil_1x1=64, fil_1x1_3x3=16, fil_3x3=128, fil_1x1_5x5=8, fil_5x5=16, fil_m_pool=16)     # third inception layer
            net = layers.GlobalAveragePooling2D()(net)                                                                      # avg pool
            
            # 1th dense layer
            net = layers.Dense(64, activation='relu')(net)                              # fully connect 
            net = layers.Dropout(0.3)(net)                                              # dropout
            # Output layer
            out = layers.Dense(self.num_classes, activation='softmax')(net)             # output layer
            
            self.model = Model(inputs = inp, outputs = out)             # assign the CNN in model
                    
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
    Is a CNN of my creation which had excellent results in image recognition.
    
-- Note 1 --
    The idea behind this CNN is to work with multiple branches using rectangular filters and then concatenate them. 
    The rectangular filters are sized at 1 so that they only work in frequency or only in time. Conv2D(filters, kernel_size=(height, width), ...)
    In our case:
    - height -> dimension along the vertical axis = frequencies in the spectrogram
    - width -> dimension along the horizontal axis = time (windows)
    E.g.:
    Conv2D(32, (1, 5)) → filter that looks at 1 frequency band × 5 frames in time
    Conv2D(32, (5, 1)) → filter that looks at 5 frequency bands × 1 frame in time

    When you have multiple branches (e.g., a 1×k filter that only works in frequency and a k×1 filter that only works in time), each one produces a feature map.
    Concatenate(axis=-1) joins these features along the channel dimension so that the next layer sees both temporal and spectral features.
    E.g.:
    Input: (time=100, freq=40, channels=1)
    Conv(1×7): (100, 40, 32) , Conv(7×1): (100, 40, 32) -> Concatenate: (100, 40, 64)
    
    MaxPools are also initially rectangular in time (k,1). This supports the idea that spectral information is very delicate.
    Compressing along the frequency too early risks "crushing" important formants or harmonic patterns.
    However, over time, you can afford to reduce the resolution because you have "redundancy—nearby" time frames often contain similar information.
    
    Rectangular filter patterns:
    - In the first layers: smaller filters -> capture local details without mixing too much (few bands, few frames).
    - In the deeper layers: larger filters -> see larger regions, for global correlations (patterns over more time + more frequencies).
    
    So what we want to achieve with this network:
    - First layers -> small rectangular filters (with a unit size) for local details without mixing too many details from different areas.
    - Middle layers -> larger rectangular filters (with a unit size) for longer correlations.
    - Final layers -> square filters to mix time and frequency together.

    This version of CNN also uses an inception module. 
    The inception module takes an input and processes it in parallel with several different filters, then concatenates the results along the channel dimension.
    In this case:
    - Conv(1×1) -> captures simple relationships / reduces channels
    - Conv(3×3) -> local patterns
    - Conv(5×5) -> broader patterns
    - MaxPool -> captures robust/invariant information
    - Concatenate -> combines all scales into a single feature map

    This allows the network to automatically decide which “representation scale” to use.
    In this case, it can be useful for capturing spectrogram patterns at different scales:
    - short spectral variations (such as bursts)
    - harmonics that extend across multiple frequencies
    - longer temporal patterns

    I chose to insert the inception module in the middle layers of the CNN after the rectangular layers.
"""