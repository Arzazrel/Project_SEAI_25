# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana
"""
import os
import time
import random
import math
import numpy as np
import tensorflow_io as tfio  
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib 
# for model
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
# for plot
import matplotlib.pyplot as plt
# import of my files
from net_classes import SirenNet_class as SirenNet

# ------------------------------------ start: global var ------------------------------------
# -- path --
dir_father = ".."               # fathere folder containing all other folders used
dir_ds_name = "dataset"         # folder containing all the dataset to load for the program
ds_name = "speech_ds"           # folder name for the target dataset
#ds_name = "test"               # folder name for the target dataset (testing purpose)
path_dir_ds = os.path.join(dir_father,dir_ds_name,ds_name)# folder containing the dataset to load (nested dataset)

dir_model_name = "model"        # folder in which there are saved the CNN model
path_check_point_model = os.path.join(dir_father,dir_model_name,"train_hdf5")  # folder in which there are saved the checkpoint for the model training

copy_ds_name = "res_ds_copy"    # the name of the folder to contain the elaborated copy of the dataset (spectral image of the audio files) -> timing and memory optimization
path_dir_copy_ds = os.path.join(dir_father,dir_ds_name,copy_ds_name)# folder containing the dataset of spectral images calculated from audio dataset

# -- audio file var --
desired_sr = 16000              # want audio as 16kHz mono
num_mel_bins = 64               # num of bins for Mel-spectrogram
num_mfccs = 13                  # num of coefficient to take
mel_f_min = 80.0                # min frequency for the bins of MEL 
mel_f_max = 7600.0              # max frequency for the bins of MEL (or desired_sr/2)

# -- command set var --
command_set = ['forward', 'backward', 'stop','go','up','down','left','right']   # array containing all the target commands
generic_class_name = "unknown"  # name for the generic class containing all the word that aren't command for this project

# -- dataset variables  --
classes = []                    # the label associated with each class will be the position that the class name will have in this array
total_audio_ds = []             # contain the path for the whole audio dataset
total_labels = []               # contain the labels of the whole dataset
train_data = []                 # contain the data choosen as train set in fit
train_label = []                # contain the labels of the file audio choosen as train set in fit
val_data = []                   # contain the data choosen as validation set in fit
val_label = []                  # contain the labels of the file audio choosen as validation set in fit
test_data = []                  # contain the data choosen as test set in evaluation
test_label = []                 # contain the labels of the file audio choosen as test set in evaluation
test_set_split = 0.2            # test set size as a percentage of the total dataset
val_set_split = 0.1             # validation set size as a percentage of the training set

# ---- model variables ----
network = None                  # contain the CNN model, default value is None
truncate_set = False            # variable which indicates whether the sets (train, test,val) must be truncate or not when divided to batch_size                     
batch_size = 32                 # batch size for training, this is the default value
data_height = 224               # default value for height of the 2D images in input to CNN (images which represent the matrix of MEL or MFCC)
data_width = 224                # default value for width the 2D images in input to CNN (images which represent the matrix of MEL or MFCC)
data_channel = 1                # default value for channel the 2D images in input to CNN (images which represent the matrix of MEL or MFCC)
epochs = 100                    # number of epochs for training, this is the deafault value
early_patience = 10             # number of epochs with no improvement after which training will be stopped 
chosen_model = 0                # indicate the model to use for training -> 0: CNN for image (SirenNet0) , 
num_test = 1                   # number of traning to execute on the same model from which the data will then be taken and averaged to obtain the general performance of the trained model

# -- status variable --
do_mfcc = False                 # if 'true' -> calculate and use MFCC for training , if 'false' -> donìt calculate MFCC and use MEL for training
erase_copy_ds = False           # if 'true' -> if the dataset copy folder is already present, delete it and recalculate everything
                                # if 'false' -> if the dataset copy folder is already present, don't delete it and don't recalculate everything

# ------------------------------------ end: global var ------------------------------------

# ------------------------------------ start: utilities method ------------------------------------
   
# method to convert time in second in a string with this format hh:mm:ss
def format_time(time):
    hours, remainder = divmod(int(time), 3600)
    minutes, seconds = divmod(remainder, 60)

    return(f"{hours:02}:{minutes:02}:{seconds:02}")
   
# method that read a data from path and get the correct shape and set the global var of height and width for the CNN input layer
def get_data_shape(path):  
    global data_height, data_width
    data = np.load(path)                    # shape: [time, mel_bins or mfcc_coeff]
    data_width, data_height = data.shape
    
   
# method to check GPU device avaible and setting
def GPU_check():
    print("-------------------- TENSORFLOW VERSION --------------------")
    print(tf.__version__)
    print("-------------------- AVAILABLE HW DEVICES --------------------")
    print("List of devices:")
    print(device_lib.list_local_devices())
    print("--------------------")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)    
    print("------------------------------------------------------------")
              
# ------------------------------------ end: utilities method ------------------------------------

# ------------------------------------ start: plot method ------------------------------------

def plot_accuracy_and_loss(train_hist, val_hist, test_hist):
    runs = range(1, len(train_hist['accuracy']) + 1)
    
    # -- plot accuracy --
    plt.figure(figsize=(10, 6))
    plt.plot(runs, train_hist['accuracy'], label='Train Accuracy')
    plt.plot(runs, val_hist['accuracy'], label='Validation Accuracy')
    plt.plot(runs, test_hist['accuracy'], label='Test Accuracy')
    
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per run')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # -- plot loss --
    plt.figure(figsize=(10, 6))
    plt.plot(runs, train_hist['loss'], label='Train Loss')
    plt.plot(runs, val_hist['loss'], label='Validation Loss')
    plt.plot(runs, test_hist['loss'], label='Test Loss')
    
    plt.xlabel('Run')
    plt.ylabel('Loss')
    plt.title('Loss per run')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_average_metrics(train_hist, val_hist, test_hist):
    def avg(lst): return round(np.mean(lst), 4)
    
    print("\nMean:")
    print(f"Train     → Accuracy: {avg(train_hist['accuracy'])}, Loss: {avg(train_hist['loss'])}")
    print(f"Validation→ Accuracy: {avg(val_hist['accuracy'])}, Loss: {avg(val_hist['loss'])}")
    print(f"Test      → Accuracy: {avg(test_hist['accuracy'])}, Loss: {avg(test_hist['loss'])}")



# method to plot accuracy and loss. arc is a dictionary with the results, 'mode' if is '0': there are fit results, if is '1': there are evaluation results
def plot_fit_result(arc,mode):
    result_dict = {}                    # dict that will contain the results to plot with the correct label/title
    # check what results there are
    if mode == 0:                       # method called with fit results
        result_dict["loss (training set)"] = arc["loss"]                    # take loss values (training set)
        result_dict["accuracy (taining set)"] = arc["accuracy"]             # take accuracy values (training set)
        if arc.get("val_loss") is not None:                         # check if there are result of validation set
            result_dict["loss (validation set)"] = arc["val_loss"]          # take loss values (validation set)
            result_dict["accuracy (validation set)"] = arc["val_accuracy"]  # take accuracy values (validation set)
    elif mode == 1:                     # method called with evaluate results
        result_dict["loss (test set)"] = arc["loss"]                        # take loss values (test set)
        result_dict["accuracy (test set)"] = arc["accuracy"]                # take accuracy values (test set)
    # plot the results
    for k,v in result_dict.items():
        plot(k,v)
        
# method to display a plot. 'title' is the tile of the plot, 'value_list' is a list of value to draw in the plot
def plot(title,value_list):
    fig = plt.figure()
    fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # force the label of  number of epochs to be integer
    plt.plot(value_list,'o-b')
    plt.title(str(title))               # plot title
    plt.xlabel("# Epochs")              # x axis title
    plt.ylabel("Value")                 # y axis title
    plt.show()
    
# method for create and plot the confusion metrix of the model trained
def confusion_matrix():
    global test_image, test_label, network, classes # global variables references
    data_test = []                                  # set of the images of the test set
    
    # create the confusion matrix, rows indicate the real class and columns indicate the predicted class 
    conf_matrix = np.zeros((len(classes),len(classes)))     # at begin values are 0
    
    # take all the images from the test set
    for img_path in test_image:
        img = cv2.imread(img_path)              # take current iamge
        if img is not None:                     # check image taken
        #check if the image is in the correct shape for the CNN (shape specified in the global variables)
            if img.shape != (img_width, img_height, img_channel):       
                dim = (img_height ,img_width)
                img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
                img = img.astype('float32') / 255                           # normalization
                data_test.append(img)                                       # add new element
            else:
                img = img.astype('float32') / 255                           # normalization
                data_test.append(img)                                       # add new element
        else:
            print("Image loading error in model_evaluate...",img_path)
                    
    # resize the set of the image
    data_test = np.array(data_test)
    data_test = data_test.reshape((len(data_test), img_width, img_height, img_channel))     
    
    predictions = network.predict(data_test)                # get the output for each sample of the test set
    # slide the prediction result and go to create the confusion matrix
    for i in range(len(data_test)):
        # test_label[i] indicate the real value of the label associated at the test_image[i] (or data_test[i]) -> is real class (row)
        # predictions[i] indicate the class value predicted by the model for the test_image[i] (or data_test[i]) -> is predicted class (column)
        # the values are in categorical format, translate in int
        conf_matrix[np.argmax(test_label[i])][np.argmax(predictions[i])] += 1                               # update value
        
    # do percentages of confusion matrix
    conf_matrix_perc = [[None for c in range(conf_matrix.shape[1])] for r in range(conf_matrix.shape[0])]   # define matrix
    
    for i in range(conf_matrix.shape[0]):                   # rows
        for j in range(conf_matrix.shape[1]):               # columns
            conf_matrix_perc[i][j] = " (" + str( round( (conf_matrix[i][j]/len(data_test))*100 ,2) ) + "%)" # calculate percentage value
    
    # plot the confusion matrix
    rows = classes                                          # contain the label of the classes showed in the rowvalues of rows          
    columns = classes                                       # contain the label of the classes showed in the rowvalues of columns   

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    
    for i in range(len(rows)):                              # rows
        for j in range(len(columns)):                       # columns
            # give the value in the confusion matrix
            ax.text(x=j, y=i, s=str(str(conf_matrix[i][j])+conf_matrix_perc[i][j]),
                           ha="center", va="center", size='x-large')
            
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Real', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()                                              # shows confusion matrix
    
# to calculate confusion matrix using dataset.from.generator 
def compute_confusion_matrix(model, dataset, steps, classes):
    # Previsione batch-wise usando il dataset
    predictions = model.predict(dataset, steps=steps)

    # Estrazione dei dati reali (labels) dal dataset (solo per i batch usati)
    true_labels = []
    for batch_idx, (_, y) in enumerate(dataset):
        true_labels.append(y.numpy())
        if batch_idx + 1 >= steps:
            break
    true_labels = np.concatenate(true_labels, axis=0)

    # Calcolo confusion matrix
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i in range(len(predictions)):
        true_idx = np.argmax(true_labels[i])
        pred_idx = np.argmax(predictions[i])
        conf_matrix[true_idx][pred_idx] += 1

    # Calcolo percentuali
    conf_matrix_perc = [[
        f" ({(conf_matrix[i][j] / np.sum(conf_matrix)) * 100:.2f}%)"
        for j in range(conf_matrix.shape[1])
    ] for i in range(conf_matrix.shape[0])]

    # Visualizzazione confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{conf_matrix[i][j]}{conf_matrix_perc[i][j]}",
                    va='center', ha='center', fontsize=12)

    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()
    

# ------------------------------------ end: plot method ------------------------------------

# ------------------------------------ start: methods for load DS ------------------------------------

def load_and_save_ds():
    global path_dir_ds, classes, total_audio_ds, total_labels, command_set, generic_class_name, copy_ds_name, path_dir_copy_ds
    
    do_preprocessing = True                     # 
    classes.append(str(generic_class_name))     # add the generic class containing all the classes that aren't target commands
    
    print("-------------------- READING DS --------------------")   # status print
    print("Read the DS: ",path_dir_ds, " and rewrite the preprocessed data in: ", path_dir_copy_ds) # status print
    
    # create the copy folder in the same folder of the original ds
    if os.path.exists(path_dir_copy_ds) and erase_copy_ds:       # check if exist
        shutil.rmtree(path_dir_copy_ds)             # erase all the data in the folder
        do_preprocessing = True
        print("Copy ds folder present, erase it.")
    elif os.path.exists(path_dir_copy_ds) and not erase_copy_ds:
        do_preprocessing = False
        print("Copy ds folder present, don't erase it and don't preprocessing data egain.")
    else:
        do_preprocessing = True
        print("Copy ds folder not present. Create it.")
        os.makedirs(path_dir_copy_ds)                   # create the folder

    if do_preprocessing:            # read from rewrited dataset
        entries = os.listdir(path_dir_ds)                   # Get the list of items in the parent folder
        subfolders = [f for f in entries if os.path.isdir(os.path.join(path_dir_ds, f))]    # Check if there are subfolders 
    else:                           # read from dataset, preprocess and rewrite in other folder
        entries = os.listdir(path_dir_copy_ds)                   # Get the list of items in the parent folder
        subfolders = [f for f in entries if os.path.isdir(os.path.join(path_dir_copy_ds, f))]    # Check if there are subfolders  
    
    if subfolders:          # there are sub-folders -> (nested dataset)
        for sub in entries:             # scroll through each subfolder
            # check if is a target command or not
            if sub in command_set:                  # is a target command, add the class 
                classes.append(str(sub))                # update classes   
                index = classes.index(str(sub))         # take index of classes, is the label of this class 
                label = sub                             # set label
            else:                                   # isn't a target command, put in generic class
                index = 0;                              # the generic class has counter in 0 position
                label = generic_class_name              # set label
            
            if do_preprocessing:                    #
                p_copy = os.path.join(path_dir_copy_ds, sub)    # path of each folder (for the DS copy folder)
                os.makedirs(p_copy, exist_ok=True)          # create sub-folder for the copy
                sub_path = os.path.join(path_dir_ds, sub)   # path of each folder (of the original folder)
            else:
                sub_path = os.path.join(path_dir_copy_ds, sub)  # path of each folder (of the copy folder)
            
            print("Read the sub-folder: ",sub, " , belong in the class: ",label)          # status print
            
            for filename in os.listdir(sub_path):
                
                if do_preprocessing and filename.endswith(".wav"):  # check if is a audio file .wav and read audio file, do preprocessing and write it
                    # load audio file
                    file_path = os.path.join(sub_path, filename)            # create the path of the current audio file
                    audio_bin = tf.io.read_file(file_path)                  # read file audio

                    waveform, sample_rate = tf.audio.decode_wav(audio_bin)  # read the binary file and decode it to WAV, waveform in [-1,1] and sample_rate of the file (frequency)
                    waveform = tf.squeeze(waveform, axis=-1)                # from (N,1) to (N,)
                        
                    # extract features from audio files
                    try:
                        data = extract_features(waveform, sample_rate)      # get features
                            
                        out_path = os.path.join(p_copy, filename.replace(".wav", ".npy"))   # create output path for the data extracted from current audio file
                        np.save(out_path, data)                             # save the data
                           
                        total_audio_ds.append(out_path)                     # add data path to the list

                    except Exception as e:
                        print("Errore con file:", file_path, e)
                        
                if not do_preprocessing and filename.endswith(".npy"):   # only read the preprocessed dataset
                        total_audio_ds.append(os.path.join(sub_path,filename))  # add data path to the list
                        
                total_labels.append(index)          # add related label index to the list
    
    # convert in numpy array
    total_audio_ds = np.array(total_audio_ds)
    total_labels = np.array(total_labels)
    
    print("\n---- dataset reading and rewriting completed ----")    # control data print
    print("\n-- Classes statistics --")
    print("Num of classes: ",len(classes))
    for i in range(len(classes)):
        print("class name: ",classes[i])
        
    print("\n-- Space statistics --")
    print("total_audio_ds",len(total_audio_ds), total_audio_ds.shape, ", required memory: ", f"{(total_audio_ds.size * total_audio_ds.itemsize / 10**6):.2f}", " MB")
    print("total_labels",len(total_labels), total_labels.shape, ", required memory: ", f"{(total_labels.size * total_labels.itemsize / 10**6):.2f}", " MB")
    
# ------------------------------------ end: methods for load DS ------------------------------------

# ------------------------------------ start: methods for extract features ------------------------------------

# method for feature extraction. Extract and return the Mel-Spectrogram or MFCC based on the value of 'do_mfcc'
def extract_features(waveform, sample_rate):
    global desired_sr, num_mel_bins, num_mfccs, mel_f_min, mel_f_max, do_mfcc

    if sample_rate != desired_sr:       # check frequency 
        waveform = tfio.audio.resample(waveform, rate_in=tf.cast(sample_rate, tf.int64), rate_out=desired_sr)   # Resample a 16kHz mono
        
    desired_len = desired_sr                # 1 s = desired_sr -> num_samples = frequency -> in this case 16000 samples
    curr_len = tf.shape(waveform)[0]        # 

    waveform = waveform[:desired_len]               # If too long → Truncate to 1 second (desired_sr samples)
    pad_len = desired_len - tf.shape(waveform)[0]   # If too short → pad with zeros up to 1 second (desired_sr samples)
    waveform = tf.pad(waveform, [[0, pad_len]])

    # Normalization [-1,1] with minimal perturbation to avoid dividing by 0 in case of empty signal
    waveform = waveform / (tf.reduce_max(tf.abs(waveform)) + 1e-9)

    # -- STFT --
    stft = tf.signal.stft(waveform, frame_length=256, frame_step=128)   # perform STFT
    spectrogram = tf.abs(stft)                                          # [num_frames, num_freq_bins]
    
    # -- Mel-Spectrogram --
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], desired_sr, mel_f_min, mel_f_max
    )
    mel_spectrogram = tf.tensordot(tf.square(spectrogram), linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    
    if do_mfcc:             # do and return MFCC
        # -- MFCC -- 
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel_spectrogram + 1e-6))
        mfccs = mfccs[..., :num_mfccs]
        return mfccs.numpy()                # return MFCC
        
    else:
        return mel_spectrogram.numpy()      # return MEL
    
# ------------------------------------ end: methods for extract features ------------------------------------

# ------------------------------------ start: methods for data augmentation ------------------------------------

# (a) Rumore di fondo
def add_noise(wave, noise_level=0.005):
    noise = tf.random.normal(shape=tf.shape(wave), mean=0.0, stddev=noise_level)
    return wave + noise

#wave_with_noise = add_noise(waveform)

# (b) Time-shifting -> Time-shift: sposta la parola dentro la finestra da 1s (±100 ms qui). Aumenta la robustezza alla posizione della keyword.
def time_shift(wave, shift_max=0.1):
    shift = int(shift_max * desired_sr)  # massimo 0.1s
    shift_amt = tf.random.uniform([], -shift, shift, dtype=tf.int32)
    return tf.roll(wave, shift_amt, axis=0)

#wave_shifted = time_shift(waveform)

# (c) Pitch shifting (con tfio)
def pitch_shift(wave, n_steps=2):
    return tfio.audio.pitch_shift(wave, rate=desired_sr, n_steps=n_steps)
    
#wave_pitched = pitch_shift(waveform, n_steps=2)
"""
Pitch shift: cambia l’altezza (tono) senza alterare troppo la durata (dipende dall’algoritmo).

n_steps=2 → due semitoni ↑. Puoi randomizzare tra [-2, +2] o simili.

Richiede tensorflow-io (pip install tensorflow-io).
Pitch shifting (cambia tono della voce).
"""

# ------------------------------------ end: methods for data augmentation ------------------------------------

# ------------------------------------ start: methods to manage set ------------------------------------

# method for preprocessing and split the dataset
def make_set_ds():
    global total_audio_ds, total_labels, train_data, train_label, val_data, val_label, test_data, test_label
    
    seed = random.randint(1, 42)            # calculate the seed
    
    # ----- preprocessing and reshape ----
    data_ds = np.array(total_audio_ds)
    labels_ds = to_categorical(total_labels,num_classes=len(classes))    # transform label in categorical format
    
    # ---- generete the training and test set ----
    train_data_temp, test_data, train_label_temp, test_label = train_test_split(data_ds, labels_ds, test_size=test_set_split , random_state=seed, shuffle=True)     # split to generate train and test set
    train_data, val_data, train_label, val_label = train_test_split(train_data_temp, train_label_temp, test_size=test_set_split , random_state=seed, shuffle=True)  # split to generate validation set from train set
    
    # information print
    print("-------------------- SETS --------------------")
    print("Make this sets:")
    print("train_data len:",len(train_data), train_data.shape)
    print("train_label len",len(train_label), train_label.shape)
    print("val_data len:",len(val_data), val_data.shape)
    print("val_label len",len(val_label), val_label.shape)
    print("test_data len:",len(test_data), test_data.shape)
    print("test_label len",len(test_label), test_label.shape)
    print("------------------------------------------------------------")


# ------------------ start: generetor function ------------------
# explanation: for large dataset with large spectrum data of audio file (like 2D images) or big batch size the memory memory may not be sufficient. 
#              To avoid memory overflow, the sets are supplied in batches via yeld istruction.
# define generator function to do the training set
def generator_train():
    # create the tensor that will contain the data
    data_tensor = []                                        # tensor that contain the images of one batch from the set
    label_tensor = []                                       # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                    # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                  # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                    # check if it has to truncate or not the set
        rest = batch_size - (len(train_data) % batch_size)  # check if the division by batch_size produce rest
    else:
        rest = batch_size                                   # set always truncated
    
    for idx in range(len(train_data)):                      # organize the sample in batch
        
        img_path = train_data[idx]                          # take data             
        label = train_label[idx]                            # take label
        data = np.load(img_path)                            # load data shape: [time, mel_bins or mfcc_coeff]
        data = np.expand_dims(data, axis=-1)                # add channel dimension -> [time, mel_bins, 1]
        
        # add new element and convert to TF tensors
        data_tensor.append(tf.convert_to_tensor(data, dtype=tf.float32))  # add new element and convert to TF tensors
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:                   #check for the rest
            # add this sample for the future (sample in the rest)
            img_rest_tensor.append(tf.convert_to_tensor(data, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(data_tensor) == batch_size:                       # check to see if batch is full (reached batch_size)
            yield tf.stack(data_tensor), tf.stack(label_tensor)                      # return the batch
            # clean list
            data_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(train_data) - 1):                       # check if the set is finished, last batch
            if rest != batch_size:                              # check if there are rest to fix
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    data_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])

                yield tf.stack(data_tensor), tf.stack(label_tensor)                  # return the last batch

# define generator function to do the validation set
def generator_val():
    # create the tensor that will contain the data
    data_tensor = []                                        # tensor that contain the images of one batch from the set
    label_tensor = []                                       # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                    # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                  # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                    # check if it has to truncate or not the set
        rest = batch_size - (len(val_data) % batch_size)    # check if the division by batch_size produce rest
    else:
        rest = batch_size                                   # set always truncated
    
    for idx in range(len(val_data)):                        # organize the sample in batch
        
        img_path = train_data[idx]                          # take data             
        label = train_label[idx]                            # take label
        data = np.load(img_path)                            # load data shape: [time, mel_bins or mfcc_coeff]
        data = np.expand_dims(data, axis=-1)                # add channel dimension -> [time, mel_bins, 1]
        
        # add new element and convert to TF tensors
        data_tensor.append(tf.convert_to_tensor(data, dtype=tf.float32))  # add new element and convert to TF tensors
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:                   #check for the rest
            # add this sample for the future
            img_rest_tensor.append(tf.convert_to_tensor(data, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(data_tensor) == batch_size:                       # check to see if batch is full (reached batch_size)
            yield tf.stack(data_tensor), tf.stack(label_tensor)                      # return the batch
            # clean list
            data_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(val_data) - 1):                           # check if the set is finished, last batch
            if rest != batch_size:                              # check if there are rest to fix
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    data_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])
                yield tf.stack(data_tensor), tf.stack(label_tensor)                  # return the last batch
        
# define generator function to do the test set
def generator_test():
    # create the tensor that will contain the data
    data_tensor = []                                        # tensor that contain the images of one batch from the set
    label_tensor = []                                       # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                    # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                  # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                    # check if it has to truncate or not the set
        rest = batch_size - (len(test_data) % batch_size)   # check if the division by batch_size produce rest
    else:
        rest = batch_size                                   # set always truncated
    
    for idx in range(len(test_data)):                       # organize the sample in batch
        
        img_path = train_data[idx]                          # take data             
        label = train_label[idx]                            # take label
        data = np.load(img_path)                            # load data shape: [time, mel_bins or mfcc_coeff]
        data = np.expand_dims(data, axis=-1)                # add channel dimension -> [time, mel_bins, 1]
        
        # add new element and convert to TF tensors
        data_tensor.append(tf.convert_to_tensor(data, dtype=tf.float32))  # add new element and convert to TF tensors
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:               #check for the rest
            # add this sample for the future
            img_rest_tensor.append(tf.convert_to_tensor(data, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(data_tensor) == batch_size:                  # check to see if batch is full (reached batch_size)
            yield tf.stack(data_tensor), tf.stack(label_tensor)                 # return the batch
            # clean list
            data_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(test_data) - 1):                     # check if the set is finished, last batch
            if rest != batch_size:                          # check if there are rest to fix
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    data_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])
                yield tf.stack(data_tensor), tf.stack(label_tensor)                  # return the last batch
                
    gpu_memory_usage_threshold(threshold_VRAM)

# ------------------ end: generetor function ------------------

# ------------------------------------ end: methods to manage set ------------------------------------

# method to create and fit model
def make_fit_model():
    global chosen_model, batch_size, num_test, epochs, early_patience, total_audio_ds, network
    
    get_data_shape(total_audio_ds[0])                   # set the correct input shape for CNN

    # Dictionaries for collecting metrics
    history_train = {'accuracy': [], 'loss': []}        # to contain value for the training set
    history_val   = {'accuracy': [], 'loss': []}        # to contain value for the validation set
    history_test  = {'accuracy': [], 'loss': []}        # to contain value for the test set
                                      # at least one parameter isn't correct
    print("------------------------ MAKE MODEL ------------------------")
    start_all_time = time.time()                        # start time for all training
    for i in range(num_test):
        print("-------- start: test num ",i," --------")
        
        make_set_ds()                                   # split and create the sets
        
        # ---- make the model -----
        # chosen_model legend ->
        if chosen_model >= 0:
            siren_model = SirenNet.SirenNet(len(classes),data_width,data_height,data_channel)   # create an instance of the SirenNet class
            siren_model.make_model(chosen_model)        # make model
            siren_model.compile_model()                 # compile model
            network = siren_model.return_model()        # return model
        else:
            siren_model = SirenNet.SirenNet(len(classes),data_width,data_height,data_channel)   # create an instance of the SirenNet class
            siren_model.make_model(0)                   # make model
            siren_model.compile_model()                 # compile model
            network = siren_model.return_model()        # return model       
            
        # create TRAIN SET using generator function and specifying shapes and dtypes
        train_set = tf.data.Dataset.from_generator(generator_train, 
                                                 output_signature=(tf.TensorSpec(shape=(batch_size ,data_width , data_height , data_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32))).repeat()
        
        # create VALIDATION SET using generator function and specifying shapes and dtypes
        val_set = tf.data.Dataset.from_generator(generator_val, 
                                                 output_signature=(tf.TensorSpec(shape=(batch_size ,data_width , data_height , data_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32))).repeat()
        # create TEST SET using generator function and specifying shapes and dtypes
        test_set = tf.data.Dataset.from_generator(generator_test, 
                                                 output_signature=(tf.TensorSpec(shape=(batch_size ,data_width , data_height , data_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32))).repeat()
                
        # ---- fit the model -----
        print("------------------------ fit model ------------------------")
        
        checkpoint = ModelCheckpoint(filepath = path_check_point_model+'/weight_seg_SirenNet_'+str(chosen_model)+".keras", verbose = 1, save_best_only = True, monitor='val_loss', mode='min') # val_loss, min, val_categorical_accuracy, max
        eStop = EarlyStopping(patience = early_patience, verbose = 1, restore_best_weights = True, monitor='val_loss')
        
        # -- calculate steps for the sets --
        # train steps
        if (len(train_data) % batch_size) == 0 or truncate_set:        # check if the division by batch_size produce rest
            train_step = int(math.floor(len(train_data) / batch_size))
        else:
            train_step = int(math.floor((len(train_data) / batch_size)) + 1)
        
        # val steps
        if (len(val_data) % batch_size) == 0 or truncate_set:            # check if the division by batch_size produce rest
            val_step = int(math.floor(len(val_data) / batch_size))
        else:
            val_step = int(math.floor((len(val_data) / batch_size)) + 1)
            
        # test steps
        if (len(test_data) % batch_size) == 0 or truncate_set:         # check if the division by batch_size produce rest
            test_step = int(math.floor(len(test_data) / batch_size))
        else:
            test_step = int(math.floor((len(test_data) / batch_size)) + 1)

        start_time = time.time()                            # start time for training
        history = network.fit(train_set,validation_data=val_set,steps_per_epoch=train_step,validation_steps=val_step, epochs=epochs, callbacks = [checkpoint, eStop])     # fit model
        end_time = time.time()                              # end time for training
        print(f"Time for training the model: ", format_time(end_time - start_time)," - of the test number: ",i)  # print time to train the model
        
        if num_test == 1:           # case only 1 fit
            plot_fit_result(history.history,0)                  # visualize the value for the fit - history.history is a dictionary - call method for plot train result
        else:                       # case more fit, statistical pourpose
            history_train['accuracy'].append(history.history['accuracy'][-1])
            history_train['loss'].append(history.history['loss'][-1])
            history_val['accuracy'].append(history.history['val_accuracy'][-1])
            history_val['loss'].append(history.history['val_loss'][-1])
            
        # -- evaluate on test set and make plots --
        test_loss, test_acc = network.evaluate(test_set, steps=test_step)       # obtain loss and accuracy metrics
        if num_test == 1:             # case only 1 fit
            dict_metrics = {'loss': test_loss, 'accuracy': test_acc}            # create a dictionary contain the metrics
            plot_fit_result(dict_metrics,1)                                     # plot the values obtained
            compute_confusion_matrix(network, test_set, test_step, classes)     # call method to obtain the confusion matrix    
        else:                       # case more fit, statistical pourpose
            history_test['loss'].append(test_loss)
            history_test['accuracy'].append(test_acc)
        
        print("-------- end: test num ",i," --------")

    end_all_time = time.time()                              # end time for all training
    
    # plot and calculate the mean in case of number of fit test done
    if n_test > 1:
        plot_accuracy_and_loss(history_train, history_val, history_test)
        print_average_metrics(history_train, history_val, history_test)
    
    print(f"Time for training all the tests: ", format_time(end_all_time - start_all_time))
    print("------------------------------------------------")

# ------------------------------------ main ------------------------------------        
if __name__ == "__main__":
    GPU_check()             # set GPU
    load_and_save_ds()      # loand ds, extract features and write it on disk
    make_fit_model()        # train
    
"""
IMPORTANTE
    guardare altre note del file ... o readme che vengono spiegate un po' di cose utilities
    
NOTE 0:
    
        
    Esatto, hai centrato il punto 
    MEL vs MFCC per CNN
    Mel-spectrogram è di solito preferito per le CNN: mantiene una rappresentazione “immagine-like” (tempo × frequenze), con valori continui positivi → si comporta come una “immagine in scala di grigi”.
    MFCC è più compatto e astratto (meno ridondanza, più robusto al rumore), ma perdi un po’ di struttura temporale-frequenziale.
    Se usi una CNN pura, Mel-spectrogram è lo standard; se usi modelli più classici (es. GMM, SVM), allora MFCC.
    VRAM & dataset strategy
    Giustissimo: non puoi tenere in GPU RAM tutte le features.
    Strategia ottimale:
    Precalcoli Mel o MFCC una sola volta.
    Le salvi su disco come immagini 2D (numpy array).
    Durante training, carichi i .npy o .npz come fai con le immagini.
    Così non ricalcoli mai più le features → guadagni velocità enorme.
"""

