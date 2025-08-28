# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

Description: program that, given an audio file, divides it into windows and then implements the model to recognize the commands contained within it.
"""
import os
import time
import random
import math
import numpy as np
import pandas as pd
import tensorflow_io as tfio  
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib 
import tensorflow as tf
from tensorflow import keras
# for plot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ------------------------------------ start: global var ------------------------------------
# -- path --
dir_father = ".."               # fathere folder containing all other folders used
dir_ds_name = "dataset"         # folder containing all the dataset to load for the program
ds_name = "speech_ds"           # folder name for the target dataset
path_dir_ds = os.path.join(dir_father,dir_ds_name,ds_name)# folder containing the dataset to load (nested dataset)

ds_audio_name = "long_audio"    # name of the folder containing the dataset of the long audio to classify (find command)
path_dir_ds_audio = path_dir_ds = os.path.join(dir_father,dir_ds_name,ds_audio_name)# folder containing the dataset of the long audio file

dir_model_name = "model"            # folder in which there are saved the CNN model
trained_dir_model_name = "trained"  # folder in which there are saved the trained CNN model
path_dir_model = os.path.join(dir_father,dir_model_name,trained_dir_model_name)  # folder in which there are saved the model
model_name = "SirenNet_1.keras"                         # nam of the model to use
model_path = os.path.join(path_dir_model,model_name)    # path of the trained model to use

# -- audio file var --
desired_sr = 16000              # want audio as 16kHz mono
window_size = 1                 # indicates the length window (chunks) for the audio file (in s)
step_size = 0.5                 # indicates the step for next window for the audio file (in s)
truncate = False                # If the audio file is not perfectly divisible by the interval in seconds of the chunks, 
                                # indicate whether to truncate the last chunk or to do the padding.
cooldown = 2                    # number of windows to skip after detecting a command
                                
# -- classification var --
confidence_threshold = 0.9      # indicates the minimum confidence threshold before detecting the command

network = None                  # contain the CNN model, default value is None

# -- command set var --
command_set = ['forward', 'backward', 'stop','go','up','down','left','right']   # array containing all the target commands
generic_class_name = "unknown"  # name for the generic class containing all the word that aren't command for this project

# -- dataset variables  --
classes = []                    # the label associated with each class will be the position that the class name will have in this array
long_audio_files = []           # list containing all the names of long files (the ones from which to extract the commands)

do_mfcc = False                 # if 'true' -> calculate and use MFCC , if 'false' -> donìt calculate MFCC and use MEL

# ------------------------------------ end: global var ------------------------------------

# ------------------------------------ start: utilities method ------------------------------------
   
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
    
# method to load the trained model
def load_model():
    global network
    
    try:
        network = tf.keras.models.load_model(model_path)
        print("Model ",str(model_path)," loaded successfully")
    except (OSError, IOError) as e:
        network = None
        print(f"Errore nel caricamento del modello: {e}")
        sys.exit(1)                     # exit to prorgam
        
# ------------------------------------ end: utilities method ------------------------------------
    
# ------------------------------------ start: dataset method ------------------------------------
    
# method to understand the correct index for each class. It use the dataset and the classes used for the training
def load_class_index():
    global classes, labels_index, generic_class_name
    
    classes.append(str(generic_class_name))     # add the generic class containing all the classes that aren't target commands
    
    print("-------------------- READING DS --------------------")   # status print
    print("Read the DS: ",path_dir_ds) # status print
    
    entries = os.listdir(path_dir_ds)                   # Get the list of items in the parent folder
    subfolders = [f for f in entries if os.path.isdir(os.path.join(path_dir_ds, f))]    # Check if there are subfolders 
 
    if subfolders:          # there are sub-folders -> (nested dataset)
        for sub in entries:             # scroll through each subfolder
            
            if sub in command_set:                      # is a target command, add the class 
                classes.append(str(sub))                # update classes   
     
    print("\n---- dataset reading completed ----")    # control data print
    print("\n-- Classes statistics --")
    print("Num of classes: ",len(classes))
    for i in range(len(classes)):
        print("class name: ",classes[i])
              
# method to read all the long audio file
def read_audio_files():
    global long_audio_files
    
    for filename in os.listdir(path_dir_ds_audio):
        if filename.endswith(".wav"):
            long_audio_files.append(filename)

# takes a random audio file from the list of long audio files
def get_audio_random():
    
    if not long_audio_files:
        return None  
    index = random.randint(0, len(long_audio_files) - 1)
    return long_audio_files[index]
# ------------------------------------ end: dataset method ------------------------------------

# ------------------------------------ start: plot method ------------------------------------

# method to plot the prediction for each window
def plot_probabilities(times, probs_all, classes):
    plt.figure(figsize=(12, 6))
    
    for ci, cls in enumerate(classes):
        plt.plot(times, probs_all[:, ci], label=cls, linewidth=1.5)
    
    plt.xlabel("Tempo (s)")
    plt.ylabel("Probabilità")
    plt.title("Probabilità predetta per finestra")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
   
# method to do the heatmap of the prediction for each window
def plot_heatmap(times, probs_all, classes):
    plt.figure(figsize=(12, 6))

    # probs_all: shape [num_frames, num_classes]
    # Trasponiamo per avere classi sull'asse Y
    plt.imshow(
        probs_all.T, 
        aspect='auto', 
        origin='lower',
        extent=[times[0], times[-1], 0, len(classes)],
        cmap="viridis"
    )

    plt.colorbar(label="Probabilità")
    plt.yticks(np.arange(len(classes)) + 0.5, classes)  # centro delle celle
    plt.xlabel("Tempo (s)")
    plt.ylabel("Classi")
    plt.title("Heatmap probabilità nel tempo")
    plt.tight_layout()
    plt.show()


# ------------------------------------ end: plot method ------------------------------------

# ------------------------------------ start: methods for extract features ------------------------------------

# method for feature extraction. Extract and return the Mel-Spectrogram or MFCC based on the value of 'do_mfcc'
def extract_features(waveform, sample_rate):
    global desired_sr, num_mel_bins, num_mfccs, mel_f_min, mel_f_max, do_mfcc

    if sample_rate != desired_sr:       # check frequency 
        waveform = tfio.audio.resample(waveform, rate_in=tf.cast(sample_rate, tf.int64), rate_out=desired_sr)   # Resample a 16kHz mono
        
    desired_len = desire_audio_len * desired_sr     # 1 s = desired_sr -> num_samples = frequency -> in this case 16000 samples
    curr_len = tf.shape(waveform)[0]                # get the lenght of the audio (= sr * seconds)

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

# method to detect commands within the audio
def find_command_from_audio():
    # sliding window parameters
    win_size = desired_sr * window_size     # default = 1 s (same length of the audio in the dataset)
    hop_size = desired_sr // step_size      # dafault = 0.5 s (half of the audio lenght)
    
    # var to plot
    probs_all = []
    times = []
    
    chosen_audio = get_audio_random()       # get a random long audio
    waveform, sr = sf.read(chosen_audio)    # load audio file
    
    if waveform.ndim > 1:                   # stereo -> mono
        waveform = np.mean(waveform, axis=1)
    
    if sr != desired_sr:                    # resampling
        waveform = tfio.audio.resample(waveform, rate_in=sr, rate_out=desired_sr).numpy()

    if len(waveform) < win_size:
        if truncate:
            return  # troppo corto, lo scarto
        else:
            waveform = np.pad(waveform, (0, win_size - len(waveform)))
            num_frames = 1
    else:
        if not truncate:
            pad_len = ((len(waveform) - win_size) % hop_size)
            if pad_len > 0:
                waveform = np.pad(waveform, (0, pad_len))
        num_frames = 1 + (len(waveform) - win_size) // hop_size
        
    print(f"Audio input: {len(waveform)/desired_sr:.2f} s, finestre = {num_frames}")
    
    skip = 0
    # scann al the windows
    for i in range(num_frames):
        if skip > 0:    # che if skip the current window (after a command)
            skip -= 1
            continue
        
        start = i * hop_size        # calculate the start of the current window
        end = start + win_size      # calculate the end of the current window
        chunk = waveform[start:end] # create the chunk (window)
        center_time = (start + end) / 2 / desired_sr    # time to plot the prediction

        # Estrai feature
        features = extract_features(chunk, desired_sr)  # extract features
        features = np.expand_dims(features, axis=0)     # [1, width, height, channel]

        # Predizione
        probs = network.predict(features, verbose=0)[0]
        pred_idx = np.argmax(probs)         # predict class (index)
        conf = probs[pred_idx]              # confidence of the predict class (index)
        
        probs_all.append(probs)
        times.append(center_time)

        if conf > confidence_threshold and classes[pred_idx] != "unknown":
            start_ms = int(start / desired_sr * 1000)
            end_ms = int(end / desired_sr * 1000)
            print(f"[{start_ms}-{end_ms} ms] -> {classes[pred_idx]} ({conf:.2f})")
            
            skip = cooldown                 # avoid to recognize again the same command
            
    probs_all = np.array(probs_all)     # convert in numpy array
    times = np.array(times)             # convert in numpy array

    # plots
    plot_probabilities(times, probs_all, classes)   # plot results
    plot_heatmap(times, probs_all, classes)         # plot heat map

# ------------------------------------ main ------------------------------------        
if __name__ == "__main__":
    GPU_check()             # set GPU
    load_model()            # load the trained model
    load_class_index()      # load the correct indeces for the classes
    read_audio_files()
    
    