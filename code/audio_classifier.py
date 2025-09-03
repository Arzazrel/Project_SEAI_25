# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

Description: program that, given an audio file, divides it into windows and then implements the model to recognize the commands contained within it.
"""
import os
import time
import random
import math
import soundfile as sf
import numpy as np
import pandas as pd
import tensorflow_io as tfio  
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib 
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
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
path_dir_ds_audio = os.path.join(dir_father,dir_ds_name,ds_audio_name)# folder containing the dataset of the long audio file

dir_model_name = "model"            # folder in which there are saved the CNN model
trained_dir_model_name = "trained"  # folder in which there are saved the trained CNN model
path_dir_model = os.path.join(dir_father,dir_model_name,trained_dir_model_name)  # folder in which there are saved the model
model_name = "SirenNet_1.keras"                         # nam of the model to use
model_path = os.path.join(path_dir_model,model_name)    # path of the trained model to use

dir_results_name = "results"    # name of the folder containing the results for this project
logfile_name = "audio_classifier_log.txt"               # name of the log file
path_logfile = os.path.join(dir_father,dir_results_name,logfile_name)   # path for the log file

# -- audio file var --
desired_sr = 16000              # want audio as 16kHz mono
num_mel_bins = 64               # num of bins for Mel-spectrogram
num_mfccs = 13                  # num of coefficient to take
mel_f_min = 80.0                # min frequency for the bins of MEL 
mel_f_max = 7600.0              # max frequency for the bins of MEL (or desired_sr/2)
window_size = 1                 # indicates the length window (chunks) for the audio file (in s)
step_size = 0.5                 # indicates the step for next window for the audio file (in s)
desire_audio_len = 1            # indicates the desired length of the audio in the database
truncate = False                # If the audio file is not perfectly divisible by the interval in seconds of the chunks, 
                                # indicate whether to truncate the last chunk or to do the padding.
cooldown = 1                    # number of windows to skip after detecting a command
basic_cooldown = False          # if 'true' -> it indicates to use the basic cooldown method, which prevents any commands from being recognized after one has been recognized.
                                # if 'false' -> it indicates to use the more advanced cooldown method, which, after a command has been recognized, blocks recognition only for the newly recognized class and not for the others.
                                
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
enable_log = True               # if 'true' -> writes statistics of the audio classification in the log , if 'false' -> doesn't write the explanation

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
        network = tf.keras.models.load_model(model_path)        # load trained model
        print("Model ",str(model_path)," loaded successfully")
    except (OSError, IOError) as e:
        network = None
        print(f"Errore nel caricamento del modello: {e}")
        sys.exit(1)                                             # exit to prorgam
        
# method to get the time in milliseconds
def now_ms():
    return time.perf_counter() * 1000.0     # milliseconds
    
# method that writes the results of the audio_classifier to the log file.
def log_to_file(chosen_audio, audio_len, num_frames, t_segm, t_feat, t_inf, t_post, enable_log=True):

    if not enable_log:      # control check
        return
        
    text_to_write = ""      # set the log text to write in the log file
    
    current_date = datetime.now()   # get current date and time
    format_date = current_date.strftime("%d/%m/%Y %H:%M:%S")
    
    text_to_write += "---- KWS on " + str(chosen_audio) + " , duration: " + str(audio_len) + " , windows: " + str(num_frames) + " in " + format_date + "---- \n"
    
    dict_segm = compute_stats(t_segm)   # get dictionary with statistics related segmentation time
    text_to_write += "Segmentation stats:\n"
    text_to_write += " - Mean (ms): " + str(dict_segm["mean_ms"]) + "\n"
    text_to_write += " - Min (ms): " + str(dict_segm["min_ms"]) + "\n"
    text_to_write += " - Max (ms): " + str(dict_segm["max_ms"]) + "\n"
    text_to_write += " - Percentile 50 (ms): " + str(dict_segm["p50_ms"]) + "\n"
    text_to_write += " - Percentile 95 (ms): " + str(dict_segm["p95_ms"]) + "\n"
    text_to_write += " - Percentile 99 (ms): " + str(dict_segm["p99_ms"]) + "\n"
    
    dict_feat = compute_stats(t_feat)   # get dictionary with statistics related features extraction time
    text_to_write += "Features extraction stats:\n"
    text_to_write += " - Mean (ms): " + str(dict_feat["mean_ms"]) + "\n"
    text_to_write += " - Min (ms): " + str(dict_feat["min_ms"]) + "\n"
    text_to_write += " - Max (ms): " + str(dict_feat["max_ms"]) + "\n"
    text_to_write += " - Percentile 50 (ms): " + str(dict_feat["p50_ms"]) + "\n"
    text_to_write += " - Percentile 95 (ms): " + str(dict_feat["p95_ms"]) + "\n"
    text_to_write += " - Percentile 99 (ms): " + str(dict_feat["p99_ms"]) + "\n"
            
    dict_inf = compute_stats(t_inf)     # get dictionary with statistics related features extraction time
    text_to_write += "Inference stats:\n"
    text_to_write += " - Mean (ms): " + str(dict_inf["mean_ms"]) + "\n"
    text_to_write += " - Min (ms): " + str(dict_inf["min_ms"]) + "\n"
    text_to_write += " - Max (ms): " + str(dict_inf["max_ms"]) + "\n"
    text_to_write += " - Percentile 50 (ms): " + str(dict_inf["p50_ms"]) + "\n"
    text_to_write += " - Percentile 95 (ms): " + str(dict_inf["p95_ms"]) + "\n"
    text_to_write += " - Percentile 99 (ms): " + str(dict_inf["p99_ms"]) + "\n"
    
    dict_post = compute_stats(t_post)   # get dictionary with statistics related post analysis and command recognition time
    text_to_write += "Reecognition stats:\n"
    text_to_write += " - Mean (ms): " + str(dict_post["mean_ms"]) + "\n"
    text_to_write += " - Min (ms): " + str(dict_post["min_ms"]) + "\n"
    text_to_write += " - Max (ms): " + str(dict_post["max_ms"]) + "\n"
    text_to_write += " - Percentile 50 (ms): " + str(dict_post["p50_ms"]) + "\n"
    text_to_write += " - Percentile 95 (ms): " + str(dict_post["p95_ms"]) + "\n"
    text_to_write += " - Percentile 99 (ms): " + str(dict_post["p99_ms"]) + "\n"
            
    # general stats
    text_to_write += "General stats:\n"
    text_to_write += " - Total segmentation time (ms): " + str(sum(t_segm)) + "\n"
    text_to_write += " - Total features extraction time (ms): " + str(sum(t_feat)) + "\n"
    text_to_write += " - Total inference time (ms): " + str(sum(t_inf)) + "\n"
    text_to_write += " - Total post time (ms): " + str(sum(t_post)) + "\n"
    text_to_write += "Window stats:\n"
    text_to_write += " - Mean time (ms): " + str(dict_segm["mean_ms"] + dict_feat["mean_ms"] + dict_inf["mean_ms"] + dict_post["mean_ms"]) + "\n"
    text_to_write += " - Min time (ms): " + str(dict_segm["min_ms"] + dict_feat["min_ms"] + dict_inf["min_ms"] + dict_post["min_ms"]) + "\n"
    text_to_write += " - Max time (ms): " + str(dict_segm["max_ms"] + dict_feat["max_ms"] + dict_inf["max_ms"] + dict_post["max_ms"]) + "\n"
    text_to_write += " -------------------------------------- \n"
    
    # write on file in append mode
    with open(path_logfile, "a", encoding="utf-8") as f:
        f.write(text_to_write)
        
# function that, given a list containing the various times for each window of the various metrics, calculates and returns the aggregated statistics
def compute_stats(arr):
    a = np.array(arr)
    return {
            "count": a.size,
            "mean_ms": float(a.mean()),
            "min_ms": float(a.min()),
            "max_ms": float(a.max()),
            "p50_ms": float(np.percentile(a,50)),
            "p95_ms": float(np.percentile(a,95)),
            "p99_ms": float(np.percentile(a,99))
        }
    
# ------------------------------------ end: utilities method ------------------------------------
    
# ------------------------------------ start: dataset method ------------------------------------
    
# method to understand the correct index for each class. It use the dataset and the classes used for the training. the correct classes indeces depends on the sort of the classes in the dataset.
def load_class_index():
    global classes, labels_index, generic_class_name
    
    classes.append(str(generic_class_name))     # add the generic class containing all the classes that aren't target commands
    
    print("-------------------- READING DS --------------------")   # status print
    print("Read the DS: ",path_dir_ds) # status print
    
    entries = os.listdir(path_dir_ds)               # Get the list of items in the parent folder
    subfolders = [f for f in entries if os.path.isdir(os.path.join(path_dir_ds, f))]    # Check if there are subfolders 
 
    if subfolders:          # there are sub-folders -> (nested dataset)
        for sub in entries:             # scroll through each subfolder
            
            if sub in command_set:                  # is a target command, add the class 
                classes.append(str(sub))            # update classes   
     
    print("\n---- dataset reading completed ----")  # control data print
    print("\n-- Classes statistics --")
    print("Num of classes: ",len(classes))
    for i in range(len(classes)):
        print("class name: ",classes[i])
    print("----------------------------------------")
              
# method to read all the long audio file
def read_audio_files():
    global long_audio_files
    
    for filename in os.listdir(path_dir_ds_audio):
        if filename.endswith(".wav"):           # check if is an audio file
            long_audio_files.append(filename)   # load the current audio file

# takes a random audio file from the list of long audio files
def get_audio_random():
    
    if not long_audio_files:        # check loaded audio files
        return None  
    index = random.randint(0, len(long_audio_files) - 1)            # calculate random index
    return os.path.join(path_dir_ds_audio, long_audio_files[index]) # get the random audio file
    
# get all the loaded audio file
def get_audio_list():
    path_list = []                  # list for the all loaded audio file
    
    if not long_audio_files:        # check loaded audio files
        return None  

    for i, text in enumerate(long_audio_files):
        path_list.append(os.path.join(path_dir_ds_audio, long_audio_files[i]))  # calculate and append all the path
        
    return path_list
    
# ------------------------------------ end: dataset method ------------------------------------

# ------------------------------------ start: plot method ------------------------------------

# method to plot the prediction for each window
def plot_probabilities(times, probs_all, classes):
    plt.figure(figsize=(12, 6))
    
    for ci, cls in enumerate(classes):
        plt.plot(times, probs_all[:, ci], label=cls, linewidth=1.5)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title("Predicted probability per window")
    plt.ylim(-0.1, 1.1)                             # a slightly larger range of probabilities to better see the lines at the maximum and minimum point
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
   
# method to do the heatmap of the prediction for each window
def plot_heatmap(times, probs_all, classes):
    plt.figure(figsize=(12, 6))

    # probs_all: shape [num_frames, num_classes], transpose to have classes on the Y-axis
    plt.imshow(
        probs_all.T, 
        aspect='auto', 
        origin='lower',
        extent=[0, times[-1] + (times[1] - times[0]), 0, len(classes)],
        cmap="viridis"
    )

    plt.colorbar(label="Probability")
    plt.yticks(np.arange(len(classes)) + 0.5, classes)
    plt.xlabel("Time (s)")
    plt.ylabel("Classes")
    plt.title("Heatmap probability over time")    
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

# method to detect commands within the audio. See the 'basic_cooldown' var that the indicatethe modality for the cooldown managing
def find_command_from_audio(chosen_audio):
    # Containers for timings (ms)
    t_segm = []                     # list to contain the timing related windows segmentation of the audio file
    t_feat = []                     # list to contain the timing related features extraction
    t_inf  = []                     # list to contain the timing related inference
    t_post = []                     # list to contain the timing related post analysis and command recognition
    
    # sliding window parameters
    win_size = desired_sr * window_size     # defaulttyi = 1 s (same length of the audio in the dataset)
    hop_size = int(desired_sr * step_size)  # dafault = 0.5 s (half of the audio lenght)
    
    # var to plot
    probs_all = []                          # contains all the probability for each window
    times = []                              # contains the central time for each window
    
    audio = tf.io.read_file(chosen_audio)
    waveform, sr = tf.audio.decode_wav(audio, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    sr = int(sr.numpy())
    
    if sr != desired_sr:                    # resampling
        waveform = tfio.audio.resample(waveform, rate_in=sr, rate_out=desired_sr).numpy()

    if len(waveform) < win_size:            # lenght check
        if truncate:
            print("The chosen audio is too short and truncate is activated")
            return                          # too short, drop it
        else:
            waveform = np.pad(waveform, (0, win_size - len(waveform)))  # padding
            num_frames = 1
            print("The chosen audio is too short, use padding.")
    else:
        if not truncate:
            pad_len = ((len(waveform) - win_size) % hop_size)   
            if pad_len > 0:                                     # check if padding is needed
                waveform = np.pad(waveform, (0, pad_len))               # padding
        
        num_frames = 1 + (len(waveform) - win_size) // hop_size         # calculate the numbers of frames (windows)
    
    audio_len = len(waveform)/desired_sr
    print("---- Start predition ----")
    print(f"Audio input-> Name: {chosen_audio} , lenght: {audio_len:.2f} s, windows = {num_frames}")
    
    if not basic_cooldown:          # case with cooldown based on command
        last_rec_command = None     # last recongnized command 
        
    num_frames = int(num_frames)
    skip = 0                        # set skip to 0, indicates the windows to skip after recognizing a command
    # scann al the windows
    for i in range(num_frames):
         
        t_segm_start = now_ms()
        start = int(i * hop_size)   # calculate the start of the current window
        end = int(start + win_size) # calculate the end of the current window    
        chunk = waveform[start:end] # create the chunk (window)
        center_time = (start + end) / 2 / desired_sr    # time to plot the prediction (center of current window)
        t_segm_end = now_ms()
        t_segm.append(t_segm_end - t_segm_start)        # add segmentation time (ms)

        t_feat_start = now_ms()
        features = extract_features(chunk, desired_sr)  # extract features
        features = np.expand_dims(features, axis=0)     # [1, width, height, channel] -> with mel and 1s window [1,124,64,1]
        t_feat_end = now_ms()
        t_feat.append(t_feat_end - t_feat_start)        # add features extraction time (ms)

        t_inf_start = now_ms()
        probs = network.predict(features, verbose=0)[0] # predict the class of the current windows
        pred_idx = np.argmax(probs)                     # get index of the class more likely
        conf = probs[pred_idx]                          # confidence of the predict class
        t_inf_end = now_ms()
        t_inf.append(t_inf_end - t_inf_start)           # add inference time (ms)
        
        probs_all.append(probs)                         # add all the probs of the current window for the plots
        times.append(center_time)                       # add the center time of current window for the plots
        
        t_post_start = now_ms()
        
        if skip > 0:                 # skip the current window (after a command) in basic cooldown scenario
            skip -= 1
            if basic_cooldown:
                print("Skip window.")
                continue
            elif not basic_cooldown and pred_idx == last_rec_command and conf > confidence_threshold:
                print("same command already recognized. skip window.")
                continue

        if conf > confidence_threshold and classes[pred_idx] != "unknown":  # check for confidence -> SEE NOTE 0
            last_rec_command = pred_idx                     # update index for the last recognized command
            start_ms = int(start / desired_sr * 1000)
            end_ms = int(end / desired_sr * 1000)
            print(f"Window {i} [{start_ms}-{end_ms} ms] -> {classes[pred_idx]} ({conf*100:.2f})")
            
            skip = cooldown                 # avoid to recognize again the same command
        
        t_post_end = now_ms()
        t_post.append(t_post_end - t_post_start)        # add post analysis and command recognition time (ms)
            
    probs_all = np.array(probs_all)     # convert in numpy array
    times = np.array(times)             # convert in numpy array
    print("---- End predition ----")

    log_to_file(chosen_audio, audio_len, num_frames, t_segm, t_feat, t_inf, t_post) # print into log file
    # plots
    plot_probabilities(times, probs_all, classes)   # plot results
    plot_heatmap(times, probs_all, classes)         # plot heat map

# ------------------------------------ main ------------------------------------        
if __name__ == "__main__":
    GPU_check()             # set GPU
    load_model()            # load the trained model
    load_class_index()      # load the correct indeces for the classes
    read_audio_files()      # read all the long audio files
    
    response = input("Do you want to classify all the audio files or just some random ones? (all,random): ").strip().lower() # ask to the user
    if response == "all":           # predict all audio file in the folder (test mode)
        
        if long_audio_files == None or len(long_audio_files) == 0:
            print("There aren't audio file to classify.")
        else:
            path_list = get_audio_list()            # get the list of audio file path
            # scan all audio file loaded
            for audio in path_list:
                find_command_from_audio(audio)      # take a new file and analyze it
            
    elif response == "random":      # predicts a random audio file until the user exits
        while True:
            chosen_audio = get_audio_random()       # get a random long audio
            find_command_from_audio(chosen_audio)   # take a new file and analyze it
        
            response = input("Do you want to continue? (yes/no): ").strip().lower() # ask to the user
            if response != "yes":
                break
    
"""
NOTE 0:
    Based on the idea that if the classification works well, the closer the current window is to the window containing the command, the more the probability of that command 
    will increase, peaking in the window corresponding to the command. Beside that probability decreases the further the current window moves from the command window.
    We need a classification system that recognizes only the commands we want (everything else will be considered unknown, even those not fully recognized by the network).
    Therefore, a certain confidence threshold is used (default 90%). To be recognized, a command must exceed the confidence threshold. 
    To avoid recognizing the same command multiple times, there will be windows where the classification will not be considered.
"""