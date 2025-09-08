# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

Description: program for creating long audio tracks by joining audio tracks from the dataset based on their class membership.
"""

import os
import time
import random
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio  

# ------------------------------------ start: global var ------------------------------------
# -- path --
dir_father = ".."               # fathere folder containing all other folders used
dir_ds_name = "dataset"         # folder containing all the dataset to load for the program
ds_name = "speech_ds"           # folder name for the target dataset
path_dir_ds = os.path.join(dir_father,dir_ds_name,ds_name)# folder containing the dataset to load (nested dataset)

ds_audio_name = "long_audio"    # name of the folder containing the dataset of the long audio to classify (find command)
path_dir_ds_audio = os.path.join(dir_father,dir_ds_name,ds_audio_name)# folder containing the dataset of the long audio file

logfile_name = "track_audio_log.txt"                        # name of the log file
path_logfile = os.path.join(path_dir_ds_audio,logfile_name) # path for the log file

# -- audio file var --
desired_sr = 16000              # want audio as 16kHz mono
desire_audio_len = 1            # indicates the desired length of the audio (chunks) in the database
min_sec_to_chuncking = 2        # indicates the minimum number of seconds for the audio track to chunk
truncate = False                # If the audio file is not perfectly divisible by the interval in seconds of the chunks, 
                                # indicate whether to truncate the last chunk or to do the padding.

# -- command set var --
command_set = ['forward', 'backward', 'stop','go','up','down','left','right']   # array containing all the target commands
generic_class_name = "unknown"  # name for the generic class containing all the word that aren't command for this project
              
# -- dataset variables  --
classes = []                    # the label associated with each class will be the position that the class name will have in this array
audio_paths_per_class = [[]]    # global list of audio file path lists for each class, index matches the class index in 'classes'

enable_log = True               # if 'true' writes the explanation of the created audio track in the log , if 'false' doesn't write the explanation

# ------------------------------------ end: global var ------------------------------------

# function to get a random file of a specific class passed as a paramete
def get_random_file(class_name):
    if class_name not in classes:               # control check
        raise ValueError(f"Clas {class_name} not found!")
    index = classes.index(class_name)           # get the class index
    
    return random.choice(audio_paths_per_class[index])  # return the random audio file

# function that loads all the names of the audio files present in the DB divided by their class
def load_ds():
    global path_dir_ds, classes, command_set, generic_class_name
    
    classes.append(str(generic_class_name))     # add the generic class containing all the classes that aren't target commands
    audio_paths_per_class.append([])            # create empty list for generic class
    
    print("-------------------- READING DS --------------------")   # status print
    print("Read the DS: ",path_dir_ds) # status print
    
    entries = os.listdir(path_dir_ds)                   # Get the list of items in the parent folder
    subfolders = [f for f in entries if os.path.isdir(os.path.join(path_dir_ds, f))]    # Check if there are subfolders 

    if subfolders:          # there are sub-folders -> (nested dataset)
        for sub in entries:             # scroll through each subfolder
            # check if is a target command or not
            if sub in command_set:                  # is a target command, add the class 
                classes.append(str(sub))                # update classes   
                audio_paths_per_class.append([])        # crea empty list for this class
                label = sub                             # set label
            else:                                   # isn't a target command, put in generic class
                label = generic_class_name              # set label
                
            index = classes.index(label)
            sub_path = os.path.join(path_dir_ds, sub)   # path of each folder (of the original folder)
            print("Read the sub-folder: ",sub, " , belong in the class: ",label)          # status print
            
            for filename in os.listdir(sub_path):
                
                if filename.endswith(".wav"):       # check if is a audio file .wav and read audio file

                    file_path = os.path.join(sub_path, filename)    # create the path of the current audio file
                    audio_paths_per_class[index].append(file_path)  # add the current audio path to its class 
    
    print("\n---- dataset reading and rewriting completed ----")    # control data print
    print("\n-- Classes statistics --")
    print("Num of classes: ",len(classes))
    for i in range(len(classes)):
        print("class name: ", classes[i], "-> Num files: ", len(audio_paths_per_class[i]))
        
# ------------------------------------ start: methods to work on audio track ------------------------------------   

# Reads and normalizes audio tracks. Returns a list of numpy arrays ready for concat.
def prepare_tracks_for_concat_tf(selected_tracks):
    
    processed_tracks = []

    for cls, path in selected_tracks:
        try:
            # read audio file
            audio = tf.io.read_file(path)
            waveform, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
            waveform = tf.squeeze(waveform, axis=-1)        # remove, single channel

            # Resample if necessary
            if sample_rate != desired_sr:
                waveform = tfio.audio.resample(waveform, rate_in=tf.cast(sample_rate, tf.int64), rate_out=desired_sr)

            # Truncate o pad
            desired_len = desire_audio_len * desired_sr
            waveform = waveform[:desired_len]
            pad_len = desired_len - tf.shape(waveform)[0]
            waveform = tf.pad(waveform, [[0, pad_len]])

            waveform = waveform / (tf.reduce_max(tf.abs(waveform)) + 1e-9)  # normalization [-1,1]
            processed_tracks.append(waveform.numpy())                       # add to list

        except Exception as e:
            print(f"Error loading file {path}: {e}")

    return processed_tracks

# Concatenates a list of tracks (already normalized). Make sure the final length is a multiple of 'desire_audio_len' s.
def concat_fixed_tracks(tracks, output_path, sr=desired_sr):
    
    audio = np.concatenate(tracks, axis=0)      # concatenate

    desired_len = len(tracks) * sr              # Check if multiple
    if len(audio) > desired_len:
        audio = audio[:desired_len]             # cut
    elif len(audio) < desired_len:              # pudding
        pad_len = desired_len - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")

    audio = audio / (np.max(np.abs(audio)) + 1e-9)  # normalize

    sf.write(output_path, audio, sr)            # save
    print(f"Saved {output_path} ({len(audio)/sr:.2f} s)")

# ------------------------------------ end: methods to work on audio track ------------------------------------ 

# ------------------------------------ start: utilities method ------------------------------------ 

# method that writes the composition of the created tracks to the log file.
def log_to_file(selected_tracks, track_name, enable_log=True):

    if not enable_log:      # control check
        return
    
    # Line construction: name_audio_track -> [ class (start - send s) , ... ] -- number of command in audio track
    num_command = 0                     # indicate the number of command to recognize in the audio file
    line = str(track_name) + " -> ["
    for i, (cls, path) in enumerate(selected_tracks):
        line += str(cls) + " (" + str(i*desire_audio_len) + "s - " + str((i+1)*desire_audio_len) + "s)"
        if i != len(selected_tracks)-1:
            line += ", "
        if cls in classes and cls != generic_class_name:
            num_command += 1

    line += "] -- num of command in the audio track: " + str(num_command) + "\n"
    
    # Writing to file in append mode
    with open(path_logfile, "a", encoding="utf-8") as f:
        f.write(line)
        
# ------------------------------------ end: utilities method ------------------------------------ 
        
# ------------------------------------ start: methods for GUI ------------------------------------
# method which asks the user how many tracks he wants to merge
def ask_number_of_tracks():
    while True:
        user_input = input("Enter the number of tracks to combine (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            return "exit"
        try:
            num = int(user_input)
            if num > 1:
                return num
            else:
                print("Please enter an integer greater than 1.")
        except ValueError:
            print("Invalid input. Please enter a positive integer greater than 1.")

# method that asks for the desired class for the current track
def ask_class_choice():
    while True:
        choice = input("Enter the name of the class you want to add a track from: ").strip()
        if choice in classes:
            return choice
        else:
            print("Invalid class name. Please choose a class from the list above.")
        
# metho for the main cicle        
def main_cicle():
    print("Welcome! Type 'exit' at any prompt to quit the program.")
    
    while True:
        num_tracks = ask_number_of_tracks()         # ask how many tracks to combine
        if num_tracks == "exit":
            print("Exiting the program. Goodbye!")
            break

        selected_tracks = []                        # the audio track selected
        
        print("\n -- Available classes:")
        for i, cls in enumerate(classes):
            print(f"{i}: {cls}")
        print("--\n")

        for i in range(num_tracks):     
            print(f"\nSelecting track {i+1} of {num_tracks}:")
            class_name = ask_class_choice()
            if class_name == "exit":
                print("Exiting the program. Goodbye!")
                return
                
            track_path = get_random_file(class_name)                    # get random file from chosen class
            selected_tracks.append((class_name, track_path))            # add the chosen class name and audio path
            print(f"Selected: {track_path} from class '{class_name}'")

        print("\nAll selected tracks:")
        for i, (cls, path) in enumerate(selected_tracks):
            print(f"{i+1}. Class: {cls}, File: {path}")
        
        break;
          
    if num_tracks != "exit" and class_name != "exit":   # all ok, ask the name of the audio file to be created
        output_name = input("\nEnter the name of the output audio file (without extension, or type 'exit' to quit): ").strip()
        if output_name.lower() == "exit":
            print("Exiting the program. Goodbye!")
            return
        
        output_path = os.path.join(path_dir_ds_audio,output_name + ".wav")  # create the output path for the created audio file
        
        tracks_array = prepare_tracks_for_concat_tf(selected_tracks)    # list for the preparated track to concatenate
        concat_fixed_tracks(tracks_array, output_path)                  # concatenate the tracks
        log_to_file(selected_tracks, output_name, enable_log)                         # append new line in the log file
    
# ------------------------------------ end: methods for GUI ------------------------------------

# ------------------------------------ main ------------------------------------        
if __name__ == "__main__":
    load_ds()       # load the possible audio file and the classes
    main_cicle()    # do the cicle to create the new audio track