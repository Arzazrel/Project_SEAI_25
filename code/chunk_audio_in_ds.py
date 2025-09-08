# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

Description: program that checks the dataset for audio files that are too large, above a certain threshold in seconds, 
            and if so, divides them into chunks of the desired length which it will then save with a progressive name.
"""

import os
import time
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio  

# ------------------------------------ start: global var ------------------------------------
# -- path --
dir_father = ".."               # fathere folder containing all other folders used
dir_ds_name = "dataset"         # folder containing all the dataset to load for the program
ds_name = "speech_ds"           # folder name for the target dataset
#ds_name = "test"               # folder name for the target dataset (testing purpose)
path_dir_ds = os.path.join(dir_father,dir_ds_name,ds_name)# folder containing the dataset to load (nested dataset)

# -- audio file var --
desired_sr = 16000              # want audio as 16kHz mono
desire_audio_len = 1            # indicates the desired length of the audio (chunks) in the database
min_sec_to_chuncking = 2        # indicates the minimum number of seconds for the audio track to chunk
truncate = False                # If the audio file is not perfectly divisible by the interval in seconds of the chunks, 
                                # indicate whether to truncate the last chunk or to do the padding.
                                
# -- statistics var --
long_audio_file_found = 0       # indicte the number of audio files found that had time greater than min_sec_to_chuncking
# ------------------------------------ end: global var ------------------------------------

# ------------------------------------ start: utilities method ------------------------------------  
   
# method to convert time in second in a string with this format hh:mm:ss
def format_time(time):
    hours, remainder = divmod(int(time), 3600)
    minutes, seconds = divmod(remainder, 60)

    return(f"{hours:02}:{minutes:02}:{seconds:02}")
    
# ------------------------------------ end: utilities method ------------------------------------

start_time = time.time()                        # start time
print("-------------------- READING DS --------------------")   # status print
print("Read the DS: ",path_dir_ds)                              # status print
    
entries = os.listdir(path_dir_ds)                   # Get the list of items in the parent folder
subfolders = [f for f in entries if os.path.isdir(os.path.join(path_dir_ds, f))]    # Check if there are subfolders 

if subfolders:          # there are sub-folders -> (nested dataset)
    for sub in entries:             # scroll through each subfolder
        
        sub_path = os.path.join(path_dir_ds, sub)   # path of each folder (of the original folder)
        print("Read the sub-folder: ",sub)# status print
            
        for filename in os.listdir(sub_path):
                
            if filename.endswith(".wav"):  # check if is a audio file .wav
                # load audio file
                file_path = os.path.join(sub_path, filename)            # create the path of the current audio file
                audio_bin = tf.io.read_file(file_path)                  # read file audio

                waveform, sample_rate = tf.audio.decode_wav(audio_bin)  # read the binary file and decode it to WAV, waveform in [-1,1] and sample_rate of the file (frequency)
                waveform = tf.squeeze(waveform, axis=-1)                # from (N,1) to (N,)
                        
                if sample_rate != desired_sr:       # check frequency 
                    waveform = tfio.audio.resample(waveform, rate_in=tf.cast(sample_rate, tf.int64), rate_out=desired_sr)   # Resample a 16kHz mono
        
                desired_len = int(desire_audio_len * desired_sr)        # 1 s = desired_sr -> num_samples = frequency -> in this case 16000 samples
                curr_len = tf.shape(waveform)[0]                        # get the lenght of the audio (= desired_sr * seconds)
                
                if curr_len >= (desired_len * min_sec_to_chuncking):    # check if the current audio file should be split into chunks
                    print("this audio file: ",filename," is long: ", curr_len / desire_audio_len, " s")
                    long_audio_file_found += 1
                    
                    num_chunks = curr_len // desired_len                # get number of chunk for this audio file
                    remainder = curr_len % desired_len                  # get if have to truncate or to pad
    
                    if remainder > 0:                                       # have to truncate or padding
                        if truncate:                                        # truncate 
                            waveform = waveform[:num_chunks * desired_len]
                        else:                                               # padding
                            num_chunks += 1
                            waveform = np.pad(waveform, (0, desired_len - remainder))
                    
                    base = os.path.splitext(os.path.basename(file_path))[0] # get base name
                    
                    for i in range(num_chunks):
                        start = i * desired_len             # calculate the start for the current chunk
                        end = start + desired_len           # calculate the end for the current chunk
                        chunk = waveform[start:end]         # create the current chunk

                        out_name = f"{base}_{i}.wav"
                        out_path = os.path.join(sub_path, out_name)
                        sf.write(out_path, chunk, desired_sr)
                        print(f"Salvato {out_path}")

                    print(f"Totale chunk salvati: {num_chunks}")

end_time = time.time()                                      # end time      
print("\n---- dataset reading and chunking completed ----") # control data print
print(f"Number of long files found: ",str(long_audio_file_found)," execution time of program: ", format_time(end_time - start_time))