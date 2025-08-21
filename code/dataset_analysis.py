# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all log, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import warnings
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow_io as tfio  
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------ start: global var ------------------------------------
# -- path --
dir_ds_name = "dataset"                             # folder containing all the dataset to load for the program
ds_name = "speech_ds"                               # folder name for the target dataset
#ds_name = "test"                               # folder name for the target dataset
path_dir_ds = os.path.join("..",dir_ds_name,ds_name)# folder containing the dataset to load (nested dataset)
# -- audio file var --
desired_sr = 16000                  # want audio as 16kHz mono
num_mel_bins = 64                   # num of bins for Mel-spectrogram
num_mfccs = 13                      # num of coefficient to take
mel_f_min = 80.0                    # min frequency for the bins of MEL 
mel_f_max = 7600.0                  # max frequency for the bins of MEL 

# -- command set var --
command_set = ['forward', 'backward', 'stop','go','up','down','left','right']   # array containing all the target commands
generic_class_name = "unknown"      # name for the generic class containing all the word that aren't command for this project

# -- dataset variables  --
classes = []                        # the label associated with each class will be the position that the class name will have in this array
total_audio_ds = []                 # contain the total audio dataset

# ------------------------------------ end: global var ------------------------------------

# ------------------------------------ start: utilities method ------------------------------------

# method to convert time in second in a string with this format hh:mm:ss
def format_time(time):
    hours, remainder = divmod(int(time), 3600)
    minutes, seconds = divmod(remainder, 60)

    return(f"{hours:02}:{minutes:02}:{seconds:02}")
    
# method to check if the audio is a audio file
def is_valid_wav(audio_bin):
    try:
        audio_bytes = audio_bin.numpy()

        if len(audio_bytes) < 4 or audio_bytes[:4] != b"RIFF":  # check if there is a header RIFF
            return False
        return True
    except Exception:
        return False
        
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

# ------------------------------------ start: methods for load DS ------------------------------------

# method to load the audio clip dataset
def load_and_analyze_audio_from_ds():
    global path_dir_ds, classes, total_audio_ds, command_set, generic_class_name
    
    time_to_load = 0;               # var indicating the time spent for loading the audio files
    time_to_extract = 0;            # var indicating the time spent to extract the features from audio files
    #total_stft_data = []            # contain the data for STFT for the whole dataset
    total_mel_data = []             # contain the data for MEL for the whole dataset
    total_mfcc_data = []            # contain the data for MFCC for the whole dataset
    total_labels = []               # contain the labels of the whole dataset
    
    # SEE -> NOTE 0 (end page)
    sums_by_class = {cls: {"stft": None, "mel": None, "mfcc": None, "count": 0} for cls in command_set}   # create dictionary containing the spectral statistic for each command class of dataset
    sums_by_class[generic_class_name] = {"stft": None, "mel": None, "mfcc": None, "count": 0}             # add  generic class
        
    classes_counter = [0] * (len(command_set) + 1)  # array of the counter for the audio belongs for each interesting class plus generic_class
    classes.append(str(generic_class_name))         # add the generic class containing all the classes that aren't target commands
    
    print("-------------------- READING DS --------------------")
    print("Read the DS: ",path_dir_ds)              # status print

    start_time = time.time()

    entries = os.listdir(path_dir_ds)                   # Get the list of items in the parent folder
    subfolders = [f for f in entries if os.path.isdir(os.path.join(path_dir_ds, f))]    # Check if there are subfolders 

    if subfolders:          # there are sub-folders -> (nested dataset)
        for sub in entries:             # scroll through each subfolder
            
            # check if is a target command or not
            if sub in command_set:                  # is a target command, add the class 
                classes.append(str(sub))                # update classes   
                index = classes.index(str(sub))         # take index of classes, is the label of this class 
                label = sub                             # set label
            else:                                   # isn't a target command, put in generic class
                index = 0;                              # the generic class has counter in 0 position
                label = generic_class_name              #
            
            print("Read the sub-folder: ",sub, " , belong in the class: ",label)          # status print
            
            sub_path = os.path.join(path_dir_ds, sub)   # path of each folder (of the original folder)
            for filename in os.listdir(sub_path):
                # load audio file
                s_t = time.time()
                file_path = os.path.join(sub_path, filename)            # create the path of the current audio file
                audio_bin = tf.io.read_file(file_path)                  # SEE -> NOTE 1
                
                if not is_valid_wav(audio_bin):                         # Controllo header WAV, check if is a audio file
                    print(f"Skipping non-WAV file: {file_path}")
                    continue                                            # go to next file
                    
                classes_counter[index] += 1             # update counter 
                
                waveform, sample_rate = tf.audio.decode_wav(audio_bin)  # read the binary file and decode it to WAV, waveform in [-1,1] and sample_rate of the file (frequency)
                waveform = tf.squeeze(waveform, axis=-1)                # from (N,1) to (N,)
                e_t = time.time()
                time_to_load += (e_t - s_t)
                
                # extract features from audio files
                try:
                    s_t = time.time()
                    stft, mel, mfcc = extract_full_features(waveform, sample_rate)  # get features
                    e_t = time.time()
                    time_to_extract += (e_t - s_t)
                    
                    # add the data to the list
                    #total_stft_data.append(stft)
                    total_mel_data.append(mel)
                    total_mfcc_data.append(mfcc)
                    total_labels.append(label)

                    # initialize accumulators with the right size 
                    if sums_by_class[label]["stft"] is None:            # class met for the first time
                        sums_by_class[label]["stft"] = np.zeros_like(stft)
                        sums_by_class[label]["mel"]  = np.zeros_like(mel)
                        sums_by_class[label]["mfcc"] = np.zeros_like(mfcc)

                    # sum matrices
                    sums_by_class[label]["stft"] += stft    # sum value for the STFT
                    sums_by_class[label]["mel"]  += mel     # sum value for the MEL
                    sums_by_class[label]["mfcc"] += mfcc    # sum value for the MFCC
                    sums_by_class[label]["count"] += 1      # update samples counter

                except Exception as e:
                    print("Errore con file:", file_path, e)
                    
    else:                   # there aren't sub-folders -> (flat dataset)
        for filename in entries:                # for each images on the current folder
            index = 0;                              # the generic class has counter in 0 position
            classes_counter[index] += 1             # update counter 
    
    end_time = time.time()
    elapsed = end_time - start_time  # in secondi
    
    # control data print
    print("\n---- dataset reading completed ----")
    print("\n-- Time statistics --")
    print("Read the DS: ",path_dir_ds, "in ",format_time(elapsed), " , 100%") 
    load_time_perc = ( time_to_load * 100 ) / elapsed    
    print("Read the audio files in: ", format_time(time_to_load), " , ",f"{load_time_perc:.2f}","%")
    extract_time_perc = ( time_to_extract * 100 ) / elapsed  
    print("extract the features from audio files in: ", format_time(time_to_extract), " , ",f"{extract_time_perc:.2f}","%")
    total_occurrence = sum(classes_counter)
    
    print("\n-- Classes statistics --")
    print("Num of classes: ",len(classes))
    for i in range(len(classes)):
        curr_class_perc = ( classes_counter[i] * 100 ) / total_occurrence
        print("class name: ",classes[i]," , class occurrence: ", classes_counter[i],"class perc: ", f"{curr_class_perc:.2f}", "%")
        
    print("\n-- Space statistics --")
    
    #total_stft_data = np.array(total_stft_data)
    total_mel_data = np.array(total_mel_data)
    total_mfcc_data = np.array(total_mfcc_data)
    total_labels = np.array(total_labels)
    
    #print("total_stft_data",len(total_stft_data), total_stft_data.shape)
    print("total_mel_data",len(total_mel_data), total_mel_data.shape)
    print("total_mfcc_data",len(total_mfcc_data), total_mfcc_data.shape)
    print("total_labels",len(total_labels), total_labels.shape)
    #print("Requied memory for STFT ds: ",f"{(total_stft_data.size * total_stft_data.itemsize / 10**6):.2f}", " MB")
    print("Requied memory for MEL ds: ",f"{(total_mel_data.size * total_mel_data.itemsize / 10**6):.2f}", " MB")
    print("Requied memory for MFCC ds: ",f"{(total_mfcc_data.size * total_mfcc_data.itemsize / 10**6):.2f}", " MB")
    
    print("-------------------- PERFORM AVG FEATURES --------------------")
    
    avg_features = {}                                   # var to contain the avg features
    for cls, feats in sums_by_class.items():            # Iterates over each class (cls) and its data packet (feats is the dictionary with "stft", "mel", "mfcc", "count").
        if feats["count"] > 0:                          # check if the class has samples
            # Calculate the arithmetic average element-wise (i.e. for each “pixel” of the frequency×time matrix). SEE -> NOTE 2
            avg_features[cls] = {                       
                "stft": feats["stft"] / feats["count"],
                "mel":  feats["mel"]  / feats["count"],
                "mfcc": feats["mfcc"] / feats["count"],
            }
        else:                                           # current class don't have samples
            avg_features[cls] = {"stft": None, "mel": None, "mfcc": None}   # default value
                
    print("------------------------------------------------------------")

    return avg_features
    
# ------------------------------------ end: methods for load DS ------------------------------------
    
# ------------------------------------ start: methods for extract features ------------------------------------
    
# method for feature extraction. Extract and return the whole STFT, Mel-Spectrogram, and MFCC
def extract_full_features(waveform, sample_rate):
    global desired_sr, num_mel_bins, num_mfccs, mel_f_min, mel_f_max

    if sample_rate != desired_sr:       # check frequency 
        waveform = tfio.audio.resample(waveform, rate_in=tf.cast(sample_rate, tf.int64), rate_out=desired_sr)   # Resample a 16kHz mono
        
    desired_len = desired_sr                # 1 s = desired_sr -> num_samples = frequency -> in this case 16000 samples
    curr_len = tf.shape(waveform)[0]        # 

    waveform = waveform[:desired_len]               # If too long → Truncate to 1 second (desired_sr samples)
    pad_len = desired_len - tf.shape(waveform)[0]   # If too short → pad with zeros up to 1 second (desired_sr samples)
    waveform = tf.pad(waveform, [[0, pad_len]])

    # Normalization [-1,1] with minimal perturbation to avoid dividing by 0 in case of empty signal - SEE -> NOTE 3
    waveform = waveform / (tf.reduce_max(tf.abs(waveform)) + 1e-9)

    # -- STFT -- SEE -> NOTE 4 
    stft = tf.signal.stft(waveform, frame_length=256, frame_step=128)   # perform STFT
    spectrogram = tf.abs(stft)                                          # [num_frames, num_freq_bins]

    # -- Mel-Spectrogram -- SEE -> NOTE 5 
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], desired_sr, mel_f_min, mel_f_max
    )
    mel_spectrogram = tf.tensordot(tf.square(spectrogram), linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # -- MFCC -- SEE -> NOTE 6 
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel_spectrogram + 1e-6))
    mfccs = mfccs[..., :num_mfccs]
        
    return spectrogram, mel_spectrogram, mfccs
    
# ------------------------------------ end: methods for extract features ------------------------------------

# method to print the avg spectrum features for each class
def print_avg_class_spectrum(avg_features):
    
    for feature_type in ["stft", "mel", "mfcc"]:        # iterate for each extracted features 
        for cls, feats in avg_features.items():
            if feats[feature_type] is not None:         # control check of the data
                data = feats[feature_type].numpy()      # convert in numpy
                
                plt.figure(figsize=(8, 5))
                
                if feature_type == "stft":      # STFT case
                    plt.title(f"{feature_type.upper()} mean - class: {cls} - T = 16 ms , Hop = 8 ms")
                    plt.imshow(
                        np.log(data.T + 1e-6), 
                        aspect='auto', 
                        origin='lower'
                    )
                    plt.colorbar(label="Value log-scaled")  
                    plt.xlabel("Time (frame)")
                    plt.ylabel("Frequency (bins) - From 0 kHz to 16 kHz")
                    
                elif feature_type == "mel":     # Mel case
                    plt.title(f"{feature_type.upper()} mean - class: {cls}")
                    plt.imshow(
                        np.log(data.T + 1e-6), 
                        aspect='auto', 
                        origin='lower'
                    )
                    plt.colorbar(label="Value log-scaled")
                    plt.xlabel("Time (frame)")
                    y_label = "Frequency (bands) - from " + str(mel_f_min) + " Hz to " + str(mel_f_max) + " Hz"
                    plt.ylabel(y_label)         # 64 bande con fmin=80 Hz, fmax=7600 Hz sono tipici a 16 kHz.
                    
                else:                           # mfcc case
                    plt.title(f"{feature_type.upper()} mean - class: {cls}")
                    plt.imshow(
                        data.T, 
                        aspect='auto', 
                        origin='lower'
                    )
                    plt.colorbar(label="Value")
                    plt.xlabel("Time (frame)")
                    plt.ylabel("Coefficients")
                    
                plt.tight_layout()
                plt.show()
                                
                """
                For STFT and Mel: Use np.log because the values ​​are always positive -> this highlights the power differences.
                For MFCC: DO NOT use np.log (they are already compressed scale values, even negative ones).
                """

# ------------------------------------ main ------------------------------------        
if __name__ == "__main__":
    GPU_check()
    avg_features = load_and_analyze_audio_from_ds() # load and analyze the dataset
    print_avg_class_spectrum(avg_features)          # print the extracted features

    
"""
NOTE 0: 
    sums_by_class is a dictionary containing the spectral statistic for each command class of dataset.
    The collected (summed) spectral data will be used together with the number of samples to obtain the mean spectral values ​​for each class (cls).
    
    sums_by_class[cls] = {
        "stft":  matrice_somma_stft  # element-wise sum of the STFTs of the class files
        "mel":   matrice_somma_mel   # element-wise sum of the Mel-spec
        "mfcc":  matrice_somma_mfcc  # element-wise sum of the MFCC
        "count": n_file              # how many elements of the class were added together (number of element for the class)
    }
    
NOTE 1: (read file)
    tf.audio.decode_wav(audio_bin) -> Decodes a WAV (PCM) file into a float32 tensor with values ​​in [-1.0, 1.0].
    Returns two things:
    audio: tensor of shape [num_samples, num_channels]
    sample_rate: scalar int32 = sample rate written in the file (does not change it, e.g., 16 kHz).
    Useful options:
    desired_channels → force mono/stereo (e.g., desired_channels=1 for mono).
    desired_samples → cut/pad up to that number of samples (does not change the sample rate).
    Read the sample from the file. If you want a different rate (e.g., 20 kHz), you must resample.
    How many samples in 1 second?
    - formula -> num_samples = sample_rate * duration_in_seconds.
    Ex:
    - 16 kHz → 16,000 samples in 1 s
    - 8 kHz → 8,000 samples in 1 s
    - 44.1 kHz → 44,100 samples in 1 s

NOTE 2: 
    The three sums_by_class always have the same shape within a class (es at 16 kHz, frame_length=256, frame_step=128, 1 s clip):
    - STFT: [num_frames, num_freq_bins] = [124, 129]
    - Mel: [num_frames, num_freq_bins] = [124, 64]
    - MFCC: [num_frames, num_coeff] = [124, 13]                                   
    Attention: 
    For the shapes to match, it is important that all clips have the same effective duration before the STFT (e.g., padding/trimming to 1 s), 
    otherwise the number of frames (dimension of spectral matrix) will change and the sum will be incorrect.

NOTE 3:
    tf.audio.decode_wav -> When TensorFlow decodes a 16-bit PCM WAV, it converts int16 integers (-32768 to +32767) to float32 between [-1.0, +1.0].
    Values ​​between [-1, 1] do not mean the entire range is used.
    Ex:
    - A file recorded at low volume may use a limited portion of the range (e.g., samples only between [-0.2, +0.2]).
    - A file with severe clipping may reach ±1.0.
    Normalization brings the maximum absolute value of the signal to exactly 1.0.
    Ex:
    - If the signal was in [-0.2, +0.2], it becomes [-1.0, +1.0].
    - If it was already [-1.0, +1.0], it remains the same.
    Advantages of this approach:
    - Consistency across examples: Audio files can have different volumes. By always bringing them to the same "maximum peak", the network doesn't have to learn to compensate for differences in volume.
    - Training robustness: It prevents the network from classifying loud or quiet volumes as specific classes, thus avoiding biases due to loudness.
    - Simple and fast.

NOTE 4:
    STFT (Short-time Fourier transform): Breaks the signal into overlapping windows and performs the FFT on each window.
    Ex with 16 kHz:
    - frame_length = 256 -> ~16 ms per window (256 / 16000)
    - frame_step = 128 -> hop ~8 ms
    The spectrogram has the form [num_frames, num_freq_bins], with num_freq_bins = 256 / 2 + 1 = 129.
    We use the modulus (amplitude) of the STFT.

NOTE 5:
    Mel-Spectrogram -> closer to human perception.
    Project the linear spectrum onto 64 Mel bands (more perceptual).
    First, we calculate power (|X|^2) and then project it with the Mel filter matrix.
    Typical values ​​used in this case -> fmin=80 Hz, fmax=7600 Hz 

NOTE 6:
    MFCC used to obtain compact features (used in lightweight KWS models).
    When calculating MFCCs (Mel-Frequency Cepstral Coefficients):
    1) Start with the spectrum;
    2) Switch to the Mel scale and take its log;
    3) Apply a DCT (Discrete Cosine Transform) and obtain the MFCC coefficients.

    The idea of ​​the DCT is to decorrelate the features and compress the information.
    The MFCC coefficients are sorted by frequency (not by intensity).
    - Coefficient 0 (c0) represents the global average energy (such as overall loudness).
    - The subsequent coefficients (c1, c2, etc.) capture slow variations in the spectrum, i.e., the coarse shape of the speech spectrum (the formants, which are crucial for distinguishing vowels and therefore words).
    - The higher coefficients (c20, c30, etc.) represent very fine variations and often indicate noise or less useful details.

    For this reason, in speech recognition and keyword spotting, the first coefficients are typically used, as they are the most significant for the spectral shape of the voice.
    The others are often ignored because they are less informative or too sensitive to noise.

    Be careful, the first coefficients are not necessarily the ones with the greatest intensity, but they are the most significant for distinguishing sounds and words (phonemes).

"""
