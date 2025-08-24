# Project_SEAI_25

## **Description** 

Project for the Symbolic and Evolutionary Artificial Intelligence exam of the AIDE master's degree at the University of Pisa, year 2024-2025.

The project involves training a neural network to recognize voice commands that will be used to remotely control a television or MP3 player.
The goal of the project is to train the neural network on a dataset of audio files containing single words, with different pronunciations.
Among these words, the main command words (es. Start, Stop, Pause, Forward, Backward) must be recognized, while the other words related to other commands must be ignored.
Once trained, the neural network will be placed in an FPGA.

General note: At the bottom of the programs there will be notes in the comments explaining the most important parts and the choices I made.

## **Hardware used**

CPU: Intel(R) Core(TM) i7-10870H CPU @ 2.20GHz 2.21 GHz RAM: 16 GB GPU: RTX 3060 6GB laptop

## **Enviroments Settings**

Python and tensorflow are used to run this codce and train the networks. 

Caution: 
Tensorflow's setting up with the nvidia drivers to use the GPU and speed up the training process is very delicate. 
I refer to the official guide at the following link: https://www.tensorflow.org/install/pip?hl=it#windows-native 
Attention on windows last tensorflow release with GPu support is 2.10. For this project I have adoperated a configuration using wsl.

Windows configuration: python = 3.10, tensorflow = 2.10, CUDA = 11.2, cuDNN = 8.1
Wsl configuration (via miniconda) python = 3.10, tensorflow = 2.19, CUDA = 12.9, cuDNN = cuDNN 9.3

I have listed my configurations for clarity, do not consider them the best or those to be replicated, check according to your environment and your needs the best setting for you.

## **Dataset Links and Description**  

The dataset used in this project is "Google Speech Commands V2" which can be downloaded at this link: https://www.kaggle.com/datasets/sylkaladin/speech-commands-v2

The Google Speech Commands Dataset v2 is a widely used benchmark for keyword spotting tasks. 
It contains approximately 105,000 one-second audio recordings of 35 short spoken words, collected from thousands of different speakers. 
The audio files are sampled at 16 kHz and stored in WAV format, providing a standardized corpus for training and evaluating speech recognition models. 
In addition to the main keywords, the dataset includes recordings of background noise and non-speech utterances (such as “silence” and “unknown”), 
which are crucial for building robust models capable of handling real-world conditions. 
This dataset is particularly suitable for developing and benchmarking lightweight neural networks designed for low-latency, on-device speech recognition applications.

The dataset will not be included in the GitHub project due to space constraints, but it can be downloaded from the link provided.

Note: Any dataset containing single-word audio files can be used effectively with minor code changes:
- Name of the dataset to download
- Commands to recognize
- Length of audio tracks
- Rates and sampling rates
These parameters are all set via global variables in the program code.

This project includes a program ('dataset_analysis.py') for performing basic analysis of audio file datasets. 
The results obtained from the analysis of this dataset under consideration are in the '\results\ds_analysis' folder

## **The folder contains:**  
  
- dataset: folder containing the datasets and their preprocessed copy
- code: folder containing all the programs used in the project that are written in Python
- model: folder where the trained network models are saved, they are saved in different formats (.keras, .h5)
- results: A folder containing all the results obtained during the project. It contains the results of the dataset analysis and the results of the network training

## **Project choices** 

In the next sub-paragraph all the various choices made within this project will be shown.

### **Command list** 

For this project, we want to achieve voice recognition of specific commands to be used to control an MP3 player or TV. 
From the dataset, I decided to implement the commands listed below, along with the possible operations associated with them.
- 'forward': move forward within the track/movie
- 'backward': go backward within the track/movie
- 'stop': stop audio/movie track playback
- 'go': start playing the audio track/movie
- 'up': turn up the volume
- 'down': turn down the volume
- 'left': moves to the next audio track/movie
- 'right': moves to the previus audio track/movie

All other words belog to the class 'unknown'.
With this division, approximately 75% of the dataset will belong to the 'unknown' class with the remaining 25% split almost equally among the 7 command classes 
(backward and farward have half as many samples as the other classes).

### **CNNs adopted** 

This section will describe the various networks created and adopted for this project and the ideas behind them.
The networks for this project are all different versions of the same class, which I have called SirenNet and which can be found in the “\code\net_classes” folder.

#### Version 0:

	Network architecture:
    CNN that takes inspiration from the GoogLeNet model. This network consists of a convolutional layer, with max pool, followed by four inception modules, 
	a fully connect layer, with dropout, and the output layer.     
	In particular, in the four inception modules there is maxpool after the first and third, and at the end there is global average pooling. 
	
	Network dimension:
		Total params: 72,785 (284.32 KB)
		Trainable params: 72,785 (284.32 KB)
		Non-trainable params: 0 (0.00 B)
	
	Descriptions:
	In this network tried to give more emphasis from first to filters with larger kernel and then, as the depth of the network increases and the input size 
	decreases give more emphasis to filters with smaller kernels. Is a CNN of my creation which had good results in image recognition. 
	It is not specifically designed for the KWS problem.
    
#### Version 1:

	This network architecture consists of:
	- First layer consisting of two parallel convolution branches. One works only on frequencies and one only on time. After a square convolution followed by MaxPooling on time.
	- Second layer consisting of two parallel convolution branches. One works only on frequencies and one only on time (larger filters than the previous layer). 
	  After a square convolution followed by MaxPooling performed on frequencies.
	- Third layer consisting of two inception modules followed by a square MaxPooling.
	- Fourth layer consisting of an inception module followed by GlobalAveragePooling2D.
	- Fifth layer is a small dense layer before the output layer.
	
	Network dimension:
		Total params: 95,561 (373.29 KB)
		Trainable params: 95,561 (373.29 KB)
		Non-trainable params: 0 (0.00 B)

	Ideas/Descriptions:
	The idea behind this CNN is to work with multiple branches using rectangular filters and then concatenate them. 
    The rectangular filters has one dimension sized at 1 so that they only work in frequency or only in time. 
    E.g.:
    Conv2D(32, (1, 5)) → filter that looks at 1 frequency band × 5 frames in time
    Conv2D(32, (5, 1)) → filter that looks at 5 frequency bands × 1 frame in time
    
    MaxPools are also initially rectangular in time (k,1). This supports the idea that spectral information is very delicate.
    Compressing along the frequency too early risks "crushing" important informations or harmonic patterns.
    However, over time, you can reduce the resolution because you have 'redundancy—nearby' time frames often contain similar information.
    
    Rectangular filter patterns:
    - In the first layers: smaller filters -> capture local details without mixing too much (few bands, few frames).
    - In the deeper layers: larger filters -> see larger regions, for global correlations (patterns over more time + more frequencies).
    
    So what we want to achieve with this network:
    - First layers -> small rectangular filters (with a unit size) for local details without mixing too many details from different areas.
    - Middle layers -> larger rectangular filters (with a unit size) for longer correlations.
    - Final layers -> square filters to mix time and frequency together.

    This version of CNN also uses an inception module which takes an input and processes it in parallel with several different filters, then concatenates the results 
	along the channel dimension. It can be useful for capturing spectrogram patterns at different scales:
    - short spectral variations (such as bursts)
    - harmonics that extend across multiple frequencies
    - longer temporal patterns
	
### **Spectral features**

To train the KWS networks, we need to extract the spectral features from the audio files in the dataset. 
These spectral features can be viewed as 2D images with which to train the network.

To obtain these features, we must first set the sample rate (in this case, 16 kHz) and their duration (in this case, 1 s).
Then, we need to calculate the STFT to obtain the spectral information that can then be further processed (to obtain MEL and MFCC).
	
#### STFT (Short-time Fourier transform): 
Breaks the signal into overlapping windows and performs the FFT on each window. In this case 16 kHz for 1 s with frame_length = 256 (~16 ms) and frame_step = 128 (~8 ms hops).
The spectrogram has the form [num_frames, num_freq_bins] (it can be considered as 2D image), with num_freq_bins = 256 / 2 + 1 = 129. 
We use the modulus (amplitude) of the STFT.
    
#### Mel-Spectrogram 
It is closer to human perception then STFT. Project the linear spectrum onto 64 Mel bands (more perceptual).
First, we calculate the SFTF module and then project it with the Mel filter matrix (which imitates human auditory perception), applying a set of triangular filters.
	
When calculating a Mel spectrogram, we specify a minimum frequency (fmin) and a maximum frequency (fmax) for the Mel filters.
In this case, I applied: 
- fmin = 80 Hz
	Frequencies below ~80 Hz often contain only low-frequency noise, vibrations, or components that are not very informative for the voice/speech.
	The human ear has very low sensitivity below 80 Hz, so those frequencies are not very useful for keyword spotting.
	By setting fmin=80 Hz, we eliminate this unhelpful part and reduce complexity.
		
- fmax = 7600 Hz 
	fmax is typically ≈ Nyquist limit. Nyquist limit, also known as the Nyquist frequency, is a crucial concept in signal processing that dictates the 
	minimum sampling rate needed to accurately represent a signal. Nyquist frequency limit = sampling rate / 2 -> in this case Nyquist = 16 kHz / 2 = 8 kHz
	This means that you cannot correctly represent frequencies above 8 kHz, because they will overlap (alias) with lower frequencies.
		
	A slightly lower Nyquist limit (in this case, 7600 Hz) is often used to:
	- avoid aliasing or instability near the Nyquist limit,
	- focus computational resources on the most informative frequencies.

The human voice (speech) has significant energy and information especially between 300 Hz and 7-8 kHz.
Frequencies < 80 Hz -> are useless/noisy and are of no help in the KWS case.
Frequencies > 7600 Hz -> have almost no useful information for speech (more noise, less discrimination between words).
The 80–7600 Hz range captures virtually all the phonetic information needed to recognize voice commands.

#### MFCC used to obtain compact features (used in lightweight KWS models).
When calculating MFCCs (Mel-Frequency Cepstral Coefficients):
1) Start with the spectrum;
2) Switch to the Mel scale and take its log;
3) Apply a DCT (Discrete Cosine Transform) and obtain the MFCC coefficients.

The idea of ​​the DCT is to decorrelate the features and compress the information.
The MFCC coefficients are sorted by frequency (not by intensity).
- Coefficient (c0) represents the global average energy (such as overall loudness).
- The subsequent coefficients (c1, c2, etc.) capture slow variations in the spectrum, i.e., the coarse shape of the speech spectrum (the formants, which are crucial for distinguishing vowels and therefore words).
- The higher coefficients (c20, c30, etc.) represent very fine variations and often indicate noise or less useful details.

For this reason, in speech recognition and keyword spotting, the first coefficients are typically used, as they are the most significant for the spectral shape of the voice.
The others are often ignored because they are less informative or too sensitive to noise.

Be careful, the first coefficients are not necessarily the ones with the greatest intensity, but they are the most significant for distinguishing sounds and words (phonemes).
	
### **CNNs training** 

The simplest and fastest method for training the network would be to read the audio files from the dataset, process them (calculate MEl or MFCC), and store the results
(spectrograms in the form of matrices) in memory. However, this method is very expensive in terms of memory and can quickly saturate the available VRAM,
leading to errors (especially with very large datasets or large spectrograms).
A second method to limit memory consumption is to store the paths for each audio file in memory and then, during runtime training, read the audio file, process it
and use it. This consumes much less memory but significantly increases the network's training time.
An analysis of the dataset revealed that the spectrogram computation time is more than 8 times longer than the audio file reading time.

A third solution I chose, which I believe represents an excellent tradeoff between VRAM usage and training time, is to preprocess the audio file the first time
the dataset is read, and then save the resulting data (MEL or MFCC) to disk. The paths to the preprocessed data saved
to disk will then be stored in memory. This way, during training, you don't have to read and process the audio files, but you can read the ready-made data directly 
(reading them is faster than audio files). This way, you can significantly reduce memory usage, at the cost of slightly increasing training time and increasing disk usage.


### **CNNs results** 

Paragraph showing the best results obtained for all versions of Siren Net tested in the project.

#### Version 0:

	First training done only once, it does not have much statistical solidity but it gives a first look at performance and training times.
	Time for training 5:18:30
	The training and validation set accuracy are both just above 95%, and the test set accuracy is just below 95%.
	(More accurate results are in the '\results\train_1\SirenNet_vers_0' folder).
    
#### Version 1:

	First training done only once, it does not have much statistical solidity but it gives a first look at performance and training times.
	Time for training

## **Developer's notes**  
  
Work In Progress

## **Developers:**  
- Alessandro Diana