

<p align="center">
<a href="https://www.stms-lab.fr/"> <img src="img/STMS-lab.png" width="14%"></a>
&nbsp;
<a href="http://ars.ircam.fr"> <img src="img/ARS_violet_hr.png" width="14%"></a>
&nbsp;
<img src="img/label_ANR_bleu_CMJN.png" width="14%">
&nbsp;
<a href="http://www.idris.fr/jean-zay/"> <img src="img/Logo_GENCI.png" width="25%"></a>
</p>
This repository is still under construction! Please check back soon

# MBExWN_Vocoder

This repository contains a python code for the Multi-Band Excited WaveNet Neural vocoder, a neural vocoder
allowing mel spectrogram inversion for speech and singing voices with varying identities, languages, 
and voice qualities.

The code and models in this repository are demonstrations of the MBExWN vocoder. For technical details please see this
[paper](https://www.mdpi.com/2078-2489/13/3/103)

The MBExWN vocoder is rather efficient and allows inverting mel spectrogams 
faster than real time on a single core of a Laptop CPU. The vocoder generates 
a near transparent audio quality for a variety of voice identities 
(speakers and singers), languages, and voice qualities. 
The audio sample rate of the generated files is 24kHz. 

### Motivation

To facilitate research into voice attribute manipulation and multi speaker synthesis 
using the mel spectrogram as voice representation, the present repository distributes inference scripts 
together with the three trained models denoted **MW-SI-FD**, **MW-SP-FD**, and **MW-VO-FD** in 
the [paper](https://www.mdpi.com/2078-2489/13/3/103)
mentioned above. An application for transposition of speech and singing signals using an auto encoder with 
bottleneck has been investigated in this [compagnon paper](https://www.mdpi.com/2078-2489/13/3/102).

### Demo Sounds

According to the perceptual tests discussed here the quality of the generated audio is close to transparent
for both speech and singing signals. The model has problems with excitation signals that are strongly 
irregular though. Please look [here](http://recherche.ircam.fr/anasyn/roebel/MBExWN_demo/index.php)
for results with a previous version of the MBExWN vocoder. An updated demo page containing examples 
for the three models distributed here will soon be added.

##  Installation

The MBExWN Vocoder can be run directly from the source directory or installed via pip (to be done)

In case you want to run it from the source directory you need to first download and install the 
pretrained models using the script in the scripts directory.

Then you need to make sure you have all necessary dependencies installed.   

```
-    pyyaml
-    scipy
-    numpy
-    tensorflow=>2.5
-    librosa>=0.8.0
-    pysndfile
-    matplotlib
```


You can install these by means of 

```shell
$ pip install -r requirements.txt
```

#### Pretrained models

Due to the download size limitations on github we do not include the pretrained models within the repos. 
The prepretrained models are available via a separate download link. These can be installed by means of running 
the shell script 

```bash
./scripts/install_MelInvWeights.sh (to do)
```

We gratefully acknowlegde the support of   models have been pretrained on the [jean-zay computer](http://www.idris.fr/jean-zay/) of  

### Computational costs

The mel inverter is sufficiently effeicient to run perform resynthesis from mel spectrograms 
two times faster than  real-time on a single core of a Laptop CPU.  

On a GPU the vocoder achieves audio synthesis up to 200 times faster 
than real time. 

### Usage

### Generating mel spectrograms

The input format of the MBExWN  vocoder are 80 channel mel spectrograms with frame rate of 80 Hz.
To generate these mel spectograms you can use the script ./bin/generate_mel.py as follows

```shell
./bin/generate_mel.py  -o OUTPUT_DIR input_audio_file [input_audio_file ...]
```

after running this command you will find mels spectrograms under the output 
directory. For each input file you will find a pickled data file with the same basename and 
the extension replaced by means of mell. The pickled data files contain python dicts
that have keys for all analysis parameters as well as for the mel specrogram itself.


#### Running mel inversion form the command line

ToDo

#### Using the MBExWN vocoder as a python package

ToDO

### How to cite

In case you use the code or the models for your own work please cite 

```
Roebel, A.; Bous, F. Neural Vocoding for Singing and Speaking Voices with the Multi-Band Excited
WaveNet. Information 2022, 1, 0. https://doi.org/
```

### Copyright

Copyright (c) 2022 IRCAM
