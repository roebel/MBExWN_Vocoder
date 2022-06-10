

<p align="center">
<a href="https://www.stms-lab.fr/"> <img src="img/STMS-lab.png" width="14%"></a>
&nbsp;
<a href="http://ars.ircam.fr"> <img src="img/ARS_violet_hr.png" width="14%"></a>
&nbsp;
<img src="img/label_ANR_bleu_CMJN.png" width="14%">
&nbsp;
<a href="http://www.idris.fr/jean-zay/"> <img src="img/Logo_GENCI.png" width="25%"></a>
</p>


## Content

**[1. Introduction](#S1)**
  * [1.1. Motivation](#S1-1)
  * [1.2. Demo sounds](#S1-2)

**[2. Installation](#S2)**
  * [2.1. Pretrained models](#S2-1)

**[3. Usage](#S3)**
  * [3.1 Generating mel spectrograms](#S3-1)
  * [3.2 Performing mel inversion](#S3-2)

**[4. Further info](#S4)**
  * [4.1 Computational costs ](#S4-1)
  * [4.2 How to cite](#S4-2)
  * [4.3 Acknowledgements](#S4-3)
  * [4.4 ChangeLog](#S4-4)
  * [4.5 Copyright](#S4-5)

----

<p align="center">  <img src="img/ModelScheme_V3.png" width="75%"> </p>

<a name="S1"></a>
## 1. Introduction


This repository contains the python sources of the Multi-Band Excited WaveNet Neural vocoder, a neural vocoder
allowing mel spectrogram inversion for speech and singing voices with varying identities, languages, 
and voice qualities.

The code and models in this repository are demonstrations of the MBExWN vocoder. For technical details please see this
[paper](https://www.mdpi.com/2078-2489/13/3/103)

The MBExWN vocoder is rather efficient and allows inverting mel spectrogams 
faster than real time on a single core of a Laptop CPU. The vocoder generates 
a near transparent audio quality for a variety of voice identities 
(speakers and singers), languages, and voice qualities. 
The audio sample rate of the generated files is 24kHz. 

<a name="S1-1"></a>
### 1.1 Motivation 

To facilitate research into voice attribute manipulation and multi speaker synthesis 
using the mel spectrogram as voice representation, the present repository distributes inference scripts 
together with the three trained models denoted **MW-SI-FD**, **MW-SP-FD**, and **MW-VO-FD** in 
the [paper](https://www.mdpi.com/2078-2489/13/3/103)
mentioned above. An application for transposition of speech and singing signals using an auto encoder with 
bottleneck has been investigated in this [compagnon paper](https://www.mdpi.com/2078-2489/13/3/102).

<a name="S1-2"></a>
### 1.2 Demo Sounds

Please see [here](http://recherche.ircam.fr/anasyn/roebel/MBExWN_demo/index.php)
for results with a previous version of the MBExWN vocoder. An updated demo page containing examples 
for the three models distributed here will soon be added.

As an example of a real world application of the vocoder you may listen to the [artificial voice in a *deep fake* interview](https://youtu.be/seXKtSGf4U8?t=20) of the famous french singer Dalida for that we used  a [DNN for voice reenactement](https://arxiv.org/pdf/2110.03744v3.pdf)
that converted the mel spectrogramms of the voice of the actress playing Dalida into the mel spectrograms that of the voice of Dalida,
followed by th WBExWN vocoder (which was not trained on Dalida's voice).

As another example [listen to a presentation of the results](https://youtu.be/FHoBEPBuSP8?t=1853) using the [bottleneck encoder](https://www.mdpi.com/2078-2489/13/3/102)
for transposing singing voices in the context of the recreation of an artificial version of the voice of the [castrato singer Farinelli](https://en.wikipedia.org/wiki/Farinelli).
For this work we used the MBExWN model trained on speech and singing voices.


<a name="S2"></a>
## 2. Installation

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

<a name="S2-1"></a>
### 2.1 Pretrained models

Due to the download size limitations on github we do not include the pretrained models within the repos. 
The prepretrained models are available via a separate download link. These can be installed by means of running 
the shell script 

```bash
./scripts/download_and_install_MBExWN_pretrained_models.sh
```

*We gratefully acknowledge the support of GENCI that made it possible to train the models 
on the super computer [jean-zay](http://www.idris.fr/jean-zay/).*


<a name="S3"></a>
## 3. Usage

<a name="S3-1"></a>
### 3.1 Generating mel spectrograms

The input format of the MBExWN  vocoder are 80 channel mel spectrograms with frame rate of 80 Hz.
To generate these mel spectograms you can use the script ./bin/generate_mel.py as follows

```shell
./bin/generate_mel.py  -o OUTPUT_DIR input_audio_file [input_audio_file ...]
```

after running this command you will find the mel spectrograms in the directory OUTPUT_DIR. 
For each input file you will find a pickled data file with the same basename and 
the extension replaced by means of mell. The pickled data files contain python dicts
that have keys for all analysis parameters as well as for the mel specrogram itself.

These files are can be read by the two further scripts that allow 

- recreating sounds from a mel spectrogram, as well as 
- visualizing a mel spectrogram

<a name="S3-1"></a>
### 3.2 Performing mel inversion

#### 3.2.1 Command line

The resynthesis of  sound files from a mel spectrograms stored in a pickle files is performed by mean sof the script 
`./bin/resynth_mel.py`. Assume you have a sound file test.wav you can perofrm an analysis/resynthesis cycle by means of

```shell
./bin/generate_mel.py  -o OUTPUT_DIR test.wav
./bin/resynth_mel.py VOICE -i OUTPUT_DIR/test.mell -o OUTPUT_DIR --format wav
```

This will create a the files `OUTPUT_DIR/test.mell` and subsequenctly `OUTPUT_DIR/syn_test.wav`.
The output soundfile name is derived from the inut mel file name by means of replacing the extension
according to the sound file format, adn addinf a prefix `syn_`.  By default the generated sound file format 
is `flac` but as shown above the output format can be changed. The sample rate of the output sound is
always 24kHz. 

The first parameter given to `resynth_mel.py` selects one of the three pretrained MBExWN models that are 
discussed in the [paper](https://www.mdpi.com/2078-2489/13/3/103). If no model is slected 
*the `resynth_mel.py` lists all available models. Currently the folowing models are available:

```
 - SING/MBExWN_SIIConv_V71g_SING_IMP0_IMPORTmod_MCFG0_WNCHA320_DCHA32_1024_DPTACT0_ADLW0.1_GMCFG5_24kHz
 - SPEECH/MBExWN_SIIConv_V71g_SPEECH_IMP0_IMPORTmod_MCFG0_WNCHA320_DCHA32_1024_DPTACT0_ADLW0.1_GMCFG5_24kHz
 - VOICE/MBExWN_SIIConv_V71g_VOICE2_WNCHA340_IMP0_WNCHA340_IMPORTmod_MCFG0_WNCHA340_DCHA32_1024_DPTACT0_ADLW0.1_GMCFG0_24kHz
 ```

To select a model you don't need to  provide the full model specification.
Any sub string of the model name will be sufficient. Accordingly, the specification `VOICE` in the 
example above selects the third model. The search of models is performed in the order of the lost and the first 
model containing the model id strig is returned. Aurrently, the three model names starting with the strings  
**SING**, **SPEECH** and **VOICE** correspond to the three models **MW-SI-FD**, **MW-SP-FD**, and **MW-VO-FD** 
from the [paper](https://www.mdpi.com/2078-2489/13/3/103) respectively.

By default `resynth_mel.py` will run on the CPU limiting the number of thread to 2. This default configuration can be 
changed using the following two command line arguments:

```
  --num_threads NUM_THREADS : selects the number of cpu threads (Default: 2)
  --use_gpu                 : performs the inference on the gpu
```

Please see `resynth_mel.py --help` for other command line arguments.

#### 3.2.2 Python package

You may also use MBExWN as a python package. The principale operation is as fllows

```python

# import the class
from MBExWN_NVoc import mel_inverter, list_models, mbexwn_version
from MBExWN_NVoc.fileio import iovar as iov

# instantiate, giving as argument a model_id, which is the same string you use to select a model on the command line
# with resynth_mel
MelInv = mel_inverter.MELInverter(model_id_or_path=model_id)
# load mel spectrogram from a file 
dd = iov.load_var(mell_file)
# properly scale the mel spectrogram, scale_mel need the full dictionary to know the 
# analysis parameters that have been used to generate the mell spectrogram
log_mel_spectrogram = MelInv.scale_mel(dd, verbose=verbose)
# synthesize the audio
syn_audio = MelInv.synth_from_mel(log_mel_spectrogram)
```

For an example please see the source code of the script `resynth_mel`.

<a name="S4"></a>
## 4. Further info


<a name="S4-1"></a>
### 4.1 Computational costs

The mel inverter is sufficiently efficient to perform resynthesis from mel spectrograms two times faster than 
real-time on a single core of a Laptop CPU.  

On a GPU the vocoder achieves audio synthesis up to 200 times faster than real time. Note that tensorflow up to version 
2.8 is apparently performing an automatic selection of the best performing kernel each time a new shape of a Conv1D or 
Conv2D operator is encountered. Therefore, MBExWN will only achieve optimal efficiency whenever the shape of a signal
is encountered for the second time (see  [here](https://github.com/tensorflow/tensorflow/issues/54456) for 
more information).

<a name="S4-2"></a>
### 4.2 How to cite

In case you use the code or the models for your own work please cite 

```
Roebel, Axel, and Frederik Bous. 2022. 
   "Neural Vocoding for Singing and Speaking Voices with the Multi-Band Excited WaveNet" 
   Information 13, no. 3: 103. https://doi.org/10.3390/info13030103 
```

<a name="S4-3"></a>
### 4.3 Acknowledgements

This research was funded by ANR project ARS, grant number ANR-19-CE38-0001-01 and 
computation were performed using HPC resources from GENCI-IDRIS (Grant 2021-AD011011177R1).

Thanks to 

- [TensorflowTTS](https://github.com/TensorSpeech/TensorflowTTS) from which the MBExWN implementation 
  has gathered an initial version of the PQMF implementation in 
  `MBExWN_NVoc/vocoder/model/tf_preprocess.py`.

- [Magenta DDSP](https://github.com/magenta/ddsp) from which the MBExWN implementation has gathered 
  the method `PulseWaveTable.stable_cumsum_and_wrap` in `MBExWN_NVoc/vocoder/model/tf_wavetable.py`.

<a name="S4-4"></a>
### 4.4 ChangeLog

- Version 1.2.3  (2022/06/10)
  - Fixed uninitialized variable.
  - Fixed missing import

- Version 1.2.2  (2022/04/10)
  - fixed display of mel spectrogram differences when mel spectra do not have exactly the same number of frames.
  - Fixed errors in function documentation.

- Version 1.2.1  (2022/03/23)
  - Fixed many problems with missing imports.
  - More consistent behavior for model selection with the command line scipts.

- Version 1.2.0  (2022/03/05)
  - Initial release.

<a name="S4-5"></a>
### 4.5 Copyright

Copyright (c) 2022 IRCAM


<p align="center">
<a href="https://www.ircam.fr/"> <img src="img/IRCAM.CP.jpg" width="20%"></a>
&nbsp;
<img src="img/cnrs.png" width="10%">
&nbsp;
<a href="http://www.idris.fr/jean-zay/"> <img src="img/LOGO_SU_HORIZ_SEUL_CMJN.png" width="20%"></a>
</p>
