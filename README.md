# EC-DL
Official code for Deep Learning-based Narrow-Band Imaging Endocytoscopic Classification for Predicting Colorectal Lesions: A Retrospective Study (Nature Communications).

## Overview
We present a computer-aided diagnosis (CAD) model for colorectal lesion classification using narrow-band imaging endocytoscopy (EC-NBI). Inspired by progressive pre-training strategies in large language models, our approach integrates generalized and in-domain pre-training with supervised deep clustering. Evaluated on an independent cohort, the model outperforms state-of-the-art supervised methods at both image and lesion levels, surpassing even experienced endoscopists in human–machine competitions. By enhancing diagnostic accuracy and consistency, this CAD system advances the clinical utility of EC-NBI and supports the goal of optical biopsy.<br><br>

<div align="center">
  <p>
    <img src="https://github.com/CorleoneJW/EC-DL/blob/main/readme_src/cover.png" alt="Cover"/>
  </p>
</div>

## System Requirements

### Hardware
**Here are the recommended hardware conditions:**
<p>
  <img src="https://img.shields.io/badge/GPU-RTX_3090-green" />
  <img src="https://img.shields.io/badge/CPU-Intel(R)_Xeon(R)_Gold_6133-green" />
</p>

### Software
<p>
  <img src="https://img.shields.io/badge/Ubuntu-22.04.3-red" />
  <img src="https://img.shields.io/badge/Visual_Studio_Code-1.99.3-green" />
  <img src="https://img.shields.io/badge/Python-3.11.0-blue" />
  <img src="https://img.shields.io/badge/Pytorch-2.3.0-blue" />
  <img src="https://img.shields.io/badge/Cudatoolkit-11.8.0-blue" />
  <img src="https://img.shields.io/badge/Timm-0.4.12-blue" />
  <img src="https://img.shields.io/badge/Baseline_timm-1.0.15-blue" />
  <img src="https://img.shields.io/badge/Pandas-1.5.3-blue" />
  <img src="https://img.shields.io/badge/Matplotlib-3.8.4-blue" />
</p>

## Installation Guide
Git clone the project (replaceable with SSH).<br>
```
$ git clone https://github.com/CorleoneJW/EC-DL.git
```
Install dependencies.<br>
```
$ conda env create -f environment.yml -n myenv
```
Prepare the data before running code.<br>

**Typical install time on a normal desktop computer: above 15 minutes.**

### Pretraining Stage
Run python stage_pretrain.py to pretrain the model.<br>
```
$ cd journalway
$ python stage_pretrain.py --data_path pretraining_data_path --finetune preload_checkpoint.pth
```
Parameters are transmitted through cmd (see details in get_args function).

### Finetuning Stage
Run python stage_finetune.py to finetune the model.<br>
```
$ cd journalway
$ python stage_finetune.py --data_path finetuning_data_path --finetune preload_checkpoint.pth
```

You could try on the demo dataset using default config.<br>
```
$ cd journalway
$ python stage_finetune.py
```

## Expected run time
Based on the recommended hardware configuration, it takes **10 ms** to process one EC image after the program starts running.

## License
This project is covered under the <a href="https://github.com/CorleoneJW/EC-DL/blob/main/LICENSE">GPL-3.0 license</a>.
