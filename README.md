#  KIT-LSTM:
## Table of contents

* [General info](#general-info)
* [Setup](#setup)
* [Original Data](#Original-Data)
* [Hyperparameters](#Hyperparameters)
* [Status](#status)
* [Contact](#contact)
 
## General info
This is the official repository of "L. Liu, V. Ortiz-Soriano, J. Neyra and J. Chen, "KIT-LSTM: Knowledge-guided Time-aware LSTM for Continuous Clinical Risk Prediction," in 2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Las Vegas, NV, USA, 2022 pp. 1086-1091.". Link: https://www.computer.org/csdl/proceedings-article/bibm/2022/09994931/1JC2nGMB5PW

For additional experimental results, please see our medRxiv paper link at: https://www.medrxiv.org/content/10.1101/2022.11.14.22282332v1

## Setup
- Python 3.7
- Pytorch 1.7.1

## Original Data
User can request the real-world data through UK CCTS (University of Kentucky, Center for Clinical and Translational Science) with approved IRB.

## Hyperparameters
We choose the hyperparameters that delivered the best performance for the validation set by doing random search for all methods.
The resulting hyperparameters for KIT-LSTM in the paper are: the size of hidden states is 8, the learning rate is 0.01, the batch size is 256, the dropout rate is 0.2.

## Status
Project is: _in progress_ for final cleaning and organizing   

## Contact
Created by Lucas J. Liu (jli394@uky.edu) - feel free to contact me!

 
