# EPFL COM-500 Mini-Project: Group Throughput Optimization in Wireless Communications via Spatial Beamforming

### Author: [Sun Haoxin](https://github.com/HaoxinSEU), [Xu Zewei](https://github.com/xuzewei28) and [Wu Yujie](https://github.com/YuJieWU99)


## Introduction
This is a repo for EPFL COM-500 SSP mini-project: Beamforming for Throughput Optimization in 5G.

We use two different ways to do beamforming for a MISO downlink communication system: matched beamforming and flexibeam.



## Project Structure
```
- data
  - data_1.mat         % DOA estimation results of dataset 1
  - data_2.mat         % DOA estimation results of dataset 2
  - original_data     % real signal received by a base station
    - data_1.npz      % dataset 1
    - data_2.npz      % dataset 2
    - README.txt
- MATLAB              % matlab source code
  - flexibeam.m       % flexibeam
  - matched_beamforming.m   % matched beamforming
- python-Notebook     % python notebook source code
  - doa.ipynb         % DOA estimation
  - flex_beam.ipynb   % flexibeam
  - matched_beam.ipynb  % matched beamforming
  - MVDR_beam.ipynb   % MVDR beamforming
  - Simulation.ipynb  % simulation with flexibeam and matched beamforming
- report.pdf          % our report for these two methods
```


## References
1. P. Hurley and M. Simeoni, "Flexibeam: Analytic spatial filtering by beamforming," 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2016, pp. 2877-2880, doi: 10.1109/ICASSP.2016.7472203.
2. Simeoni, Matthieu & Hurley, Paul. (2016). Beamforming towards regions of interest for multi-site mobile networks. 
3. Krim, H., & Viberg, M. (1996). Two decades of array signal processing research: the parametric approach. IEEE signal processing magazine, 13(4), 67-94.
