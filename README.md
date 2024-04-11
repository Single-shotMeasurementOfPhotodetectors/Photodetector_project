# Photodetector_project
This repository includes sample pixel data and operational codes for the Paper submission "Verifiable single-shot measurement of photon detectors".

## Data pre-processing and sample pixel data
We provide the code to pre-process the collected data using pixel 1 as an illustrative example. Relavant documents can be found in the folder "data_preprocessing".
Collected data for both training (estimation process) and testing (Viterbi detection process) are under the folder "data_preprocessing/raw_data_for_alignment". 
Files include the following:
1. Dark current measurements: to estimate the system offset
2. Photodetector output corresponding to designed input pattern: to extract estimates of all parameters
3. A designed input pattern to the photodetector: support the estimation process
4. Photodetector output corresponding to random inputs: to test the extracted parameter estimates
5. Group truth of the random inputs: to help evaluate the detect results

## Data pre-processing code
We provide code to perform the data pre-processing

## Main operational code
We provide following codes for different purposes:
1. Estimation code: extract parameter estimations
2. Reconstruction code: sythesize outputs of a detector based on the parameter estimates
3. Detection code: do the sequence detection based on the extracted parameters
