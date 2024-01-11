# Photodetector_project
This repository includes sample data sets and operational results, data pre-processing code, and main operational code.

## Sample data sets
We provide data sets for 10 pixels. Each pixel has following data sets:
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
