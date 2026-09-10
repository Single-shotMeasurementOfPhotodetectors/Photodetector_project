# Single-shot Measurement of Photodetectors -- Operational Codes
This repository contains the sample pixel data and operational codes for the paper submission titled "Single-shot measurement of semiconductor photodetectors". In this repository, photodetectors are referred to as pixels because they serve as the individual elements of a focal plane array.

## All-in-One Demo Code for Output Reconstruction

For readers who wish to **quickly reproduce the plots** without first delving into the underlying code or datasets, we recommend the following steps:

1. **Download** the folder `raw_data_for_alignment`, which contains the sample inputs and measurements for pixel 1.  
2. **Create** a folder named `data_processed`, which will store the processed data.  
3. **Download and run** the script `output_reconstruction_all_in_one.py`.  

During execution, the script will display **prompt messages** indicating the procedures being carried out. Relevant estimated parameters will also be shown in the output messages.  

Upon completion, the program generates a figure that **reconstructs (predicts) the output samples**  using parameters extracted from both single-shot and traditional measurements, and compares them with the ground-truth measurements. We present a Monte Carlo simulation of such outputs to highlight similarity (Figure 4), whereas in this code, we show one sample output.



## Data Pre-processing and Sample Pixel Data
We offer code for pre-processing the collected data, using pixel 1 as an illustrative example. 

### Data for Training and Testing
Both training (estimation process) and testing (Viterbi detection process) data are located in the folder `raw_data_for_alignment`. The files in this folder include:

1. `training_input.csv`: A 420-bit binary sequence, repeated to generate training outputs.
2. `training_output_sample_pix1.mat`: Collected sample-level outputs used for parameter estimation. Each input bit corresponds to 8 sample-level outputs due to an oversampling ratio of 8.
3. `test_input_secX.csv`: Ground truth for testing, composed of three sections of random binary input sequences used to generate test outputs. 
Zeros are added before the random bits to facilitate smooth data alignment, and they will be removed during testing.
4. `test_output_sample_pix1.mat`: Collected sample-level outputs used for sequence detection. Each input bit corresponds to 4 sample-level outputs due to an oversampling ratio of 4.


### Data Pre-processing Code

Run `data_alignment.py` to align the input bits and output samples based on correlation and manual checks. The aligned data will be saved to the folder `data_processed`.

Saved files include:

1. `bit_input_training_secX.csv`: Aligned input bits for parameter estimation, consisting of 8 sections. In each section, the first input bit is aligned to the first 8 output samples 
in the corresponding sample-level outputs.
2. `sample_output_training_secX.csv`: Aligned output samples for parameter estimation, consisting of 8 sections.
3. `bit_input_test_secX.csv`: Aligned input bits for sequence detection, consisting of 3 sections. In each section, the first input bit is aligned to the first 4 output samples in the 
corresponding sample-level outputs.
4. `sample_output_test_secX.csv`: Aligned output samples for sequence detection, consisting of 3 sections.
5. `bit_input_without_pilot_test_secX.csv`: Input bits for detection after removal of padded zeros from `bit_input_test_secX.csv`.
6. `sample_output_without_pilot_test_secX.csv`: Output samples for detection after removal of outputs associated with padded zeros from `sample_output_test_secX.csv`.

We use `bit_input_without_pilot_test_secX.csv` and `sample_output_without_pilot_test_secX.csv` in our tests, but include `bit_input_test_secX.csv` and `sample_output_test_secX.csv`, 
which can also be used in the detection process. Including padded zero sections in these files facilitates a simpler detection process compared to the tests presented in the paper.


## Operational Codes

### Estimation, Reconstruction and Detection
Run `Estimation_regeneration_detection` after implementing the data alignment. The code takes the following tunable values in addition to the saved data for training and testing:

1. `m`: Grouping size in Stage 1 of the estimation problem. This value is set to 20 in our case.
2. `oversampling_training`: Oversampling ratio for the estimation process. This value should match the actual data collection process.
3. `conv_factor`: The collected outputs are in the unit of counts. This value translates the counts to the number of particles. 
If the collected outputs are already measured in the number of particles, this value should be set to 1.
4. `eta_set`: The collected outputs are subject to an arbitrary offset due to the collection process. This value is to be subtracted during the implementation. 
In other words, if the saved data output is X counts, the number of particles is `(X - eta_set) * conv_factor`. If there is no such offset, this value should be set to 0.
5. `ts`: Sampling interval, in seconds.
6. `max_viterbi_bit`: Maximum constraint length in the Viterbi decoding, set to 15. If the constraint length is larger than 15, it is cut off at 15.
7. `alpha_traditional`: Input intensity (particles/sec) obtained from the traditional method.
8. `delta_traditional`: Dark current intensity (particles/sec) obtained from the traditional method.
9. `td_traditional`: Detector response time (seconds) obtained from the traditional method.
10. `mu_traditional`: Gain obtained from the traditional method.
11. `sigma_traditional`: Sample-level noise (particles) obtained from the traditional method.

After all operations, the obtained results will be saved to the folder `results`. Specifically, the saved results include:

1. `results/estimation/estimated_parameters.csv`: This file contains the estimated parameter values of the proposed method.
2. `results/reconstruction/reconstructed_data.csv`: Here, the reconstructed sample-level outputs based on parameters of the proposed/traditional method, as well as the actual outputs, are saved.
3. `results/reconstruction/ROC.csv`: Save the True Positive (TP) rates and False Positive (FP) rates of the reconstructed and actual outputs.
4. `results/reconstruction/AUR.csv`: This file stores the Area Under the ROC (AUR) curves calculated based on the ROC curves.
5. `results/detection/error_rates.csv`: Contains the testing process results during the sequence detection, including accuracy, false positive rates, false negative rates, true positive rates, and true negative rates.

