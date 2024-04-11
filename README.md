# Measurement of Photodetector -- Operational Codes
This repository contains the sample pixel data and operational codes for the paper submission titled "Verifiable Single-Shot Measurement of Photon Detectors".

## Data Pre-processing and Sample Pixel Data
We offer code for pre-processing the collected data, using pixel 1 as an illustrative example. Relevant documents can be found in the folder `data_preprocessing`.

### Data for Training and Testing
Both training (estimation process) and testing (Viterbi detection process) data are located in the folder `data_preprocessing/raw_data_for_alignment`. The files in this folder include:

1. **training_input.csv**: a 420-bit binary sequence, repeated to generate training outputs.
2. **training_output_sample_pix1.mat**: collected sample-level outputs used for parameter estimation. Each input bit corresponds to 8 sample-level outputs due to an oversampling ratio of 8.
3. **test_input_secX.csv**: ground truth for testing, composed of three sections of random binary input sequences used to generate test outputs. Zeros are added before the random bits to facilitate smooth data alignment, and they will be removed during testing.
4. **test_output_sample_pix1.mat**: collected sample-level outputs used for sequence detection. Each input bit corresponds to 4 sample-level outputs due to an oversampling ratio of 4.


### Data Pre-processing Code

Run ``data_alignment.py`` to align the input bits and output samples based on correlation and manual checks. Plots indicating the alignment will be generated, and the aligned data will be saved to the folder `data_processed`.

Saved files include:

1. **bit_input_training_secX.csv**: Aligned input bits for parameter estimation, consisting of 8 sections. In each section, the first input bit is aligned to the first 8 output samples in the corresponding sample-level outputs.
2. **sample_output_training_secX.csv**: Aligned output samples for parameter estimation, consisting of 8 sections.
3. **bit_input_test_secX.csv**: Aligned input bits for sequence detection, consisting of 3 sections. In each section, the first input bit is aligned to the first 4 output samples in the corresponding sample-level outputs.
4. **sample_output_test_secX.csv**: Aligned output samples for sequence detection, consisting of 3 sections.
5. **bit_input_without_pilot_test_secX.csv**: Input bits for detection after removal of padded zeros from **bit_input_test_secX.csv**.
6. **sample_output_without_pilot_test_secX.csv**: Output samples for detection after removal of outputs associated with padded zeros from **sample_output_test_secX.csv**.

Additionally, we include **bit_input_test_secX.csv** and **sample_output_test_secX.csv**, which can also be used in the detection process. Including padded zero sections in these files facilitates a simpler detection process compared to the method presented in the paper.