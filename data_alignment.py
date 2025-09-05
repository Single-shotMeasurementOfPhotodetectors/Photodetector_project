from typing import List
import scipy.io
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


class sampleAlignment:
    
    def __init__(
        self, 
        oversampling_ratio: int, 
        sample_match_indices: List[int], 
        shifts: List[int], 
        data_type: str
    ):
        """
        Initialize reconstruction parameters.
    
        Parameters
        ----------
        oversampling_ratio : int
            Oversampling factor of the output relative to the input bits.
        sample_match_indices : List[int]
            Alignment indices for each section. If not provided, 
            the algorithm will estimate alignment automatically 
            via correlation.
        shifts : List[int]
            Manual shift adjustments for each section. While automatic 
            alignment provides an approximate match, fine-tuning may be 
            required to achieve precise input-output alignment.
        data_type : str
            Type of the data (e.g., 'training', 'test').
        """

        self.oversampling = oversampling_ratio
        self.sample_match = sample_match_indices
        self.shift = shifts
        self.type = data_type

        
    def sampleAlign(self, sample_input: List[int], sample_output: List[List[float]]):
        """
        Align (oversampled) output samples to input bits and save processed data.
    
        Parameters
        ----------
        sample_input : List[int]
            Original input bits.
        sample_output : List[List[float]] or List[float]
            Output samples to be aligned.
        """

        # Determine number of sections
        secs = len(sample_output)
        if secs > 100:
            secs = 1  # For single long lists
    
        sample_period = len(sample_input)
    
        for sec in range(secs):
            # Extract the current section
            sample_output_sec = sample_output[sec] if secs > 1 else sample_output
    
            # Determine alignment index
            if self.sample_match:
                # Use predefined alignment if available
                sample_match_idx = self.sample_match[sec]
            else:
                # Automatic alignment via correlation
                mean_out = sum(sample_output_sec) / len(sample_output_sec)
                sample_output_centered = [ele - mean_out for ele in sample_output_sec]
    
                nperiod = len(sample_output_centered) // sample_period
                # Truncate and reshape for averaging
                sample_output_period = np.mean(
                    np.array(sample_output_centered[:nperiod * sample_period]).reshape(nperiod, sample_period),
                    axis=0
                )
    
                # Center input: map {0,1} -> {-1,1} and repeat
                sample_input_centered = [2 * ele - 1 for ele in sample_input] * 2
    
                # Compute correlation to find best alignment
                corr = [
                    np.sum(np.array(sample_input_centered[i:i+sample_period]) * np.array(sample_output_period))
                    for i in range(sample_period)
                ]
                sample_match_idx = corr.index(max(corr))
    
            # Apply manual shift if specified
            shift_idx = self.shift[sec]
            sample_match_start = sample_match_idx + shift_idx
    
            # Compute samples to delete to align with oversampling
            if sample_match_start % self.oversampling != 0:
                sample_end = ((sample_match_start // self.oversampling) + 1) * self.oversampling - 1
                sample_to_delete = sample_end - sample_match_start + 1
            else:
                sample_end = sample_match_start - 1
                sample_to_delete = 0
    
            # Truncate output to aligned section
            sample_output_sectioned = sample_output_sec[sample_to_delete:]
            sample_output_len = (len(sample_output_sectioned) // self.oversampling) * self.oversampling
            sample_output_sectioned = sample_output_sectioned[:sample_output_len]
            bit_output_len = sample_output_len // self.oversampling
    
            # Prepare input aligned to output
            sample_input_head = sample_input[sample_end+1:]
            sample_input_head = sample_input_head[:(len(sample_input_head) // self.oversampling) * self.oversampling]
    
            nperiod_input = (sample_output_len - len(sample_input_head)) // sample_period
            sample_input_sectioned = sample_input_head + sample_input * nperiod_input
            sample_input_sectioned += sample_input[:sample_output_len - len(sample_input_sectioned)]
    
            # Average over oversampling to get bit-level input
            bit_input_sectioned = [
                int(ele) for ele in np.mean(
                    np.array(sample_input_sectioned).reshape(bit_output_len, self.oversampling), axis=1
                )
            ]
    
            # Plot to visually check the alignment:
            # - Input value 1 corresponds to a rising trend in the output
            # - Input value 0 corresponds to a falling/decaying trend in the output
            # - self.shift can be adjusted to fine-tune the alignment
            fig, ax1 = plt.subplots(figsize=(20, 12))
            color = 'tab:red'
            ax1.set_xlabel('sample index')
            ax1.set_ylabel('sample level output', color=color)
            idx_list = list(np.arange(0,500,1))
            ax1.plot(list(np.array(sample_output_sectioned)[idx_list]), color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.xaxis.set_major_locator(MultipleLocator(160))
            ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax1.grid(which='major', color='#CCCCCC', linestyle='--')
            ax1.grid(which='minor', color='#CCCCCC', linestyle=':')
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('sample level input', color=color) # by sample level input, we mean oversampled bits
            ax2.plot(list(np.array(sample_input_sectioned)[idx_list]), color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()
            plt.rc('axes', labelsize=20, titlesize=20)
            plt.show()
    
            # Save aligned bit input
            bit_input_file = f'data_processed/bit_input_{self.type}{"_sec"+str(sec) if secs>1 else ""}.csv'
            with open(bit_input_file, 'w', newline='') as csvfile:
                csv.writer(csvfile).writerow(bit_input_sectioned)
    
            # Save aligned sample output
            sample_output_file = f'data_processed/sample_output_{self.type}{"_sec"+str(sec) if secs>1 else ""}.csv'
            with open(sample_output_file, 'w', newline='') as csvfile:
                csv.writer(csvfile).writerow(sample_output_sectioned)
    
        print(f"Data for {self.type} have been pre-processed and saved")

    
    
    def DeletePilots(self, bit_input: List[int], bit_input_pattern: List[int], sample_output: List[int], npadded: int):
        # test data is collected with padding zeros at the head
        pattern_len = len(bit_input_pattern)
        padded_len = pattern_len + npadded
        input_len = len(bit_input)
        for imatch in range(len(bit_input) - pattern_len):
            bit_input_sec = bit_input[imatch:imatch+pattern_len]
            corr = np.sum(np.array(bit_input_sec) - np.array(bit_input_pattern))
            if corr == 0:
                break
        bit_delete = []
        if imatch >= npadded:
            bit_delete.extend(range(imatch-npadded,imatch))
        else:
            bit_delete.extend(range(imatch))
        i = 0
        while i*padded_len + imatch + pattern_len < input_len:
            bit_delete.extend(range(i*padded_len+imatch+pattern_len,min(i*padded_len+imatch+pattern_len+npadded,input_len)))
            i += 1
        sample_delete = []
        for ele in bit_delete:
            for item in range(ele*self.oversampling, (ele+1)*self.oversampling):
                sample_delete.append(item)
            
        updated_bit_input = [value for index, value in enumerate(bit_input) if index not in bit_delete]
        updated_sample_output = [value for index, value in enumerate(sample_output) if index not in sample_delete]
        
        bit_input_file = 'data_processed/bit_input_without_pilot_' + self.type + '.csv';
        with open(bit_input_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)           
            csv_writer.writerow(updated_bit_input)
        
        sample_output_file = 'data_processed/sample_output_without_pilot_' + self.type + '.csv';
        with open(sample_output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)           
            csv_writer.writerow(updated_sample_output)
        

#%% ----------------------------------------------------------------------
# User-adjustable parameters for measurement and processing settings
# ----------------------------------------------------------------------

# -------------------------------
# Parametric estimation stage
# -------------------------------
oversampling_est = 8  # Oversampling ratio during parametric estimation
# Indices of collected sample-level outputs for alignment in each section
# sample_match_indices_est = [133, 2462, 1294, 682, 2922, 2858, 1522, 568]
# If alignment is unknown, leave as an empty list:
sample_match_indices_est = []
# number of estimation data sections
nsec_est = 8 
# Shift values to fine-tune input-output alignment (one per section)
sample_shifts_est = [3] * nsec_est

# -------------------------------
# Viterbi detection stage
# -------------------------------
oversampling_det = 4  # Oversampling ratio during Viterbi detection
# Indices of collected sample-level outputs for alignment in each section
# sample_match_indices_det = [1428, 567, 495]
# If alignment is unknown, leave as an empty list:
sample_match_indices_det = []
# number of detection data sections
nsec_det = 3
# Shift values to correct correlation-based alignment
sample_shifts_det = [1] * nsec_det

# -------------------------------
# Zero-padding for testing data
# -------------------------------
# Number of zeros added to assist alignment in testing data;
# these will be removed during processing
npadded = 20



#%% --------------------------------------------------------------------
# Preprocessing for Estimation (Training) Data
# ----------------------------------------------------------------------

# Load sample-level output data for training (estimation)
est_output = scipy.io.loadmat('raw_data_for_alignment/training_output_sample_pix1.mat')
sample_output_est = [list(ele) for ele in est_output['y']]

# Load input sequence pattern (one period)
est_input_file = 'raw_data_for_alignment/training_input.csv'
with open(est_input_file, newline='') as f:
    bit_input_est = [int(ele) for ele in next(csv.reader(f))]

# Upsample input sequence according to the oversampling ratio
sample_input_est = [bit for bit in bit_input_est for _ in range(oversampling_est)]

# Initialize sample alignment object for estimation data
estimation_data = sampleAlignment(
    oversampling_ratio=oversampling_est,
    sample_match_indices=sample_match_indices_est,
    shifts=sample_shifts_est,
    data_type='training'
)

# Align samples and save preprocessed estimation data
estimation_data.sampleAlign(sample_input_est, sample_output_est)


#%% --------------------------------------------------------------------
# Preprocessing for Detection (Testing) Data
# ----------------------------------------------------------------------

# Load sample-level output data for testing (detection)
det_output = scipy.io.loadmat('raw_data_for_alignment/test_output_sample_pix1.mat')
sample_output_det = [list(ele) for ele in det_output['y']]

# Process each detection section separately
for detect_sec in range(nsec_det):
    # Load input sequence for this section
    det_input_file = f'raw_data_for_alignment/test_input_sec{detect_sec}.csv'
    with open(det_input_file, newline='') as f:
        bit_input_det = [int(ele) for ele in next(csv.reader(f))]
    
    # Upsample input sequence according to oversampling for detection
    sample_input_det = [bit for bit in bit_input_det for _ in range(oversampling_det)]
    
    # Use section-specific alignment indices and shifts
    sample_match_indices_det_temp = [sample_match_indices_det[detect_sec]] if sample_match_indices_det else []
    sample_shifts_det_temp = [sample_shifts_det[detect_sec]]
    
    # Initialize sample alignment object for this detection section
    detection_data = sampleAlignment(
        oversampling_ratio=oversampling_det,
        sample_match_indices=sample_match_indices_det_temp,
        shifts=sample_shifts_det_temp,
        data_type=f'test_sec{detect_sec}'
    )
    
    # Align output samples for this section
    sample_output_det_temp = [int(ele) for ele in sample_output_det[detect_sec]]
    detection_data.sampleAlign(sample_input_det, sample_output_det_temp)
    
    # Handle padded pilots: remove initial npadded bits from input pattern
    bit_input_det_pattern = bit_input_det[npadded:]
    
    # Retrieve saved aligned input and output data
    bit_input_file = f'data_processed/bit_input_test_sec{detect_sec}.csv'
    with open(bit_input_file, newline='') as f:
        bit_input_det_aligned = [int(ele) for ele in next(csv.reader(f))]
    
    sample_output_file = f'data_processed/sample_output_test_sec{detect_sec}.csv'
    with open(sample_output_file, newline='') as f:
        sample_output_det_aligned = [int(ele) for ele in next(csv.reader(f))]
    
    # Delete padded pilot bits and update aligned sequences
    detection_data.DeletePilots(
        bit_input_det_aligned,
        bit_input_det_pattern,
        sample_output_det_aligned,
        npadded
    )
