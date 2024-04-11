from typing import List
import scipy.io
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


class sampleAlignment:
    
    def __init__(self, oversampling_ratio, sample_match_indices: List[int], shifts: List[int], data_type: str):
        # suppose the alignment is known and fixed
        self.oversampling = oversampling_ratio
        self.sample_match = sample_match_indices
        self.shift = shifts
        self.type = data_type
        
    def sampleAlign(self, sample_input: List[int], sample_output: List[List[float]]):        
        # suppose the alignment is unknown for all sections of output samples
        secs = len(sample_output)
        # we only expect a few sections of output data, if secs is large, it indicates there is only one section
        if secs > 100:
            secs = 1
        sample_period = len(sample_input)
        for sec in range(secs):
            if secs > 1:
                sample_output_sec = sample_output[sec]
            else:
                sample_output_sec = sample_output
            
            # if fixed, directly employ the correct match
            if self.sample_match:
                sample_match_idx = self.sample_match[sec]
            else:
                # recenter the input and output to calculate the correlation
                mean_sample_output = sum(sample_output_sec)/len(sample_output_sec)
                sample_output_sec_centered = [ele - mean_sample_output for ele in sample_output_sec]
                nperiod = int(np.floor(len(sample_output_sec)/sample_period))
                sample_output_sec_period = sample_output_sec_centered[:nperiod*sample_period]
                sample_output_sec_period = np.array(sample_output_sec_period).reshape(nperiod,sample_period)
                sample_output_sec_period = list(np.mean(sample_output_sec_period, axis = 0))
                sample_input_centered = [ele*2-1 for ele in sample_input]*2
                # create an empty list to store the correlation
                corr = []
                for sample_idx in range(sample_period):
                    sample_input_sec = sample_input_centered[sample_idx:sample_idx+sample_period]
                    corr.append(np.sum(np.array(sample_input_sec) * np.array(sample_output_sec_period)))
                sample_match_idx = corr.index(max(corr))
            shift_idx = self.shift[sec]

            sample_match_start = sample_match_idx + shift_idx
            if sample_match_start % 8 != 0:
                sample_end = (int(np.floor(sample_match_start)/self.oversampling) + 1)*self.oversampling - 1;
                sample_to_delete = sample_end - sample_match_start + 1
            else:
                sample_end = sample_match_start-1
                sample_to_delete = 0
            
            sample_output_sectioned = sample_output_sec[sample_to_delete:]
            sample_output_len = int(np.floor(len(sample_output_sectioned)/self.oversampling)*self.oversampling)
            bit_output_len = int(sample_output_len/self.oversampling)
            sample_output_sectioned = sample_output_sectioned[:sample_output_len]
            sample_input_sectioned_head = sample_input[sample_end+1:]
            sample_input_sectioned_head = sample_input_sectioned_head[:int(np.floor(len(sample_input_sectioned_head)/self.oversampling))*self.oversampling]
            nperiod_input = int(np.floor((sample_output_len - len(sample_input_sectioned_head))/sample_period))
            sample_input_sectioned = sample_input_sectioned_head + sample_input*nperiod_input
            sample_input_sectioned = sample_input_sectioned + sample_input[:sample_output_len - len(sample_input_sectioned)]
            bit_input_sectioned = list(np.mean(np.array(sample_input_sectioned).reshape(bit_output_len,self.oversampling), axis = 1))
            bit_input_sectioned = [int(ele) for ele in bit_input_sectioned]
            
            
            # plot to check the alignment
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
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('sample level input', color=color)  # we already handled the x-label with ax1
            ax2.plot(list(np.array(sample_input_sectioned)[idx_list]), color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()
            plt.rc('axes', labelsize=20, titlesize=20)
            plt.show()
            
            
            if secs > 1:
                bit_input_aligned = 'data_processed/bit_input_' + self.type + '_sec' + str(sec) + '.csv';
            else:
                bit_input_aligned = 'data_processed/bit_input_' + self.type + '.csv';
            with open(bit_input_aligned, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)           
                csv_writer.writerow(bit_input_sectioned)
                
            if secs > 1:
                sample_output_aligned = 'data_processed/sample_output_' + self.type + '_sec' + str(sec) + '.csv';
            else:
                sample_output_aligned = 'data_processed/sample_output_' + self.type + '.csv';
            with open(sample_output_aligned, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)           
                csv_writer.writerow(sample_output_sectioned)
                
        print("Data for " + self.type + " have been pre-processed and saved")
    
    
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
        


# preprocessing for estimation data
# define the oversampling ratio
oversampling_est = 8
# correct the correlation based alignment
shift = 1

# training (estimation) data alignment
# data is periodic with periods specified by input sequence
est_output = scipy.io.loadmat('raw_data_for_alignment/training_output_154fW_34836uS_sample_pix1.mat')
sample_output_est = list(est_output['y'])
sample_output_est = [list(ele) for ele in sample_output_est]

# input sequence pattern (one period)
est_input = 'raw_data_for_alignment/training_input.csv'
with open(est_input, newline='') as bit_input_est:
    bit_input_est = list(csv.reader(bit_input_est))[0]
bit_input_est = [int(ele) for ele in bit_input_est]    
sample_input_est = [ele for ele in bit_input_est for i in range(oversampling_est)]        

# we collect 8 sections of sample-level outputs, these are indices recorded for alignment in each section
sample_match_indices_est = [133, 2462, 1294, 682, 2922, 2858, 1522, 568]
# sample_match_indices_est = []
sample_shifts_est = [2 + shift]*8 # need to manually verify the alignment
estimation_data = sampleAlignment(oversampling_est, sample_match_indices_est, sample_shifts_est, 'training')
estimation_data.sampleAlign(sample_input_est, sample_output_est)


# preprocessing for detection data 
# define the oversampling ratio
oversampling_det = 4

# testing (detection) data alignment
# data is periodic with periods specified by input sequence
det_output = scipy.io.loadmat('raw_data_for_alignment/test_output_154fW_34836uS_sample_pix1.mat')
sample_output_det = list(det_output['y'])

# input sequence pattern (one period)
sample_match_indices_det = [1428, 567, 495]
sample_shifts_det = [0 + shift]*3 #0
npadded = 20
for detect_sec in range(3):
    # Load data, combine the sequence (although they are not well correlated)
    det_input = 'raw_data_for_alignment/test_input_sec' + str(detect_sec) + '.csv';
    with open(det_input, newline='') as bit_input_det:
        bit_input_det = list(csv.reader(bit_input_det))[0]
    bit_input_det = [int(ele) for ele in bit_input_det]  
    sample_input_det = [ele for ele in bit_input_det for i in range(oversampling_det)]
    
    sample_match_indices_det_temp = [sample_match_indices_det[detect_sec]]
    sample_shifts_det_temp = [sample_shifts_det[detect_sec]]
    
    detection_data = sampleAlignment(oversampling_det, sample_match_indices_det_temp, sample_shifts_det_temp, 'test_sec' + str(detect_sec))
    sample_output_det_temp = sample_output_det[detect_sec]
    sample_output_det_temp = [int(ele) for ele in sample_output_det_temp]
    detection_data.sampleAlign(sample_input_det, sample_output_det_temp)


    # suppose there are padded pilots delete
    bit_input_det_pattern = bit_input_det[npadded:]
    # retrieve the saved the output samples and input bits with padded patterns
    bit_input_file = 'data_processed/bit_input_test_sec' + str(detect_sec) + '.csv';
    with open(bit_input_file, newline='') as bit_input_det_aligned:
        bit_input_det_aligned = list(csv.reader(bit_input_det_aligned))[0]
    bit_input_det_aligned = [int(ele) for ele in bit_input_det_aligned]  
    sample_output_file = 'data_processed/sample_output_test_sec' + str(detect_sec) + '.csv';
    with open(sample_output_file, newline='') as sample_output_det_aligned:
        sample_output_det_aligned = list(csv.reader(sample_output_det_aligned))[0]
    sample_output_det_aligned = [int(ele) for ele in sample_output_det_aligned]  
    
    detection_data.DeletePilots(bit_input_det_aligned, bit_input_det_pattern, sample_output_det_aligned, npadded)