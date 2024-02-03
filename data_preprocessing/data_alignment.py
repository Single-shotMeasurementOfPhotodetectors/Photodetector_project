from typing import List
import scipy.io
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


class FrameAlignment:
    
    def __init__(self, oversampling_ratio, frame_match_indices: List[int], shifts: List[int], data_type: str):
        # suppose the alignment is known and fixed
        self.oversampling = oversampling_ratio
        self.frame_match = frame_match_indices
        self.shift = shifts
        self.type = data_type
        
    def FrameAlign(self, frame_input: List[int], frame_output: List[List[float]]):        
        # suppose the alignment is unknown for all sections of output frames
        secs = len(frame_output)
        # we only expect a few sections of output data, if secs is large, it indicates there is only one section
        if secs > 100:
            secs = 1
        frame_period = len(frame_input)
        for sec in range(secs):
            if secs > 1:
                frame_output_sec = frame_output[sec]
            else:
                frame_output_sec = frame_output
            
            # if fixed, directly employ the correct match
            if self.frame_match:
                frame_match_idx = self.frame_match[sec]
            else:
                # recenter the input and output to calculate the correlation
                mean_frame_output = sum(frame_output_sec)/len(frame_output_sec)
                frame_output_sec_centered = [ele - mean_frame_output for ele in frame_output_sec]
                nperiod = int(np.floor(len(frame_output_sec)/frame_period))
                frame_output_sec_period = frame_output_sec_centered[:nperiod*frame_period]
                frame_output_sec_period = np.array(frame_output_sec_period).reshape(nperiod,frame_period)
                frame_output_sec_period = list(np.mean(frame_output_sec_period, axis = 0))
                frame_input_centered = [ele*2-1 for ele in frame_input]*2
                # create an empty list to store the correlation
                corr = []
                for frame_idx in range(frame_period):
                    frame_input_sec = frame_input_centered[frame_idx:frame_idx+frame_period]
                    corr.append(np.sum(np.array(frame_input_sec) * np.array(frame_output_sec_period)))
                frame_match_idx = corr.index(max(corr))
            shift_idx = self.shift[sec]

            frame_match_start = frame_match_idx + shift_idx
            if frame_match_start % 8 != 0:
                frame_end = (int(np.floor(frame_match_start)/self.oversampling) + 1)*self.oversampling - 1;
                frame_to_delete = frame_end - frame_match_start + 1
            else:
                frame_end = frame_match_start-1
                frame_to_delete = 0
            
            frame_output_sectioned = frame_output_sec[frame_to_delete:]
            frame_output_len = int(np.floor(len(frame_output_sectioned)/self.oversampling)*self.oversampling)
            bit_output_len = int(frame_output_len/self.oversampling)
            frame_output_sectioned = frame_output_sectioned[:frame_output_len]
            frame_input_sectioned_head = frame_input[frame_end+1:]
            frame_input_sectioned_head = frame_input_sectioned_head[:int(np.floor(len(frame_input_sectioned_head)/self.oversampling))*self.oversampling]
            nperiod_input = int(np.floor((frame_output_len - len(frame_input_sectioned_head))/frame_period))
            frame_input_sectioned = frame_input_sectioned_head + frame_input*nperiod_input
            frame_input_sectioned = frame_input_sectioned + frame_input[:frame_output_len - len(frame_input_sectioned)]
            bit_input_sectioned = list(np.mean(np.array(frame_input_sectioned).reshape(bit_output_len,self.oversampling), axis = 1))
            bit_input_sectioned = [int(ele) for ele in bit_input_sectioned]
            
            # fig, ax1 = plt.subplots(figsize=(20, 12))
            # color = 'tab:red'
            # ax1.set_xlabel('Frame index')
            # ax1.set_ylabel('Frame level output', color=color)
            # idx_list = list(np.arange(0,500,1))
            # ax1.plot(list(np.array(frame_output_sectioned)[idx_list]), color=color)
            # ax1.tick_params(axis='y', labelcolor=color)
            # ax1.xaxis.set_major_locator(MultipleLocator(160))
            # ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
            # ax1.grid(which='major', color='#CCCCCC', linestyle='--')
            # ax1.grid(which='minor', color='#CCCCCC', linestyle=':')
            
            # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            # color = 'tab:blue'
            # ax2.set_ylabel('Frame level input', color=color)  # we already handled the x-label with ax1
            # ax2.plot(list(np.array(frame_input_sectioned)[idx_list]), color=color)
            # ax2.tick_params(axis='y', labelcolor=color)
            
            # fig.tight_layout()
            # plt.rc('axes', labelsize=20, titlesize=20)
            # plt.show()
            
            
            if secs > 1:
                bit_input_aligned = 'data_pre/bit_input_' + self.type + '_sec' + str(sec) + '.csv';
            else:
                bit_input_aligned = 'data_pre/bit_input_' + self.type + '.csv';
            # Open the file in write mode with newline='' to avoid extra newlines
            with open(bit_input_aligned, 'w', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)           
                # Write the list as a single row in the CSV file
                csv_writer.writerow(bit_input_sectioned)
                
            if secs > 1:
                frame_output_aligned = 'data_pre/frame_output_' + self.type + '_sec' + str(sec) + '.csv';
            else:
                frame_output_aligned = 'data_pre/frame_output_' + self.type + '.csv';
            # Open the file in write mode with newline='' to avoid extra newlines
            with open(frame_output_aligned, 'w', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)           
                # Write the list as a single row in the CSV file
                csv_writer.writerow(frame_output_sectioned)
                
        print("Data for " + self.type + " have been pre-processed and saved")
    
    
    def DeletePilots(self, bit_input: List[int], bit_input_pattern: List[int], frame_output: List[int], npadded: int):
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
        frame_delete = []
        for ele in bit_delete:
            for item in range(ele*self.oversampling, (ele+1)*self.oversampling):
                frame_delete.append(item)
            
        updated_bit_input = [value for index, value in enumerate(bit_input) if index not in bit_delete]
        updated_frame_output = [value for index, value in enumerate(frame_output) if index not in frame_delete]
        
        bit_input_file = 'data_pre/bit_input_without_pilot_' + self.type + '.csv';
        # Open the file in write mode with newline='' to avoid extra newlines
        with open(bit_input_file, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)           
            # Write the list as a single row in the CSV file
            csv_writer.writerow(updated_bit_input)
        
        frame_output_file = 'data_pre/frame_output_without_pilot_' + self.type + '.csv';
        # Open the file in write mode with newline='' to avoid extra newlines
        with open(frame_output_file, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)           
            # Write the list as a single row in the CSV file
            csv_writer.writerow(updated_frame_output)
        


# preprocessing for estimation data
# define the oversampling ratio
oversampling_est = 8
pixel = '208_13'
shift = 1

# training (estimation) data alignment
# data is periodic with periods specified by input sequence
est_output = scipy.io.loadmat('420bitfiles_20240104_154fW_'+pixel+'_34836uS.mat')
frame_output_est = list(est_output['y'])
frame_output_est = [list(ele) for ele in frame_output_est]

# input sequence pattern (one period)
est_input = '420_bit_inv.csv'
with open(est_input, newline='') as bit_input_est:
    bit_input_est = list(csv.reader(bit_input_est))[0]
bit_input_est = [int(ele) for ele in bit_input_est]    
frame_input_est = [ele for ele in bit_input_est for i in range(oversampling_est)]        

frame_match_indices_est = [133, 2462, 1294, 682, 2922, 2858, 1522, 568]
# frame_match_indices_est = []
frame_shifts_est = [2 + shift]*8 #2
estimation_data = FrameAlignment(oversampling_est, frame_match_indices_est, frame_shifts_est, 'est')
estimation_data.FrameAlign(frame_input_est, frame_output_est)


# preprocessing for detection data
# define the oversampling ratio
oversampling_det = 4

# testing (detection) data alignment
# data is periodic with periods specified by input sequence
det_output = scipy.io.loadmat('400TestFiles_20240104_154fW_'+pixel+'_34836uS.mat')
frame_output_det = list(det_output['y'])

# input sequence pattern (one period)
frame_match_indices_det = [1428, 567, 495]
frame_shifts_det = [0 + shift]*3 #0
npadded = 20
for detect_sec in range(3):
    # Load data, combine the sequence (although they are not well correlated)
    det_input = 'random_sequence400_' + str(detect_sec) + '_8fpb.csv';
    with open(det_input, newline='') as bit_input_det:
        bit_input_det = list(csv.reader(bit_input_det))[0]
    bit_input_det = [int(ele) for ele in bit_input_det]  
    frame_input_det = [ele for ele in bit_input_det for i in range(oversampling_det)]
    
    frame_match_indices_det_temp = [frame_match_indices_det[detect_sec]]
    frame_shifts_det_temp = [frame_shifts_det[detect_sec]]
    
    detection_data = FrameAlignment(oversampling_det, frame_match_indices_det_temp, frame_shifts_det_temp, 'det_sec' + str(detect_sec))
    frame_output_det_temp = frame_output_det[detect_sec]
    frame_output_det_temp = [int(ele) for ele in frame_output_det_temp]
    detection_data.FrameAlign(frame_input_det, frame_output_det_temp)


    # suppose there are padded pilots delete
    bit_input_det_pattern = bit_input_det[npadded:]
    # retrieve the saved the output frames and input bits with padded patterns
    bit_input_file = 'data_pre/bit_input_det_sec' + str(detect_sec) + '.csv';
    with open(bit_input_file, newline='') as bit_input_det_aligned:
        bit_input_det_aligned = list(csv.reader(bit_input_det_aligned))[0]
    bit_input_det_aligned = [int(ele) for ele in bit_input_det_aligned]  
    frame_output_file = 'data_pre/frame_output_det_sec' + str(detect_sec) + '.csv';
    with open(frame_output_file, newline='') as frame_output_det_aligned:
        frame_output_det_aligned = list(csv.reader(frame_output_det_aligned))[0]
    frame_output_det_aligned = [int(ele) for ele in frame_output_det_aligned]  
    
    detection_data.DeletePilots(bit_input_det_aligned, bit_input_det_pattern, frame_output_det_aligned, npadded)