import math
import itertools
import csv

import numpy as np
import scipy.io
from scipy import optimize

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from typing import List



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
            
            
            # # plot to check the alignment
            # fig, ax1 = plt.subplots(figsize=(20, 12))
            # color = 'tab:red'
            # ax1.set_xlabel('sample index')
            # ax1.set_ylabel('sample level output', color=color)
            # idx_list = list(np.arange(0,500,1))
            # ax1.plot(list(np.array(sample_output_sectioned)[idx_list]), color=color)
            # ax1.tick_params(axis='y', labelcolor=color)
            # ax1.xaxis.set_major_locator(MultipleLocator(160))
            # ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
            # ax1.grid(which='major', color='#CCCCCC', linestyle='--')
            # ax1.grid(which='minor', color='#CCCCCC', linestyle=':')
            
            # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            # color = 'tab:blue'
            # ax2.set_ylabel('sample level input', color=color)  # we already handled the x-label with ax1
            # ax2.plot(list(np.array(sample_input_sectioned)[idx_list]), color=color)
            # ax2.tick_params(axis='y', labelcolor=color)
            
            # fig.tight_layout()
            # plt.rc('axes', labelsize=20, titlesize=20)
            # plt.show()
            
            
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
        

def pre_processing(k, m, s, w):
    n_tilde = int(np.floor(len(s) / m))

    s_temp = s[:n_tilde*m]
    # sum over s_bit to obtain x values
    s_temp = np.reshape(s_temp, (n_tilde, m))
    s_temp = np.transpose(s_temp)
    x = np.sum(s_temp, axis=0)

    # sum over w_sample to obtain y values
    w_temp = w[:n_tilde * m * k]
    w_temp = np.reshape(w_temp, (n_tilde, m * k))
    w_temp = np.transpose(w_temp)
    y = np.sum(w_temp, axis=0) / m

    return x, y


def init_params(x, y, tau, k, m):
    # count the occurance of x
    counts = np.bincount(x)
    # find the two most frequent x values
    v0, v1= counts.argsort()[-2:]
        
    y0 = []
    y1 = []
    for (idx, value) in enumerate(x):
        if value == v0:
            y0.append(y[idx])
        elif value == v1:
            y1.append(y[idx])

    bar_y0 = np.mean(y0)
    bar_y1 = np.mean(y1)
    s_y0 = np.var(y0)
    s_y1 = np.var(y1)

    init_mu = m * abs(s_y1 - s_y0) / abs(bar_y1 - bar_y0)
    init_alpha = m * abs(bar_y1 - bar_y0) / init_mu / tau / abs(v1 - v0)
    init_delta = abs(m / tau / init_mu * bar_y0 - init_alpha * v0) / m
    init_sigmas = abs(s_y0 - init_mu**2/m**2*(init_alpha * v0 + m*init_delta)*tau)*m/k

    return init_alpha, init_delta, init_mu, init_sigmas


def cost_function(n, k, m, x, y, tau, beta, gamma, mu):

    var = mu*tau/m**2*(beta*x+m*gamma)
    mean = tau/m*(beta*x+m*gamma)
    
    # in the first stage, no leakage is considered
    cost = np.sum(np.log(var))
    cost += np.sum((y-mean)**2/var)

    return cost/n


def post_process_adm(varx_min, n_sec, k, m, tau, x, y):
    beta_conv = varx_min[0]
    gamma_conv = varx_min[1]
    mu_conv = m ** 2 / n_sec / tau * np.sum(
        (y - tau / m * (beta_conv * x + m * gamma_conv)) ** 2 / (beta_conv * x + m * gamma_conv))

    alpha_conv = beta_conv / mu_conv
    delta_conv = gamma_conv / mu_conv
    cost_conv = cost_function(n_sec, k, m, x, y, tau, beta_conv, gamma_conv, mu_conv)

    print(f"Optimial values: alpha={alpha_conv}, delta={delta_conv}, mu={mu_conv},cost={cost_conv}\n")


def post_process_taud(taun_groupv, ts, cost_taud):
    sigma_conv = np.sqrt(np.exp(cost_taud))
    cost_conv = np.exp(cost_taud) / sigma_conv ** 2 + 2 * np.log(sigma_conv)
    print(f"Optimial values: taud={taun_groupv[0] * ts}, sigma={sigma_conv}, cost={cost_conv}\n")
    return sigma_conv


def cnts_to_electrons(w, eta, conv_factor):
    w = [i - eta for i in list(w)]
    w = [conv_factor * i for i in list(w)]
    return w


def optimization(s_bit, w_out, x, y, ts, tau, alpha_init, delta_init, mu_init, sigmas_init, oversampling_est):
    print("Stage 1 optimization, return beta, gamma.\n")
    # in stage 1, no ISI is considered, hence all data are concatenated for the Stage 1 optimization
    x_con = np.array(list(itertools.chain.from_iterable(x)))
    y_con = np.array(list(itertools.chain.from_iterable(y)))
    nsec = len(x_con)

    def compute_loss_adm(varx):
        beta = varx[0]
        gamma = varx[1]
        # if the data is at the beginning of a data collection
        # the leakage is not properly taken care of, hence neglect these samples
        # view mu as a function of beta and gamma
        mu = m ** 2 / nsec / tau * np.sum(
            (y_con - tau / m * (beta * x_con + m * gamma)) ** 2 / (beta * x_con + m * gamma)
        )

        cost = cost_function(nsec, oversampling_est, m, x_con, y_con, tau, beta, gamma, mu)
        
        if beta < 0 or gamma < 0:
            cost += 1e9

        return cost

    bounds_adm = [(0, np.inf), (0, np.inf)]
    # test with optimization solvers (testing 3 optimizers and choose the one with the smallest cost)
    result_adm = dict()
    result_adm['NM'] = optimize.minimize(compute_loss_adm, [beta_init, gamma_init], method="Nelder-Mead")
    print("Nelder-Mead method starting from the initial guess.")
    print(f"Convergent results is {result_adm['NM'].x} with cost {result_adm['NM'].fun}")

    result_adm['BFGS'] = optimize.minimize(compute_loss_adm, [beta_init, gamma_init], bounds=bounds_adm,
                                           method="L-BFGS-B")
    print("L-BFGS method starting from the initial guess.")
    print(f"Convergent results is {result_adm['BFGS'].x} with cost {result_adm['BFGS'].fun}")

    minimizer_kwargs = {"method": "L-BFGS-B"}
    result_adm['basin'] = optimize.basinhopping(compute_loss_adm, [beta_init, gamma_init],
                                                minimizer_kwargs=minimizer_kwargs, niter=200)
    print("Basinhopping method starting from the initial guess.")
    print(f"Convergent results is {result_adm['basin'].x} with cost {result_adm['basin'].fun}")
    
    costs_adm = {}
    for key in result_adm:
        if not key in costs_adm:
            costs_adm[key] = result_adm[key].fun
        
    # get the smallest cost and do the post_process
    best_minimizer = min(costs_adm, key=costs_adm.get)
    result_adm_best = result_adm[best_minimizer]
    
    post_process_adm(result_adm_best.x, nsec, oversampling_est, m, tau, x_con, y_con)
    beta0 = result_adm_best.x[0]
    gamma0 = result_adm_best.x[1]
    

    print("Stage 2 optimization, return taud, mu, sigma.\n")
    
    # use the estimated parameters to estimate tau_D
    # calculate D
    R = 500  # R is large enough to count all Q>0, conservatively chosen here for td <= 0.5

    def compute_loss_tms(vary):
        # taun = taud/ts
        tn = vary[0]
        mu = vary[1]
        sigma = vary[2]
        
        # go back to resolve the ambiguity of mu
        qtemp = 1 - 1 * tn * (1 - np.exp(-1 / tn))
        qj = np.array([qtemp])
        for i in range(R - 1):
            qtemp = tn * (
                    np.exp(-(i + 2) / tn) + np.exp(-i / tn) - 2 * np.exp(-(i + 1) / tn)
            )
            # Q_val  is Q in the formulation
            qj = np.append(qj, qtemp)
        
        # i_col index the data collections
        cost = 0
        # calculate the effective length of w_out
        n_w_eff = 0
        for i_col in range(len(x)):
            s_temp = s_bit[i_col]
            w_temp = w_out[i_col]
            seq_sample = np.repeat(s_temp, oversampling_est)
            Dvar = (beta0 * seq_sample + gamma0) * ts
    
            # check_err = np.sum(Q) - mu0
            for i in range(len(w_temp)):
                if i <= R - 1:
                    Dmul = Dvar[:i + 1]
                    Dmul = np.flip(Dmul)
                    Qmul1 = qj[:i + 1]
                    Qmul2 = mu*np.square(qj[: i+1])
                else:
                    Dmul = Dvar[i - R + 1:i + 1]
                    Dmul = np.flip(Dmul)
                    Qmul1 = qj
                    Qmul2 = mu*np.square(qj)
                mean = np.sum(Dmul * Qmul1)
                var = np.sum(Dmul * Qmul2) + sigma**2
                # if the data is at the beginning of a data collection
                # the leakage is not properly taken care of, hence neglect these samples
                neg = 20
                if i >= neg:
                    cost += np.log(var) + (w_temp[i] - mean) ** 2/var
            n_w_eff += len(w_temp) - neg
        
        if mu < 0.1 or tn < 0.5:
            cost += 1e9

        return cost/n_w_eff
    
    # test with local optimization solvers
    tn_init = 0.5

    result_com = optimize.minimize(compute_loss_tms, [tn_init, mu_init, np.sqrt(sigmas_init)], method="Nelder-Mead")
    print(
        "Nelder-Mead method starting from initial guess")
    result_com.x[2] = abs(result_com.x[2])
    print(result_com.x)
    print(result_com.fun)

    td_0 = result_com.x[0]*ts
    mu_0 = result_com.x[1]
    sigma_0 = result_com.x[2]
    print(
        f"Convergent results: alpha: {beta0 / mu_0}, delta: {gamma0 / mu_0}, "
        f"mu: {mu_0}, sigma: {sigma_0}, td: {td_0}\n")

    return beta0 / mu_0, gamma0 / mu_0, td_0, mu_0, sigma_0**2


def bit_processing(k, w):
    # sum over w_sample to obtain y values
    w_temp = np.reshape(w, (int(len(w)/k), k))
    y = np.sum(w_temp, axis=1)

    return y

def padded_bin(i, width):
    s = bin(i)
    return s[2:].zfill(width)

def sequence_detection(alpha, delta, mu, td, sigma, seq_bit, y_bit, K, tau):
    # K-1 bits are interference
    # K is the constraint length
    Q = []
    Q.append(mu*(1 - td/tau*(1 - np.exp(-tau/td))))
    for kk in range(1,K):
        Q.append(mu*td/tau * np.exp(-(kk+1)*tau/td) * (1-np.exp(tau/td))**2)
    
    # use information of seq_bit for sanity check only
    bit_group = {}   
    for kk in range(K-1,len(seq_bit)):
        bit_sec = seq_bit[kk-K+1:kk+1]
        key_sec = ''.join(str(ele) for ele in bit_sec)
        if key_sec in bit_group:
            bit_group[key_sec].append(y_bit[kk])
        else:
            bit_group[key_sec] = [y_bit[kk]]
        
    key_set = []
    key_decode_set = []
    exp_group = {}
    state_output = {}
    for i in range(2**K):
        key_sec = padded_bin(i,K)
        key_set.append(key_sec)
        if i < 2**(K-1):
            key_sec_decode = padded_bin(i,K-1)
            key_decode_set.append(key_sec_decode)
        bit_sec = [int(ele) for ele in key_sec]
        D_sec = [(alpha*m+delta)*tau for m in bit_sec]
        output_expect_mean = np.sum(np.array(D_sec)*np.array(Q))
        output_expect_var = np.sum(np.array(D_sec)*(np.array(Q)**2)) + sigma**2
        state_output[key_sec] = [output_expect_mean,output_expect_var]
        if key_sec in bit_group:
            # compute the true output mean and variance for sanity check only
            output_sec = bit_group[key_sec]
            output_mean = np.mean(output_sec)
            output_var = np.var(output_sec)
            exp_group[key_sec] = [[output_expect_mean,output_expect_var],[output_mean, output_var]]
           
    # calculate the state evolution
    key_decode_set.sort()
    # test the sequence check
    pre_state = {}
    for key in key_decode_set:
        pre_state[key] = math.inf
    pre_state[key_decode_set[0]] = 0
    state_evolve = [pre_state]
    # calculate the trellis
    for output_bit in range(len(seq_bit)):
        curr_state = {}
        for state in key_decode_set: 
            key_curr = [int(ele) for ele in state]
            pre_key1 = key_curr[1:] + [0]
            pre_key1 = ''.join(str(ele) for ele in pre_key1)
            pre_key2 = key_curr[1:] + [1]
            pre_key2 = ''.join(str(ele) for ele in pre_key2)
            pre_key1_ISI = key_curr + [0]
            pre_key1_ISI = ''.join(str(ele) for ele in pre_key1_ISI)
            pre_key2_ISI = key_curr + [1]
            pre_key2_ISI = ''.join(str(ele) for ele in pre_key2_ISI)
            curr_state[state] = min(
                pre_state[pre_key1]+ (y_bit[output_bit] - state_output[pre_key1_ISI][0])**2/2/state_output[pre_key1_ISI][1] + np.log(state_output[pre_key1_ISI][1])/2,
                pre_state[pre_key2]+ (y_bit[output_bit] - state_output[pre_key2_ISI][0])**2/2/state_output[pre_key2_ISI][1] + np.log(state_output[pre_key2_ISI][1])/2
                ) 
        state_evolve.append(curr_state)
        pre_state = curr_state
     
    # backtrace the trellis to find out the sequence
    seq_rev = []    
    pre_keys = []                            
    for bit_check in range(len(seq_bit),0,-1):
        path_metric = list(state_evolve[bit_check].values())
        if not pre_keys:
            idx = path_metric.index(min(path_metric))
        else:
            path_metric_arr = np.array(path_metric)
            path_metric_temp = list(path_metric_arr[pre_keys])
            if path_metric_temp[0] < path_metric_temp[1]:
                idx = pre_keys[0]
            else:
                idx = pre_keys[1]
        key_curr = key_decode_set[idx]
        key_curr = [int(ele) for ele in key_curr]
        seq_rev.append(key_curr[0])
        pre_key1 = key_curr[1:] + [0]
        pre_key1 = ''.join(str(ele) for ele in pre_key1)
        pre_key1 = key_decode_set.index(pre_key1)
        pre_key2 = key_curr[1:] + [1]
        pre_key2 = ''.join(str(ele) for ele in pre_key2)
        pre_key2 = key_decode_set.index(pre_key2)
        pre_keys = [pre_key1, pre_key2]
            
    seq_recover= list(reversed(seq_rev))
    
    return seq_recover


def rate_calculation(seq_bit, seq_recover):
    seq_truth = np.array(seq_bit)
    seq_detect = np.array(seq_recover)
    TP = np.sum(np.where(np.logical_and(seq_truth, seq_detect), 1, 0))
    TN = seq_truth.shape[0] - np.sum(np.where(np.logical_xor(seq_truth, seq_detect), 1, 0)) - TP
    FN = np.sum(np.where(np.logical_and(seq_truth, np.logical_not(seq_detect)), 1, 0))
    FP = seq_truth.shape[0] - TP - TN - FN
    ACC = TP + TN
    N = ACC + FN + FP
    rates = [ACC/N, FP/N, FN/N, TP/N, TN/N]
    
    return rates

def output_generation(seq_bit, k, alpha, delta, mu, sigma, td, tau):
    # append zeros at beginning to reach stationary stage
    seq_add = list(np.zeros(10))
    seq_bit = seq_add + seq_bit
    length = len(seq_bit)
    # each bit is oversampled with a factor k
    seq_sample = np.repeat(seq_bit, k)
    # generate particle arrival times
    # input 1 -> photon arrival + dark current
    # input 0 -> dark current
    time_arr = []
    number_arr = []
    # tau is bit interval
    tau_sample = tau / k
    for n in range(length * k):
        if seq_sample[n] == 0:
            # exponential inter-arrival times
            ti = np.random.exponential(1 / delta, (1, int(10 * delta * tau_sample)))
        else:
            # exponential inter-arrival times
            ti = np.random.exponential(1 / (alpha + delta), (1, int(10 * (alpha + delta) * tau_sample)))

        # plus time offset corresponding to each input bit
        t_temp = np.cumsum(ti) + n * tau_sample
        # cutoff by interval end
        t_arr = t_temp[t_temp < (n + 1) * tau_sample]
        # record all the arrival times
        time_arr.extend(t_arr)
        number_arr.append(len(t_arr))

    # generate the output
    # approximate how many intervals need to consider for integration
    # the integration of impulse response from 0 to n*t is M*(1-exp(-n*t/td))
    n_inter = 0
    # define the output counts
    cnt = 0
    epsilon = 1e-5
    while abs(cnt - mu) > epsilon:
        n_inter += 1
        cnt = mu * (1 - math.exp(-n_inter * tau_sample / td))

    # iterate over all particle arrivals and calculate their integration over the subsequent n_iter + 1 intervals
    y_out = np.zeros(len(seq_sample) + n_inter + 1)
    for i in range(len(time_arr)):
        y_temp = np.zeros(len(seq_sample) + n_inter + 1)
        tp = time_arr[i]
        #  ni is the index of the interval the  i-th particle arrives
        ni = int(np.floor(tp / tau_sample))
        # check = ((ni + 1) * tau_sample - tp)
        y_temp[ni] = mu - mu * math.exp(-((ni + 1) * tau_sample - tp) / td)
        for j in range(1, n_inter):
            y_temp[ni + j] = mu * math.exp(-((ni + j) * tau_sample - tp) / td) - mu * math.exp(
                -((ni + j + 1) * tau_sample - tp) / td)
        y_out = y_out + y_temp

    # add offset to output
    # remove the added 10 bits of redundancy
    y_out = y_out[10*k:length * k] + np.random.normal(0, sigma, (length-10) * k)

    # return the sample level output
    return y_out


def sample_to_bit_out(w, k):
    nbit = int(len(w) / k)
    # sum over s_sample to obtain s_bit values
    w_temp = np.reshape(w, (nbit, k))
    w_temp = np.transpose(w_temp)
    y_bit = np.sum(w_temp, axis=0)
    return y_bit

        
def sequence_metrics(params_proposed, params_traditional, seq_bit, output_bit, K, tau, oversampling):
    alpha_proposed, delta_proposed, mu_proposed, td_proposed, sigma_proposed = params_proposed
    seq_proposed = sequence_detection(alpha_proposed, delta_proposed, mu_proposed, td_proposed, np.sqrt(oversampling*sigma_proposed**2), seq_bit, output_bit, K, tau)
    rates_proposed = rate_calculation(seq_bit, seq_proposed)
    
    thresh = np.mean(np.array(output_bit))
    seq_thresh = np.where(np.array(output_bit)>thresh, 1, 0)
    rates_thresh = rate_calculation(seq_bit, seq_thresh)

    alpha_traditional, delta_traditional, mu_traditional, td_traditional, sigma_traditional = params_traditional
    seq_traditional = sequence_detection(alpha_traditional, delta_traditional, mu_traditional, td_traditional, np.sqrt(oversampling*sigma_traditional**2), seq_bit, output_bit, K, tau)
    rates_traditional = rate_calculation(seq_bit, seq_traditional)

    detection_results = [rates_proposed, rates_traditional, rates_thresh]
    return detection_results


# preprocessing for estimation data
# define the oversampling ratio
oversampling_training = 8
# correct the correlation based alignment
shift = 1  # default as 0

# training (estimation) data alignment
# data is periodic with periods specified by input sequence
est_output = scipy.io.loadmat('raw_data_for_alignment/training_output_sample_pix1.mat')
sample_output_est = list(est_output['y'])
sample_output_est = [list(ele) for ele in sample_output_est]

# input sequence pattern (one period)
est_input = 'raw_data_for_alignment/training_input.csv'
with open(est_input, newline='') as bit_input_est:
    bit_input_est = list(csv.reader(bit_input_est))[0]
bit_input_est = [int(ele) for ele in bit_input_est]    
sample_input_est = [ele for ele in bit_input_est for i in range(oversampling_training)]        

'''
We collect 8 sections of sample-level outputs, each section has its own alignment shift
Without any prior information of the alignment, please run 

sample_match_indices_est = []

Algorithm will then try to find out the optimal alignment automatically.
If the alignment is determined, directly given the alignment shifts will save the computation time
'''
# These are indices recorded for alignment in each section
sample_match_indices_est = [133, 2462, 1294, 682, 2922, 2858, 1522, 568]

sample_shifts_est = [2 + shift]*8 # need to manually verify the alignment
estimation_data = sampleAlignment(oversampling_training, sample_match_indices_est, sample_shifts_est, 'training')
estimation_data.sampleAlign(sample_input_est, sample_output_est)


# preprocessing for detection data 
# define the oversampling ratio
oversampling_testing = 4

# testing (detection) data alignment
# data is periodic with periods specified by input sequence
det_output = scipy.io.loadmat('raw_data_for_alignment/test_output_sample_pix1.mat')
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
    sample_input_det = [ele for ele in bit_input_det for i in range(oversampling_testing)]
    
    sample_match_indices_det_temp = [sample_match_indices_det[detect_sec]]
    sample_shifts_det_temp = [sample_shifts_det[detect_sec]]
    
    detection_data = sampleAlignment(oversampling_testing, sample_match_indices_det_temp, sample_shifts_det_temp, 'test_sec' + str(detect_sec))
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
    
    
    
#%%
# main function
# define m, grouping size
m = 20

conv_factor = 10.3759765625 # 1.03759765625, 10.3759765625, 31.1279296875, 213.62
eta_set = 3652.27
# define sample level interval
ts = 34836e-6
max_viterbi_bit = 15

alpha_sequential = 676.5725634
delta_sequential = 186.3316544
td_sequential = 0.41714
mu_sequential = 69.75224132
sigma_sequential = 70

tau = ts * oversampling_training
tau_detect = ts * oversampling_testing 
params_sequential = [alpha_sequential, delta_sequential, mu_sequential, td_sequential, sigma_sequential]

#%% implement the parametric estimation  
    
# Initialize containers
sbit, wsample, xsec, ysec = [], [], [], []
x_con, y_con = [], []
n_group = 0

# --- Load training data (8 sections) ---
# Each section contains:
#   - Bit input sequence (s)
#   - Sample-level detector output (w)
# After preprocessing, obtain aggregated features (x, y).
for sec in range(8):
    # Load input bit sequence
    sfile = f"data_processed/bit_input_training_sec{sec}.csv"
    with open(sfile, newline='') as input_bit:
        stemp = list(map(int, list(csv.reader(input_bit))[0]))
    sbit.append(stemp)

    # Load detector output samples
    wfile = f"data_processed/sample_output_training_sec{sec}.csv"
    with open(wfile, newline='') as input_sample:
        wtemp = [int(float(ele)) for ele in list(csv.reader(input_sample))[0]]

    # Convert raw counts to electrons
    wtemp = cnts_to_electrons(wtemp, eta_set, conv_factor)

    # Preprocess input-output pairs for Stage 1
    xtemp, ytemp = pre_processing(oversampling_training, m, stemp, wtemp)
    n_group += len(xtemp)

    # Store results
    wsample.append(wtemp)
    xsec.append(xtemp)
    ysec.append(ytemp)
    x_con.extend(xtemp)
    y_con.extend(ytemp)
    
# Convert concatenated lists into arrays
xarr = np.array(x_con)
yarr = np.array(y_con)

# --- Parameter initialization ---
alpha_init, delta_init, mu_init, sigmas_init = init_params(
    xarr, yarr, tau, oversampling_training, m
)
beta_init  = alpha_init * mu_init
gamma_init = delta_init * mu_init
cost_init  = cost_function(
    n_group, oversampling_training, m, xarr, yarr, tau,
    beta_init, gamma_init, mu_init
)

print(
    f"Initial parameter estimates:\n"
    f"  alpha = {alpha_init:.6f}, delta = {delta_init:.6f}, "
    f"mu = {mu_init:.6f}, cost = {cost_init:.6f}\n"
)

# --- First optimization run ---
alpha_conv, delta_conv, td_conv, mu_conv, sigmas_conv = optimization(
    sbit, wsample, xsec, ysec, ts, tau,
    alpha_init, delta_init, mu_init, sigmas_init,
    oversampling_training
)

# --- Iterative refinement ---
niter = 10
err   = 1
i     = 1
conv  = "Y"

while err > 5e-4:
    alpha_temp, delta_temp, td_temp, mu_temp, sigmas_temp = optimization(
        sbit, wsample, xsec, ysec, ts, tau,
        alpha_conv, delta_conv, mu_conv, sigmas_conv,
        oversampling_training
    )

    # Relative error across all parameters
    err = max(
        abs(alpha_temp - alpha_conv) / alpha_conv,
        abs(delta_temp - delta_conv) / delta_conv,
        abs(td_temp - td_conv)     / td_conv,
        abs(mu_temp - mu_conv)     / mu_conv,
    )
    print(f"Iteration {i}: difference = {err:.2e}\n")

    # Update parameters
    alpha_conv, delta_conv, td_conv, mu_conv, sigmas_conv = (
        alpha_temp, delta_temp, td_temp, mu_temp, sigmas_temp
    )

    # Convergence check
    if i > niter:
        print("Convergence not achieved within 10 iterations.\n")
        conv = "N"
        break
    i += 1

# Pack final results into a convenient list
params_single_shot = [
    alpha_conv, delta_conv, mu_conv, td_conv, np.sqrt(sigmas_conv)
]

#%%
ybit_test = []
sbit_test = []
wsample_test_list = []
stest_list = []

for detect_sec in range(3):
    # --- Load detector output samples ---
    randomfile = f"data_processed/sample_output_without_pilot_test_sec{detect_sec}.csv"
    with open(randomfile, newline='') as input_file:
        randomtemp = list(map(int, list(csv.reader(input_file))[0]))
    
    # Convert sample-level outputs to bit-level, then to electrons
    wsample_test_list.append(
        cnts_to_electrons(randomtemp, eta_set, conv_factor)
    )
    randomtemp = sample_to_bit_out(randomtemp, oversampling_testing)
    randomtemp = cnts_to_electrons(randomtemp, eta_set*oversampling_testing, conv_factor)
    ybit_test.extend(randomtemp)

    # --- Load ground truth input bits ---
    detectfile = f"data_processed/bit_input_without_pilot_test_sec{detect_sec}.csv"
    with open(detectfile, newline='') as input_file:
        detecttemp = list(map(int, list(csv.reader(input_file))[0]))
    stest_list.append(detecttemp)
    sbit_test.extend(detecttemp)


#%% ==========================================================================
# Regeneration of output samples
# ==========================================================================
# When reconstructing outputs, section connections are neglected

# --- Generate synthetic outputs ---
synthetic_sample_single_shot = output_generation(
    stest_list[0], oversampling_testing,
    alpha_conv, delta_conv, mu_conv, np.sqrt(sigmas_conv),
    td_conv, tau_detect
)

synthetic_sample_sequential = output_generation(
    stest_list[0], oversampling_testing,
    alpha_sequential, delta_sequential, mu_sequential, sigma_sequential,
    td_sequential, tau_detect
)



#%% ==========================================================================
# Plot reconstructed outputs vs. ground truth
# ==========================================================================
for i in range(2):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.xaxis.set_major_locator(MultipleLocator(160))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    ax.grid(which="minor", color="#CCCCCC", linestyle=":")

    idx_list = np.arange(i * 500 + 200, i * 500 + 400)

    ax.plot(idx_list, np.array(synthetic_sample_single_shot)[idx_list], label="Regenerated output (single_shot)")
    ax.plot(idx_list, np.array(synthetic_sample_sequential)[idx_list], label="Regenerated output (sequential)")
    ax.plot(idx_list, np.array(wsample_test_list[0])[idx_list], label="True output")

    ax.set_ylabel("Sample level output")
    ax.set_title(f"Data collection {i}")
    ax.legend(fontsize=20, loc="lower right")
    plt.rc("axes", labelsize=20, titlesize=20)




