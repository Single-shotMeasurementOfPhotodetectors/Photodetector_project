import numpy as np
import csv
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import itertools
import math

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


def cnts_to_photons(w, eta, conv_factor):
    w = [i - eta for i in list(w)]
    w = [conv_factor * i for i in list(w)]
    return w


def optimization(s_bit, w_out, x, y, ts, tau, alpha_init, delta_init, mu_init, sigmas_init, oversampling_training):
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

        cost = cost_function(nsec, oversampling_training, m, x_con, y_con, tau, beta, gamma, mu)
        
        if beta < 0 or gamma < 0:
            cost += 1e9

        return cost

    bounds_adm = [(0, np.inf), (0, np.inf)]
    # test with optimization solvers (testing 3 optimizers and choose the one with the smallest cost)
    result_adm = dict()
    result_adm['NM'] = optimize.minimize(compute_loss_adm, [beta_init, gamma_init], method="Nelder-Mead")
    print("Nelder-Mead method (local optimization) starting from the initial guess, may collapse "
          "when problem is ill-conditioned.")
    print(f"Convergent results is {result_adm['NM'].x} with cost {result_adm['NM'].fun}")

    result_adm['BFGS'] = optimize.minimize(compute_loss_adm, [beta_init, gamma_init], bounds=bounds_adm,
                                           method="L-BFGS-B")
    print("L-BFGS method starting from initial guess")
    print(f"Convergent results is {result_adm['BFGS'].x} with cost {result_adm['BFGS'].fun}")

    minimizer_kwargs = {"method": "L-BFGS-B"}
    result_adm['basin'] = optimize.basinhopping(compute_loss_adm, [beta_init, gamma_init],
                                                minimizer_kwargs=minimizer_kwargs, niter=200)
    print("Basinhopping method (global optimization) starting from the initial guess.")
    print(f"Convergent results is {result_adm['basin'].x} with cost {result_adm['basin'].fun}")
    
    costs_adm = {}
    for key in result_adm:
        if not key in costs_adm:
            costs_adm[key] = result_adm[key].fun
        
    # get the smallest cost and do the post_process
    best_minimizer = min(costs_adm, key=costs_adm.get)
    result_adm_best = result_adm[best_minimizer]
    
    post_process_adm(result_adm_best.x, nsec, oversampling_training, m, tau, x_con, y_con)
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
            seq_sample = np.repeat(s_temp, oversampling_training)
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
        "Nelder-Mead method starting from initial guess (this method may collapse when problem is ill-conditioned)")
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

def TP_FP_cal(seq_input, seq_output, th):
    seq_input = np.array(seq_input)
    seq_output = np.array(seq_output)
    TPR = []
    FPR = []
    for i in range(th.shape[0]):
        th_temp = th[i]
        seq_detect = seq_output > th_temp
        seq_truth = seq_input > 0
        TP = np.sum(np.where(np.logical_and(seq_truth, seq_detect), 1, 0))
        TN = seq_truth.shape[0] - np.sum(np.where(np.logical_xor(seq_truth, seq_detect), 1, 0)) - TP
        FN = np.sum(np.where(np.logical_and(seq_truth, np.logical_not(seq_detect)), 1, 0))
        FP = seq_truth.shape[0] - TP - TN - FN
        TPR.append(TP/(TP + FN))
        FPR.append(FP/(FP + TN))
    return [TPR, FPR]

def AUR_cal(ROC):
    TP_diff = np.diff(ROC[0])
    AUR_lower = -np.sum(np.array(ROC[1][:-1])*TP_diff)
    AUR_upper = -np.sum(np.array(ROC[1][1:])*TP_diff)
    return (AUR_lower + AUR_upper)/2
    

def write_lists_to_csv(file_path, names, *lists):
    # Combine the names and lists into a list of rows
    rows = [names] + list(zip(*lists))

    # Open the CSV file in write mode
    with open(file_path, 'w', newline='') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)

        # Write the rows to the CSV file
        csv_writer.writerows(rows)
        
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



#%%
# main function
# define m, grouping size
m = 20
# oversampling ratio
oversampling_training = 8
oversampling_testing = 4
conv_factor = 10.3759765625 # 1.03759765625, 10.3759765625, 31.1279296875, 213.62
eta_set = 3652.27
# define sample level interval
ts = 34836e-6
max_viterbi_bit = 15

alpha_traditional = 676.5725634
delta_traditional = 186.3316544
td_traditional = 0.41714
mu_traditional = 69.75224132
sigma_traditional = 70

tau = ts * oversampling_training
tau_detect = ts * oversampling_testing
params_traditional = [alpha_traditional, delta_traditional, mu_traditional, td_traditional, sigma_traditional]

#%% implement the parametric estimation

s = []
w = []
x = []
y = []
x_con = []
y_con = []
n_group = 0
# Load data, combine the sequence (sequence around the connection will be deleted to account for unknown )
for sec in range(8):
    sfile = 'data_processed/bit_input_training_sec' + str(sec) + '.csv';
    with open(sfile, newline='') as input_bit:
        stemp = list(csv.reader(input_bit))[0]
    stemp = list(map(int, stemp))
    s.append(stemp)
    
    wfile = 'data_processed/sample_output_training_sec' + str(sec) + '.csv';
    with open(wfile, newline='') as input_bit:
        wtemp = list(csv.reader(input_bit))[0]
    wtemp = [int(float(ele)) for ele in wtemp]
    wtemp = cnts_to_photons(wtemp, eta_set, conv_factor)
    xtemp, ytemp = pre_processing(oversampling_training, m, stemp, wtemp)
    n_group += len(xtemp)
    w.append(wtemp)
    x.append(xtemp)
    x_con.extend(xtemp)
    y_con.extend(ytemp)
    y.append(ytemp)
    
xarr = np.array(x_con)
yarr = np.array(y_con)
alpha_init, delta_init, mu_init, sigmas_init = init_params(xarr, yarr, tau, oversampling_training, m)
beta_init = alpha_init * mu_init
gamma_init = delta_init * mu_init
cost_init = cost_function(n_group, oversampling_training, m, xarr, yarr, tau, beta_init, gamma_init, mu_init)
print(f"Estimation initial values:\nalpha={alpha_init}, delta={delta_init}, mu={mu_init}, cost={cost_init}\n")

alpha_conv, delta_conv, td_conv, mu_conv, sigmas_conv = optimization(s, w, x, y, ts, tau, alpha_init, delta_init, mu_init, sigmas_init, oversampling_training)

niter = 10
err = 1
i = 1
conv = 'Y'
while err > 5e-4:
    alpha_temp, delta_temp, td_temp, mu_temp, sigmas_temp = optimization(s, w, x, y, ts, tau, alpha_conv, delta_conv, mu_conv, sigmas_conv, oversampling_training)
    err = max(abs(alpha_temp - alpha_conv) / alpha_conv, abs(delta_temp - delta_conv) / delta_conv, abs(td_temp - td_conv) / td_conv,
              abs(mu_temp - mu_conv) / mu_conv)
    print(f"Err is {err} in interation {i}\n")
    alpha_conv, delta_conv, td_conv, mu_conv, sigmas_conv = alpha_temp, delta_temp, td_temp, mu_temp, sigmas_temp
    if i > niter:
        print("Does not converge in 10 iterations\n")
        conv = 'N'
        break
    i += 1

csv_file_path = "results/estimation/estimated_parameters.csv"
write_lists_to_csv(csv_file_path, ['alpha','delta','td','mu','sigma','conv or not'], [alpha_conv],[delta_conv],[td_conv],[mu_conv],[np.sqrt(sigmas_conv)],[conv])
params_proposed = [alpha_conv, delta_conv, mu_conv, td_conv, np.sqrt(sigmas_conv)]

#%%
y_bit_proposed = []
seq_bit = []
for detect_sec in range(3):
    randomfile = 'data_processed/sample_output_without_pilot_test_sec' + str(detect_sec) + '.csv'
    with open(randomfile, newline='') as input_bit:
        randomtemp = list(csv.reader(input_bit))[0]
    randomtemp = list(map(int, randomtemp))
    randomtemp = sample_to_bit_out(randomtemp, oversampling_testing)
    randomtemp = cnts_to_photons(randomtemp, eta_set*oversampling_testing, conv_factor)
    y_bit_proposed.extend(randomtemp)
    
    detectfile = 'data_processed/bit_input_without_pilot_test_sec' + str(detect_sec) + '.csv'
    with open(detectfile, newline='') as input_bit:
        detecttemp = list(csv.reader(input_bit))[0]
    detecttemp = list(map(int, detecttemp))
    seq_bit.extend(detecttemp)


#%%
# test the sequence detection
# on the training set
K_train = min(max(int(5*td_conv/tau),5),max_viterbi_bit)  
s_bit_train = [item for sub_s in s for item in sub_s]
w_sample_train = [item for sub_w in w for item in sub_w]
output_bit_train = [sum(w_sample_train[i:i+8]) for i in range(0, len(w_sample_train), oversampling_training)]   
rates_train = sequence_metrics(params_proposed, params_traditional, s_bit_train, output_bit_train, K_train, tau, oversampling_training)

# on the test set       
# consider one photon arrival affect the current bit output and the subsequent K-1 bit outputs
K_test = min(max(int(5*td_conv/tau_detect),5),max_viterbi_bit)      
rates_test = sequence_metrics(params_proposed, params_traditional, seq_bit, y_bit_proposed, K_test, tau_detect, oversampling_testing)

csv_file_path = "results/detection/error_rates.csv"
label = ['ACC', 'FP', 'FN', 'TP', 'TN']
write_lists_to_csv(csv_file_path, ['Label','Train_proposed','Train_traditional','Train_thresh','Test_proposed','Test_traditional','Test_thresh'], label, rates_train[0],rates_train[1],rates_train[2], rates_test[0],rates_test[1],rates_test[2])

#%%
# test the regeneration of the output
# when re-constructing the outputs, we neglect the connection of sections

synthetic_sample_proposed = output_generation(s[0], oversampling_training, alpha_conv, delta_conv, mu_conv, np.sqrt(sigmas_conv), td_conv, tau)
synthetic_sample_traditional = output_generation(s[0], oversampling_training, alpha_traditional, delta_traditional, mu_traditional, sigma_traditional, td_traditional, tau)

s_sample = [selem for selem in s[0] for j in range(oversampling_training) ]
csv_file_path = "results/reconstruction/reconstructed_data.csv"
write_lists_to_csv(csv_file_path, ['Sample_input','Actual','Proposed','Traditional'], s_sample, w[0],synthetic_sample_proposed,synthetic_sample_traditional)


# plot the comparison of data
for i in range(2):
    fig,ax = plt.subplots(figsize=(10, 8))
    ax.xaxis.set_major_locator(MultipleLocator(160))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    idx_list = list(np.arange(i*500+200,i*500+400,1))
    # plt.plot(idx_list,list(np.array(s1_sample)[idx_list]*samplediff+samplemin), label='sample level input', linewidth=0.8)
    plt.plot(idx_list,list(np.array(synthetic_sample_proposed)[idx_list]), label='Regenerated output (Est)')
    plt.plot(idx_list,list(np.array(synthetic_sample_traditional)[idx_list]), label='Regenerated output (Exp)')
    plt.plot(idx_list,list(np.array(w[0])[idx_list]), label='True output')
    plt.ylabel('Sample level output')
    plt.rc('axes', labelsize=20, titlesize=20)
    plt.title(f"Data collection {i}")
    plt.legend(fontsize="20", loc ="lower right")

#%% ROC curves

recons_bit_proposed = [sum(synthetic_sample_proposed[i:i+8]) for i in range(0, len(synthetic_sample_proposed), 8)]
recons_bit_traditional = [sum(synthetic_sample_traditional[i:i+8]) for i in range(0, len(synthetic_sample_traditional), 8)]
true_bit = [sum(w[0][i:i+8]) for i in range(0, len(w[0]), 8)]
minn = min(true_bit)
maxx = max(true_bit)
th = np.linspace(minn,maxx,4000,endpoint=False)
ROC_true = TP_FP_cal(s[0],true_bit,th)
ROC_proposed = TP_FP_cal(s[0],recons_bit_proposed,th)
ROC_traditional = TP_FP_cal(s[0],recons_bit_traditional,th)
csv_file_path = "results/reconstruction/ROC.csv"
write_lists_to_csv(csv_file_path, ['TP_true','FP_true','TP_proposed','FP_proposed','TP_traditional','FP_traditional'], ROC_true[0],ROC_true[1], ROC_proposed[0],ROC_proposed[1], ROC_traditional[0],ROC_traditional[1])

plt.figure()
plt.plot(ROC_true[0],ROC_true[1],'-',linewidth=7) 
plt.plot(ROC_proposed[0],ROC_proposed[1],'-',linewidth=4)
plt.plot(ROC_traditional[0],ROC_traditional[1],'-',linewidth=2)
plt.legend(['True output','Estimation reconstruction','Experimental reconstruction'])
plt.title("FP vs TP for pixel 286,128")
plt.xlabel('TP')
plt.ylabel('FP')
plt.ylim([-0.01,1])

AUR_true = AUR_cal(ROC_true)
AUR_proposed = AUR_cal(ROC_proposed)
AUR_traditional = AUR_cal(ROC_traditional)

csv_file_path = "results/reconstruction/AUR.csv"
write_lists_to_csv(csv_file_path, ['Actual','Proposed','Traditional'], [AUR_true], [AUR_proposed], [AUR_traditional])


Err_train = [1-rates_train[0][0],1-rates_train[1][0],1-rates_train[2][0]]
Err_test = [1-rates_test[0][0],1-rates_test[1][0],1-rates_test[2][0]]







