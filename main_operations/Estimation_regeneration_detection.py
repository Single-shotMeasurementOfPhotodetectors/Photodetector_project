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


def optimization(s_bit, w_out, x, y, ts, tau, alpha_init, delta_init, mu_init, sigmas_init):
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

        cost = cost_function(nsec, k, m, x_con, y_con, tau, beta, gamma, mu)
        
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
    
    post_process_adm(result_adm_best.x, nsec, k, m, tau, x_con, y_con)
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
            seq_sample = np.repeat(s_temp, k)
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
    Q = []
    Q.append(mu*(1 - td/tau*(1 - np.exp(-tau/td))))
    for k in range(1,K):
        Q.append(mu*td/tau * np.exp(-(k+1)*tau/td) * (1-np.exp(tau/td))**2)
    
    # use information of seq_bit for sanity check only
    bit_group = {}   
    for k in range(K-1,len(seq_bit)):
        bit_sec = seq_bit[k-K+1:k+1]
        key_sec = ''.join(str(ele) for ele in bit_sec)
        if key_sec in bit_group:
            bit_group[key_sec].append(y_bit[k])
        else:
            bit_group[key_sec] = [y_bit[k]]
        
    key_set = []
    exp_group = {}
    state_output = {}
    for i in range(2**K):
        key_sec = padded_bin(i,K)
        key_set.append(key_sec)
        bit_sec = [int(ele) for ele in key_sec]
        bit_sec.reverse()
        D_sec = [(alpha*m+delta)*tau for m in bit_sec]
        output_expect_mean = np.sum(np.array(D_sec)*np.array(Q))
        output_expect_var = np.sum(np.array(D_sec)**2*np.array(Q) + sigma**2)
        state_output[key_sec] = [output_expect_mean,output_expect_var]
        if key_sec in bit_group:
            output_sec = bit_group[key_sec]
            output_mean = np.mean(output_sec)
            output_var = np.var(output_sec)
            exp_group[key_sec] = [[output_expect_mean,output_expect_var],[output_mean, output_var]]
           
    # calculate the state evolution
    key_set.sort()
    # test the sequence check
    pre_state = {}
    for key in key_set:
        pre_state[key] = math.inf
    pre_state[key_set[0]] = 0
    state_evolve = [pre_state]
    # calculate the trellis
    for output_bit in range(len(seq_bit)):
        curr_state = {}
        for state in key_set: 
            key_curr = [int(ele) for ele in state]
            pre_key1 = [0] + key_curr[:-1]
            pre_key1 = ''.join(str(ele) for ele in pre_key1)
            pre_key2 = [1] + key_curr[:-1]
            pre_key2 = ''.join(str(ele) for ele in pre_key2)
            curr_state[state] = min(pre_state[pre_key1],pre_state[pre_key2]) + (y_bit[output_bit] - state_output[state][0])**2/2/state_output[state][1] + np.log(state_output[state][1])/2
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
        key_curr = key_set[idx]
        key_curr = [int(ele) for ele in key_curr]
        seq_rev.append(key_curr[-1])
        pre_key1 = [0] + key_curr[:-1]
        pre_key1 = ''.join(str(ele) for ele in pre_key1)
        pre_key1 = key_set.index(pre_key1)
        pre_key2 = [1] + key_curr[:-1]
        pre_key2 = ''.join(str(ele) for ele in pre_key2)
        pre_key2 = key_set.index(pre_key2)
        pre_keys = [pre_key1, pre_key2]
            
    seq_recover= list(reversed(seq_rev))
    
    return seq_recover


def rate_calculation(seq_bit, seq_recover):
    err = [seq_bit[i] - seq_recover[i] for i in range(len(seq_bit))]
    FP = err.count(-1)/len(seq_bit)
    TP = (sum(seq_bit) - err.count(1))/len(seq_bit)
    FN = err.count(1)/len(seq_bit)
    TN = (len(seq_bit) - sum(seq_bit) - err.count(-1))/len(seq_bit)
    ACC = TP + TN
    rates = [ACC, FP, FN, TP, TN]
    
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

    # return the frame level output
    return y_out


def frame_to_bit_out(w, k):
    nbit = int(len(w) / k)
    # sum over s_frame to obtain s_bit values
    w_temp = np.reshape(w, (nbit, k))
    w_temp = np.transpose(w_temp)
    y_bit = np.sum(w_temp, axis=0)
    return y_bit




# main function
# define m, grouping size
m = 20
# oversampling ratio
k = 8
conv_factor = 10.3759765625 # 1.03759765625, 10.3759765625, 31.1279296875, 213.62
# define frame level interval
ts = 34836e-6
tau = ts * k
eta_set = 3656.05
tau_detect = ts * 4
max_viterbi_bit = 15


s = []
w = []
x = []
y = []
x_con = []
y_con = []
n_group = 0
# Load data, combine the sequence (sequence around the connection will be deleted to account for unknown )
for sec in range(8):
    sfile = 'bit_input_est_sec' + str(sec) + '.csv';
    with open(sfile, newline='') as input_bit:
        stemp = list(csv.reader(input_bit))[0]
    stemp = list(map(int, stemp))
    s.append(stemp)
    
    wfile = 'frame_output_est_sec' + str(sec) + '.csv';
    with open(wfile, newline='') as input_bit:
        wtemp = list(csv.reader(input_bit))[0]
    wtemp = [int(float(ele)) for ele in wtemp]
    wtemp = cnts_to_photons(wtemp, eta_set, conv_factor)
    xtemp, ytemp = pre_processing(k, m, stemp, wtemp)
    n_group += len(xtemp)
    w.append(wtemp)
    x.append(xtemp)
    x_con.extend(xtemp)
    y_con.extend(ytemp)
    y.append(ytemp)
    
xarr = np.array(x_con)
yarr = np.array(y_con)
alpha_init, delta_init, mu_init, sigmas_init = init_params(xarr, yarr, tau, k, m)
beta_init = alpha_init * mu_init
gamma_init = delta_init * mu_init
cost_init = cost_function(n_group, k, m, xarr, yarr, tau, beta_init, gamma_init, mu_init)
print(f"Estimation initial values:\nalpha={alpha_init}, delta={delta_init}, mu={mu_init}, cost={cost_init}\n")

alpha_conv, delta_conv, td_conv, mu_conv, sigmas_conv = optimization(s, w, x, y, ts, tau, alpha_init, delta_init, mu_init, sigmas_init)

niter = 10
err = 1
i = 1
conv = 'Y'
while err > 5e-4:
    alpha_temp, delta_temp, td_temp, mu_temp, sigmas_temp = optimization(s, w, x, y, ts, tau, alpha_conv, delta_conv, mu_conv, sigmas_conv)
    err = max(abs(alpha_temp - alpha_conv) / alpha_conv, abs(delta_temp - delta_conv) / delta_conv, abs(td_temp - td_conv) / td_conv,
              abs(mu_temp - mu_conv) / mu_conv)
    print(f"Err is {err} in interation {i}\n")
    alpha_conv, delta_conv, td_conv, mu_conv, sigmas_conv = alpha_temp, delta_temp, td_temp, mu_temp, sigmas_temp
    if i > niter:
        print("Does not converge in 10 iterations\n")
        conv = 'N'
        break
    i += 1

estimation_results = [alpha_conv, delta_conv, td_conv, mu_conv, sigmas_conv, conv]

# test for different detection data offset
y_bit_est = []
seq_bit = []
for detect_sec in range(3):
    randomfile = 'frame_output_without_pilot_det_sec' + str(detect_sec) + '.csv'
    with open(randomfile, newline='') as input_bit:
        randomtemp = list(csv.reader(input_bit))[0]
    randomtemp = list(map(int, randomtemp))
    randomtemp = frame_to_bit_out(randomtemp, 4)
    randomtemp = cnts_to_photons(randomtemp, eta_set*4, conv_factor)
    y_bit_est.extend(randomtemp)
    
    detectfile = 'bit_input_without_pilot_det_sec' + str(detect_sec) + '.csv'
    with open(detectfile, newline='') as input_bit:
        detecttemp = list(csv.reader(input_bit))[0]
    detecttemp = list(map(int, detecttemp))
    seq_bit.extend(detecttemp)

       
# consider one photon arrival affect the current bit output and the subsequent K-1 bit outputs
K = min(max(int(5*td_conv/tau_detect),5),max_viterbi_bit)      
seq_est = sequence_detection(alpha_conv, delta_conv, mu_conv, td_conv, np.sqrt(sigmas_conv), seq_bit, y_bit_est, K, tau_detect)
rates_est = rate_calculation(seq_bit, seq_est)
thresh_est = np.mean(np.array(y_bit_est))
seq_thresh_est = [1 if ele>thresh_est else 0 for ele in y_bit_est]
rates_thresh = rate_calculation(seq_bit, seq_thresh_est)

detection_results = [rates_est, rates_thresh]

# test the regeneration of the output
# when re-constructing the outputs, we neglect the connection of sections
synthetic_frame = output_generation(s[0], k, alpha_conv, delta_conv, mu_conv, np.sqrt(sigmas_conv), td_conv, tau)

# plot the comparison of data
for i in range(5):
    fig,ax = plt.subplots(figsize=(10, 8))
    ax.xaxis.set_major_locator(MultipleLocator(160))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    idx_list = list(np.arange(i*500,i*500+200,1))
    # plt.plot(idx_list,list(np.array(s1_frame)[idx_list]*framediff+framemin), label='Frame level input', linewidth=0.8)
    plt.plot(idx_list,list(np.array(synthetic_frame)[idx_list]), label='True output')
    plt.plot(idx_list,list(np.array(w[0])[idx_list]), label='Regenerated output')
    plt.ylabel('Frame level output')
    plt.rc('axes', labelsize=20, titlesize=20)
    plt.title(f"Data collection {i}")
    plt.legend(fontsize="20", loc ="lower right")











