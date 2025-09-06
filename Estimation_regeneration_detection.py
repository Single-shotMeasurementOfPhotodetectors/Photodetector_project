import numpy as np
import csv
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import itertools
import math


def pre_processing(k, m, s, w):
    """
    Pre-process input and output sequences by aggregating bits/samples.

    Parameters
    ----------
    k : int
        Oversampling ratio.
    m : int
        Number of bits to aggregate in stage 1.
    s : list[int]
        Input bit sequence.
    w : list[float]
        Output sample sequence.

    Returns
    -------
    x : np.ndarray of int
        Aggregated input values (sum over m input bits).
    y : np.ndarray of float
        Aggregated output values (average over m*k output samples, scaled by m).
    """
    # Convert lists to numpy arrays
    s = np.asarray(s, dtype=int)
    w = np.asarray(w, dtype=float)

    # Number of complete groups that fit in the input
    n_groups = len(s) // m

    # --- Aggregate input bits ---
    # Reshape into (n_groups, m) and sum across each row
    s_trimmed = s[:n_groups * m].reshape(n_groups, m)
    x = s_trimmed.sum(axis=1).astype(int)

    # --- Aggregate output samples ---
    # Reshape into (n_groups, m*k) and sum across each row
    w_trimmed = w[:n_groups * m * k].reshape(n_groups, m * k)
    y = (w_trimmed.sum(axis=1) / m).astype(float)

    return x, y




def init_params(x, y, tau, k, m):
    """
    Initialize model parameters from aggregated input/output data (Stage 1).
    See Section 1.3.1 in the supplementary material.

    Parameters
    ----------
    x : np.ndarray of int
        Aggregated input values (sum over m input bits).
    y : np.ndarray of float
        Aggregated output values (average over m*k output samples, scaled by m).
    tau : float
        Bit interval.
    k : int
        Oversampling ratio.
    m : int
        Number of bits to aggregate in stage 1.

    Returns
    -------
    init_alpha : float
        Initialized value for light intensity.
    init_delta : float
        Initialized value for dark current intensity.
    init_mu : float
        Initialized value for internal gain.
    init_sigmas : float
        Initialized value for noise standard deviation.
    """
    # Count how many times each input value occurs
    counts = np.bincount(x)

    # Identify the two most frequent input values
    v0, v1 = counts.argsort()[-2:]

    # Extract corresponding output samples for those two input values
    y0 = y[x == v0]
    y1 = y[x == v1]

    # Compute mean and variance for each group
    bar_y0, bar_y1 = np.mean(y0), np.mean(y1)
    var_y0, var_y1 = np.var(y0), np.var(y1)

    # Estimate internal gain (mu) based on variance difference
    init_mu = m * abs(var_y1 - var_y0) / abs(bar_y1 - bar_y0)

    # Estimate light intensity (alpha) based on mean difference
    init_alpha = (m * abs(bar_y1 - bar_y0)) / (init_mu * tau * abs(v1 - v0))

    # Estimate dark current intensity (delta)
    init_delta = abs((m / (tau * init_mu)) * bar_y0 - init_alpha * v0) / m

    # Estimate noise standard deviation (sigmas)
    init_sigmas = abs(
        var_y0 - (init_mu**2 / m**2) * (init_alpha * v0 + m * init_delta) * tau
    ) * m / k

    return init_alpha, init_delta, init_mu, init_sigmas



def cost_function(n, k, m, x, y, tau, beta, gamma, mu):
    """
    Compute the log-loss function for Stage 1.
    See Section 1.2.1 in the supplementary material.

    Parameters
    ----------
    n : int
        Number of input bits.
    k : int
        Oversampling ratio.
    m : int
        Number of bits aggregated in stage 1.
    x : np.ndarray of int
        Aggregated input values (sum over m input bits).
    y : np.ndarray of float
        Aggregated output values (average over m*k output samples, scaled by m).
    tau : float
        Bit interval.
    beta : float
        Product of internal gain and light intensity (mu * alpha).
    gamma : float
        Product of internal gain and dark current intensity (mu * delta).
    mu : float
        Internal gain.

    Returns
    -------
    float
        Log-loss value. This is the optimization objective in Stage 1.
    """

    # Compute variance and mean for each aggregated sample
    # Formula derived from noise model (Stage 1, no leakage considered)
    var = mu * tau / m**2 * (beta * x + m * gamma)
    mean = tau / m * (beta * x + m * gamma)

    # Compute log-loss: sum of log-variance + normalized squared errors
    log_loss = np.sum(np.log(var)) + np.sum((y - mean)**2 / var)

    # Return normalized cost (per input bit)
    return log_loss / n



def post_process_adm(varx_min, n_sec, k, m, tau, x, y):
    """
    Extract parameters of interest from the optimization results (Stage 1).
    See Section 1.2.1 in the supplementary material.

    Parameters
    ----------
    varx_min : array of float
        Optimization results [beta_conv, gamma_conv].
    n_sec : int
        Number of sections (n_sec = n/m).
    k : int
        Oversampling factor.
    m : int
        Number of bits aggregated in stage 1.
    x : np.ndarray of int
        Aggregated input values (sum over m input bits).
    y : np.ndarray of float
        Aggregated output values (average over m*k output samples, scaled by m).
    tau : float
        Bit interval.

    Extracted values
    -------
    alpha_conv : float
        Convergent solution for light intensity.
    delta_conv : float
        Convergent solution for dark current intensity.
    cost_conv : float
        Convergent cost value in stage 1.
    """
    
    # Extract optimization variables
    beta_conv, gamma_conv = varx_min[0], varx_min[1]

    # Compute mu from stage 1 formulation
    mu_conv = (m**2 / n_sec / tau) * np.sum(
        (y - tau / m * (beta_conv * x + m * gamma_conv))**2 / (beta_conv * x + m * gamma_conv)
    )

    # Compute convergent parameters
    alpha_conv = beta_conv / mu_conv
    delta_conv = gamma_conv / mu_conv

    # Compute cost for the converged parameters
    cost_conv = cost_function(n_sec, k, m, x, y, tau, beta_conv, gamma_conv, mu_conv)

    # Print summary of results
    print(f"Optimal values: alpha={alpha_conv}, delta={delta_conv}, mu={mu_conv}, cost={cost_conv}\n")
    


def cnts_to_electrons(w, eta, conv_factor):
    """
    Convert raw detector counts to number of electrons.
    Our readouts are not in units of electrons; this function translates counts to electrons. 
    You may not need this conversion if samples are already measured in number of electrons.

    Parameters
    ----------
    w : list of float
        Raw counts measured by the detector.
    eta : float
        Offset counts to be subtracted.
    conv_factor : float
        Conversion factor to map corrected counts to number of electrons.

    Returns
    -------
    w_electrons : list of float
        Estimated number of electrons corresponding to the input counts.
    
    Notes
    -----
    The original detector counts do not directly represent electrons.
    Number of electrons is calculated as: electrons = conv_factor * (counts - eta)
    """

    # Step 1: Remove baseline offset (dark current)
    w_corrected = [i - eta for i in w]

    # Step 2: Convert corrected counts to number of electrons
    w_electrons = [conv_factor * i for i in w_corrected]

    return w_electrons



def optimization(s_bit, w_out, x, y, ts, tau, alpha_init, delta_init, mu_init, sigmas_init, oversampling_training):
    """
    Perform the two-stage optimazation
    See Section 1.2.1 and 1.2.2 in the supplementary material.

    Parameters
    ----------
    s_bit : list[int]
        Input bit sequence.
    w_out : list[float]
        Output sample sequence.
    x : np.ndarray of int
        Aggregated input values (sum over m input bits).
    y : np.ndarray of float
        Aggregated output values (average over m*k output samples, scaled by m).
    ts : float
        Sampling interval (ts = tau/k).
    tau : float
        Bit interval.
    alpha_init : float
        Initialized value for light intensity.
    delta_init : float
        Initialized value for dark current intensity.
    mu_init : float
        Initialized value for internal gain.
    sigmas_init : float
        Initialized value for noise standard deviation.
    oversampling_training : int
        Oversampling ratio.

    Returns
    -------
    beta0 / mu_0: float
        Estimated light intensity
    gamma0 / mu_0: float
        Estimated dark current intesity
    td_0: float
        Estimated device response time
    mu_0: float
        Estimated internal gain
    sigma_0**2: float
        Estimated noise variance

    """
    print("Stage 1 optimization: estimating beta and gamma.\n")

    # ----------------------------------------------------------------------
    # Step 1: Concatenate all input and output data
    # Stage 1 ignores ISI, so all sections are combined
    # ----------------------------------------------------------------------
    x_con = np.array(list(itertools.chain.from_iterable(x)))
    y_con = np.array(list(itertools.chain.from_iterable(y)))
    nsec = len(x_con)
    
    # ----------------------------------------------------------------------
    # Step 2: Define loss function for ADM (alpha, delta, mu) stage 1
    # ----------------------------------------------------------------------
    def compute_loss_adm(varx):
        """
        Loss function for stage 1 optimization.
        varx = [beta, gamma]
        """
        beta, gamma = varx
    
        # Compute mu based on beta and gamma
        # Neglect initial samples where leakage may not be properly captured
        mu = (m ** 2 / nsec / tau) * np.sum(
            (y_con - tau / m * (beta * x_con + m * gamma)) ** 2 / (beta * x_con + m * gamma)
        )
    
        # Compute cost
        cost = cost_function(nsec, oversampling_training, m, x_con, y_con, tau, beta, gamma, mu)
    
        # Penalize negative parameters
        if beta < 0 or gamma < 0:
            cost += 1e9
    
        return cost
    
    # ----------------------------------------------------------------------
    # Step 3: Set bounds and perform optimizations with multiple solvers
    # ----------------------------------------------------------------------
    bounds_adm = [(0, np.inf), (0, np.inf)]
    result_adm = {}
    
    # 1) Nelder-Mead (local optimization, may fail for ill-conditioned problems)
    result_adm['NM'] = optimize.minimize(compute_loss_adm, [beta_init, gamma_init], method="Nelder-Mead")
    print("Nelder-Mead method:")
    print(f"Converged results: {result_adm['NM'].x}, cost: {result_adm['NM'].fun}")
    
    # 2) L-BFGS-B (local optimization with bounds)
    result_adm['BFGS'] = optimize.minimize(compute_loss_adm, [beta_init, gamma_init],
                                           bounds=bounds_adm, method="L-BFGS-B")
    print("L-BFGS-B method:")
    print(f"Converged results: {result_adm['BFGS'].x}, cost: {result_adm['BFGS'].fun}")
    
    # 3) Basinhopping (global optimization)
    minimizer_kwargs = {"method": "L-BFGS-B"}
    result_adm['basin'] = optimize.basinhopping(
        compute_loss_adm, [beta_init, gamma_init],
        minimizer_kwargs=minimizer_kwargs, niter=200
    )
    print("Basinhopping method (global optimization):")
    print(f"Converged results: {result_adm['basin'].x}, cost: {result_adm['basin'].fun}")
    
    # ----------------------------------------------------------------------
    # Step 4: Select the best result based on the lowest cost
    # ----------------------------------------------------------------------
    costs_adm = {key: result_adm[key].fun for key in result_adm}
    best_minimizer = min(costs_adm, key=costs_adm.get)
    result_adm_best = result_adm[best_minimizer]
    
    # ----------------------------------------------------------------------
    # Step 5: Post-process the best solution
    # ----------------------------------------------------------------------
    post_process_adm(result_adm_best.x, nsec, oversampling_training, m, tau, x_con, y_con)
    
    # Save best parameters for further stages
    beta0, gamma0 = result_adm_best.x

    
    
    print("Stage 2 optimization: estimating taud, mu, sigma.\n")

    # ----------------------------------------------------------------------
    # Step 1: Set parameters
    # ----------------------------------------------------------------------
    R = 500  # Maximum number of past samples to include in Q, chosen conservatively for td <= 0.5
    
    # ----------------------------------------------------------------------
    # Step 2: Define loss function for stage 2
    # ----------------------------------------------------------------------
    def compute_loss_tms(vary):
        """
        Loss function for Stage 2 optimization.
        
        Parameters
        ----------
        vary : array
            [tn, mu, sigma] where:
            tn : float, normalized tau_d / ts
            mu : float
            sigma : float
        
        Returns
        -------
        cost : float
            Normalized negative log-likelihood cost
        """
        tn, mu, sigma = vary
    
        # ------------------------------------------------------------------
        # Step 2a: Compute the Q sequence
        # ------------------------------------------------------------------
        # Initial Q value
        qtemp = 1 - tn * (1 - np.exp(-1 / tn))
        qj = np.array([qtemp])
    
        # Calculate Q for R-1 steps
        for i in range(R - 1):
            qtemp = tn * (np.exp(-(i + 2)/tn) + np.exp(-i/tn) - 2 * np.exp(-(i + 1)/tn))
            qj = np.append(qj, qtemp)
    
        # ------------------------------------------------------------------
        # Step 2b: Compute cost across all data collections
        # ------------------------------------------------------------------
        cost = 0
        n_w_eff = 0  # Effective number of samples considered

        for i_col in range(len(x)):
            s_temp = s_bit[i_col]
            w_temp = w_out[i_col]
    
            # Upsample input sequence
            seq_sample = np.repeat(s_temp, oversampling_training)
            Dvar = (beta0 * seq_sample + gamma0) * ts
    
            # Compute contribution of each output sample
            for i in range(len(w_temp)):
                # Select the appropriate Q and D sequences
                if i <= R - 1:
                    Dmul = np.flip(Dvar[:i + 1])
                    Qmul1 = qj[:i + 1]
                    Qmul2 = mu * qj[:i + 1] ** 2
                else:
                    Dmul = np.flip(Dvar[i - R + 1:i + 1])
                    Qmul1 = qj
                    Qmul2 = mu * qj ** 2
    
                # Compute mean and variance of the predicted output
                mean = np.sum(Dmul * Qmul1)
                var = np.sum(Dmul * Qmul2) + sigma ** 2
    
                # Neglect first few samples where leakage is not properly captured
                neg = 20
                if i >= neg:
                    cost += np.log(var) + (w_temp[i] - mean) ** 2 / var
    
            n_w_eff += len(w_temp) - neg

        # Penalize unrealistic parameters
        if mu < 0.1 or tn < 0.5:
            cost += 1e9
    
        return cost / n_w_eff
    
    # ----------------------------------------------------------------------
    # Step 3: Perform local optimization using Nelder-Mead
    # ----------------------------------------------------------------------
    tn_init = 0.5  # Initial guess for tn
    result_com = optimize.minimize(compute_loss_tms, [tn_init, mu_init, np.sqrt(sigmas_init)], method="Nelder-Mead")
    
    print("Nelder-Mead method (may fail if problem is ill-conditioned):")
    # Ensure sigma is positive
    result_com.x[2] = abs(result_com.x[2])
    print("Optimized parameters:", result_com.x)
    print("Cost:", result_com.fun)
    
    # ----------------------------------------------------------------------
    # Step 4: Save convergent parameters
    # ----------------------------------------------------------------------
    td_0 = result_com.x[0] * ts
    mu_0 = result_com.x[1]
    sigma_0 = result_com.x[2]
    
    print(
        f"Convergent results: alpha = {beta0 / mu_0}, delta = {gamma0 / mu_0}, "
        f"mu = {mu_0}, sigma = {sigma_0}, td = {td_0}\n"
    )

    return beta0 / mu_0, gamma0 / mu_0, td_0, mu_0, sigma_0**2


def padded_bin(i, width):
    # Convert integer i to binary string
    s = bin(i)
    
    # Remove '0b' prefix and pad with leading zeros to match desired width
    return s[2:].zfill(width)



def sequence_detection(alpha, delta, mu, td, sigma, seq_bit, y_bit, K, tau):
    """
    Perform sequence detection using Viterbi algorithm.
    See Secion 2 in the supplementary material.
    
    Parameters
    ----------
    alpha, delta, mu, td, sigma : float
        System parameters (light/dark intensity, internal gain, device response time, and noise STD).
    seq_bit : list of int
        Input bit sequence.
    y_bit : list of float
        Observed (measured) outputs.
    K : int
        Constraint length (number of bits considered in the trellis state).
    tau : float
        Bit interval.

    Returns
    -------
    seq_recover : list of int
        Recovered input bit sequence.
    """

    # Step 1: Compute ISI coefficients Q for K bits
    Q = []
    Q.append(mu * (1 - td / tau * (1 - np.exp(-tau / td))))  # First coefficient
    for kk in range(1, K):
        Q.append(mu * td / tau * np.exp(-(kk + 1) * tau / td) * (1 - np.exp(tau / td)) ** 2)

    # Step 2: Group observed outputs by bit patterns (for sanity check only)
    bit_group = {}
    for kk in range(K - 1, len(seq_bit)):
        bit_sec = seq_bit[kk - K + 1 : kk + 1]
        key_sec = ''.join(str(ele) for ele in bit_sec)
        if key_sec in bit_group:
            bit_group[key_sec].append(y_bit[kk])
        else:
            bit_group[key_sec] = [y_bit[kk]]

    # Step 3: Build trellis states and expected outputs
    key_set = []          # All K-bit state keys
    key_decode_set = []   # All (K-1)-bit state keys
    exp_group = {}        # Expected vs true means/vars (for sanity check)
    state_output = {}     # Expected output mean/variance for each state

    for i in range(2 ** K):
        key_sec = padded_bin(i, K)
        key_set.append(key_sec)

        # Build (K-1)-bit state keys
        if i < 2 ** (K - 1):
            key_sec_decode = padded_bin(i, K - 1)
            key_decode_set.append(key_sec_decode)

        # Convert bit string to list of ints
        bit_sec = [int(ele) for ele in key_sec]

        # Expected output mean/variance for this state
        D_sec = [(alpha * m + delta) * tau for m in bit_sec]
        output_expect_mean = np.sum(np.array(D_sec) * np.array(Q))
        output_expect_var = np.sum(np.array(D_sec) * (np.array(Q) ** 2)) + sigma ** 2
        state_output[key_sec] = [output_expect_mean, output_expect_var]

        # Optional sanity check with observed outputs
        if key_sec in bit_group:
            output_sec = bit_group[key_sec]
            output_mean = np.mean(output_sec)
            output_var = np.var(output_sec)
            exp_group[key_sec] = [
                [output_expect_mean, output_expect_var],
                [output_mean, output_var],
            ]

    # Step 4: Initialize trellis (forward path)
    key_decode_set.sort()
    pre_state = {key: math.inf for key in key_decode_set}
    pre_state[key_decode_set[0]] = 0  # Start from all-zero state
    state_evolve = [pre_state]

    # Step 5: Forward recursion (update state metrics)
    for output_bit in range(len(seq_bit)):
        curr_state = {}
        for state in key_decode_set:
            key_curr = [int(ele) for ele in state]

            # Build predecessor states
            pre_key1 = ''.join(str(ele) for ele in (key_curr[1:] + [0]))
            pre_key2 = ''.join(str(ele) for ele in (key_curr[1:] + [1]))

            # Build ISI-extended predecessors
            pre_key1_ISI = ''.join(str(ele) for ele in (key_curr + [0]))
            pre_key2_ISI = ''.join(str(ele) for ele in (key_curr + [1]))

            # Path metric update using Gaussian log-likelihood
            curr_state[state] = min(
                pre_state[pre_key1]
                + (y_bit[output_bit] - state_output[pre_key1_ISI][0]) ** 2
                / (2 * state_output[pre_key1_ISI][1])
                + np.log(state_output[pre_key1_ISI][1]) / 2,
                pre_state[pre_key2]
                + (y_bit[output_bit] - state_output[pre_key2_ISI][0]) ** 2
                / (2 * state_output[pre_key2_ISI][1])
                + np.log(state_output[pre_key2_ISI][1]) / 2,
            )
        state_evolve.append(curr_state)
        pre_state = curr_state

    # Step 6: Backtrace to recover most likely bit sequence
    seq_rev = []
    pre_keys = []
    for bit_check in range(len(seq_bit), 0, -1):
        path_metric = list(state_evolve[bit_check].values())

        # Choose the best path (handle first step separately)
        if not pre_keys:
            idx = path_metric.index(min(path_metric))
        else:
            path_metric_arr = np.array(path_metric)
            path_metric_temp = list(path_metric_arr[pre_keys])
            idx = pre_keys[0] if path_metric_temp[0] < path_metric_temp[1] else pre_keys[1]

        # Extract current key and prepend recovered bit
        key_curr = [int(ele) for ele in key_decode_set[idx]]
        seq_rev.append(key_curr[0])

        # Update predecessor indices for backtracking
        pre_key1 = ''.join(str(ele) for ele in (key_curr[1:] + [0]))
        pre_key2 = ''.join(str(ele) for ele in (key_curr[1:] + [1]))
        pre_keys = [key_decode_set.index(pre_key1), key_decode_set.index(pre_key2)]

    # Reverse to obtain final recovered sequence
    seq_recover = list(reversed(seq_rev))

    return seq_recover



def rate_calculation(seq_bit, seq_recover):
    """
    Calculate detection accuracy metrics by comparing true vs. recovered sequences.
    
    Returns normalized rates for: [Accuracy, False Positive, False Negative, True Positive, True Negative].
    """
    seq_truth = np.array(seq_bit)      # Ground truth sequence
    seq_detect = np.array(seq_recover) # Recovered (detected) sequence
    
    # Count true positives (TP): both truth and detected are 1
    TP = np.sum(np.where(np.logical_and(seq_truth, seq_detect), 1, 0))
    # Count true negatives (TN): both truth and detected are 0
    TN = seq_truth.shape[0] - np.sum(np.where(np.logical_xor(seq_truth, seq_detect), 1, 0)) - TP
    # Count false negatives (FN): truth=1 but detected=0
    FN = np.sum(np.where(np.logical_and(seq_truth, np.logical_not(seq_detect)), 1, 0))
    # Count false positives (FP): truth=0 but detected=1
    FP = seq_truth.shape[0] - TP - TN - FN

    # Accuracy = TP + TN; total samples = N
    ACC = TP + TN
    N = ACC + FN + FP

    # Return normalized rates
    rates = [ACC/N, FP/N, FN/N, TP/N, TN/N]    
    return rates


def output_generation(seq_bit, k, alpha, delta, mu, sigma, td, tau):
    """
    Reconstruct sample-level output sequence based on input bits and estimated system parameters.
    Models photon/electron arrivals as a Poisson process with exponential inter-arrival times
    and integrates impulse response over time.

    Parameters
    ----------
    seq_bit : list of int
        Input bit sequence (0 or 1).
    k : int
        Oversampling factor (samples per bit).
    alpha : float
        Intensity scaling for bit=1 (signal).
    delta : float
        Dark current intensity (baseline when bit=0).
    mu : float
        Internal gain
    sigma : float
        Standard deviation of Gaussian noise added to output.
    td : float
        Device response time.
    tau : float
        Bit interval.

    Returns
    -------
    y_out : np.ndarray
        Generated sample-level output signal.
    """

    # --- Preprocessing ---
    # Add 10 zero bits to the beginning so the system reaches a stationary state
    seq_add = list(np.zeros(10))
    seq_bit = seq_add + seq_bit
    length = len(seq_bit)

    # Oversample input sequence by factor k
    seq_sample = np.repeat(seq_bit, k)

    # Tau per sample (smaller interval within each bit)
    tau_sample = tau / k

    # --- Particle arrival time generation ---
    time_arr = []   # store arrival times
    number_arr = [] # store number of arrivals per interval
    for n in range(length * k):
        if seq_sample[n] == 0:
            # Dark current only
            ti = np.random.exponential(1 / delta, (1, int(10 * delta * tau_sample)))
        else:
            # Signal + dark current
            ti = np.random.exponential(1 / (alpha + delta), (1, int(10 * (alpha + delta) * tau_sample)))

        # Arrival times shifted by bit interval offset
        t_temp = np.cumsum(ti) + n * tau_sample
        # Keep only arrivals within current interval
        t_arr = t_temp[t_temp < (n + 1) * tau_sample]

        # Save results
        time_arr.extend(t_arr)
        number_arr.append(len(t_arr))

    # --- Impulse response integration ---
    # Determine number of intervals required to capture impulse response tail
    n_inter = 0
    cnt = 0
    epsilon = 1e-5
    while abs(cnt - mu) > epsilon:
        n_inter += 1
        cnt = mu * (1 - math.exp(-n_inter * tau_sample / td))

    # Allocate output array
    y_out = np.zeros(len(seq_sample) + n_inter + 1)

    # For each particle arrival, compute contribution to subsequent intervals
    for tp in time_arr:
        y_temp = np.zeros(len(seq_sample) + n_inter + 1)
        ni = int(np.floor(tp / tau_sample))  # interval index where particle arrives

        # Contribution in arrival interval
        y_temp[ni] = mu - mu * math.exp(-((ni + 1) * tau_sample - tp) / td)

        # Contributions in subsequent intervals
        for j in range(1, n_inter):
            y_temp[ni + j] = (
                mu * math.exp(-((ni + j) * tau_sample - tp) / td)
                - mu * math.exp(-((ni + j + 1) * tau_sample - tp) / td)
            )
        y_out += y_temp

    # --- Postprocessing ---
    # Remove initial 10 zero-bit padding, add Gaussian noise
    y_out = y_out[10 * k:length * k] + np.random.normal(0, sigma, (length - 10) * k)

    return y_out



def sample_to_bit_out(w, k):
    """
    Convert sample-level data back to bit-level values by summing samples per bit.
    
    Parameters
    ----------
    w : array-like
        Sample-level data sequence.
    k : int
        Oversampling factor (number of samples per bit).
        
    Returns
    -------
    y_bit : np.ndarray
        Bit-level values obtained by summing samples within each bit interval.
    """
    nbit = int(len(w) / k)                  # number of bits
    w_temp = np.reshape(w, (nbit, k))       # reshape into (nbit, k) for grouping
    w_temp = np.transpose(w_temp)           # transpose to align bits across rows
    y_bit = np.sum(w_temp, axis=0)          # sum across k samples per bit
    return y_bit


def TP_FP_cal(seq_input, seq_output, th):
    """
    Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    for different decision thresholds.
    
    Parameters
    ----------
    seq_input : array-like
        Ground-truth binary input sequence.
    seq_output : array-like
        Continuous detection output values.
    th : np.ndarray
        Array of thresholds to evaluate.
        
    Returns
    -------
    [TPR, FPR] : list of lists
        TPR and FPR values for each threshold.
    """
    seq_input = np.array(seq_input)
    seq_output = np.array(seq_output)
    TPR = []
    FPR = []

    for i in range(th.shape[0]):
        th_temp = th[i]
        seq_detect = seq_output > th_temp    # detected sequence given threshold
        seq_truth = seq_input > 0            # ground truth (boolean mask)

        # Compute confusion matrix elements
        TP = np.sum(np.where(np.logical_and(seq_truth, seq_detect), 1, 0))
        TN = seq_truth.shape[0] - np.sum(np.where(np.logical_xor(seq_truth, seq_detect), 1, 0)) - TP
        FN = np.sum(np.where(np.logical_and(seq_truth, np.logical_not(seq_detect)), 1, 0))
        FP = seq_truth.shape[0] - TP - TN - FN

        # Compute rates
        TPR.append(TP / (TP + FN))           # sensitivity (true positive rate)
        FPR.append(FP / (FP + TN))           # false alarm rate (false positive rate)

    return [TPR, FPR]


def AUR_cal(ROC):
    """
    Calculate the Area Under the ROC curve (AUR) using trapezoidal approximation.
    
    Parameters
    ----------
    ROC : list [TPR, FPR]
        ROC curve data: TPR (y-axis) and FPR (x-axis).
        
    Returns
    -------
    float
        Approximated AUR value.
    """
    TP_diff = np.diff(ROC[0])                   # differences in TPR
    AUR_lower = -np.sum(np.array(ROC[1][:-1]) * TP_diff)
    AUR_upper = -np.sum(np.array(ROC[1][1:]) * TP_diff)
    return (AUR_lower + AUR_upper) / 2          # trapezoid average


def write_lists_to_csv(file_path, names, *lists):
    """
    Write multiple lists into a CSV file with headers.
    
    Parameters
    ----------
    file_path : str
        Path to the output CSV file.
    names : list of str
        Column names (header row).
    *lists : lists
        Data lists to write as columns.
    """
    # Combine header and data
    rows = [names] + list(zip(*lists))

    # Write to CSV
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)
        

def sequence_metrics(params_proposed, params_traditional, seq_bit, output_bit, K, tau, oversampling):
    """
    Evaluate detection performance of proposed, traditional, and threshold-based methods.
    
    Parameters
    ----------
    params_proposed : tuple
        Proposed (single-shot measurement) model parameters (alpha, delta, mu, td, sigma).
    params_traditional : tuple
        Traditional (sequential) model parameters (alpha, delta, mu, td, sigma).
    seq_bit : array-like
        Ground truth bit sequence.
    output_bit : array-like
        Observed noisy output sequence.
    K : int
        Oversampling factor.
    tau : float
        Bit interval.
    oversampling : int
        Oversampling factor for noise scaling.
        
    Returns
    -------
    detection_results : list
        Detection rates [proposed, traditional, threshold-based].
    """
    # --- Proposed method ---
    alpha_proposed, delta_proposed, mu_proposed, td_proposed, sigma_proposed = params_proposed
    seq_proposed = sequence_detection(
        alpha_proposed, delta_proposed, mu_proposed, td_proposed,
        np.sqrt(oversampling * sigma_proposed**2),
        seq_bit, output_bit, K, tau
    )
    rates_proposed = rate_calculation(seq_bit, seq_proposed)
    
    # --- Threshold-based method ---
    thresh = np.mean(np.array(output_bit))                  # threshold = mean output
    seq_thresh = np.where(np.array(output_bit) > thresh, 1, 0)
    rates_thresh = rate_calculation(seq_bit, seq_thresh)

    # --- Traditional method ---
    alpha_traditional, delta_traditional, mu_traditional, td_traditional, sigma_traditional = params_traditional
    seq_traditional = sequence_detection(
        alpha_traditional, delta_traditional, mu_traditional, td_traditional,
        np.sqrt(oversampling * sigma_traditional**2),
        seq_bit, output_bit, K, tau
    )
    rates_traditional = rate_calculation(seq_bit, seq_traditional)

    # Collect results
    detection_results = [rates_proposed, rates_traditional, rates_thresh]
    return detection_results



#%% ==========================================================================
# User-adjustable parameters for estimation, regeneration, and detection
# ==========================================================================

# --- General settings ---
# Number of bits grouped together as one section in Stage 1
m = 20

# Oversampling ratios (training = estimation stage, testing = detection stage)
oversampling_training = 8
oversampling_testing = 4

# Conversion from raw detector counts to electrons:
# - If measurements are already in electrons, set conv_factor = 1 and eta_set = 0
# - Otherwise, adjust conv_factor and eta_set based on system calibration
conv_factor = 10.3759765625     # counts-to-electrons conversion factor (system-specific)
eta_set     = 3652.27           # baseline offset in counts (system-specific)

# Sample-level interval (time per raw detector sample)
ts = 34836e-6   # seconds

# Viterbi decoder settings
max_viterbi_bit = 15             # maximum ISI (constraint length)

# Bit-level intervals
tau        = ts * oversampling_training   # training (estimation stage)
tau_detect = ts * oversampling_testing    # testing (Viterbi decoding stage)


# --- Parameters from traditional sequential estimation methods ---
alpha_traditional = 676.5725634
delta_traditional = 186.3316544
td_traditional    = 0.41714
mu_traditional    = 69.75224132
sigma_traditional = 70

# Pack into parameter list for convenience
params_traditional = [
    alpha_traditional,
    delta_traditional,
    mu_traditional,
    td_traditional,
    sigma_traditional
]


#%% ==========================================================================
# Parametric Estimation
# ==========================================================================

# Initialize containers
s, w, x, y = [], [], [], []
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
    s.append(stemp)

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
    w.append(wtemp)
    x.append(xtemp)
    y.append(ytemp)
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
    s, w, x, y, ts, tau,
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
        s, w, x, y, ts, tau,
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
    print(f"Iteration {i}: error = {err:.2e}\n")

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

# --- Save estimated parameters ---
csv_file_path = "results/estimation/estimated_parameters.csv"
write_lists_to_csv(
    csv_file_path,
    ["alpha", "delta", "td", "mu", "sigma", "converged"],
    [alpha_conv],
    [delta_conv],
    [td_conv],
    [mu_conv],
    [np.sqrt(sigmas_conv)],
    [conv],
)

# Pack final results into a convenient list
params_proposed = [
    alpha_conv, delta_conv, mu_conv, td_conv, np.sqrt(sigmas_conv)
]


#%% ==========================================================================
# Load test data (without pilot)
# ==========================================================================
y_bit_proposed = []
seq_bit = []

for detect_sec in range(3):
    # --- Load detector output samples ---
    randomfile = f"data_processed/sample_output_without_pilot_test_sec{detect_sec}.csv"
    with open(randomfile, newline='') as input_file:
        randomtemp = list(map(int, list(csv.reader(input_file))[0]))
    
    # Convert sample-level outputs to bit-level, then to electrons
    randomtemp = sample_to_bit_out(randomtemp, oversampling_testing)
    randomtemp = cnts_to_electrons(randomtemp, eta_set * oversampling_testing, conv_factor)
    y_bit_proposed.extend(randomtemp)

    # --- Load ground truth input bits ---
    detectfile = f"data_processed/bit_input_without_pilot_test_sec{detect_sec}.csv"
    with open(detectfile, newline='') as input_file:
        detecttemp = list(map(int, list(csv.reader(input_file))[0]))
    seq_bit.extend(detecttemp)


#%% ==========================================================================
# Sequence detection performance evaluation
# ==========================================================================

# --- On training set ---
# Consider ISI across ~5*td/tau bits, but clip to [5, max_viterbi_bit]
K_train = min(max(int(5 * td_conv / tau), 5), max_viterbi_bit)

# Flatten input/output sequences
s_bit_train = [bit for sub_s in s for bit in sub_s]
w_sample_train = [sample for sub_w in w for sample in sub_w]

# Convert training samples to bit-level outputs
output_bit_train = [
    sum(w_sample_train[i : i + oversampling_training])
    for i in range(0, len(w_sample_train), oversampling_training)
]

# Compute detection metrics
rates_train = sequence_metrics(
    params_proposed, params_traditional,
    s_bit_train, output_bit_train,
    K_train, tau, oversampling_training
)


# --- On test set ---
# Consider ISI across ~5*td/tau_detect bits
K_test = min(max(int(5 * td_conv / tau_detect), 5), max_viterbi_bit)

rates_test = sequence_metrics(
    params_proposed, params_traditional,
    seq_bit, y_bit_proposed,
    K_test, tau_detect, oversampling_testing
)


#%% ==========================================================================
# Save detection results
# ==========================================================================

csv_file_path = "results/detection/error_rates.csv"
labels = ["ACC", "FP", "FN", "TP", "TN"]

write_lists_to_csv(
    csv_file_path,
    ["Label",
     "Train_proposed", "Train_traditional", "Train_thresh",
     "Test_proposed",  "Test_traditional",  "Test_thresh"],
    labels,
    rates_train[0], rates_train[1], rates_train[2],
    rates_test[0],  rates_test[1],  rates_test[2],
)


#%% ==========================================================================
# Regeneration of output samples
# ==========================================================================
# When reconstructing outputs, section connections are neglected

# --- Generate synthetic outputs ---
synthetic_sample_proposed = output_generation(
    s[0], oversampling_training,
    alpha_conv, delta_conv, mu_conv, np.sqrt(sigmas_conv),
    td_conv, tau
)

synthetic_sample_traditional = output_generation(
    s[0], oversampling_training,
    alpha_traditional, delta_traditional, mu_traditional, sigma_traditional,
    td_traditional, tau
)

# --- Save reconstruction data ---
s_sample = [bit for bit in s[0] for _ in range(oversampling_training)]
csv_file_path = "results/reconstruction/reconstructed_data.csv"

write_lists_to_csv(
    csv_file_path,
    ["Sample_input", "Actual", "Proposed", "Traditional"],
    s_sample, w[0], synthetic_sample_proposed, synthetic_sample_traditional
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

    ax.plot(idx_list, np.array(synthetic_sample_proposed)[idx_list], label="Regenerated output (Est)")
    ax.plot(idx_list, np.array(synthetic_sample_traditional)[idx_list], label="Regenerated output (Exp)")
    ax.plot(idx_list, np.array(w[0])[idx_list], label="True output")

    ax.set_ylabel("Sample level output")
    ax.set_title(f"Data collection {i}")
    ax.legend(fontsize=20, loc="lower right")
    plt.rc("axes", labelsize=20, titlesize=20)


#%% ==========================================================================
# ROC curve computation
# ==========================================================================

# --- Aggregate samples into bit-level outputs ---
bit_block = oversampling_training
recons_bit_proposed = [sum(synthetic_sample_proposed[i:i+bit_block]) for i in range(0, len(synthetic_sample_proposed), bit_block)]
recons_bit_traditional = [sum(synthetic_sample_traditional[i:i+bit_block]) for i in range(0, len(synthetic_sample_traditional), bit_block)]
true_bit = [sum(w[0][i:i+bit_block]) for i in range(0, len(w[0]), bit_block)]

# --- Thresholds ---
minn, maxx = min(true_bit), max(true_bit)
th = np.linspace(minn, maxx, 4000, endpoint=False)

# --- Compute ROC curves ---
ROC_true = TP_FP_cal(s[0], true_bit, th)
ROC_proposed = TP_FP_cal(s[0], recons_bit_proposed, th)
ROC_traditional = TP_FP_cal(s[0], recons_bit_traditional, th)

# --- Save ROC results ---
csv_file_path = "results/reconstruction/ROC.csv"
write_lists_to_csv(
    csv_file_path,
    ["TP_true", "FP_true", "TP_proposed", "FP_proposed", "TP_traditional", "FP_traditional"],
    ROC_true[0], ROC_true[1],
    ROC_proposed[0], ROC_proposed[1],
    ROC_traditional[0], ROC_traditional[1],
)

# --- Plot ROC curves ---
plt.figure(figsize=(8, 6))
plt.plot(ROC_true[0], ROC_true[1], "-", linewidth=7, label="True output")
plt.plot(ROC_proposed[0], ROC_proposed[1], "-", linewidth=4, label="Estimation reconstruction")
plt.plot(ROC_traditional[0], ROC_traditional[1], "-", linewidth=2, label="Experimental reconstruction")

plt.title("FP vs TP for pixel 286,128")
plt.xlabel("TP")
plt.ylabel("FP")
plt.ylim([-0.01, 1])
plt.legend()


#%% ==========================================================================
# Area under ROC (AUR)
# ==========================================================================

AUR_true = AUR_cal(ROC_true)
AUR_proposed = AUR_cal(ROC_proposed)
AUR_traditional = AUR_cal(ROC_traditional)

csv_file_path = "results/reconstruction/AUR.csv"
write_lists_to_csv(
    csv_file_path,
    ["Actual", "Proposed", "Traditional"],
    [AUR_true], [AUR_proposed], [AUR_traditional],
)


#%% ==========================================================================
# Error rate summaries (Train/Test)
# ==========================================================================
Err_train = [1 - rates_train[j][0] for j in range(3)]
Err_test = [1 - rates_test[j][0] for j in range(3)]







