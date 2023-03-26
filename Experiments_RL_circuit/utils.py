import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as nn
import PySpice as ps

# simulate ECG data for 50 patients, 20 seconds and 200 Hz with different heart_rates from Normal(70, 10) distribution
def simulate_ecg_data(n_patients=50, duration=20, sampling_rate=200, heart_rate_mean=70, heart_rate_std=10):
    # create a list of heart rates
    heart_rates = np.random.normal(heart_rate_mean, heart_rate_std, n_patients)
    # create a list of ECG signals
    ecg_signals = []
    # create a list of ECG signals
    for heart_rate in heart_rates:
        # create a list of ECG signals
        ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate)
        # append the list of ECG signals
        ecg_signals.append(ecg_signal)
    # return the list of ECG signals
    return ecg_signals


# generate a dataframe of voltage conditional on ecg_data using multivariate gaussian distribution
def generate_voltage(ecg_data):
    voltages = []
    for ecg_signal in ecg_data:
        # create a list of voltages
        voltage = np.random.normal(2*ecg_signal, 0.2)
        # append the list of voltages
        voltages.append(voltage)
    # return the list of voltages
    return voltages



# get signal rate and other heart rate variability metrices from ECG data as our static features
def get_signal_rate(ecg_data):
    # create a list of metrics
    features = []
    for ecg_signal in ecg_data:
        ecg_processed, info = nk.ecg_process(ecg_signal, sampling_rate=200)
        hrv =  nk.hrv(ecg_processed, sampling_rate=200, show=False)
        features.append(hrv)
    # return the list of hrv metrics
    return features




# write a transformer encoder function for ECG data to get dynamic feature V (the output of the last encoder layer) and two static features R and L
def transformer_encoder(ecg_data, static_features, n_layers=6, n_heads=8, d_model=64, d_ff=256, dropout=0.1):
    # create a list of dynamic features
    dynamic_features = []
    # create a list of static features
    static_features = []
    # create a list of static features
    for ecg_signal in ecg_data:
        # create a list of dynamic features
        dynamic_feature = nk.ecg_encode(ecg_signal, sampling_rate=200, method="transformer", n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
        # append the list of dynamic features
        dynamic_features.append(dynamic_feature)
    # return the list of dynamic features and static features
    return dynamic_features, static_features

# write the ecg_encode function to return a dynamic feature V (the output of the last encoder layer) and two static features R and L


