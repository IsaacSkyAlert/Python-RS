"""
Created on 2024
@author: A. Isaac
"""
import numpy as np
# DEFINICIÓN DE VARIABLES: Vectores en Mayúsculas y escalares en minúsculas.
# EJEMPLO:New_Data=Vector=[1 2 3 ... n];sensitivity_instrumet=escalar = 1.0
_sampling_rate = 100
def Remove_Trend(Signal):
    import numpy as np
    mean = np.mean(Signal)
    Detrend_Signal = Signal - mean
    return Detrend_Signal

_acc_sensitivity_instrumet = 384500/100
def Tranform_Counts_to_Gals(Data):
    New_Data = Data/_acc_sensitivity_instrumet
    return New_Data

_vel_sensitivity_instrumet = 399650000/100
def Tranform_Counts_to_Vel(Data):
    New_Data = Data/_vel_sensitivity_instrumet
    return New_Data

nsta = int(15 * _sampling_rate)
nlta = int(40 * _sampling_rate)
clean_value = 6*nlta
constant_norm_sta = 1. / nsta 
constant_norm_lta = 1. / nlta
complement_sta = 1 - constant_norm_sta 
complement_lta = 1 - constant_norm_lta
def STA_LTA(Signal):
    ndat = len(Signal)
    sta = 0.
    lta = np.finfo(0.0).tiny 
    Signal = np.square(Signal)
    Caracteristic_Function = np.zeros(ndat, dtype=np.float64)
    for i in range(1, ndat):
        sta = constant_norm_sta * Signal[i] + complement_sta * sta
        lta = constant_norm_lta * Signal[i] + complement_lta * lta
        Caracteristic_Function[i] = np.sqrt((sta / lta)**2) - 1
    Caracteristic_Function[:clean_value] = 0
    return Caracteristic_Function
    
thr_on = 0.80
def Anomaly_Did_Start(STA_LTA_Signal):
    for i in range(nlta, len(STA_LTA_Signal)):
        if STA_LTA_Signal[i] > thr_on:
            return i
    return None
 
thr_off = 0.001 
def Anomaly_Did_End(STA_LTA_Signal, n_act):
    if n_act <= 0:
        return None
    n_start = n_act + 1
    for j in range(n_start, len(STA_LTA_Signal)):
        if STA_LTA_Signal[j] < thr_off:
            return j
    return None
    
def Get_Slice(Signal, i, j):
    Slice = Signal[i:j]
    return Slice

def Get_Max_Amplitude(Signal):
    max_amplitude = max(Signal**2)
    return round(np.sqrt(max_amplitude),2)

def Compute_PGA(a_ENZ,a_ENN,a_ENE):
    PGA = (a_ENZ + a_ENN + a_ENE)/3
    return round(PGA,2)

def SkyAlert_Scale(gals):
    if gals < 1:
        return "Imperceptible"
    elif 1 <= gals <= 5:
        return "Débil"
    elif 5.01 <= gals <= 21:
        return "Leve"
    elif 21.01 <= gals <= 35:
        return "Moderado"
    elif 35.01 <= gals <= 77:
        return "Fuerte"
    elif 77.01 <= gals <= 156:
        return "Violento"
    else:
        return "Severo"
