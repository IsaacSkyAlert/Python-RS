"""
Created on 2024
@author: A. Isaac
"""
# DEFINICIÓN DE VARIABLES: Vectores en Mayúsculas y escalares en minúsculas.
# EJEMPLO:New_Data=Vector=[1 2 3 ... n];sensitivity_instrumet=escalar = 1.2
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

nsta = 40
nlta = 600
clean_value = 150
constant_norm_sta = 1. / nsta 
constant_norm_lta = 1. / nlta
complement_sta = 1 - constant_norm_sta 
complement_lta = 1 - constant_norm_lta
def STA_LTA(Signal):
    import numpy as np
    ndat = len(Signal)
    sta = 0.
    lta = np.finfo(0.0).tiny 
    Signal = np.square(Signal)
    Caracteristic_Function = np.zeros(ndat, dtype=np.float64)
    for i in range(1, ndat):
        sta = constant_norm_sta * Signal[i] + complement_sta * sta
        lta = constant_norm_lta * Signal[i] + complement_lta * lta
        Caracteristic_Function[i] = sta / lta
    Caracteristic_Function[:clean_value] = 0 
    return Caracteristic_Function
 
thr_on = 6.62
def Anomaly_Did_Start(Signal):
    STA_LTA_Signal = STA_LTA(Signal)
    for i in range(len(STA_LTA_Signal)):
        if STA_LTA_Signal[i] > thr_on:
            return i
    return 0
    
thr_off = 0.3    
def Anomaly_Did_End(Signal, n_act):
    if n_act <= 0:
        return None
    
    STA_LTA_Signal = STA_LTA(Signal)
    n_start = n_act + 1
    for j in range(n_start,len(STA_LTA_Signal)):
        if STA_LTA_Signal[j] < thr_off:
            return j
    return 0

def Detect_Anomaly_Plot(Signal, start_time):
    import numpy as np
    import matplotlib.pyplot as plt
    from obspy.core import UTCDateTime
    STA_LTA_Signal = STA_LTA(Signal)
    Time = np.arange(len(Signal)) / _sampling_rate
    start_time_utc = UTCDateTime(start_time)
    n_act = Anomaly_Did_Start(Signal)
    act_time = start_time_utc + (n_act / _sampling_rate)
    if n_act >0:
        n_des = Anomaly_Did_End(Signal, n_act)
        des_time = start_time_utc + (n_des / _sampling_rate)
    else:
        n_des = 0
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(Time, Signal)
    ax[0].set_xlabel('Tiempo [s]')
    ax[0].set_ylabel('Amplitud')
    ax[0].set_title(f'Inicio de señal en UTC:{start_time}')
    if n_act > 0 and n_des > 0:
        ax[0].axvline(x=n_act/_sampling_rate, color='r', linestyle='--', 
                  label=f'Inicio de anomalía a las {act_time.datetime.strftime("%H:%M:%S")}')
        ax[0].axvline(x=n_des/_sampling_rate, color='b', linestyle='--', 
                  label=f'Fin de anomalía a las {des_time.datetime.strftime("%H:%M:%S")}')
    else:
        ax[0].axvline(x=0, color='w', label='Sin anomalía')
    ax[0].legend()
    ax[1].plot(Time, STA_LTA_Signal, label='STA/LTA Signal')
    ax[1].axhline(y=thr_on, color='r', linestyle='--', 
                  label=f'Umbral activación={thr_on}', alpha=0.5)
    ax[1].axhline(y=thr_off, color='b', linestyle='--', 
                  label=f'Umbral desactivación={thr_off}', alpha=0.5)
    ax[1].set_xlabel('Tiempo [s]')
    ax[1].set_ylabel('STA/LTA')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

        
def Detect_Anomaly_Multi_Plot(EHZ_Counts,ENN_Counts,ENE_Counts,ENZ_Counts, start_time):
    import numpy as np
    import matplotlib.pyplot as plt
    from obspy.core import UTCDateTime
    EHZ = Tranform_Counts_to_Vel(EHZ_Counts)
    ENN = Tranform_Counts_to_Gals(ENN_Counts)
    ENE = Tranform_Counts_to_Gals(ENE_Counts)
    ENZ = Tranform_Counts_to_Gals(ENZ_Counts)
    STA_LTA_EHZ = STA_LTA(EHZ)
    STA_LTA_ENN = STA_LTA(ENN)
    STA_LTA_ENE = STA_LTA(ENE)
    STA_LTA_ENZ = STA_LTA(ENZ)
    Time_EHZ = np.arange(len(EHZ)) / _sampling_rate
    Time_ENN = np.arange(len(ENN)) / _sampling_rate
    Time_ENE = np.arange(len(ENE)) / _sampling_rate
    Time_ENZ = np.arange(len(ENZ)) / _sampling_rate
    start_time_utc = UTCDateTime(start_time)
    n_act = Anomaly_Did_Start(EHZ) 
    act_time = start_time_utc + (n_act / _sampling_rate)
    if n_act >0:
        n_des = Anomaly_Did_End(EHZ, n_act)
        des_time = start_time_utc + (n_des / _sampling_rate)
    else:
        n_des = 0
    n_act_n = Anomaly_Did_Start(ENN)
    n_act_e = Anomaly_Did_Start(ENE)
    act_time_n = start_time_utc + (n_act_n / _sampling_rate)
    if n_act_n > 0 and n_act_e > 0:
        n_des_n = Anomaly_Did_End(ENN, n_act_n)
        n_des_e = Anomaly_Did_End(ENE, n_act_e)
        des_time_n = start_time_utc + (n_des_n / _sampling_rate)
    else:
        n_des_n = 0
        n_des_e = 0
    fig, ax = plt.subplots(4, 1, figsize=(14, 10))
    ax[0].plot(Time_EHZ, EHZ, label='EHZ')
    ax[0].set_xlabel('Tiempo [s]')
    ax[0].set_ylabel('Velocidad [cm/s]')
    ax[0].set_title(f'Inicio de señal en UTC:{start_time}')
    if n_act > 0 and n_des > 0:
        ax[0].axvline(x=n_act/_sampling_rate, color='r', linestyle='--', 
                  label=f'Inicio de anomalía a las {act_time.datetime.strftime("%H:%M:%S")}')
        ax[0].axvline(x=n_des/_sampling_rate, color='b', linestyle='--', 
                  label=f'Fin de anomalía a las {des_time.datetime.strftime("%H:%M:%S")}')
    else:
        ax[0].axvline(x=0, color='w', label='Sin anomalía')
    ax[0].legend(loc='upper right')
    ax[1].plot(Time_EHZ, STA_LTA_EHZ, label='STA/LTA EHZ')
    ax[1].axhline(y=thr_on, color='r', linestyle='--', 
                  label=f'Umbral activación={thr_on}', alpha=0.5)
    ax[1].axhline(y=thr_off, color='b', linestyle='--', 
                  label=f'Umbral desactivación={thr_off}', alpha=0.5)
    ax[1].set_xlabel('Tiempo [s]')
    ax[1].set_ylabel('STA/LTA')
    ax[1].legend(loc='upper right')
    ax[2].plot(Time_ENN, ENN, "g-", label='ENN')
    ax[2].plot(Time_ENE, ENE, "m-", label='ENE')
    ax[2].plot(Time_ENZ, ENZ, "k-", label='ENZ')
    ax[2].set_xlabel('Tiempo [s]')
    ax[2].set_ylabel('Aceleración [cm/s*s]')
    ax[2].set_title(f'Inicio de señal en UTC:{start_time}')
    if n_des_n > 0 and n_des_e > 0:
        ax[2].axvline(x=n_act_n/_sampling_rate, color='r', linestyle='--', 
                  label=f'Inicio de anomalía a las {act_time_n.datetime.strftime("%H:%M:%S")}')
        ax[2].axvline(x=n_des_n/_sampling_rate, color='b', linestyle='--', 
                  label=f'Fin de anomalía a las {des_time_n.datetime.strftime("%H:%M:%S")}')
    else:
        ax[2].axvline(x=0, color='w', label='Sin anomalía')
    ax[2].legend(loc='upper right')
    ax[3].plot(Time_ENN, STA_LTA_ENN,"g-", label='STA/LTA ENN')
    ax[3].plot(Time_ENE, STA_LTA_ENE,"m-", label='STA/LTA ENE')
    ax[3].plot(Time_ENZ, STA_LTA_ENZ,"k-", label='STA/LTA ENZ')
    ax[3].axhline(y=thr_on, color='r', linestyle='--', 
                  label=f'Umbral activación={thr_on}', alpha=0.5)
    ax[3].axhline(y=thr_off, color='b', linestyle='--', 
                  label=f'Umbral desactivación={thr_off}', alpha=0.5)
    ax[3].set_xlabel('Tiempo [s]')
    ax[3].set_ylabel('STA/LTA')
    ax[3].legend(loc='upper right')
    plt.tight_layout()
    plt.show()
