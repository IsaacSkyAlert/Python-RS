from obspy import read
import numpy as np
from obspy.core import UTCDateTime
from RSAnomalyDetectFunctionsv2 import *

start_time = "2024-02-14T12:10:00"
EHZ_Stream = read('M2.8-14-02-2024-EHZ.sac')
EHZ_Counts = EHZ_Stream[0].data
print(EHZ_Counts)
ENN_Stream = read('M2.8-14-02-2024-ENN.sac')
ENN_Counts = ENN_Stream[0].data
print(ENN_Counts)
ENE_Stream = read('M2.8-14-02-2024-ENE.sac')
ENE_Counts = ENE_Stream[0].data
print(ENE_Counts)
ENZ_Stream = read('M2.8-14-02-2024-ENZ.sac')
ENZ_Counts = ENZ_Stream[0].data
print(ENZ_Counts)

st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
st.decimate(2, strict_length=False, no_filter=True)
print(st)
z_signal = st[0].data
n = len(EHZ_Counts)
noise1 = np.random.rand(n)*4e4
noise2 = np.random.rand(n)*3e4
noise3 = np.random.rand(n)*2e4
noise4 = np.random.rand(n)*1e4

# Prueba 1:
n_act = Anomaly_Did_Start(z_signal)  
print(n_act)
n_des = Anomaly_Did_End(z_signal, n_act)
print(n_des)
Detect_Anomaly_Plot(z_signal, start_time)

# Prueba 2:
print(Anomaly_Did_Start(Signal=ENN_Counts))
print(Anomaly_Did_Start(Signal=ENE_Counts))
print(Anomaly_Did_Start(Signal=ENZ_Counts))
Detect_Anomaly_Multi_Plot(EHZ_Counts, ENN_Counts, 
                          ENE_Counts, ENZ_Counts, start_time)

# Prueba 3:
Detect_Anomaly_Multi_Plot(EHZ_Counts=noise1, ENN_Counts=noise2, 
                          ENE_Counts=noise3, ENZ_Counts=noise4, 
                          start_time="2024-02-14T12:10:00")

# Prueba 4:
mu, sigma = 0, 1  # Media y desviación estándar del ruido
ruido1 = np.random.normal(mu, sigma, n)*4e2
ruido2 = np.random.normal(mu, sigma, n)*3e2
ruido3 = np.random.normal(mu, sigma, n)*2e2
ruido4 = np.random.normal(mu, sigma, n)*1e2

def generar_picos_esporadicos(ruido, longitud, num_picos):
    for _ in range(num_picos):
        indice_pico = np.random.randint(0, longitud)
        amplitud_pico = np.random.uniform(5000, 10000)
        pico = np.zeros(longitud)
        pico[indice_pico] = amplitud_pico
        for i in range(indice_pico+1, longitud):
            pico[i] = -amplitud_pico * np.exp(-(i - indice_pico))
        # Agregar el pico a la señal de ruido
        ruido += pico
    return ruido

ruido1_picos = generar_picos_esporadicos(ruido1, longitud=n, num_picos=4)
ruido2_picos = generar_picos_esporadicos(ruido2, longitud=n, num_picos=3)
ruido3_picos = generar_picos_esporadicos(ruido3, longitud=n, num_picos=2)
ruido4_picos = generar_picos_esporadicos(ruido4, longitud=n, num_picos=2)

Detect_Anomaly_Multi_Plot(ruido1_picos, ruido2_picos, 
                          ruido3_picos, ruido4_picos, 
                          start_time="2024-02-14T12:10:00")
