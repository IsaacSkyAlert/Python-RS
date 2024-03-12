import unittest
from RSAnomalyDetectFunctionsv2 import *
from obspy import read
import numpy as np

start_time = "2024-02-14T12:10:00"
EHZ_Stream = read('M2.8-14-02-2024-EHZ.sac')
EHZ_Counts = EHZ_Stream[0].data
ENN_Stream = read('M2.8-14-02-2024-ENN.sac')
ENN_Counts = ENN_Stream[0].data
ENE_Stream = read('M2.8-14-02-2024-ENE.sac')
ENE_Counts = ENE_Stream[0].data
ENZ_Stream = read('M2.8-14-02-2024-ENZ.sac')
ENZ_Counts = ENZ_Stream[0].data
n = len(EHZ_Counts)
st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
st.decimate(2, strict_length=False, no_filter=True)
z_signal = st[0].data
Noise1 = np.random.rand(n)*4e4
Noise2 = np.random.rand(n)*3e4
Noise3 = np.random.rand(n)*2e4
Noise4 = np.random.rand(n)*1e4


class Test_Transform_Counts_To_Gals(unittest.TestCase):
    def test_max_value(self):
            Data = ENZ_Counts  
            New_Data = Tranform_Counts_to_Gals(Data)
            self.assertAlmostEqual(max(New_Data), 21.85, delta=0.01)
            
    def test_length(self):
            Data = ENZ_Counts  
            New_Data = Tranform_Counts_to_Gals(Data)
            self.assertEqual(len(New_Data), len(Data),
                             f"Las longitudes deberían ser las mismas.")

    
class Test_Detection_Anomaly(unittest.TestCase):
    def test_anomaly_did_start_negative(self):
        Signal1 = Noise1
        Signal2 = Noise2
        Signal3 = Noise3
        Signal4 = Noise4
        n_act1 = Anomaly_Did_Start(Signal1)
        n_act2 = Anomaly_Did_Start(Signal2)
        n_act3 = Anomaly_Did_Start(Signal3)
        n_act4 = Anomaly_Did_Start(Signal4)
        self.assertEqual(n_act1, 0,
                           f"El valor devuelto debería ser cero.")
        self.assertEqual(n_act2, 0,
                           f"El valor devuelto debería ser cero.")
        self.assertEqual(n_act3, 0,
                           f"El valor devuelto debería ser cero.")
        self.assertEqual(n_act4, 0,
                           f"El valor devuelto debería ser cero.")
    def test_anomaly_did_start_positive(self):
        Signal1 = EHZ_Counts
        Signal2 = ENN_Counts
        Signal3 = ENE_Counts
        Signal4 = ENZ_Counts
        n_act1 = Anomaly_Did_Start(Signal1)
        n_act2 = Anomaly_Did_Start(Signal2)
        n_act3 = Anomaly_Did_Start(Signal3)
        n_act4 = Anomaly_Did_Start(Signal4)
        self.assertGreater(n_act1, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        self.assertGreater(n_act2, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        self.assertGreater(n_act3, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        self.assertGreater(n_act4, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        

    def test_anomaly_did_end_negative(self):
        Signal1 = EHZ_Counts
        n_des1 = Anomaly_Did_End(Signal1,0)
        n_des2 = Anomaly_Did_End(Signal1,0.0)
        n_des3 = Anomaly_Did_End(Signal1,-1)
        n_des4 = Anomaly_Did_End(Signal1,-5.0)
        self.assertEqual(n_des1, None,
                           f"El valor devuelto debería ser None.")
        self.assertEqual(n_des2, None,
                           f"El valor devuelto debería ser None.")
        self.assertEqual(n_des3, None,
                           f"El valor devuelto debería ser None.")
        self.assertEqual(n_des4, None,
                           f"El valor devuelto debería ser None.")
        
    def test_anomaly_did_end_positive(self):
        Signal1 = EHZ_Counts
        Signal2 = ENN_Counts
        Signal3 = ENE_Counts
        Signal4 = ENZ_Counts
        n_act1 = Anomaly_Did_Start(Signal1)
        n_act2 = Anomaly_Did_Start(Signal2)
        n_act3 = Anomaly_Did_Start(Signal3)
        n_act4 = Anomaly_Did_Start(Signal4)
        n_des1 = Anomaly_Did_End(Signal1,n_act1)
        n_des2 = Anomaly_Did_End(Signal2,n_act2)
        n_des3 = Anomaly_Did_End(Signal3,n_act3)
        n_des4 = Anomaly_Did_End(Signal4,n_act4)
        print(n_act1,n_act2,n_act3,n_act4)
        print(n_des1,n_des2,n_des3,n_des4)
        self.assertGreater(n_des1, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        self.assertGreater(n_des2, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        self.assertGreater(n_des3, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        self.assertGreater(n_des4, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        
    def test_anomaly_did_end_positive_noise(self):
        Signal1 = Noise1
        Signal2 = Noise2
        Signal3 = Noise3
        Signal4 = Noise4
        n_des1 = Anomaly_Did_End(Signal1,50)
        n_des2 = Anomaly_Did_End(Signal2,150)
        n_des3 = Anomaly_Did_End(Signal3,180)
        n_des4 = Anomaly_Did_End(Signal4,200)
        print(n_des1,n_des2,n_des3,n_des4)
        self.assertGreater(n_des1, 0,
                           f"El valor devuelto debería ser mayor que cero.")
        self.assertEqual(n_des2, 0,
                           f"El valor devuelto debería ser cero.")
        self.assertEqual(n_des3, 0,
                           f"El valor devuelto debería ser cero.")
        self.assertEqual(n_des4, 0,
                           f"El valor devuelto debería ser cero.")
        
if __name__ == '__main__':
    unittest.main()
