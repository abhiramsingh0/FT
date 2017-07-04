import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy
##################################################################
np.random.seed(0)
last_index = 1.0
n_samples = 1024
time = np.linspace(0, last_index, n_samples)
###################################################################
no_cycles = 100
s1 = np.sin(no_cycles * 2 * np.pi * time)  # Signal 1 : sinusoidal signal
S = s1
S += 0.1 * np.random.normal(size=s1.shape)  # Add noise
###################################################################
f, t, sxx = signal.spectrogram(S,n_samples)
plt.pcolormesh(t, f, sxx)
plt.ylabel('frequency Hz')
plt.xlabel('time sec')
plt.show()
