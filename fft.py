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
no_cycles = 50
s1 = np.sin(no_cycles * 2 * np.pi * time)  # Signal 1 : sinusoidal signal
S = s1
S += 0.2 * np.random.normal(size=s1.shape)  # Add noise
#s2 = np.sign(np.sin(no_cycles * 2 * np.pi * time))  # Signal 2 : square signal
#s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
#####################################################################
s_fft = np.fft.fft(S)/n_samples
freq_scale = np.fft.fftfreq(n_samples, last_index/n_samples)
freq = np.abs(freq_scale[0:n_samples/2+1])
########################################################################
#plt.plot(time,s1)
#plt.plot(freq_scale,np.imag(s1_fft))
#plt.plot(freq_scale,np.real(s1_fft))
#plt.plot(freq_scale,s1_fft)
f,ax = plt.subplots(4,1)
ax[0].plot(time,S)
ax[0].set_ylabel('signal')
ax[1].plot(freq_scale,s_fft)
ax[1].set_ylabel('complex exponential')
ax[2].plot(freq_scale,np.imag(s_fft))
ax[2].set_ylabel('cosine part')
ax[3].plot(freq_scale,np.real(s_fft))
ax[3].set_ylabel('sine part')
plt.show()
#S = np.c_[s1, s2, s3]
#S += 0.2 * np.random.normal(size=S.shape)  # Add noise

#S /= S.std(axis=0)  # Standardize data
# Mix data
#A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
#X = np.dot(S, A.T)  # Generate observations
