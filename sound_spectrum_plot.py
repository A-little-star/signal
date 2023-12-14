#%%
import numpy as np
import soundfile as sf
import wave
import matplotlib.pyplot as plt

T = 31

wav, sample_rate = sf.read('test.wav')
time = np.arange(0, len(wav)) * (1.0 / sample_rate)
plt.subplot(211)
plt.plot(time, wav)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
# %%
plt.subplot(212)
plt.specgram(wav, NFFT=512, Fs=sample_rate, window=np.hanning(M=512), noverlap=256)
plt.show()