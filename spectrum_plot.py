#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

x = np.arange(1, 5, 0.02)
y = np.cos(2 * np.pi * 1 * x)
plt1 = plt.subplot(2, 1, 1)
plt.plot(x, y)

freq = np.linspace(0, 50 / 2, 101)
specs = np.fft.rfft(y) / 200 * 2
plt2 = plt.subplot(2, 1, 2)
plt.subplots_adjust(hspace=0.4)
plt.plot(freq, specs)
plt.show
# %%
