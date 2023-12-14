import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window

def init_kernel(win_len, win_inc, fft_len, win_type):
    if win_type == None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)
    
    fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]
    real = np.real(fourier_basis)
    imag = np.imag(fourier_basis)
    kernel = np.concatenate([real, imag], 1).T
    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))

class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hann', feature_type='real'):
        super(ConvSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        self.win_len = win_len
        self.win_inc = win_inc
        self.win_type = win_type
        self.feature_type = feature_type

        kernel, _ = init_kernel(win_len, win_inc, fft_len, win_type)
        self.register_buffer('weight', kernel)

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        outputs = F.conv1d(inputs, self.weight, stride=self.win_inc)
        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.fft_len // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase

if __name__=='__main__':
    wav = torch.randn(1, 16000*4)
    stft = ConvSTFT(win_len=400, win_inc=100, fft_len=512, win_type='hann', feature_type='complex')
    specs = stft(wav)
    print(specs)