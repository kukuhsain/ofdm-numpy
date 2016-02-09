# import numpy, scipy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# inisialisasi
fo = 24000000
to = 1. / fo

t = np.arange(0, 8 * to, .01 * to)
carr_real = np.cos(2 * np.pi * fo * t)
carr_imaj = np.sin(2 * np.pi * fo * t)

#plt.plot(t, carr_real)
#plt.plot(t, carr_imaj)
#plt.show()

# input data biner
inp = np.array([0,0,1,0,0,1,1,0,1,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1])

# mapping dan s/p
h = 0
mapping = np.zeros([8, 1], dtype=complex)

for k in range(0, inp.size, 4):
    temp_real = np.array([inp[k], inp[k+1]])
    temp_imaj = np.array([inp[k+2], inp[k+3]])
    
    if np.all(temp_real == np.array([0, 0])):
        mapping[h] = -3
    elif np.all(temp_real == np.array([0, 1])):
        mapping[h] = -1
    elif np.all(temp_real == np.array([1, 1])):
        mapping[h] = 1
    else:
        mapping[h] = 3

    if np.all(temp_imaj == np.array([0, 0])):
        mapping.imag[h] = -3
    elif np.all(temp_imaj == np.array([0, 1])):
        mapping.imag[h] = -1
    elif np.all(temp_imaj == np.array([1, 1])):
        mapping.imag[h] = 1
    else:
        mapping.imag[h] = 3
        
    h += 1

# ifft
ifftmap = fftpack.ifft(mapping.T)
ifftmap = ifftmap.T

# interpolasi
inter_real = ifftmap.real
inter_imaj = ifftmap.imag

#k = np.ones(100)
interpol_real = np.zeros(800)
interpol_imaj = np.zeros(800)

#   real
for m in range(8):
    for n in range(100):
        o = (m * 100 + n)
        interpol_real[o] = inter_real[m]

#   imajiner
for m in range(8):
    for n in range(100):
        o = (m * 100 + n)
        interpol_imaj[o] = inter_imaj[m]

# perkalian sebelum transmit
channel_real = carr_real * interpol_real
channel_imaj = carr_imaj * interpol_imaj

# transmit
channel = channel_real + channel_imaj

plt.plot(t, channel_real)
plt.plot(t, channel_imaj)
plt.plot(t, channel)
plt.show()

# receive
#   real
rec_real = channel * carr_real

#   imajiner
rec_imaj = channel * carr_imaj

# integral
integral = np.zeros(8, dtype=complex)

#   real
for m in range(8):
    o = m * 100
    integral[m] = np.sum(rec_real[o:o+100]) * 2 / 100

#   imajiner
for m in range(8):
    o = m * 100
    integral.imag[m] = np.sum(rec_imaj[o:o+100]) * 2 / 100

# fft
rec_fft = fftpack.fft(integral)
rec_fft = np.around(rec_fft)

# demapping dan p/s
out = np.zeros(32)

for k in range(8):
    temp_real = rec_fft.real[k]
    temp_imaj = rec_fft.imag[k]
    
    h = k*4
    
    if np.all(temp_real == -3):
        out[h:h+2] = np.array([0, 0])
    elif np.all(temp_real == -1):
        out[h:h+2] = np.array([0, 1])
    elif np.all(temp_real == 1):
        out[h:h+2] = np.array([1, 1])
    else:
        out[h:h+2] = np.array([1, 0])

    if np.all(temp_imaj == -3):
        out[h+2:h+4] = np.array([0, 0])
    elif np.all(temp_imaj == -1):
        out[h+2:h+4] = np.array([0, 1])
    elif np.all(temp_imaj == 1):
        out[h+2:h+4] = np.array([1, 1])
    else:
        out[h+2:h+4] = np.array([1, 0])

# output
out = out.astype(int)
