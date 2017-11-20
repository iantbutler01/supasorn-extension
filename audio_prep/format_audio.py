import subprocess
import math
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
from sys import argv, exit
from glob import glob
from os import mkdir, path
from multiprocessing import Pool
import re

NFFT=512

def get_mfccs(mel_filter_banks):
    num_ceps = 13
    mfccs = dct(mel_filter_banks, norm='ortho', axis=1)[:, 1:num_ceps+1]
    nframes, ncoeffs = mfccs.shape
    n = np.arange(ncoeffs)
    cepp_lifter = 22
    lift = 1+(cepp_lifter/2)*np.sin(np.pi * n / cepp_lifter)
    mfccs *= lift
    return mfccs

def apply_mel_scale_filter(power_spectrum, sample_rate):
    low_freq_mel = 0
    nfilt = 40
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    _bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(_bin[m - 1])
        f_m = int(_bin[m])             # center
        f_m_plus = int(_bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - _bin[m - 1]) / (_bin[m] - _bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (_bin[m + 1] - k) / (_bin[m + 1] - _bin[m])
    filter_banks = np.dot(power_spectrum, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    return 20 * np.log10(filter_banks)


def get_spectra(bytes_slice):
    A = np.fft.rfft(bytes_slice, NFFT)
    power_spectra = np.abs(A)**2
    return power_spectra

def signal_to_frames(signal, step_size, sample_rate, ms=0.025, st=0.01):
    sig_len = len(signal)
    frame_len = int(round(sample_rate * ms))
    frame_step = int(sample_rate * st)
    num_frames = int(np.ceil(float(np.abs(sig_len - frame_len)) / frame_step))  # Make sure that we have at least 1 frame
    pad_len =  num_frames * frame_step + frame_len
    z = np.zeros((pad_len - sig_len))
    padded_signal = np.append(signal, z)
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    indices = indices.astype(np.int32, copy=False)
    frames = padded_signal[indices]
    frames *= np.hamming(frame_len)
    return frames

def process(file_path):
    if not re.search(r'normalized', file_path):
        return
    folder = file_path.split('/')[-2]
    sample_rate, signal = scipy.io.wavfile.read(file_path)
    print(sample_rate)
    step = int(math.floor(0.025*sample_rate))
    frames = signal_to_frames(signal, step, sample_rate)
    power_spectra = get_spectra(frames)
    energy = np.where(power_spectra == 0, np.finfo(float).eps, power_spectra)
    energy = np.log(energy)[:, 1:16]
    applied_mel_scale = apply_mel_scale_filter(power_spectra, sample_rate)
    mfccs = get_mfccs(applied_mel_scale)
    output = np.concatenate([mfccs, energy], 1)
    file_name = file_path.split('/')[-1].split('.')[0]
    if not path.exists(f'./mfccs/{folder}'):
        mkdir(f'./mfccs/{folder}')
    np.savetxt(f'./mfccs/{folder}/{file_name}', output)

def main():
    if len(argv) < 2:
        print('Need to supply path to audio files.')
        exit(-1)

    if not path.exists(f'./mfccs'):
        mkdir('./mfccs')
    audio_paths = glob(argv[1]+"/**/*")
    with Pool() as po:
        po.map(process, audio_paths)


if __name__ == '__main__':
    main()





