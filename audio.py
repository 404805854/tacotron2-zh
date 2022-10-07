import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch


def dc_notch_filter(wav):
    # code from speex
    notch_radius = 0.982
    den = notch_radius ** 2 + 0.7 * (1 - notch_radius) ** 2
    b = np.array([1, -2, 1]) * notch_radius
    a = np.array([1, -2 * notch_radius, den])
    return signal.lfilter(b, a, wav)


def save_wav(wav, path, sample_rate):
    wav = dc_notch_filter(wav)
    wav = wav / np.abs(wav).max() * 0.999
    f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
    f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
    wav = f1 * f2
    # proposed by @dsmiller
    wavfile.write(path, sample_rate, wav.astype(np.int16))


def mel_to_audio(infer_result_mels, default):
    from tacotron2_common.layers import TacotronSTFT
    taco_stft = TacotronSTFT(
        default.filter_length, default.hop_length, default.win_length,
        default.n_mel_channels, default.sampling_rate, default.mel_fmin,
        default.mel_fmax)
    mel_decompress = taco_stft.spectral_de_normalize(infer_result_mels)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * default.max_wav_value

    from tacotron2_common.audio_processing import griffin_lim
    audio = griffin_lim(torch.autograd.Variable(
        spec_from_mel[:, :, :-1]), taco_stft.stft_fn, default.n_iters)

    audio = audio.squeeze().cpu().numpy()

    return audio
