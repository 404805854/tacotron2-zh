import pprint
import ast
import argparse

from tacotron2_common.audio_processing import griffin_lim


class HParams(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def parse(self, string):
        # Overrides hparams from a comma-separated string of name=value pairs
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self


griffin_lim_hp = HParams(
    ###########################################################################################################################################

    # Audio
    # Audio parameters are the most important parameters to tune when using this work on your personal data. Below are the beginner steps to adapt
    # this work to your personal data:
    #	1- Determine my data sample rate: First you need to determine your audio sample_rate (how many samples are in a second of audio). This can be done using sox: "sox --i <filename>"
    #		(For this small tuto, I will consider 24kHz (24000 Hz), and defaults are 22050Hz, so there are plenty of examples to refer to)
    #	2- set sample_rate parameter to your data correct sample rate
    #	3- Fix win_size and and hop_size accordingly: (Supposing you will follow our advice: 50ms window_size, and 12.5ms frame_shift(hop_size))
    #		a- win_size = 0.05 * sample_rate. In the tuto example, 0.05 * 24000 = 1200
    #		b- hop_size = 0.25 * win_size. Also equal to 0.0125 * sample_rate. In the tuto example, 0.25 * 1200 = 0.0125 * 24000 = 300 (Can set frame_shift_ms=12.5 instead)
    #	4- Fix n_fft, num_freq and upsample_scales parameters accordingly.
    #		a- n_fft can be either equal to win_size or the first power of 2 that comes after win_size. I usually recommend using the latter
    #			to be more consistent with signal processing friends. No big difference to be seen however. For the tuto example: n_fft = 2048 = 2**11
    #		b- num_freq = (n_fft / 2) + 1. For the tuto example: num_freq = 2048 / 2 + 1 = 1024 + 1 = 1025.
    #		c- For WaveNet, upsample_scales products must be equal to hop_size. For the tuto example: upsample_scales=[15, 20] where 15 * 20 = 300
    #			it is also possible to use upsample_scales=[3, 4, 5, 5] instead. One must only keep in mind that upsample_kernel_size[0] = 2*upsample_scales[0]
    #			so the training segments should be long enough (2.8~3x upsample_scales[0] * hop_size or longer) so that the first kernel size can see the middle
    #			of the samples efficiently. The length of WaveNet training segments is under the parameter "max_time_steps".
    #	5- Finally comes the silence trimming. This very much data dependent, so I suggest trying preprocessing (or part of it, ctrl-C to stop), then use the
    #		.ipynb provided in the repo to listen to some inverted mel/linear spectrograms. That will first give you some idea about your above parameters, and
    #		it will also give you an idea about trimming. If silences persist, try reducing trim_top_db slowly. If samples are trimmed mid words, try increasing it.
    #	6- If audio quality is too metallic or fragmented (or if linear spectrogram plots are showing black silent regions on top), then restart from step 2.
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality

    # Mel spectrogram
    n_fft=1024,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    win_size=800,
    # 22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
    sample_rate=16000,
    # Can replace hop_size parameter. (Recommended: 12.5)
    frame_shift_ms=None,
    # The power of the spectrogram magnitude (1. for energy, 2. for power)
    magnitude_power=2.,

    # Mel and Linear spectrograms normalization/scaling and clipping
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    signal_normalization=True,
    # Only relevant if mel_normalization = True
    allow_clipping_in_normalization=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    symmetric_mels=True,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion,
    max_abs_value=4.,

    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-120,
    ref_level_db=20,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmin=55,
    fmax=7600,  # To be increased/reduced depending on data.

    # Griffin Lim
    # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    power=1.5,
    # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
    griffin_lim_iters=60,
    ###########################################################################################################################################
)

diy_size, batch_size = [(96, 48), (128, 40), (256, 36), (384, 32)][1]


# Default hyperparameters
default = HParams(
    input="./test.txt",
    output="./output",
    epochs=100,
    epochs_per_checkpoint=5,
    data_loader_workers=2,
    batch_size=batch_size,

    datasets="|".join(["BZNSYP", "AISHELL3"]),
    training_files="./data/metadata.csv",
    validation_files="",

    text_cleaners="basic_cleaners",
    sampling_rate=griffin_lim_hp.sample_rate,
    filter_length=griffin_lim_hp.n_fft,
    hop_length=griffin_lim_hp.hop_size,
    win_length=griffin_lim_hp.win_size,
    n_mel_channels=griffin_lim_hp.num_mels,
    mel_fmin=griffin_lim_hp.fmin,
    mel_fmax=griffin_lim_hp.fmax,
    n_iters=60,
    max_wav_value=32768.0,

    ###########################################################################################################################################

    # Tacotron

    symbols_embedding_dim=diy_size,

    # Encoder parameters
    encoder_kernel_size=5,
    encoder_n_convolutions=3,
    encoder_embedding_dim=diy_size,  # 192, #384,
    # spk_embedding_dim=96,

    # Decoder parameters
    n_frames_per_step=1,
    decoder_rnn_dim=diy_size,  # 288, #480,
    prenet_dim=256,
    max_decoder_steps=1000,
    gate_threshold=0.5,
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,

    # Attention
    attention_dim=128,
    attention_rnn_dim=diy_size,
    attention_location_n_filters=32,
    attention_location_kernel_size=31,

    # Mel-post processing network parameters
    postnet_embedding_dim=512,  # 256, #512,
    postnet_kernel_size=5,
    postnet_n_convolutions=5,  # 2, #5,


    # Alternative Attention Modules
    attention_mode=["default", "GMM", "FA", "SMA", "DCA", "GMM_Evo"][2],
    # GMM Attention parameters
    gmm_kernel=5,
    delta_bias=1.0,
    sigma_bias=10.0,
    # GMM Attention Evo
    GMM_Evo_K=8,
    GMM_Evo_Version='1',
    ###########################################################################################################################################
)

default_args = []
for k, value in default.items():
    key = k.replace("_", "-")
    default_args.append(f"--{key}={value}")
