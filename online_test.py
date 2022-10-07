
#Load the Tacotron2 model pre-trained on LJ Speech dataset and prepare it for inference:
import torch
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()


#Load pretrained WaveGlow model
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()


#Now, let's make the model say:
text = "Hello world, I missed you so much."


#Format the input using utility methods
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
sequences, lengths = utils.prepare_input_sequence([text])


#Run the chained models:
with torch.no_grad():
    mel, _, _ = tacotron2.infer(sequences, lengths)
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050


#You can write it to a file and listen to it
from scipy.io.wavfile import write
write("audio.wav", rate, audio_numpy)


'''
#Alternatively, play it right away in a notebook with IPython widgets
from IPython.display import Audio
Audio(audio_numpy, rate=rate)
'''
