import torchaudio
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
signal, fs = torchaudio.load('audio.wav')
embeddings = classifier.encode_batch(signal)
print(embeddings.shape)
