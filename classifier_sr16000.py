import torchaudio
from speechbrain.pretrained import EncoderClassifier


classifier = EncoderClassifier.from_hparams(
    source="/content/best_model/",
    hparams_file='hparams_inference.yaml',
    savedir="/content/best_model/",)

# Perform classification
audio_file = '/content/drive/MyDrive/Colab Notebooks/VOICEACTRESS100_005.wav'
signal, fs = torchaudio.load(audio_file)  # test_speaker: 5789
output_probs, score, index, text_lab = classifier.classify_batch(signal)
print('Target: 5789, Predicted: ' + text_lab[0])
