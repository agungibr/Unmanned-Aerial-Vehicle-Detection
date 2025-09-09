import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def _load_audio(self, file_path):
        signal, sr = librosa.load(file_path, sr=None) 
        if sr != self.sample_rate:
            signal = librosa.resample(y=signal, orig_sr=sr, target_sr=self.sample_rate)
        return signal

    def _pad_or_truncate(self, signal, num_expected_samples):
        if len(signal) > num_expected_samples:
            signal = signal[:num_expected_samples]
        elif len(signal) < num_expected_samples:
            signal = np.pad(signal, (0, num_expected_samples - len(signal)), "constant")
        return signal

    def extract(self, file_path, duration_sec=5):
        signal = self._load_audio(file_path)

        num_expected_samples = int(self.sample_rate * duration_sec)
        signal = self._pad_or_truncate(signal, num_expected_samples)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram