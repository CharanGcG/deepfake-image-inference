# inference/preprocess.py
import librosa
import numpy as np
import torch
from .config import SAMPLE_RATE, N_MELS

# Add default spectrogram params (match your training config if possible)
N_MELS = 256          # 128 → 256 (double vertical resolution)
N_FFT = 2048          # 1024 → 2048 (better frequency detail)
HOP_LENGTH = 128      # 256 → 128 (double horizontal resolution)


def load_audio_as_tensor(path, fixed_seconds=None, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Returns:
      wav_tensor: torch.FloatTensor shape (1, T)  (AASIST.Model expects (B, T))
      log_mel: numpy array (n_mels, time) for visualization
    Uses keyword args for librosa.feature.melspectrogram to avoid signature issues.
    """
    # load audio (librosa.load always returns float32 numpy array)
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    if fixed_seconds is not None:
        target_len = int(sr * fixed_seconds)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

    # compute mel spectrogram — use keyword args to avoid positional-arg signature issues
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr_out,
        n_fft=2048,
        hop_length=128,
        n_mels=256,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # waveform tensor shape: (1, T)
    wav_tensor = torch.from_numpy(y).float().unsqueeze(0)

    return wav_tensor, log_mel
