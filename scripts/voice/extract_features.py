import librosa
import numpy as np


def extract_mfcc(file_path, n_mfcc=13):

    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None
