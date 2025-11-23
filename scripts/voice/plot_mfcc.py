import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_mfcc(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig('mfcc.png')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python plot_mfcc.py caminho_do_arquivo.wav")
    else:
        plot_mfcc(sys.argv[1])
