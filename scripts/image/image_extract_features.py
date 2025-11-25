import cv2
import numpy as np


def extract_image_features(file_path, bins=32):
    try:
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError("Imagem não pode ser carregada.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
        hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()

        hist = np.concatenate([hist_r, hist_g, hist_b])

        hist = hist / np.sum(hist)

        return hist

    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None
