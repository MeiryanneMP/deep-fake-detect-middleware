from image_extract_features import extract_image_features
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np

if __name__ == "__main__":

    load_dotenv()

    data_dir = Path(os.getenv("IMAGE_DATA_DIR"))
    if not data_dir or not data_dir.exists():
        raise ValueError("Variável ou pasta não existe")

    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))

    if not image_files:
        print(f"Nenhum arquivo de imagem encontrado em {data_dir}")
    else:
        all_features = []
        file_names = []

        for img_file in image_files:
            print(f"Processando {img_file.name}...")
            features = extract_image_features(img_file)

            if features is not None:
                all_features.append(features)
                file_names.append(img_file.name)
                print(f"Features extraídas com sucesso de {img_file.name}")
            else:
                print(f"Falha na extração das features de {img_file.name}")

        if all_features:
            all_features_array = np.array(all_features, dtype=object)

            output_path = Path(__file__).parent.parent.parent / \
                "data" / "image-npy" / "features.npy"
            np.save(output_path, all_features_array)

            np.save(Path(__file__).parent.parent.parent / "data" / "image-npy" /
                    "file_names.npy", np.array(file_names))

            print(f"\nAs features foram salvas em {output_path}")
