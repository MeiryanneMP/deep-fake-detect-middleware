from extract_features import extract_mfcc
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data" / "voice-people"

    audio_files = list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))

    if not audio_files:
        print(f"Nenhum arquivo de áudio encontrado em {data_dir}")
    else:
        all_features = []
        file_names = []

        for audio_file in audio_files:
            print(f"Processando {audio_file.name}...")
            features = extract_mfcc(audio_file)
            if features is not None:
                all_features.append(features)
                file_names.append(audio_file.name)
                print(f"MFCC extraídos com sucesso de {audio_file.name}")
            else:
                print(f"Falha na extração dos MFCCs de {audio_file.name}")

        if all_features:
            all_features_array = np.array(all_features, dtype=object)

            output_path = Path(__file__).parent.parent.parent / \
                "data" / "voice-npy" / "mfccs.npy"
            np.save(output_path, all_features_array)

            np.save(Path(__file__).parent.parent.parent / "data" / "voice-npy" /
                    "file_names.npy", np.array(file_names))

            print(f"\nTodos os MFCCs foram salvos em {output_path}")
