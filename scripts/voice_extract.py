from extract_features import extract_mfcc
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "voice-people"

    audio_files = list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))

    if not audio_files:
        print(f"Nenhum arquivo de áudio encontrado em {data_dir}")
    else:
        for audio_file in audio_files:
            print(f"Processando {audio_file.name}...")
            features = extract_mfcc(audio_file)
            if features is not None:
                print(f"MFCC extraídos com sucesso de {audio_file.name}")
            else:
                print(f"Falha na extração dos MFCCs de {audio_file.name}")
