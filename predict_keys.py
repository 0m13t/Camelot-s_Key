#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import torch
import librosa
import numpy as np

# Zorg dat dataset.py en eval.py in dezelfde map staan als dit script
try:
    from dataset import CAMELOT_MAPPING
    from eval import load_model
except ImportError as e:
    print(f"Fout: Kan ondersteunende bestanden niet laden ({e})")
    sys.exit(1)

def parse_args():
    default_model_path = Path('checkpoints') / 'keynet.pt'
    parser = argparse.ArgumentParser(description="AI Key Prediction voor alle audioformaten.")
    parser.add_argument('-f', '--path', type=str, required=True, help="Pad naar bestand of map.")
    parser.add_argument('-m', '--model_path', type=str, default=str(default_model_path), help="Pad naar keynet.pt")
    parser.add_argument('--device', type=str, default="cpu", help="Device: 'cpu' of 'cuda'.")
    return parser.parse_args()

def get_audio_list(path):
    """Vindt alle ondersteunde audiobestanden, niet alleen mp3."""
    SUPPORTED_EXTS = {".mp3", ".wav", ".aiff", ".aif", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".wma"}
    path = Path(path)
    files = []
    
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    elif path.is_dir():
        for ext in SUPPORTED_EXTS:
            # Zoek naar zowel kleine letters als hoofdletters extensies
            files.extend(list(path.glob(f"*{ext}")))
            files.extend(list(path.glob(f"*{ext.upper()}")))
    
    return sorted(list(set(files))) # Verwijder dubbelen en sorteer

def preprocess_audio(audio_path, sample_rate=44100, n_bins=105, hop_length=8820):
    """
    Laadt audio (FLAC, WAV, MP3, etc.) via librosa en zet het om naar 
    het CQT-spectrogram dat de AI verwacht.
    """
    # Librosa is veel robuuster voor FLAC/WAV op Windows dan torchaudio
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    # Bereken het Constant-Q Transform (CQT) spectrogram
    cqt = librosa.cqt(y, sr=sample_rate, hop_length=hop_length, 
                      n_bins=n_bins, bins_per_octave=24, fmin=65)
    
    spec = np.abs(cqt)
    spec = np.log1p(spec)

    # De AI verwacht exact 103 bins (we verwijderen de laatste 2 van de 105)
    spec_tensor = torch.tensor(spec[:, 0:-2], dtype=torch.float32)
    
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0) # Voeg kanaal-dimensie toe
    return spec_tensor

def camelot_output(pred_idx):
    """Zet de AI-output (0-23) om naar Camelot (1A-12B)."""
    idx = (pred_idx % 12) + 1
    mode = "A" if pred_idx < 12 else "B"
    camelot_str = f"{idx}{mode}"

    # Zoek de muzikale naam op in de mapping van de maker
    names = [k for k, v in CAMELOT_MAPPING.items() if v == pred_idx]
    key_text = "/".join(sorted(set(names))) if names else "Unknown"
    return camelot_str, key_text

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    if not os.path.exists(args.model_path):
        print(f"Error: AI Model niet gevonden op {args.model_path}")
        return

    # Laad het getrainde brein
    model = load_model(args.model_path, device)
    model.eval()

    files = get_audio_list(args.path)
    if not files:
        print(f"Geen ondersteunde bestanden gevonden in: {args.path}")
        return

    # Print een tabel die de hoofd-app kan uitlezen
    print("-" * 30)
    for p in files:
        try:
            spec = preprocess_audio(p)
            spec = spec.to(device).unsqueeze(0) # Batch dimensie toevoegen

            with torch.no_grad():
                output = model(spec)
                pred = int(torch.argmax(output, dim=1).cpu().item())

            camelot, key_name = camelot_output(pred)
            # Belangrijk: Print de filenaam en resultaten voor de hoofd-app
            print(f"RESULT|{p.name}|{camelot}|{key_name}")
            
        except Exception as e:
            print(f"ERROR|{p.name}|{str(e)}")
    print("-" * 30)

if __name__ == "__main__":
    main()