import os
import re
import shutil
import threading
import json
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES

import numpy as np
import librosa
from pydub import AudioSegment

import mutagen
from mutagen import File as MutagenFile
from mutagen.flac import FLAC
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TIT2, TPE1, TBPM, TKEY, ID3NoHeaderError
from mutagen.wave import WAVE

# --- THEME SETUP ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")  

# --- AUDIO ANALYSIS & UTILS ---
AUDIO_EXTS = {".mp3", ".wav", ".aiff", ".aif", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".wma"}

def slugify(text: str) -> str:
    """Normalizes string for filesystem-safe filenames."""
    text = str(text).strip()
    return re.sub(r"-+", "-", re.sub(r"\s+", "-", re.sub(r'[<>:"/\\|?*]+', "", text))).strip("-._ ")

def safe_text(value, fallback="Unknown"):
    """Validates metadata strings."""
    if value is None: return fallback
    value = str(value).strip()
    return value if value else fallback

def get_title_and_artist(file_path: str):
    """Extracts Title and Artist from file metadata."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    fallback_title = " ".join(base_name.replace("-", " ").replace("_", " ").split())
    fallback_artist = "Unknown Artist"
    try:
        audio = MutagenFile(file_path, easy=True)
        if audio is None: return fallback_title, fallback_artist
        title = safe_text(audio.get("title", [fallback_title])[0], fallback_title)
        artist = safe_text(audio.get("artist", audio.get("albumartist", [fallback_artist]))[0], fallback_artist)
        return title, artist
    except Exception: return fallback_title, fallback_artist

def estimate_bpm(y: np.ndarray, sr: int) -> int:
    """Analyzes audio rhythm and normalizes to 70-180 BPM."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo)[0])))
    if bpm < 70: bpm *= 2
    elif bpm > 180: bpm = int(round(bpm / 2))
    return max(1, bpm)

def get_base_dir():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

def get_ai_keys(folder_path, log_func):
    """Runs AI key detection directly inside the app process."""
    keys_dict = {}
    try:
        base_dir = get_base_dir()
        model_path = os.path.join(base_dir, "checkpoints", "keynet.pt")

        if not os.path.exists(model_path):
            log_func(f"Critical Error: Missing model: {model_path}")
            return keys_dict

        # Import here so PyInstaller includes it
        from predict_keys import get_audio_list, load_model, preprocess_audio, camelot_output
        import torch

        device = torch.device("cpu")
        model = load_model(model_path, device)
        model.eval()

        files = get_audio_list(folder_path)
        if not files:
            log_func(f"No supported audio files found in: {folder_path}")
            return keys_dict

        log_func("AI analysis started...")

        for p in files:
            try:
                spec = preprocess_audio(p)
                spec = spec.to(device).unsqueeze(0)

                with torch.no_grad():
                    output = model(spec)
                    pred = int(torch.argmax(output, dim=1).cpu().item())

                camelot, key_name = camelot_output(pred)
                keys_dict[p.name] = (key_name, camelot)
                log_func(f"AI OK: {p.name} -> {camelot} | {key_name}")

            except Exception as e:
                log_func(f"AI ERROR: {p.name} -> {e}")

    except Exception as e:
        log_func(f"System Error: {e}")

    return keys_dict

def write_metadata(file_path: str, title: str, artist: str, bpm: int, musical_key: str, camelot: str, log_func):
    """
    Writes specific DJ-friendly metadata:
    Title Tag: Key-BPM
    Artist Tag: SongName-OriginalArtist
    """
    ext = os.path.splitext(file_path)[1].lower()
    new_title = f"{camelot}-{bpm}"
    new_artist = f"{title}-{artist}"
    
    try:
        if ext == ".mp3":
            # 1. Update EasyID3 tags for general compatibility
            try:
                audio = EasyID3(file_path)
            except ID3NoHeaderError:
                mutagen.File(file_path).add_tags()
                audio = EasyID3(file_path)
            
            audio["title"] = [new_title]
            audio["artist"] = [new_artist]
            audio["bpm"] = [str(bpm)]
            audio.save()

            # 2. Update Standard ID3 tags (Crucial for Windows Explorer visibility)
            id3 = ID3(file_path)
            id3.add(TIT2(encoding=3, text=[new_title]))   # Title
            id3.add(TPE1(encoding=3, text=[new_artist]))  # Contributing Artist
            id3.add(TKEY(encoding=3, text=[musical_key])) # Musical Key
            id3.save(file_path, v2_version=3)

        elif ext == ".flac":
            audio = FLAC(file_path)
            # FLAC uses Vorbis Comments (Title and Artist are standard here)
            audio["title"] = [new_title]
            audio["artist"] = [new_artist]
            audio["bpm"] = [str(bpm)]
            audio["initialkey"] = [musical_key]
            audio["djid_camelot"] = [camelot]
            audio.save()

        elif ext == ".wav":
            audio = WAVE(file_path)
            if audio.tags is None:
                audio.add_tags()
            # WAV files are tricky; we force standard ID3 frames into the RIFF header
            audio.tags.add(TIT2(encoding=3, text=[new_title]))
            audio.tags.add(TPE1(encoding=3, text=[new_artist]))
            audio.tags.add(TBPM(encoding=3, text=[str(bpm)]))
            audio.tags.add(TKEY(encoding=3, text=[musical_key]))
            audio.save()

        log_func(f"      Tags Injected: {new_title} | {new_artist}")
    except Exception as e:
        log_func(f"      Metadata Error: {e}")

def convert_audio(src_path: str, dest_path: str, target_format: str):
    """Transcodes audio via FFmpeg."""
    fmt = target_format.replace(".", "")
    audio = AudioSegment.from_file(src_path)
    audio.export(dest_path, format=fmt)

def process_folder(source_root: str, dest_root: str, target_format: str, log_func, completion_callback):
    """Main processing pipeline."""
    try:
        os.makedirs(dest_root, exist_ok=True)
        files = [os.path.join(r, f) for r, d, fs in os.walk(source_root) for f in fs if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
        if not files: raise RuntimeError("No audio files found.")
        
        log_func("Initializing AI Analysis...")
        ai_keys = get_ai_keys(source_root, log_func)

        copied_count, skipped_count = 0, 0
        for src_path in files:
            fn = os.path.basename(src_path)
            if fn not in ai_keys:
                log_func(f"Skipped: {fn}")
                skipped_count += 1
                continue

            log_func(f"Processing: {fn}")
            try:
                musical_key, camelot = ai_keys[fn]
                duration = librosa.get_duration(path=src_path)
                y, sr = librosa.load(src_path, sr=None, mono=True, offset=max(0, (duration-60)/2), duration=60)
                bpm = estimate_bpm(y, sr)
                title, artist = get_title_and_artist(src_path)

                _, orig_ext = os.path.splitext(src_path)
                final_ext = target_format if target_format != "Original" else orig_ext
                
                # Filename logic: Key-BPM-Title-Artist
                new_fn = f"{camelot}-{bpm}-{slugify(title)}-{slugify(artist)}{final_ext}"
                dest_path = os.path.join(dest_root, new_fn)
                
                counter = 1
                base_name = os.path.splitext(dest_path)[0]
                while os.path.exists(dest_path):
                    dest_path = f"{base_name}({counter}){final_ext}"
                    counter += 1

                if target_format == "Original" or orig_ext == target_format:
                    shutil.copy2(src_path, dest_path)
                else:
                    log_func(f"  -> Converting to {final_ext.upper()}...")
                    convert_audio(src_path, dest_path, target_format)

                write_metadata(dest_path, title, artist, bpm, musical_key, camelot, log_func)
                copied_count += 1
            except Exception as e:
                log_func(f"  Error: {e}")
                skipped_count += 1

        log_func("Processing Complete.\n")
        completion_callback(True, f"Finished.\nSuccess: {copied_count}\nSkipped: {skipped_count}")
    except Exception as e: completion_callback(False, str(e))

class TkinterDnD_CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class App(TkinterDnD_CTk):
    def __init__(self):
        super().__init__()
        self.title("Community Audio Studio - AI Edition")
        self.geometry("950x750")
        self.config = self.load_config()
        self.source_path, self.dest_path = ctk.StringVar(), ctk.StringVar()
        self.format_choice, self.custom_ffmpeg_path = ctk.StringVar(value="Original"), ctk.StringVar(value=self.config.get("ffmpeg_path", ""))
        self.setup_ffmpeg_locator()
        self.build_ui()

    def load_config(self):
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f: return json.load(f)
            except: pass
        return {"ffmpeg_path": ""}

    def setup_ffmpeg_locator(self):
        path = self.custom_ffmpeg_path.get()
        bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "ffmpeg.exe")
        found = path if os.path.isfile(path) else (bundled if os.path.isfile(bundled) else shutil.which("ffmpeg"))
        if found:
            bin_dir = os.path.dirname(found)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            AudioSegment.converter = found
            ffprobe = os.path.join(bin_dir, "ffprobe.exe")
            if os.path.isfile(ffprobe): AudioSegment.ffprobe = ffprobe

    def build_ui(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        self.tab_main, self.tab_settings = self.tabview.add("Audio Processor"), self.tabview.add("Settings")

        # Layout
        for i, (label, var) in enumerate([("Source Folder:", self.source_path), ("Output Folder:", self.dest_path)]):
            ctk.CTkLabel(self.tab_main, text=label, font=("Roboto", 14, "bold")).grid(row=i, column=0, padx=15, pady=10, sticky="e")
            entry = ctk.CTkEntry(self.tab_main, textvariable=var, width=450, height=40)
            entry.grid(row=i, column=1, padx=15, pady=10)
            ctk.CTkButton(self.tab_main, text="Browse", width=100, command=lambda v=var: v.set(filedialog.askdirectory())).grid(row=i, column=2, padx=15, pady=10)
            entry.drop_target_register(DND_FILES)
            entry.dnd_bind('<<Drop>>', lambda e, v=var: v.set(e.data.strip('{}')))

        ctk.CTkLabel(self.tab_main, text="Output Format:", font=("Roboto", 14, "bold")).grid(row=2, column=0, sticky="e", padx=15, pady=10)
        ctk.CTkOptionMenu(self.tab_main, variable=self.format_choice, values=["Original", ".mp3", ".wav"]).grid(row=2, column=1, sticky="w", padx=15, pady=10)

        self.run_btn = ctk.CTkButton(self.tab_main, text="START PROCESSING", font=("Roboto", 16, "bold"), height=45, command=self.start_process)
        self.run_btn.grid(row=3, column=0, columnspan=3, pady=20)

        self.log_box = ctk.CTkTextbox(self.tab_main, width=850, height=350, font=("Consolas", 13))
        self.log_box.grid(row=4, column=0, columnspan=3, padx=15, pady=10, sticky="nsew")
        self.tab_main.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(self.tab_settings, text="FFmpeg Path:", font=("Roboto", 16, "bold")).pack(pady=10)
        ctk.CTkEntry(self.tab_settings, textvariable=self.custom_ffmpeg_path, width=500).pack(pady=5)
        ctk.CTkButton(self.tab_settings, text="Save Settings", command=self.save_settings).pack(pady=20)

    def save_settings(self):
        with open("config.json", "w") as f: json.dump({"ffmpeg_path": self.custom_ffmpeg_path.get()}, f)
        self.setup_ffmpeg_locator()
        messagebox.showinfo("Config", "Saved.")

    def log(self, msg: str):
        self.after(0, lambda: [self.log_box.insert("end", msg + "\n"), self.log_box.see("end")])

    def start_process(self):
        if not self.source_path.get() or not self.dest_path.get(): return
        self.log_box.delete("1.0", "end")
        self.run_btn.configure(state="disabled", text="RUNNING...")
        threading.Thread(target=process_folder, args=(self.source_path.get(), self.dest_path.get(), self.format_choice.get(), self.log, lambda s, m: self.after(0, self.finish, s, m)), daemon=True).start()

    def finish(self, success, msg):
        self.run_btn.configure(state="normal", text="START PROCESSING")
        messagebox.showinfo("Status", msg)

if __name__ == "__main__":
    App().mainloop()