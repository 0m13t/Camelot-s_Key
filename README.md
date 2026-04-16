# CDJ-Legacy-Metadata-Assistant

An AI-powered audio processor designed for DJs using legacy hardware (Pioneer CDJ-1000, CDJ-800, etc.). This tool analyzes your music library for Key and BPM, then injects that data directly into the file metadata and filenames so they are visible on small hardware displays.

## The Problem
Older CDJs don't have the advanced sorting and analysis of modern Rekordbox setups. Finding the right key or tempo involves a lot of guesswork or manual prep, I wanted to reduce this to bare minimum to have more fun playing on my old cdj800's

## The Solution
This app automates the workflow:
1.  AI Key Detection: Uses a CNN (KeyNet) to predict the musical key with high accuracy.
2.  BPM Estimation: Analyzes the rhythm to find the tempo.
3.  Camelot Conversion: Automatically converts keys to the Camelot Wheel (e.g., 8A, 11B).
4.  Metadata Injection: Renames files and updates tags to:
    Title: `[Key]-[BPM]` (e.g., `8A-124`)
    Artist: `[Song Name]-[Original Artist]`

This allows you to see the mix-compatibility at a glance on the CDJ screen.

## Installation

### Prerequisites
* **Python 3.10+**
* **FFmpeg**: Required for audio processing. (Place `ffmpeg.exe` in a `bin/` folder or add it to your System PATH).

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/CDJ-Legacy-Metadata-Assistant.git](https://github.com/YOUR_USERNAME/CDJ-Legacy-Metadata-Assistant.git)
   cd CDJ-Legacy-Metadata-Assistant