# YT Summary Pro

A Python script to generate AI-powered summaries and analyses of YouTube videos or local audio files.

## Features

- Download YouTube video audio.
- Transcribe audio using OpenAI Whisper models.
- Generate AI analysis (rating, summary, recommendation, content density, key timestamps) using DeepSeek or OpenAI.
- Output interactive HTML reports.
- Caching for downloaded audio and AI analysis.

## Dependencies

- `yt-dlp`
- `openai`
- `transformers`
- `torch`
- `numpy`
- `ffmpeg` (for audio extraction from local video files)

Install Python dependencies using:
```bash
pip install yt-dlp openai transformers torch numpy
```
Ensure `ffmpeg` is installed and accessible in your system's PATH.

## Setup

1.  **API Keys:**
    This script requires an API key for either DeepSeek or OpenAI, depending on your `--api-provider` choice.

    -   **Environment Variables (Recommended):** Set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` in your environment.
        ```bash
        export DEEPSEEK_API_KEY="YOUR_DEEPSEEK_KEY"
        # or
        export OPENAI_API_KEY="YOUR_OPENAI_KEY"
        ```
    -   **`api_keys.txt` (Local Fallback):** Create a file named `api_keys.txt` in the script's directory with your keys:
        ```
        DEEPSEEK_API_KEY=YOUR_DEEPSEEK_KEY
        OPENAI_API_KEY=YOUR_OPENAI_KEY
        ```
        **NOTE:** `api_keys.txt` is ignored by Git and will not be committed to your repository.

2.  **Output Directory:**
    The script will create an `output/` directory by default to store generated HTML reports. This directory is also ignored by Git.

## Usage

```bash
python ytsummary.py -h
```

Example:
```bash
python ytsummary.py -u "https://www.youtube.com/watch?v=VIDEO_ID" --api-provider deepseek
```
