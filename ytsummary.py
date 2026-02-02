#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import os
import re
import sys
import time
import torch
import numpy as np
import subprocess
import platform
import threading
import unicodedata
from datetime import datetime
from yt_dlp import YoutubeDL
from openai import OpenAI
from transformers import pipeline

# ======================
# Helper Functions
# ======================
def format_timestamp(seconds):
    try:
        return str(datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S'))
    except:
        return "00:00:00"

def slugify(value):
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)

def load_api_key(key_name):
    # Try loading from environment variable first
    env_key = os.environ.get(key_name.upper())
    if env_key:
        return env_key

    # Fallback to api_keys.txt
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        key_file = os.path.join(script_dir, "api_keys.txt")
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                for line in f:
                    if line.strip() and '=' in line:
                        key, value = line.split('=', 1)
                        if key.strip().upper() == key_name.upper():
                            return value.strip().strip('"').strip("'")
        return None
    except Exception as e:
        print(f"Error reading API key file: {str(e)}")
        return None


def loading_animation(stop_event, message="Processing"):
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{animation[i]} {message}")
        sys.stdout.flush()
        time.sleep(0.1)
        i = (i + 1) % len(animation)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
    sys.stdout.flush()

def start_loading(message="Processing"):
    stop_event = threading.Event()
    thread = threading.Thread(target=loading_animation, args=(stop_event, message))
    thread.daemon = True
    thread.start()
    return stop_event

def download_audio(url):
    output_path = os.path.expanduser(f"~/.audio_cache/{hashlib.md5(url.encode()).hexdigest()}.mp3")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"Using cached audio: {output_path}")
        return output_path

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace('.mp3', '.%(ext)s'),
        'quiet': True,
    }

    stop_event = start_loading("Downloading audio")
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            temp_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            os.rename(temp_path, output_path)
            return output_path
    finally:
        stop_event.set()

def transcribe_audio(audio_path, model_size='small'):
    transcript_cache = audio_path + ".json"
    if os.path.exists(transcript_cache):
        print(f"⚡ Using cached transcript: {transcript_cache}")
        import json
        with open(transcript_cache, 'r') as f:
            return json.load(f)

    start_time = time.time()
    stop_event = start_loading("Transcribing audio")
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}",
            device="mps",
            torch_dtype=torch.float16,
            model_kwargs={"attn_implementation": "sdpa"}
        )
        result = pipe(audio_path, chunk_length_s=30, batch_size=4, return_timestamps=True)
        segments = []
        for chunk in result["chunks"]:
            start = chunk.get("timestamp", (0, 0))[0] or 0
            end = chunk.get("timestamp", (0, 0))[1] or start + 30
            segments.append({"text": chunk.get("text", ""), "timestamp": (start, end)})
            transcript_data = {"text": result["text"], "segments": segments}
        import json
        with open(transcript_cache, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        elapsed = time.time() - start_time
        print(f"🕒 Transcription completed in {elapsed:.2f} seconds")
        return transcript_data
    finally:
        stop_event.set()

def parse_ai_response(response):
    result = {'rating': 'N/A', 'summary': '', 'recommendation': '', 'density': '', 'timestamps': []}
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('Rating:'):
            result['rating'] = line.split(': ')[1]
        elif line.startswith('Summary:'):
            result['summary'] = line.split(': ')[1]
        elif line.startswith('Recommendation:'):
            result['recommendation'] = line.split(': ')[1]
        elif line.startswith('Density:'):
            result['density'] = line.split(': ')[1]
        elif line.startswith('- ['):
            match = re.search(r'\[([\d:]+)\]', line)
            if match:
                timestamp = match.group(1)
                topic = re.sub(r'^-\s*\[[\d:]+\]\s*', '', line)
                result['timestamps'].append({'time': timestamp, 'topic': topic})
    return result

def generate_html_report(result):
    css = """
    <style>
    .transcript { margin-top: 2rem; padding: 1rem; background: #f9f9f9; font-family: monospace; }
    .transcript-line { margin-bottom: 0.5rem; }
    .transcript-timestamp { color: #555; margin-right: 1rem; }
    </style>
    """
    body = f"""
    <h2>{result['title']}</h2>
    <p><strong>Rating:</strong> {result['rating']} | <strong>Density:</strong> {result['density']}</p>
    <p><strong>Summary:</strong> {result['summary']}</p>
    <p><strong>Recommendation:</strong> {result['recommendation']}</p>
    <ul>
    {''.join([f"<li><a href='{result['url']}&t={time_to_seconds(t['time'])}s'>[{t['time']}]</a> {t['topic']}</li>" for t in result['timestamps']])}
    </ul>
    <div class="transcript">
    {''.join([f"<div class='transcript-line'><span class='transcript-timestamp'>{format_timestamp(s['timestamp'][0])}-{format_timestamp(s['timestamp'][1])}</span><span>{s['text']}</span></div>" for s in result['transcript_segments']])}
    </div>
    """
    return f"<html><head><meta charset='utf-8'>{css}</head><body>{body}</body></html>"

def print_system_info():
    print("System Info:")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    try:
        gpu_info = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode()
        gpu_name = re.search(r"Chipset Model: (.*)", gpu_info).group(1)
        print(f"GPU: {gpu_name.strip()}")
    except Exception as e:
        print(f"GPU Info Error: {str(e)}")

def is_playlist(url):
    with YoutubeDL({'quiet': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return 'entries' in info
        except:
            return False

def get_playlist_videos(url):
    ydl_opts = {'extract_flat': True, 'quiet': True, 'force_generic_extractor': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return [f"https://youtube.com/watch?v={entry['id']}" for entry in info['entries']]

def time_to_seconds(timestamp):
    try:
        parts = list(map(int, timestamp.split(':')))
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except:
        return 0

def get_audio_duration(file_path):
    import wave, contextlib
    with contextlib.closing(wave.open(file_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def generate_analysis(transcript_text, api_key, provider):
    import hashlib, json
    cache_key = hashlib.md5(transcript_text.encode()).hexdigest()
    cache_path = os.path.expanduser(f"~/.audio_cache/{cache_key}.analysis.json")
    if os.path.exists(cache_path):
        print(f"⚡ Using cached AI analysis: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    prompt = """Analyze this transcript and provide:
    1. Rating (1-10)
    2. 1-sentence summary
    3. Watch recommendation
    4. Content density percentage
    5. Key timestamps with topics

    Format:
    Rating: X/10
    Summary: [summary]
    Recommendation: [Watch/Skip] - [reason]
    Density: XX%
    Timestamps:
    - [HH:MM:SS] Topic"""

    if provider == 'deepseek':
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        model_name = "deepseek-chat"
    else:
        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        model_name = "gpt-4o"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript_text}
        ]
    )
    result = parse_ai_response(response.choices[0].message.content)
    with open(cache_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result

def main():
    print_system_info()

    parser = argparse.ArgumentParser(description='YT Summary Pro: AI-Powered Video Analysis')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-u', '--url', help='YouTube URL (video or playlist)')
    input_group.add_argument('-i', '--input-file', help='Local audio file path (mp3, m4a, wav, etc.)')

    parser.add_argument('-o', '--output', default='output', help='Output directory for HTML files')
    parser.add_argument('--api-provider', choices=['deepseek', 'openai'], default='openai', help='API provider')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--model', choices=['base', 'small', 'medium'], default='small', help='Whisper model size')
    parser.add_argument('--parallel', type=int, default=4, help='Number of workers')
    parser.add_argument('--keep-audio', action='store_true', help='Keep downloaded audio files')

    args = parser.parse_args()
    api_key = args.api_key or load_api_key(f"{args.api_provider.upper()}_API_KEY") or input(f"Enter {args.api_provider.capitalize()} API key: ").strip()
    os.makedirs(args.output, exist_ok=True)

    tasks = []
    if args.url:
        video_urls = get_playlist_videos(args.url) if is_playlist(args.url) else [args.url]
        for url in video_urls:
            tasks.append((url, None))
    elif args.input_file:
        tasks.append((None, args.input_file))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_audio, url, file_path, args, api_key) for url, file_path in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                output_file = os.path.join(args.output, f"{slugify(result['title'])}.html")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(generate_html_report(result))
                print(f"✅ Report saved to {output_file}")

def process_audio(url, file_path, args, api_key):
    try:
        def extract_audio_from_video(video_path):
            wav_path = os.path.splitext(video_path)[0] + ".wav"
            cmd = f"ffmpeg -y -i \"{video_path}\" -ar 16000 -ac 1 -f wav \"{wav_path}\""
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return wav_path

        if url:
            with YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title, duration = info.get('title', 'Unknown'), info.get('duration', 0)
                audio_path = download_audio(url)
        else:
            audio_path = extract_audio_from_video(file_path)
            title, duration = os.path.basename(file_path), get_audio_duration(audio_path)

        print(f"📺 Title: {title}")
        transcript = transcribe_audio(audio_path, args.model)
        analysis = generate_analysis(transcript['text'], api_key, args.api_provider)

        if not args.keep_audio and url and os.path.exists(audio_path):
            os.remove(audio_path)

        print("📝 Quick Summary:")
        print(f"Rating        : {analysis['rating']}")
        print(f"Summary       : {analysis['summary']}")
        print(f"Recommendation: {analysis['recommendation']}")
        print(f"Density       : {analysis['density']}")
        print("Timestamps:")
        for t in analysis['timestamps']:
            print(f" - [{t['time']}] {t['topic']}")

        return {
            'title': title,
            'duration': duration,
            'url': url or "Local File",
            'transcript_segments': transcript['segments'],
            **analysis
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == '__main__':
    main()
