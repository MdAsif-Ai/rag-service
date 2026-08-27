import os
import logging
import subprocess
from typing import List, Optional
from google import genai
from google.genai import types
from groq import Groq
from app.ingestion.loaders.base import DocumentLoader, ParsedSection

logger = logging.getLogger(__name__)

class GeminiVisionLoader(DocumentLoader):
    """Loads images (PNG, JPG) or PDFs of handwritten notes using Gemini natively."""
    def __init__(self, is_pdf: bool = False):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.is_pdf = is_pdf

    def load(self, file_path: str) -> List[ParsedSection]:
        sections = []
        
        # Read the raw bytes of the file
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        # Determine the correct MIME type
        if self.is_pdf:
            mime_type = "application/pdf"
        elif file_path.lower().endswith(".png"):
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"
            
        markdown = self._analyze_file(file_bytes, mime_type)
        
        sections.append(ParsedSection(
            content=markdown,
            source_type="gemini_vision",
            metadata={"original_file": os.path.basename(file_path)}
        ))
            
        return sections

    def _analyze_file(self, file_bytes: bytes, mime_type: str) -> str:
        prompt = (
            "You are an expert educational assistant. "
            "Carefully transcribe all text in this document, including handwriting. "
            "If it contains math equations, transcribe them in LaTeX. "
            "If it is a diagram, explain it in detail. Format output in clean Markdown."
        )
        try:
            # Pass the raw file bytes directly to Gemini
            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
                ]
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Vision failed: {e}")
            raise RuntimeError(f"Gemini failed to process file: {e}")


class GroqAudioLoader(DocumentLoader):
    """Loads audio files (MP3, WAV) and YouTube URLs using Groq Whisper."""
    def __init__(self, url: Optional[str] = None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.url = url

    def load(self, file_path: str) -> List[ParsedSection]:
        audio_path = file_path

        if self.url:
            audio_path = "temp_youtube_audio.mp3"
            logger.info(f"Downloading audio from YouTube: {self.url}")
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3", 
                "-o", audio_path, self.url
            ], check=True)

        logger.info(f"Transcribing audio with Groq Whisper: {audio_path}")
        
        with open(audio_path, "rb") as file:
            transcript = self.client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )

        if self.url and os.path.exists(audio_path):
            os.remove(audio_path)

        sections = []
        for segment in transcript.segments:
            start_time = segment.get("start", 0)
            text = segment.get("text", "").strip()
            if text:
                mins, secs = divmod(int(start_time), 60)
                timestamp = f"[{mins:02d}:{secs:02d}]"
                sections.append(ParsedSection(
                    content=f"{timestamp} {text}",
                    source_type="audio",
                    metadata={"start_time": start_time}
                ))
        return sections


class VideoLoader(DocumentLoader):
    """Extracts audio from video files (MP4, MKV) and transcribes via Groq Whisper."""
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load(self, file_path: str) -> List[ParsedSection]:
        audio_path = "temp_video_audio.mp3"
        logger.info(f"Extracting audio from video: {file_path}")
        
        subprocess.run([
            "ffmpeg", "-i", file_path, "-vn", "-acodec", "libmp3lame", "-ab", "128k", audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logger.info(f"Transcribing video audio with Groq Whisper: {audio_path}")
        with open(audio_path, "rb") as file:
            transcript = self.client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )

        if os.path.exists(audio_path):
            os.remove(audio_path)

        sections = []
        for segment in transcript.segments:
            start_time = segment.get("start", 0)
            text = segment.get("text", "").strip()
            if text:
                mins, secs = divmod(int(start_time), 60)
                timestamp = f"[{mins:02d}:{secs:02d}]"
                sections.append(ParsedSection(
                    content=f"{timestamp} {text}",
                    source_type="video",
                    metadata={"start_time": start_time}
                ))
        return sections