"""
Scholarship Voice Assistant - Voice Pipeline
=============================================
STT (Groq Whisper) and TTS (Bhashini/Google Cloud) integration.
"""

import asyncio
import base64
import io
import time
from pathlib import Path
from typing import Optional, AsyncGenerator, Union
import httpx
try:
    import edge_tts
except ImportError:
    edge_tts = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config import get_config

logger = get_logger()
config = get_config()


class SpeechToText:
    """
    Speech-to-Text using Groq Whisper API.
    Ultra-fast transcription optimized for voice assistants.
    """
    
    def __init__(self):
        """Initialize the STT service."""
        self.api_key = config.groq.api_key
        self.model = config.groq.whisper_model
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def transcribe(
        self, 
        audio_data: bytes,
        language: str = "hi",  # Default to Hindi, handles Hinglish well
        prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Raw audio bytes (WAV, MP3, or WebM format)
            language: Language code (hi, en, or auto-detect)
            prompt: Optional prompt to guide transcription
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.api_key:
            logger.error("❌ Groq API key not configured for STT")
            return None
        
        start_time = time.time()
        
        try:
            client = await self._get_client()
            
            # Try to convert WebM to WAV if needed
            processed_audio = audio_data
            audio_filename = "audio.webm"
            audio_mime = "audio/webm"
            
            # Check if it's WebM (starts with WEBM signature or is not WAV)
            if not audio_data[:4] == b'RIFF':
                # Try fast ffmpeg subprocess first (no Python overhead)
                try:
                    process = await asyncio.create_subprocess_exec(
                        'ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-ar', '16000', '-ac', '1', 'pipe:1',
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    stdout, _ = await process.communicate(input=audio_data)
                    if stdout and len(stdout) > 44:  # Valid WAV has at least header
                        processed_audio = stdout
                        audio_filename = "audio.wav"
                        audio_mime = "audio/wav"
                        logger.info(f"⚡ Fast ffmpeg conversion: {len(audio_data)} -> {len(processed_audio)} bytes")
                except FileNotFoundError:
                    # ffmpeg not installed, try pydub fallback
                    logger.info("📦 ffmpeg not found, using pydub fallback")
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
                        wav_io = io.BytesIO()
                        audio_segment.export(wav_io, format="wav")
                        processed_audio = wav_io.getvalue()
                        audio_filename = "audio.wav"
                        audio_mime = "audio/wav"
                        logger.info(f"📦 Pydub conversion: {len(audio_data)} -> {len(processed_audio)} bytes")
                    except ImportError:
                        logger.warning("⚠️ pydub not installed, sending raw audio")
                    except Exception as conv_err:
                        logger.warning(f"⚠️ Audio conversion failed: {conv_err}, sending raw audio")
                except Exception as ffmpeg_err:
                    logger.warning(f"⚠️ ffmpeg conversion failed: {ffmpeg_err}, trying pydub")
                    # Fallback to pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
                        wav_io = io.BytesIO()
                        audio_segment.export(wav_io, format="wav")
                        processed_audio = wav_io.getvalue()
                        audio_filename = "audio.wav"
                        audio_mime = "audio/wav"
                    except Exception:
                        pass  # Use raw audio
            
            # Prepare the audio file
            files = {
                "file": (audio_filename, processed_audio, audio_mime),
            }
            
            data = {
                "model": self.model,
                "response_format": "text",
            }
            
            # Add language hint (helps with Indian accents)
            if language != "auto":
                data["language"] = language
            
            # Add prompt for better context
            if prompt:
                data["prompt"] = prompt
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            
            logger.api_call("Groq Whisper", "transcriptions")
            
            response = await client.post(
                self.api_url,
                files=files,
                data=data,
                headers=headers
            )
            
            if response.status_code == 200:
                text = response.text.strip()
                elapsed = (time.time() - start_time) * 1000
                logger.latency("STT", elapsed)
                logger.user_speech(text[:100] + "..." if len(text) > 100 else text)
                return text
            else:
                logger.error(f"❌ STT failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error_with_context("STT", e, f"Audio size: {len(audio_data)} bytes")
            return None
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class TextToSpeech:
    """
    Text-to-Speech using Bhashini API with Google Cloud fallback.
    Natural Hindi voices for Indian context.
    """
    
    def __init__(self):
        """Initialize the TTS service."""
        self.bhashini_config = config.bhashini
        self.edge_tts_config = config.edge_tts
        self.google_config = config.google
        self._client: Optional[httpx.AsyncClient] = None
        self._google_client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def synthesize(
        self,
        text: str,
        language: str = "hi",  # hi for Hindi, en for English
        voice_gender: str = "female"
    ) -> Optional[bytes]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language code (hi or en)
            voice_gender: Voice gender (female or male)
            
        Returns:
            Audio bytes (MP3/WAV format) or None if failed
        """
        # Try Edge TTS first (Highest Quality)
        if self.edge_tts_config.is_configured() and edge_tts is not None:
            audio = await self._synthesize_edge_tts(text, language)
            if audio:
                return audio
        
        # Try Bhashini second
        if self.bhashini_config.is_configured():
            audio = await self._synthesize_bhashini(text, language)
            if audio:
                return audio
            logger.warning("⚠️ Bhashini TTS failed, trying Google Cloud")
        
        # Fallback to Google Cloud
        if self.google_config.is_tts_configured():
            return await self._synthesize_google(text, language, voice_gender)
        
        # Last resort: return None
        logger.error("❌ No TTS service available")
        return None
    
    async def _synthesize_edge_tts(
        self,
        text: str,
        language: str = "hi"
    ) -> Optional[bytes]:
        """
        Synthesize using Microsoft Edge TTS (Free Neural Voices).
        Top quality for Hindi/Hinglish.
        """
        start_time = time.time()
        try:
            # Select voice
            voice = self.edge_tts_config.voice_hindi if language == "hi" else self.edge_tts_config.voice_english
            
            communicate = edge_tts.Communicate(text, voice, rate=self.edge_tts_config.rate, pitch=self.edge_tts_config.pitch)
            
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            
            elapsed = (time.time() - start_time) * 1000
            logger.latency("EdgeTTS", elapsed)
            return bytes(audio_data)
            
        except Exception as e:
            logger.error_with_context("EdgeTTS", e, f"Text: {text[:50]}...")
            return None
    
    async def _synthesize_bhashini(
        self,
        text: str,
        language: str = "hi"
    ) -> Optional[bytes]:
        """
        Synthesize using Bhashini API.
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Audio bytes or None
        """
        start_time = time.time()
        
        try:
            client = await self._get_client()
            
            # Bhashini TTS endpoint
            url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
            
            headers = {
                "Authorization": self.bhashini_config.api_key,
                "Content-Type": "application/json"
            }
            
            # Map language codes
            lang_map = {
                "hi": "hi",
                "en": "en",
                "hinglish": "hi"  # Use Hindi voice for Hinglish
            }
            source_lang = lang_map.get(language, "hi")
            
            payload = {
                "pipelineTasks": [
                    {
                        "taskType": "tts",
                        "config": {
                            "language": {
                                "sourceLanguage": source_lang
                            },
                            "serviceId": self.bhashini_config.tts_service_id,
                            "gender": "female"
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": text}]
                }
            }
            
            logger.api_call("Bhashini", "TTS")
            
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract audio from response
                audio_content = result.get("pipelineResponse", [{}])[0]
                audio_base64 = audio_content.get("audio", [{}])[0].get("audioContent")
                
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    elapsed = (time.time() - start_time) * 1000
                    logger.latency("Bhashini TTS", elapsed)
                    return audio_bytes
            
            logger.warning(f"⚠️ Bhashini TTS: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error_with_context("Bhashini TTS", e, f"Text: {text[:50]}...")
            return None
    
    async def _synthesize_google(
        self,
        text: str,
        language: str = "hi",
        voice_gender: str = "female"
    ) -> Optional[bytes]:
        """
        Synthesize using Google Cloud TTS REST API.
        
        Args:
            text: Text to synthesize
            language: Language code
            voice_gender: Voice gender
            
        Returns:
            Audio bytes or None
        """
        start_time = time.time()
        
        # Check if we have API key
        api_key = self.google_config.api_key
        if not api_key:
            logger.error("❌ Google API key not configured for TTS")
            return None
        
        try:
            client = await self._get_client()
            
            # Select voice based on language
            if language == "hi":
                voice_name = self.google_config.tts_voice_hindi
                language_code = "hi-IN"
            else:
                voice_name = self.google_config.tts_voice_english
                language_code = "en-IN"
            
            # Google TTS REST API endpoint
            url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
            
            payload = {
                "input": {"text": text},
                "voice": {
                    "languageCode": language_code,
                    "name": voice_name
                },
                "audioConfig": {
                    "audioEncoding": "MP3",
                    "speakingRate": 1.0
                }
            }
            
            logger.api_call("Google Cloud", "TTS")
            
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                audio_base64 = result.get("audioContent")
                
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    elapsed = (time.time() - start_time) * 1000
                    logger.latency("Google TTS", elapsed)
                    return audio_bytes
            else:
                logger.error(f"❌ Google TTS error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error_with_context("Google TTS", e, f"Text: {text[:50]}...")
            return None
    
    async def close(self):
        """Close HTTP clients."""
        if self._client:
            await self._client.aclose()


class VoicePipeline:
    """
    Combined voice pipeline managing STT and TTS.
    Handles language detection and routing.
    """
    
    def __init__(self):
        """Initialize the voice pipeline."""
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
    
    async def speech_to_text(
        self,
        audio_data: bytes,
        hint_language: str = "hi"
    ) -> Optional[str]:
        """
        Convert speech to text.
        
        Args:
            audio_data: Audio bytes
            hint_language: Expected language for better accuracy
            
        Returns:
            Transcribed text or None
        """
        # Add context prompt for scholarship domain
        prompt = "Scholarship, engineering, medical, SC, ST, OBC, AICTE, NTSE, INSPIRE, NSP"
        
        return await self.stt.transcribe(
            audio_data=audio_data,
            language=hint_language,
            prompt=prompt
        )
    
    async def text_to_speech(
        self,
        text: str,
        detected_language: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            detected_language: Language detected from user's speech
            
        Returns:
            Audio bytes or None
        """
        # Auto-detect language from text if not provided
        if detected_language is None:
            detected_language = self._detect_language(text)
        
        return await self.tts.synthesize(
            text=text,
            language=detected_language
        )
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on script.
        
        Args:
            text: Input text
            
        Returns:
            Language code (hi or en)
        """
        # Check for Devanagari script
        devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        
        # If more than 20% Devanagari, treat as Hindi
        if devanagari_count > len(text) * 0.2:
            return "hi"
        
        return "en"  # Default to English/Hinglish voice
    
    async def close(self):
        """Close all clients."""
        await self.stt.close()
        await self.tts.close()


# Singleton instance
_voice_pipeline: Optional[VoicePipeline] = None

def get_voice_pipeline() -> VoicePipeline:
    """Get the global voice pipeline instance."""
    global _voice_pipeline
    if _voice_pipeline is None:
        _voice_pipeline = VoicePipeline()
    return _voice_pipeline
