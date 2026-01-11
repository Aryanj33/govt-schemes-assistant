"""
Scholarship Voice Assistant - Configuration Module
===================================================
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Find and load .env file
def find_env_file() -> Optional[Path]:
    """Search for .env file in common locations."""
    possible_paths = [
        Path(__file__).parent.parent.parent / "config" / ".env",
        Path(__file__).parent.parent.parent / ".env",
        Path.cwd() / "config" / ".env",
        Path.cwd() / ".env",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None

# Load environment variables
env_path = find_env_file()
if env_path:
    load_dotenv(env_path)

@dataclass
class GroqConfig:
    """Groq API configuration for Whisper STT and Llama LLM."""
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    whisper_model: str = "whisper-large-v3-turbo"
    llm_model: str = "llama-3.1-8b-instant"  # Faster 8B model for lower latency
    llm_temperature: float = 0.3  # Lower for speed and consistency
    llm_max_tokens: int = 60  # Very short for concise conversational responses
    
    def is_configured(self) -> bool:
        return bool(self.api_key)

@dataclass
class LiveKitConfig:
    """LiveKit server configuration."""
    url: str = field(default_factory=lambda: os.getenv("LIVEKIT_URL", "ws://localhost:7880"))
    api_key: str = field(default_factory=lambda: os.getenv("LIVEKIT_API_KEY", "devkey"))
    api_secret: str = field(default_factory=lambda: os.getenv("LIVEKIT_API_SECRET", "secret"))
    room_name: str = "scholarship-assistant"
    
    def is_configured(self) -> bool:
        return bool(self.url and self.api_key and self.api_secret)

@dataclass
class TwilioConfig:
    """Twilio telephony configuration."""
    account_sid: str = field(default_factory=lambda: os.getenv("TWILIO_ACCOUNT_SID", ""))
    auth_token: str = field(default_factory=lambda: os.getenv("TWILIO_AUTH_TOKEN", ""))
    phone_number: str = field(default_factory=lambda: os.getenv("TWILIO_PHONE_NUMBER", ""))
    
    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.phone_number)

@dataclass
class BhashiniConfig:
    """Bhashini TTS API configuration."""
    user_id: str = field(default_factory=lambda: os.getenv("BHASHINI_USER_ID", ""))
    api_key: str = field(default_factory=lambda: os.getenv("BHASHINI_API_KEY", ""))
    pipeline_id: str = field(default_factory=lambda: os.getenv("BHASHINI_PIPELINE_ID", ""))
    tts_service_id: str = "ai4bharat/indic-tts-coqui-indo_women-gpu--t4"
    
    def is_configured(self) -> bool:
        return bool(self.user_id and self.api_key)

@dataclass
class EdgeTTSConfig:
    """Edge TTS configuration (Free high-quality neural voices)."""
    voice_hindi: str = "hi-IN-SwaraNeural"  # Excellent female Hindi voice
    voice_english: str = "en-IN-NeerjaNeural"  # Excellent Indian English voice
    rate: str = "+0%"  # Speed adjustment
    pitch: str = "+0Hz"  # Pitch adjustment
    
    def is_configured(self) -> bool:
        return True  # Always available as it's free/public

@dataclass
class ElevenLabsConfig:
    """ElevenLabs TTS configuration for high-quality neural voices."""
    api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "pMsXg93S9C6U60rV9x9G"))  # Natural female voice
    model_id: str = "eleven_multilingual_v2"  # Best quality for Hindi/Hinglish
    
    def is_configured(self) -> bool:
        return bool(self.api_key)

@dataclass
class GoogleConfig:
    """Google Cloud configuration for TTS fallback and Gemini LLM."""
    credentials_path: str = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    tts_voice_hindi: str = "hi-IN-Wavenet-A"  # Female Hindi voice
    tts_voice_english: str = "en-IN-Wavenet-A"  # Female Indian English voice
    gemini_model: str = "gemini-2.0-flash-exp"
    
    def is_tts_configured(self) -> bool:
        # TTS works with either credentials file OR API key
        has_creds = bool(self.credentials_path) and Path(self.credentials_path).exists()
        has_api_key = bool(self.api_key)
        return has_creds or has_api_key
    
    def is_gemini_configured(self) -> bool:
        return bool(self.api_key)

@dataclass
class DataConfig:
    """Data and RAG configuration."""
    scholarships_path: Path = field(default_factory=lambda: Path(
        os.getenv("SCHOLARSHIPS_DATA_PATH", 
                  str(Path(__file__).parent.parent.parent / "data" / "processed" / "schemes.json"))
    ))
    faiss_index_path: Path = field(default_factory=lambda: Path(
        os.getenv("FAISS_INDEX_PATH",
                  str(Path(__file__).parent.parent.parent / "data" / "embeddings" / "faiss_index"))
    ))
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_results: int = 1  # Single best match for speed
    
    def ensure_directories(self):
        """Create data directories if they don't exist."""
        self.scholarships_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)

@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    
    # Sub-configurations
    groq: GroqConfig = field(default_factory=GroqConfig)
    livekit: LiveKitConfig = field(default_factory=LiveKitConfig)
    twilio: TwilioConfig = field(default_factory=TwilioConfig)
    bhashini: BhashiniConfig = field(default_factory=BhashiniConfig)
    edge_tts: EdgeTTSConfig = field(default_factory=EdgeTTSConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of configuration issues found
        """
        issues = []
        
        # Check required configurations
        if not self.groq.is_configured():
            issues.append("⚠️  GROQ_API_KEY not set - STT and primary LLM will not work")
        
        if not self.livekit.is_configured():
            issues.append("⚠️  LiveKit not fully configured - voice streaming may fail")
        
        # Check TTS options
        if not self.bhashini.is_configured() and not self.google.is_tts_configured():
            issues.append("⚠️  No TTS configured - need either Bhashini or Google Cloud TTS")
        
        # Check LLM fallback
        if not self.groq.is_configured() and not self.google.is_gemini_configured():
            issues.append("❌ No LLM configured - at least one of Groq or Gemini required")
        
        return issues
    
    def print_status(self):
        """Print configuration status for debugging."""
        print("\n" + "="*50)
        print("📋 CONFIGURATION STATUS")
        print("="*50)
        print(f"🔧 Debug Mode: {self.debug}")
        print(f"📝 Log Level: {self.log_level}")
        print(f"🌐 Port: {self.port}")
        print()
        print(f"🎤 Groq STT/LLM: {'✅ Configured' if self.groq.is_configured() else '❌ Not configured'}")
        print(f"🔊 LiveKit: {'✅ Configured' if self.livekit.is_configured() else '❌ Not configured'}")
        print(f"📞 Twilio Phone: {'✅ Configured' if self.twilio.is_configured() else '❌ Not configured'}")
        print(f"🗣️  Bhashini TTS: {'✅ Configured' if self.bhashini.is_configured() else '❌ Not configured'}")
        print(f"☁️  Google TTS: {'✅ Configured' if self.google.is_tts_configured() else '❌ Not configured'}")
        print(f"🤖 Gemini LLM: {'✅ Configured' if self.google.is_gemini_configured() else '❌ Not configured'}")
        print()
        
        issues = self.validate()
        if issues:
            print("⚠️  Configuration Issues:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("✅ All configurations valid!")
        print("="*50 + "\n")

# Global config instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config
