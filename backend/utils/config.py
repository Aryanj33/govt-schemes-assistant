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
    # Text path: 70B for high-quality, longer answers (Discover / chat).
    llm_model: str = field(default_factory=lambda: os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"))
    llm_temperature: float = 0.1
    llm_max_tokens: int = 200
    # Voice path: 8b-instant is ~3x faster TTFT (80-150ms vs 250-400ms).
    # Quality is fine for short Hinglish replies; we already ground on RAG.
    voice_llm_model: str = field(default_factory=lambda: os.getenv("GROQ_VOICE_MODEL", "llama-3.1-8b-instant"))
    # 220 tokens ≈ 150 words — enough for 2-3 grounded sentences with safety
    # headroom so the model never gets cut off mid-sentence by max_tokens.
    # System prompts still enforce brevity; this is just the hard-stop ceiling.
    voice_llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("GROQ_VOICE_MAX_TOKENS", "220")))
    voice_llm_temperature: float = 0.15
    
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
    webhook_base_url: str = field(default_factory=lambda: os.getenv("WEBHOOK_BASE_URL", ""))

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
    """ElevenLabs TTS configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "tA6LGZpsqStKtSaGiXND"))
    model_id: str = "eleven_turbo_v2_5"  # Fastest model for voice calls

    def is_configured(self) -> bool:
        return bool(self.api_key)

@dataclass
class CartesiaConfig:
    """Cartesia TTS configuration for ultra-low-latency neural voices."""
    api_key: str = field(default_factory=lambda: os.getenv("CARTESIA_API_KEY", ""))
    voice_id: str = field(default_factory=lambda: os.getenv("CARTESIA_VOICE_ID", "694f9389-aac1-45b6-b726-9d9369183238"))  # Default Cartesia voice
    # sonic-2 supports Hindi + English + Hinglish natively; sonic-english is English-only
    model_id: str = field(default_factory=lambda: os.getenv("CARTESIA_MODEL_ID", "sonic-2"))
    
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
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast and lightweight model
    top_k_results: int = 5  # Top 5 results for better context
    
    def ensure_directories(self):
        """Create data directories if they don't exist."""
        self.scholarships_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)

@dataclass
class SessionConfig:
    """Session persistence configuration."""
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", ""))
    ttl_seconds: int = field(default_factory=lambda: int(os.getenv("SESSION_TTL_SECONDS", "3600")))
    history_limit: int = field(default_factory=lambda: int(os.getenv("SESSION_HISTORY_LIMIT", "12")))
    session_prefix: str = field(default_factory=lambda: os.getenv("SESSION_PREFIX", "sarkari_mitra"))
    
    def is_redis_configured(self) -> bool:
        return bool(self.redis_url)

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
    cartesia: CartesiaConfig = field(default_factory=CartesiaConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    
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
        print(f"🔊 ElevenLabs TTS: {'✅ Configured' if self.elevenlabs.is_configured() else '❌ Not configured'}")
        print(f"🎙️  Cartesia TTS: {'✅ Configured' if self.cartesia.is_configured() else '❌ Not configured'}")
        print(f"☁️  Google TTS: {'✅ Configured' if self.google.is_tts_configured() else '❌ Not configured'}")
        print(f"🤖 Gemini LLM: {'✅ Configured' if self.google.is_gemini_configured() else '❌ Not configured'}")
        print(f"🗂️  Redis Sessions: {'✅ Configured' if self.session.is_redis_configured() else '⚠️ Memory fallback'}")
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
