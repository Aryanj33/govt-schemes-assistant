"""
Scholarship Voice Assistant - LiveKit Agent
============================================
Main orchestration using LiveKit Agents framework.
Handles real-time voice conversation flow.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, AsyncGenerator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger, setup_logging
from utils.config import get_config

logger = get_logger()
config = get_config()

# Check for LiveKit availability
try:
    from livekit import rtc, api
    from livekit.agents import (
        JobContext,
        JobProcess,
        WorkerOptions,
        cli,
        llm,
    )
    from livekit.agents.voice_assistant import VoiceAssistant
    from livekit.plugins import silero
    HAS_LIVEKIT_AGENTS = True
except ImportError:
    HAS_LIVEKIT_AGENTS = False
    logger.warning("⚠️ LiveKit Agents not installed. Using simplified pipeline.")


class ScholarshipVoiceAgent:
    """
    Main voice agent for scholarship assistance.
    Orchestrates STT → LLM → TTS pipeline.
    """
    
    def __init__(self):
        """Initialize the voice agent."""
        self.rag = None
        self.conversation_handler = None
        self.voice_pipeline = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return
        
        logger.info("🚀 Initializing Scholarship Voice Agent...")
        
        # Import here to avoid circular imports
        from rag.scholarship_rag import get_scholarship_rag
        from agent.conversation_handler import get_conversation_handler
        from agent.voice_pipeline import get_voice_pipeline
        
        # Initialize RAG
        self.rag = get_scholarship_rag()
        if not self.rag.is_ready:
            logger.info("📚 Building scholarship index...")
            self.rag.build_index()
        logger.info(f"✅ RAG ready with {self.rag.vectorstore.size} scholarships")
        
        # Initialize conversation handler
        self.conversation_handler = get_conversation_handler()
        await self.conversation_handler.initialize()
        logger.info("✅ Conversation handler ready")
        
        # Initialize voice pipeline
        self.voice_pipeline = get_voice_pipeline()
        logger.info("✅ Voice pipeline ready")
        
        self._initialized = True
        logger.info("✅ Scholarship Voice Agent initialized!")
    
    async def process_audio(self, audio_data: bytes) -> Optional[bytes]:
        """
        Process incoming audio and generate voice response.
        
        Args:
            audio_data: Raw audio bytes from user
            
        Returns:
            Response audio bytes or None
        """
        await self.initialize()
        
        start_time = time.time()
        
        # Step 1: Speech to Text
        user_text = await self.voice_pipeline.speech_to_text(audio_data)
        if not user_text:
            logger.warning("⚠️ STT returned empty result")
            return None
        
        stt_time = time.time()
        logger.latency("STT Total", (stt_time - start_time) * 1000)
        
        # Step 2: Generate LLM Response
        response_text = await self.conversation_handler.generate_response(user_text)
        
        llm_time = time.time()
        logger.latency("LLM Total", (llm_time - stt_time) * 1000)
        
        # Step 3: Text to Speech
        audio_response = await self.voice_pipeline.text_to_speech(response_text)
        
        tts_time = time.time()
        logger.latency("TTS Total", (tts_time - llm_time) * 1000)
        
        # Total end-to-end
        logger.latency("End-to-End", (tts_time - start_time) * 1000)
        
        return audio_response
    
    async def process_audio_stream(
        self, 
        audio_data: bytes
    ) -> 'AsyncGenerator[bytes, None]':
        """
        Stream audio responses as sentences are generated.
        Achieves <500ms first-chunk latency through concurrent TTS.
        
        Pipeline:
        1. STT (blocking) → Get user text
        2. LLM Stream → Sentences to TTS Queue
        3. TTS Worker → Audio chunks to Output Queue
        4. Yield audio chunks as they're ready
        
        Args:
            audio_data: Raw audio bytes from user
            
        Yields:
            Audio chunks (MP3) for each sentence
        """
        await self.initialize()
        
        start_time = time.time()
        
        # Step 1: Speech to Text (blocking - unavoidable)
        user_text = await self.voice_pipeline.speech_to_text(audio_data)
        if not user_text:
            logger.warning("⚠️ STT returned empty result")
            return
        
        stt_time = time.time()
        logger.latency("STT", (stt_time - start_time) * 1000)
        
        # Create queues for concurrent processing
        sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        
        # Start TTS worker (processes sentences → audio concurrently)
        tts_task = asyncio.create_task(
            self._tts_worker(sentence_queue, audio_queue, start_time)
        )
        
        # Stream sentences from LLM → sentence queue
        sentence_count = 0
        async for sentence in self.conversation_handler.generate_response_stream(user_text):
            if sentence.strip():
                sentence_count += 1
                logger.info(f"📝 Sentence {sentence_count}: {sentence[:50]}...")
                await sentence_queue.put(sentence)
        
        # Signal end of sentences
        await sentence_queue.put(None)
        
        # Yield audio chunks as they're ready
        first_audio_time = None
        chunk_count = 0
        
        while True:
            audio_chunk = await audio_queue.get()
            if audio_chunk is None:
                break
            
            chunk_count += 1
            
            if first_audio_time is None:
                first_audio_time = time.time()
                logger.latency("First Audio Chunk", (first_audio_time - start_time) * 1000)
            
            yield audio_chunk
        
        # Wait for TTS worker to complete
        await tts_task
        
        total_time = time.time() - start_time
        logger.latency("Total Streaming", total_time * 1000)
        logger.info(f"✅ Streamed {chunk_count} audio chunks in {total_time:.2f}s")
    
    async def _tts_worker(
        self, 
        sentence_queue: asyncio.Queue,
        audio_queue: asyncio.Queue,
        start_time: float
    ):
        """
        TTS Worker: Process sentences to audio concurrently.
        Allows overlapping TTS with LLM streaming.
        """
        sentence_num = 0
        
        while True:
            sentence = await sentence_queue.get()
            
            if sentence is None:
                # Signal end of audio
                await audio_queue.put(None)
                break
            
            sentence_num += 1
            tts_start = time.time()
            
            # Convert sentence to audio
            audio = await self.voice_pipeline.text_to_speech(sentence)
            
            if audio:
                tts_elapsed = (time.time() - tts_start) * 1000
                logger.latency(f"TTS Sentence {sentence_num}", tts_elapsed)
                await audio_queue.put(audio)
            else:
                logger.warning(f"⚠️ TTS failed for sentence {sentence_num}")
    
    async def handle_text_message(self, text: str) -> str:
        """
        Handle text message (for testing without voice).
        
        Args:
            text: User's text message
            
        Returns:
            Assistant's text response
        """
        await self.initialize()
        return await self.conversation_handler.generate_response(text)
    
    def reset_session(self):
        """Reset for new conversation session."""
        if self.conversation_handler:
            self.conversation_handler.reset_conversation()


# Global agent instance
_agent: Optional[ScholarshipVoiceAgent] = None

def get_agent() -> ScholarshipVoiceAgent:
    """Get the global agent instance."""
    global _agent
    if _agent is None:
        _agent = ScholarshipVoiceAgent()
    return _agent


# LiveKit Agent Entry Point (when using livekit-agents framework)
if HAS_LIVEKIT_AGENTS:
    
    async def entrypoint(ctx: JobContext):
        """
        LiveKit Agents entrypoint.
        Called when a participant joins the room.
        """
        logger.info(f"🔗 Participant joined: {ctx.room.name}")
        
        # Initialize our agent
        agent = get_agent()
        await agent.initialize()
        
        # Create VAD for voice activity detection
        vad = silero.VAD.load()
        
        # For LiveKit Agents with custom STT/TTS, we need to implement
        # the full pipeline. This is a simplified version.
        
        @ctx.room.on("track_subscribed")
        async def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant
        ):
            """Handle incoming audio track."""
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            
            logger.info(f"🎤 Audio track from {participant.identity}")
            
            # Process audio frames
            audio_stream = rtc.AudioStream(track)
            
            # Buffer for collecting audio
            audio_buffer = bytearray()
            is_speaking = False
            silence_frames = 0
            
            async for frame in audio_stream:
                # Check VAD
                speech_probability = await vad.detect(frame)
                
                if speech_probability > 0.5:
                    is_speaking = True
                    silence_frames = 0
                    audio_buffer.extend(frame.data)
                elif is_speaking:
                    silence_frames += 1
                    audio_buffer.extend(frame.data)
                    
                    # End of speech (500ms silence ~ 25 frames at 20ms)
                    if silence_frames > 25:
                        logger.info(f"📝 Processing {len(audio_buffer)} bytes of audio")
                        
                        # Process the complete utterance
                        response_audio = await agent.process_audio(bytes(audio_buffer))
                        
                        if response_audio:
                            # Publish response audio
                            # Note: In full implementation, create AudioSource and publish
                            logger.info("🔊 Sending response audio")
                        
                        # Reset
                        audio_buffer = bytearray()
                        is_speaking = False
                        silence_frames = 0
        
        # Wait for disconnect
        await ctx.room.disconnected
        logger.info("🔌 Room disconnected")
        agent.reset_session()
    
    
    def run_livekit_agent():
        """Run as LiveKit Agent worker."""
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
            )
        )


# Simplified HTTP-based agent (fallback when LiveKit Agents not available)
async def run_simple_server():
    """
    Run a simple HTTP server for testing without LiveKit.
    Accepts audio via POST, returns audio response.
    """
    from aiohttp import web
    
    agent = get_agent()
    await agent.initialize()
    
    async def handle_audio(request: web.Request) -> web.Response:
        """Handle audio POST request (non-streaming, legacy)."""
        try:
            audio_data = await request.read()
            response_audio = await agent.process_audio(audio_data)
            
            if response_audio:
                return web.Response(
                    body=response_audio,
                    content_type="audio/mpeg"
                )
            else:
                return web.Response(status=500, text="Processing failed")
                
        except Exception as e:
            logger.error(f"❌ Request error: {e}")
            return web.Response(status=500, text=str(e))
    
    async def handle_audio_stream(request: web.Request) -> web.StreamResponse:
        """
        Handle audio POST with streaming response.
        Returns audio chunks as they're generated for <500ms first-chunk latency.
        """
        try:
            audio_data = await request.read()
            
            # Prepare streaming response
            response = web.StreamResponse(
                status=200,
                headers={
                    'Content-Type': 'audio/mpeg',
                    'Transfer-Encoding': 'chunked',
                    'Cache-Control': 'no-cache',
                }
            )
            await response.prepare(request)
            
            # Stream audio chunks as they're generated
            chunk_count = 0
            async for audio_chunk in agent.process_audio_stream(audio_data):
                chunk_count += 1
                await response.write(audio_chunk)
                logger.info(f"📡 Sent audio chunk {chunk_count}: {len(audio_chunk)} bytes")
            
            await response.write_eof()
            logger.info(f"✅ Streaming complete: {chunk_count} chunks sent")
            return response
            
        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            import traceback
            traceback.print_exc()
            return web.Response(status=500, text=str(e))
    
    async def handle_text(request: web.Request) -> web.Response:
        """Handle text POST request (for testing)."""
        try:
            data = await request.json()
            text = data.get("text", "")
            
            response = await agent.handle_text_message(text)
            
            return web.json_response({
                "response": response,
                "session_state": {
                    "category": agent.conversation_handler.state.preferred_category,
                    "state": agent.conversation_handler.state.preferred_state,
                    "course": agent.conversation_handler.state.preferred_course
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Request error: {e}")
            return web.Response(status=500, text=str(e))
    
    async def handle_reset(request: web.Request) -> web.Response:
        """Reset conversation session."""
        agent.reset_session()
        return web.json_response({"status": "reset"})
    
    async def handle_health(request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "scholarships_loaded": agent.rag.vectorstore.size if agent.rag else 0
        })
    
    # Create app
    app = web.Application()
    app.router.add_post("/audio", handle_audio)
    app.router.add_post("/audio/stream", handle_audio_stream)  # Low-latency streaming endpoint
    app.router.add_post("/text", handle_text)
    app.router.add_post("/reset", handle_reset)
    app.router.add_get("/health", handle_health)
    
    # Add Twilio phone call support if configured
    if config.twilio.is_configured():
        from telephony.twilio_handler import TwilioCallHandler, setup_twilio_routes
        twilio_handler = TwilioCallHandler(
            account_sid=config.twilio.account_sid,
            auth_token=config.twilio.auth_token,
            voice_agent=agent
        )
        setup_twilio_routes(app, twilio_handler)
        
        # Pre-generate greeting audio for instant playback
        async def pregenerate_on_startup(app):
            await twilio_handler.pregenerate_greeting()
        app.on_startup.append(pregenerate_on_startup)
        
        logger.info(f"📞 Twilio phone calls enabled: {config.twilio.phone_number}")
    else:
        logger.info("📞 Twilio not configured - phone calls disabled")
    
    # Serve frontend static files
    frontend_path = Path(__file__).parent.parent.parent / "frontend"
    if frontend_path.exists():
        app.router.add_static('/css/', frontend_path / 'css', name='css')
        app.router.add_static('/js/', frontend_path / 'js', name='js')
        
        async def handle_index(request):
            return web.FileResponse(frontend_path / 'index.html')
        app.router.add_get('/', handle_index)
        logger.info(f"📁 Serving frontend from {frontend_path}")
    
    # Enable CORS
    import aiohttp_cors
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(app.router.routes()):
        if not hasattr(route, 'resource') or '/css/' not in str(route.resource) and '/js/' not in str(route.resource):
            try:
                cors.add(route)
            except ValueError:
                pass  # Skip routes that don't support CORS
    
    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", config.port)
    
    logger.info(f"🌐 Simple server running on http://0.0.0.0:{config.port}")
    logger.info("📡 Endpoints: POST /audio, POST /text, POST /reset, GET /health")
    
    await site.start()
    
    # Keep running
    while True:
        await asyncio.sleep(3600)


def main():
    """Main entry point."""
    setup_logging(level=config.log_level)
    config.print_status()
    
    if HAS_LIVEKIT_AGENTS and config.livekit.is_configured():
        logger.info("🎙️ Starting LiveKit Agent mode...")
        run_livekit_agent()
    else:
        logger.info("🌐 Starting Simple HTTP Server mode...")
        asyncio.run(run_simple_server())


if __name__ == "__main__":
    main()
