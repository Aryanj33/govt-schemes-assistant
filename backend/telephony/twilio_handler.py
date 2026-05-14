"""
Twilio Phone Call Handler
Handles incoming phone calls and connects them to the voice pipeline.
"""

import asyncio
import time
import uuid
from typing import Optional, Dict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from aiohttp import web
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from utils.logger import get_logger
from utils.config import get_config

logger = get_logger()

# How long to wait for silence after user stops speaking (seconds).
# Lowered from '2' to '1' — gives a much snappier feel without truncating.
# Users very rarely pause more than 1s mid-sentence.
SPEECH_TIMEOUT = '1'
# Use 'experimental_conversations' for more accurate transcription on Hinglish.
SPEECH_MODEL = 'experimental_conversations'


class TwilioCallHandler:
    """Handles Twilio phone calls and streams audio to/from AI pipeline."""

    GREETING_TEXT = "Namaste! Main Vidya hoon. Bataiye, main aapki kaise madad kar sakti hoon?"

    def __init__(self, account_sid: str, auth_token: str, voice_agent, webhook_base_url: str = ""):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.client = Client(account_sid, auth_token)
        self.voice_agent = voice_agent
        self.base_url = webhook_base_url.rstrip("/")
        self.audio_cache: Dict[str, bytes] = {}
        self._greeting_audio_id: Optional[str] = None

    def _audio_url(self, audio_id: str) -> str:
        """Build absolute URL for Twilio <Play> tags."""
        return f"{self.base_url}/twilio/audio/{audio_id}"

    def _cache_audio(self, audio_bytes: bytes, prefix: str = "") -> str:
        """Store audio in cache and return its ID."""
        audio_id = f"{prefix}{uuid.uuid4()}" if not prefix else f"{prefix}-{uuid.uuid4().hex[:8]}"
        self.audio_cache[audio_id] = audio_bytes
        return audio_id

    # ── Startup ──────────────────────────────────────────────────────

    async def pregenerate_greeting(self):
        """Pre-generate greeting audio at startup for instant first-call playback."""
        try:
            logger.info("🔊 Pre-generating greeting audio...")
            audio_bytes = await self.voice_agent.voice_pipeline.text_to_speech(
                self.GREETING_TEXT, detected_language="hi"
            )
            if audio_bytes:
                audio_id = f"greeting-{uuid.uuid4().hex[:8]}"
                self.audio_cache[audio_id] = audio_bytes
                self._greeting_audio_id = audio_id
                logger.info(f"✅ Greeting audio cached ({len(audio_bytes)} bytes)")
        except Exception as e:
            logger.error(f"❌ Failed to pre-generate greeting: {e}")

    # ── Audio serving ────────────────────────────────────────────────

    async def serve_audio(self, request: web.Request) -> web.Response:
        """Serve cached audio files to Twilio."""
        audio_id = request.match_info.get('audio_id')
        audio_data = self.audio_cache.get(audio_id)

        if not audio_data:
            logger.warning(f"⚠️ Audio not found: {audio_id}")
            return web.Response(status=404)

        # Keep greeting audio (reused), delete one-time audio after serving
        if not audio_id.startswith("greeting-"):
            del self.audio_cache[audio_id]

        # Detect content type from audio header
        if audio_data[:3] in (b'\xff\xfb\x90', b'\xff\xfb\xa0') or audio_data[:2] == b'\xff\xfb' or audio_data[:3] == b'ID3':
            content_type = 'audio/mpeg'
        else:
            content_type = 'audio/wav'

        return web.Response(body=audio_data, content_type=content_type)

    # ── Call handlers ────────────────────────────────────────────────

    async def handle_incoming_call(self, request: web.Request) -> web.Response:
        """Handle incoming/outbound call — play greeting, start listening."""
        try:
            logger.info("📞 Incoming call received")

            # Reset conversation for new call
            self.voice_agent.conversation_handler.reset_conversation()

            response = VoiceResponse()

            # Gather with greeting audio
            gather = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout=SPEECH_TIMEOUT,
                speech_model=SPEECH_MODEL,
                bargeIn=True,
                enhanced=True,
            )

            if self._greeting_audio_id and self._greeting_audio_id in self.audio_cache:
                gather.play(self._audio_url(self._greeting_audio_id))
            else:
                gather.say(self.GREETING_TEXT, voice='Polly.Aditi', language='hi-IN')

            response.append(gather)

            # Fallback: if user doesn't speak after greeting, wait a bit more
            gather2 = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout=SPEECH_TIMEOUT,
                timeout=6,
            )
            gather2.say("Boliye, main sun rahi hoon.", voice='Polly.Aditi', language='hi-IN')
            response.append(gather2)

            # If still nothing, end call gracefully
            response.say("Koi input nahi mila. Phir kabhi call kijiye. Dhanyavaad!", voice='Polly.Aditi', language='hi-IN')
            response.hangup()

            return web.Response(text=str(response), content_type='text/xml')

        except Exception as e:
            import traceback; traceback.print_exc()
            logger.error(f"❌ Critical Error in handle_incoming_call: {e}")
            response = VoiceResponse()
            response.say("Maaf kijiye, system mein problem hai.", voice='Polly.Aditi', language='hi-IN')
            return web.Response(text=str(response), content_type='text/xml')

    async def process_speech(self, request: web.Request) -> web.Response:
        """
        Core turn handler: STT result → LLM → TTS → <Play> back to user.
        Falls back to Polly if TTS fails. Always keeps conversation alive.
        """
        try:
            data = await request.post()
            speech_result = data.get('SpeechResult', '')

            logger.info(f"🎤 User said: {speech_result}")

            if not speech_result:
                # No speech detected — just listen again
                resp = VoiceResponse()
                gather = Gather(
                    input='speech',
                    action='/twilio/process-speech',
                    language='hi-IN',
                    speech_timeout=SPEECH_TIMEOUT,
                    timeout=6,
                )
                gather.say("Maaf kijiye, sun nahi payi. Dobara boliye.", voice='Polly.Aditi', language='hi-IN')
                resp.append(gather)
                # If still no speech, end
                resp.say("Dhanyavaad! Phir kabhi call kijiye.", voice='Polly.Aditi', language='hi-IN')
                resp.hangup()
                return web.Response(text=str(resp), content_type='text/xml')

            overall_start = time.time()

            # ── LLM ──
            t_llm = time.time()
            response_text = await self.voice_agent.conversation_handler.generate_response(speech_result)
            llm_ms = (time.time() - t_llm) * 1000
            logger.info(f"⏱️ LLM: {llm_ms:.0f}ms | {len(response_text)} chars | {response_text[:80]}")

            final_text = response_text or "Maaf kijiye, kuch problem aa gayi."

            # ── TTS — use voice pipeline for natural-sounding audio ──
            t_tts = time.time()
            tts_audio = None
            try:
                tts_audio = await self.voice_agent.voice_pipeline.text_to_speech(
                    final_text, detected_language="hi"
                )
            except Exception as tts_err:
                logger.warning(f"⚠️ TTS failed, falling back to Polly: {tts_err}")

            tts_ms = (time.time() - t_tts) * 1000
            elapsed = time.time() - overall_start

            # ── Build TwiML ──
            response = VoiceResponse()
            gather = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout=SPEECH_TIMEOUT,
                bargeIn=True,
            )

            if tts_audio:
                audio_id = self._cache_audio(tts_audio, prefix="resp")
                gather.play(self._audio_url(audio_id))
                logger.info(f"⚡ Total: {elapsed:.2f}s (LLM {llm_ms:.0f}ms, TTS {tts_ms:.0f}ms) | Pipeline")
            else:
                gather.say(final_text, voice='Polly.Aditi', language='hi-IN')
                logger.info(f"⚡ Total: {elapsed:.2f}s (LLM {llm_ms:.0f}ms) | Polly fallback")

            response.append(gather)

            # Fallback Gather — if user didn't speak during audio, keep listening
            gather2 = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout=SPEECH_TIMEOUT,
                timeout=8,
            )
            response.append(gather2)

            # If still no speech after 2 Gathers, prompt once more then end
            response.say("Kya aap wahan hain?", voice='Polly.Aditi', language='hi-IN')
            gather3 = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout=SPEECH_TIMEOUT,
                timeout=5,
            )
            response.append(gather3)

            response.say("Dhanyavaad! Phir kabhi call kijiye.", voice='Polly.Aditi', language='hi-IN')
            response.hangup()

            return web.Response(text=str(response), content_type='text/xml')

        except Exception as e:
            logger.error(f"❌ Error processing speech: {e}")
            import traceback; traceback.print_exc()
            response = VoiceResponse()
            response.say("Maaf kijiye, problem aa gayi.", voice='Polly.Aditi', language='hi-IN')
            return web.Response(text=str(response), content_type='text/xml')

    async def handle_call_status(self, request: web.Request) -> web.Response:
        """Handle call status updates from Twilio."""
        try:
            data = await request.post()
            call_sid = data.get('CallSid')
            call_status = data.get('CallStatus')
            logger.info(f"📊 Call {call_sid} status: {call_status}")

            # Clean up audio cache on call end to prevent memory leak
            if call_status in ('completed', 'failed', 'no-answer', 'canceled'):
                # Keep only greeting audio
                to_delete = [k for k in self.audio_cache if not k.startswith("greeting-")]
                for k in to_delete:
                    del self.audio_cache[k]
                if to_delete:
                    logger.info(f"🧹 Cleaned {len(to_delete)} cached audio entries")

            return web.Response(text='OK')

        except Exception as e:
            logger.error(f"❌ Error handling call status: {e}")
            return web.Response(text='OK')

    async def initiate_outbound_call(self, request: web.Request) -> web.Response:
        """POST /twilio/call  { "phone": "+919876543210" }"""
        try:
            data = await request.json()
            to_number = data.get("phone", "").strip()
            if not to_number:
                return web.json_response({"error": "phone number is required"}, status=400)

            if not to_number.startswith("+"):
                to_number = "+" + to_number

            cfg = get_config()
            from_num = "+" + cfg.twilio.phone_number.lstrip("+")
            base_url = cfg.twilio.webhook_base_url.rstrip("/")

            if not base_url:
                return web.json_response(
                    {"error": "WEBHOOK_BASE_URL not set in .env"},
                    status=503,
                )

            call = self.client.calls.create(
                to=to_number,
                from_=from_num,
                url=f"{base_url}/twilio/voice",
                status_callback=f"{base_url}/twilio/status",
                status_callback_method="POST",
            )

            logger.info(f"📞 Outbound call initiated: {call.sid} → {to_number}")
            return web.json_response({"sid": call.sid, "status": call.status})

        except Exception as e:
            logger.error(f"❌ Failed to initiate outbound call: {e}")
            return web.json_response({"error": str(e)}, status=500)


def setup_twilio_routes(app: web.Application, call_handler: TwilioCallHandler):
    """Add Twilio webhook routes to the app."""
    app.router.add_post('/twilio/voice', call_handler.handle_incoming_call)
    app.router.add_post('/twilio/process-speech', call_handler.process_speech)
    app.router.add_post('/twilio/status', call_handler.handle_call_status)
    app.router.add_get('/twilio/audio/{audio_id}', call_handler.serve_audio)
    app.router.add_post('/twilio/call', call_handler.initiate_outbound_call)
    logger.info("📱 Twilio routes configured (Voice + Audio Serving + Outbound Calls)")
