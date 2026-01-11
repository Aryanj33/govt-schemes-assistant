"""
Twilio Phone Call Handler
Handles incoming phone calls and connects them to the voice pipeline.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional, Dict
from aiohttp import web
from twilio.twiml.voice_response import VoiceResponse, Gather, Say, Play
from twilio.rest import Client

logger = logging.getLogger(__name__)


class TwilioCallHandler:
    """Handles Twilio phone calls and streams audio to/from AI pipeline."""
    
    GREETING_TEXT = "Namaste! Main Vidya hoon. Bataiye, main aapki kaise madad kar sakti hoon?"
    
    def __init__(self, account_sid: str, auth_token: str, voice_agent):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.client = Client(account_sid, auth_token)
        self.voice_agent = voice_agent
        self.active_calls = {}  # Track active call sessions
        self.audio_cache: Dict[str, bytes] = {}  # Cache generated audio for playback
        self._greeting_audio_id: Optional[str] = None  # Pre-cached greeting
        
    async def pregenerate_greeting(self):
        """Pre-generate and cache greeting audio for instant playback."""
        try:
            logger.info("🔊 Pre-generating greeting audio...")
            audio_bytes = await self.voice_agent.voice_pipeline.text_to_speech(
                self.GREETING_TEXT, detected_language="hi"
            )
            if audio_bytes:
                audio_id = "greeting-" + str(uuid.uuid4())[:8]
                self.audio_cache[audio_id] = audio_bytes
                self._greeting_audio_id = audio_id
                logger.info(f"✅ Greeting audio cached ({len(audio_bytes)} bytes)")
        except Exception as e:
            logger.error(f"❌ Failed to pre-generate greeting: {e}")
        
    async def _generate_audio_response(self, text: str) -> VoiceResponse:
        """Generate TwiML with <Play> tag for ElevenLabs audio."""
        response = VoiceResponse()
        
        try:
            # Generate audio using the voice pipeline (uses ElevenLabs if configured)
            # Use 'hi' as hint language for proper Indian accent handling
            audio_bytes = await self.voice_agent.voice_pipeline.text_to_speech(text, detected_language="hi")
            
            if audio_bytes:
                # Store audio in cache
                audio_id = str(uuid.uuid4())
                self.audio_cache[audio_id] = audio_bytes
                
                # Schedule cleanup (optional, naive implementation)
                # In prod, use Redis or TTL cache
                
                # Use <Play> to stream our custom audio
                response.play(f'/twilio/audio/{audio_id}')
            else:
                # Fallback to Polly if generation fails
                logger.warning("⚠️ TTS generation failed, falling back to Polly")
                response.say(text, voice='Polly.Aditi', language='hi-IN')
                
        except Exception as e:
            logger.error(f"❌ Error generating audio response: {e}")
            response.say(text, voice='Polly.Aditi', language='hi-IN')
            
        return response

    async def serve_audio(self, request: web.Request) -> web.Response:
        """Serve cached audio files to Twilio."""
        audio_id = request.match_info.get('audio_id')
        audio_data = self.audio_cache.get(audio_id)
        
        if not audio_data:
            return web.Response(status=404)
            
        # Don't delete greeting audio (it's reused)
        if not audio_id.startswith("greeting-"):
            del self.audio_cache[audio_id]
        
        return web.Response(body=audio_data, content_type='audio/mpeg')

    async def handle_incoming_call(self, request: web.Request) -> web.Response:
        """
        Handle incoming phone call from Twilio.
        This is called when someone dials your Twilio number.
        """
        try:
            # Get call data from Twilio
            data = await request.post()
            call_sid = data.get('CallSid')
            from_number = data.get('From')
            
            logger.info(f"📞 Incoming call from {from_number} (SID: {call_sid})")
            
            response = VoiceResponse()
            
            # Use pre-cached greeting if available (instant playback)
            if self._greeting_audio_id and self._greeting_audio_id in self.audio_cache:
                response.play(f'/twilio/audio/{self._greeting_audio_id}')
                logger.info("⚡ Using cached greeting (0ms TTS)")
            else:
                # Fallback: generate on-the-fly
                response = await self._generate_audio_response(self.GREETING_TEXT)
            
            # Start gathering speech input
            gather = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout='auto',
                speech_model='phone_call'
            )
            response.append(gather)
            
            # If no input, say goodbye (Fallback)
            response.say(
                "Koi input nahi mila. Dhanyavaad!",
                voice='Polly.Aditi',
                language='hi-IN'
            )
            
            return web.Response(
                text=str(response),
                content_type='text/xml'
            )
            
        except Exception as e:
            logger.error(f"❌ Error handling incoming call: {e}")
            response = VoiceResponse()
            response.say("Sorry, technical problem ho gayi hai. Phir se try karein.")
            return web.Response(text=str(response), content_type='text/xml')
    
    async def process_speech(self, request: web.Request) -> web.Response:
        """
        Process speech input from user and generate AI response.
        """
        try:
            data = await request.post()
            speech_result = data.get('SpeechResult', '')
            call_sid = data.get('CallSid')
            
            logger.info(f"🎤 User said: {speech_result}")
            
            if not speech_result:
                response = await self._generate_audio_response("Aapka sawaal samajh nahi aaya. Kripya dobara boliye.")
                response.redirect('/twilio/voice')
                return web.Response(text=str(response), content_type='text/xml')
            
            # Get AI response using existing conversation handler
            ai_response = await self.voice_agent.conversation_handler.generate_response(
                speech_result
            )
            
            logger.info(f"🤖 AI response: {ai_response[:100]}...")
            
            # Create response with AI answer (ElevenLabs Audio)
            response = await self._generate_audio_response(ai_response)
            
            # Gather more input
            gather = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout='auto'
            )
            response.append(gather)
            
            # If no more input, goodbye
            response.say(
                "Dhanyavaad! Aapka din shubh ho.",
                voice='Polly.Aditi',
                language='hi-IN'
            )
            response.hangup()
            
            return web.Response(text=str(response), content_type='text/xml')
            
        except Exception as e:
            logger.error(f"❌ Error processing speech: {e}")
            response = VoiceResponse()
            response.say("Maaf kijiye, problem aa gayi.", voice='Polly.Aditi')
            return web.Response(text=str(response), content_type='text/xml')
    
    async def handle_call_status(self, request: web.Request) -> web.Response:
        """Handle call status updates (for logging)."""
        try:
            data = await request.post()
            call_sid = data.get('CallSid')
            call_status = data.get('CallStatus')
            
            logger.info(f"📊 Call {call_sid} status: {call_status}")
            
            if call_status == 'completed':
                # Clean up active calls
                if call_sid in self.active_calls:
                    del self.active_calls[call_sid]
            
            return web.Response(text='OK')
            
        except Exception as e:
            logger.error(f"❌ Error handling call status: {e}")
            return web.Response(text='OK')


def setup_twilio_routes(app: web.Application, call_handler: TwilioCallHandler):
    """Add Twilio webhook routes to the app."""
    app.router.add_post('/twilio/voice', call_handler.handle_incoming_call)
    app.router.add_post('/twilio/process-speech', call_handler.process_speech)
    app.router.add_post('/twilio/status', call_handler.handle_call_status)
    # New route for serving dynamic audio
    app.router.add_get('/twilio/audio/{audio_id}', call_handler.serve_audio)
    logger.info("📱 Twilio routes configured (Voice + Audio Serving)")
