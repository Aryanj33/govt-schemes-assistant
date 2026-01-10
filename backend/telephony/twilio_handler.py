"""
Twilio Phone Call Handler
Handles incoming phone calls and connects them to the voice pipeline.
"""

import asyncio
import json
import logging
from typing import Optional
from aiohttp import web
from twilio.twiml.voice_response import VoiceResponse, Gather, Say
from twilio.rest import Client

logger = logging.getLogger(__name__)


class TwilioCallHandler:
    """Handles Twilio phone calls and streams audio to/from AI pipeline."""
    
    def __init__(self, account_sid: str, auth_token: str, voice_agent):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.client = Client(account_sid, auth_token)
        self.voice_agent = voice_agent
        self.active_calls = {}  # Track active call sessions
        
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
            
            # Create TwiML response
            response = VoiceResponse()
            
            # Greet the caller
            response.say(
                "Namaste! Main Vidya. Kis cheez mein madad chahiye?",
                voice='Polly.Aditi',
                language='hi-IN'
            )
            
            # Start gathering speech input
            gather = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout='auto',
                speech_model='phone_call'
            )
            gather.say(
                "Aap apna sawaal puchiye.",
                voice='Polly.Aditi',
                language='hi-IN'
            )
            response.append(gather)
            
            # If no input, say goodbye
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
                response = VoiceResponse()
                response.say(
                    "Aapka sawaal samajh nahi aaya. Kripya dobara boliye.",
                    voice='Polly.Aditi',
                    language='hi-IN'
                )
                response.redirect('/twilio/voice')
                return web.Response(text=str(response), content_type='text/xml')
            
            # Get AI response using existing conversation handler
            ai_response = await self.voice_agent.conversation_handler.generate_response(
                speech_result
            )
            
            logger.info(f"🤖 AI response: {ai_response[:100]}...")
            
            # Create response with AI answer
            response = VoiceResponse()
            response.say(
                ai_response,
                voice='Polly.Aditi',
                language='hi-IN'
            )
            
            # Gather more input
            gather = Gather(
                input='speech',
                action='/twilio/process-speech',
                language='hi-IN',
                speech_timeout='auto'
            )
            # Don't say anything here, just listen for next input
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
            response.say(
                "Maaf kijiye, problem aa gayi. Kripya phir se try karein.",
                voice='Polly.Kajal',
                language='hi-IN'
            )
            return web.Response(text=str(response), content_type='text/xml')
    
    async def handle_call_status(self, request: web.Request) -> web.Response:
        """Handle call status updates (for logging)."""
        try:
            data = await request.post()
            call_sid = data.get('CallSid')
            call_status = data.get('CallStatus')
            
            logger.info(f"📊 Call {call_sid} status: {call_status}")
            
            if call_status == 'completed':
                # Clean up any session data
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
    logger.info("📱 Twilio routes configured")
