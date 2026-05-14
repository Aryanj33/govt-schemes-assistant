"""
Sarkari Mitra Backend Server
FastAPI server with all required endpoints for Frontend2
Optimized for voice call performance with connection pooling
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import os
import sys
import time
import json
import base64
import re
from pathlib import Path
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from dotenv import load_dotenv
import uuid
import asyncio
import httpx

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Import conversation handler
from agent.conversation_handler import get_conversation_handler
from rag.scholarship_rag import STATE_NAMES
from runtime_metrics import get_runtime_metrics
from session_store import get_session_store

metrics = get_runtime_metrics()
session_store = get_session_store()

# Initialize Twilio client
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
twilio_from_number = os.getenv("TWILIO_PHONE_NUMBER", "").replace("++", "+")
webhook_base_url = os.getenv("WEBHOOK_BASE_URL", "")

twilio_client = None
call_status_store: Dict[str, dict] = {}
call_sessions: Dict[str, dict] = {}  # Store call session data including language
phone_conversation_handlers: Dict[str, Any] = {}  # Store conversation handlers per phone number

# Audio cache for Twilio TTS — Twilio <Play> needs a URL, so we cache audio bytes and serve them
twilio_audio_cache: Dict[str, bytes] = {}
twilio_greeting_audio_id: Optional[str] = None

# Connection pool for better performance
http_client: Optional[httpx.AsyncClient] = None


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]

async def get_http_client() -> httpx.AsyncClient:
    """Get or create HTTP client with connection pooling."""
    global http_client
    if http_client is None or http_client.is_closed:
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
    return http_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    global twilio_client, http_client
    
    # ── Startup ──────────────────────────────────────────────────────
    # Initialize HTTP client
    await get_http_client()
    await session_store.initialize()
    
    # Initialize Twilio
    if twilio_account_sid and twilio_auth_token:
        try:
            twilio_client = Client(twilio_account_sid, twilio_auth_token)
            print(f"✅ Twilio initialized: {twilio_from_number}")
        except Exception as e:
            print(f"⚠️  Twilio initialization failed: {e}")
    else:
        print("⚠️  Twilio credentials not configured")
    
    # OPTIMIZATION: Preload conversation handler and RAG index
    print("📚 Preloading RAG index and conversation handler...")
    handler = get_conversation_handler("startup")
    await handler.initialize()
    
    # Force RAG index load and warm up
    if not handler.rag.is_ready:
        print("🔨 Building RAG index...")
        handler.rag.build_index()
    
    # Warm up with dummy query to ensure everything is loaded
    try:
        _ = await handler.rag.search_parallel("test scholarship", top_k=1, rerank=False)
        print("✅ RAG index preloaded and warmed up")
    except Exception as e:
        print(f"⚠️  RAG warmup failed: {e}")
    
    # Warm up the voice LLM endpoint so the first real call doesn't pay cold-start.
    if handler.groq_client:
        try:
            from utils.config import get_config as _get_cfg
            _cfg = _get_cfg()
            await handler.groq_client.chat.completions.create(
                model=_cfg.groq.voice_llm_model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0.0,
            )
            print(f"✅ Voice LLM warmed up ({_cfg.groq.voice_llm_model})")
        except Exception as e:
            print(f"⚠️  Voice LLM warmup failed: {e}")
    
    print("✅ Conversation handler pre-initialized")
    
    # Note: TTS pre-generation is skipped — _generate_tts_audio() returns
    # None by design (uses Polly.Aditi fallback for zero-latency Twilio calls).
    print("ℹ️  Using Polly.Aditi for Twilio greeting (fastest option)")
    
    yield  # ── Application runs here ─────────────────────────────────
    
    # ── Shutdown ─────────────────────────────────────────────────────
    if http_client:
        await http_client.aclose()
        print("✅ HTTP client closed")
    await session_store.close()

# Initialize FastAPI app with lifespan
app = FastAPI(title="Sarkari Mitra API", version="2.0.0", lifespan=lifespan)


# ── Twilio TTS helpers ────────────────────────────────────────────────

async def _generate_tts_audio(text: str) -> Optional[bytes]:
    """
    Generate TTS audio bytes.
    Skipping Cartesia/ElevenLabs for speed — returns None so Twilio uses
    its built-in Polly.Aditi voice (fastest, zero extra latency).
    """
    # Return None immediately → caller falls back to Polly.Aditi <Say>
    # which is rendered server-side by Twilio with zero network latency.
    print(f"⚡ Using fast Polly.Aditi for: {text[:60]}...")
    return None


def _cache_twilio_audio(audio_bytes: bytes, prefix: str = "resp") -> str:
    """Cache audio bytes and return an ID for Twilio <Play> URL."""
    audio_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    twilio_audio_cache[audio_id] = audio_bytes
    return audio_id


def _twilio_audio_url(audio_id: str) -> str:
    """Build absolute URL for Twilio <Play> tags."""
    base = webhook_base_url.rstrip("/")
    return f"{base}/twilio/audio/{audio_id}"


@app.get("/twilio/audio/{audio_id}")
async def serve_twilio_audio(audio_id: str):
    """Serve cached TTS audio to Twilio <Play> tags."""
    audio_data = twilio_audio_cache.get(audio_id)
    if not audio_data:
        raise HTTPException(status_code=404, detail="Audio not found")

    # Keep greeting audio (reused), delete one-time response audio after serving
    if not audio_id.startswith("greeting-"):
        twilio_audio_cache.pop(audio_id, None)

    # Detect content type
    content_type = "audio/mpeg" if audio_data[:3] == b"ID3" or audio_data[:2] == b"\xff\xfb" else "audio/wav"
    return Response(content=audio_data, media_type=content_type)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class SearchRequest(BaseModel):
    query: str = ""
    categories: Optional[List[str]] = None
    category: Optional[str] = None
    level: Optional[str] = None
    state: Optional[str] = None
    session_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)

class TextRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ResetRequest(BaseModel):
    session_id: Optional[str] = None

class CallRequest(BaseModel):
    phone_number: str
    language: Optional[str] = "en"


def _category_matches_scheme(requested_categories: List[str], scheme: Dict[str, Any]) -> bool:
    requested = [category.lower() for category in requested_categories]
    scheme_categories = scheme.get('categories') or scheme.get('category') or []
    if isinstance(scheme_categories, str):
        scheme_categories = [scheme_categories]

    searchable_text = " ".join([
        " ".join(str(item) for item in scheme_categories),
        " ".join(scheme.get('tags', [])) if isinstance(scheme.get('tags'), list) else str(scheme.get('tags', '')),
        str(scheme.get('name', '')),
        str(scheme.get('details', '')),
        str(scheme.get('eligibility', '')),
    ]).lower()

    alias_map = {
        "sc": ["scheduled caste", "sc", "dalit"],
        "st": ["scheduled tribe", "st", "tribal"],
        "obc": ["other backward", "obc", "backward class"],
        "women": ["women", "woman", "girl", "female", "mahila"],
        "farmers": ["farmer", "kisan", "agriculture"],
        "students": ["student", "scholarship", "education"],
        "divyang": ["divyang", "disabled", "disability", "pwd"],
        "general": ["general"],
        "msme": ["msme", "business", "enterprise", "entrepreneur"],
    }

    for category in requested:
        candidates = alias_map.get(category, [category])
        if any(candidate in searchable_text for candidate in candidates):
            return True
    return False


def _scheme_matches_state(requested_state: str, scheme: Dict[str, Any], strict: bool = False) -> bool:
    level = str(scheme.get("level", "")).lower()
    if level == "central":
        return True

    requested = requested_state.lower().strip()
    searchable_text = " ".join([
        str(scheme.get("name", "")),
        str(scheme.get("description", "")),
        str(scheme.get("details", "")),
        str(scheme.get("eligibility", "")),
        str(scheme.get("benefits", "")),
        str(scheme.get("documents", "")),
        " ".join(scheme.get("tags", [])) if isinstance(scheme.get("tags"), list) else str(scheme.get("tags", "")),
        " ".join(scheme.get("category", [])) if isinstance(scheme.get("category"), list) else str(scheme.get("category", "")),
        " ".join(scheme.get("categories", [])) if isinstance(scheme.get("categories"), list) else str(scheme.get("categories", "")),
    ]).lower()

    if requested in searchable_text:
        return True

    other_state_mentioned = any(
        state in searchable_text
        for state in STATE_NAMES
        if state != requested and len(state) > 4
    )
    if other_state_mentioned:
        return False

    return not strict

# Placeholder endpoints - will be implemented
@app.post("/search")
async def search_schemes(request: SearchRequest):
    """Search for government schemes with filters"""
    trace_id = new_trace_id()
    started_at = time.time()
    print(f"🔍 [{trace_id}] Search request: query='{request.query}', categories={request.categories}, level={request.level}, state={request.state}")
    
    try:
        # Get conversation handler for RAG access
        session_id = request.session_id or "search-anonymous"
        conversation_handler = get_conversation_handler(session_id)
        await conversation_handler.initialize()
        
        if not conversation_handler.rag.is_ready:
            conversation_handler.rag.build_index()
        
        # Build search query
        search_query = request.query or ""
        
        # Build filters
        filters = {}
        if request.state:
            filters['state'] = request.state
        requested_categories = request.categories or ([request.category] if request.category else None)
        result_limit = max(1, min(request.limit, 20))

        # Perform RAG search
        results = await conversation_handler.rag.search_parallel(
            search_query, 
            top_k=max(result_limit * 2, 10),  # Get more results for filtering
            filters=filters, 
            rerank=True
        )
        
        # Filter results by categories and level
        filtered_schemes = []
        legacy_results = []
        for scheme, score in results:
            if request.state and not _scheme_matches_state(request.state, scheme, strict=True):
                continue

            # Category filtering
            if requested_categories:
                if not _category_matches_scheme(requested_categories, scheme):
                    continue
            
            # Level filtering
            if request.level:
                scheme_level = scheme.get('level', '').lower()
                if request.level.lower() not in scheme_level:
                    continue
            
            # Format scheme data
            formatted_scheme = {
                "id": scheme.get('id', ''),
                "name": scheme.get('name', ''),
                "description": scheme.get('description') or scheme.get('details', ''),
                "details": scheme.get('details', ''),
                "level": scheme.get('level', ''),
                "categories": scheme.get('categories') or scheme.get('category') or [],
                "category": scheme.get('category') or scheme.get('categories') or [],
                "state": scheme.get('state', ''),
                "eligibility": scheme.get('eligibility', ''),
                "documents": scheme.get('documents', []),
                "applicationLink": scheme.get('applicationLink') or scheme.get('source', ''),
                "application_process": scheme.get('application_process', ''),
                "amount": scheme.get('benefits', scheme.get('award_amount', '')),
                "benefits": scheme.get('benefits', scheme.get('award_amount', '')),
                "tags": scheme.get('tags', []),
                "source": scheme.get('source', ''),
                "relevance": int(score * 100)  # Convert to percentage
            }
            
            filtered_schemes.append(formatted_scheme)
            legacy_results.append([formatted_scheme, score])

        # Limit results
        filtered_schemes = filtered_schemes[:result_limit]
        legacy_results = legacy_results[:result_limit]
        
        print(f"✅ [{trace_id}] Search complete: {len(filtered_schemes)} schemes found")
        metrics.record("search", (time.time() - started_at) * 1000, success=True)
        
        profile_summary = "No profile info yet"
        handler_state = getattr(conversation_handler, "state", None)
        if handler_state and hasattr(handler_state, "get_profile_summary"):
            profile_summary = handler_state.get_profile_summary()

        return {
            "schemes": filtered_schemes,
            "results": legacy_results,
            "total": len(filtered_schemes),
            "message": f"Found {len(filtered_schemes)} schemes matching your criteria",
            "session_id": session_id,
            "trace_id": trace_id,
            "profile_summary": profile_summary,
        }
        
    except Exception as e:
        print(f"❌ [{trace_id}] Search error: {e}")
        import traceback
        traceback.print_exc()
        metrics.record("search", (time.time() - started_at) * 1000, success=False)
        return {
            "schemes": [],
            "results": [],
            "total": 0,
            "message": f"Search failed: {str(e)}",
            "trace_id": trace_id,
        }

@app.post("/text")
async def text_chat(request: TextRequest):
    """Text chat with session management for browser interface - OPTIMIZED with streaming"""
    trace_id = new_trace_id()
    started_at = time.time()
    print(f"💬 [{trace_id}] Text chat request: {request.message[:50]}..., session: {request.session_id}")
    
    try:
        # Get session-specific conversation handler
        session_id = request.session_id or f"text-{int(time.time())}"
        conversation_handler = get_conversation_handler(session_id)
        await conversation_handler.initialize()
        
        # OPTIMIZATION: Use streaming for faster perceived response
        # Collect all chunks for non-streaming clients
        response_chunks = []
        async for chunk in conversation_handler.generate_response_stream(request.message, buffer_sentences=True):
            response_chunks.append(chunk)
        
        ai_response = " ".join(response_chunks)
        
        print(f"🤖 [{trace_id}] Text response generated: {ai_response[:100]}...")
        metrics.record("text", (time.time() - started_at) * 1000, success=True)
        
        return {
            "response": ai_response,
            "session_id": session_id,
            "schemes": conversation_handler.state.last_scholarships[:3],
            "profile_summary": conversation_handler.state.get_profile_summary(),
            "last_scheme_count": len(conversation_handler.state.last_scholarships),
            "trace_id": trace_id,
        }
        
    except Exception as e:
        print(f"❌ [{trace_id}] Text chat error: {e}")
        import traceback
        traceback.print_exc()
        metrics.record("text", (time.time() - started_at) * 1000, success=False)
        raise HTTPException(status_code=500, detail=f"Text chat failed: {str(e)}")

@app.post("/text/stream")
async def text_chat_stream(request: TextRequest):
    """STREAMING text chat for real-time responses (SSE)"""
    trace_id = new_trace_id()
    started_at = time.time()
    print(f"💬 [{trace_id}] Streaming text chat request: {request.message[:50]}..., session: {request.session_id}")
    
    async def generate_sse():
        try:
            # Get session-specific conversation handler
            session_id = request.session_id or f"text-{int(time.time())}"
            conversation_handler = get_conversation_handler(session_id)
            await conversation_handler.initialize()
            
            # Send session ID first
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'trace_id': trace_id})}\n\n"
            
            # Stream response chunks
            async for chunk in conversation_handler.generate_response_stream(request.message, buffer_sentences=True):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # Send completion signal
            metrics.record("text_stream", (time.time() - started_at) * 1000, success=True)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            print(f"❌ [{trace_id}] Streaming error: {e}")
            metrics.record("text_stream", (time.time() - started_at) * 1000, success=False)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate_sse(), media_type="text/event-stream")

@app.post("/audio")
async def process_audio(audio: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """Process uploaded audio for browser-based voice chat - FAST MODE"""
    trace_id = new_trace_id()
    started_at = time.time()
    print(f"🎤 [{trace_id}] Audio upload received: {audio.filename}, size: {audio.size}, session: {session_id}")
    
    try:
        # Read audio data
        audio_data = await audio.read()
        print(f"📊 Audio data size: {len(audio_data)} bytes")
        
        # Validate audio format
        if not audio.content_type or not any(fmt in audio.content_type.lower() for fmt in ['audio', 'webm', 'wav', 'mp3']):
            raise HTTPException(status_code=400, detail="Invalid audio format. Supported: WebM, WAV, MP3")
        
        # Get conversation handler
        conversation_handler = get_conversation_handler()
        
        # Use Groq Whisper for transcription
        if not conversation_handler.groq_client:
            await conversation_handler.initialize()
        
        if not conversation_handler.groq_client:
            raise HTTPException(status_code=503, detail="Speech recognition service not available")
        
        # Transcribe audio using Groq Whisper
        print("🎯 Starting transcription...")
        transcription_response = await conversation_handler.groq_client.audio.transcriptions.create(
            file=("audio.webm", audio_data, audio.content_type or "audio/webm"),
            model="whisper-large-v3-turbo",
            language="hi"  # Hindi/Hinglish
        )
        
        transcription = transcription_response.text.strip()
        print(f"📝 Transcription: {transcription}")
        
        if not transcription:
            return {
                "transcription": "",
                "response": "मुझे आपकी आवाज़ सुनाई नहीं दी। कृपया फिर से बोलें।",
                "session_id": session_id or "temp-session-id"
            }
        
        # Use session-based handler for stateful conversation with RAG
        if not session_id:
            session_id = f"voice-{int(time.time())}"
        conversation_handler = get_conversation_handler(session_id)
        await conversation_handler.initialize()
        
        # Generate AI response using full pipeline (with RAG + conversation history)
        print("⚡ Generating AI response (full pipeline)...")
        ai_response = await conversation_handler.generate_response(transcription, language="hinglish")
        
        print(f"✅ Audio processing complete")
        metrics.record("audio", (time.time() - started_at) * 1000, success=True)
        return {
            "transcription": transcription,
            "response": ai_response,
            "session_id": session_id,
            "profile_summary": conversation_handler.state.get_profile_summary(),
            "last_scheme_count": len(conversation_handler.state.last_scholarships),
            "trace_id": trace_id,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [{trace_id}] Audio processing error: {e}")
        import traceback
        traceback.print_exc()
        metrics.record("audio", (time.time() - started_at) * 1000, success=False)
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
    finally:
        # Clean up
        if hasattr(audio, 'file'):
            audio.file.close()

@app.post("/audio/stream")
async def stream_audio(audio: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """Stream audio response for browser-based voice chat - FAST MODE with premium TTS"""
    print(f"🎤 Audio stream request: {audio.filename}, session: {session_id}")
    
    try:
        # Read and transcribe audio (same as /audio endpoint)
        audio_data = await audio.read()
        
        # Validate audio format
        if not audio.content_type or not any(fmt in audio.content_type.lower() for fmt in ['audio', 'webm', 'wav', 'mp3']):
            raise HTTPException(status_code=400, detail="Invalid audio format. Supported: WebM, WAV, MP3")
        
        # Use session-based handler for stateful conversation with RAG
        if not session_id:
            session_id = f"stream-{int(time.time())}"
        conversation_handler = get_conversation_handler(session_id)
        await conversation_handler.initialize()
        
        if not conversation_handler.groq_client:
            raise HTTPException(status_code=503, detail="Speech recognition service not available")
        
        # Transcribe audio
        print("🎯 Transcribing audio for streaming...")
        transcription_response = await conversation_handler.groq_client.audio.transcriptions.create(
            file=("audio.webm", audio_data, audio.content_type or "audio/webm"),
            model="whisper-large-v3-turbo",
            language="hi"
        )
        
        transcription = transcription_response.text.strip()
        print(f"📝 Transcription: {transcription}")
        
        if not transcription:
            error_text = "Mujhe aapki awaaz sunaai nahi di. Kripya phir se bolein."
            return await generate_premium_tts_stream(error_text)
        
        # Generate AI response using full pipeline (with RAG + conversation history)
        print("⚡ Generating AI response for streaming (full pipeline)...")
        ai_response = await conversation_handler.generate_response(transcription, language="hinglish")
        
        # Stream premium TTS response
        return await generate_premium_tts_stream(ai_response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Audio streaming error: {e}")
        import traceback
        traceback.print_exc()
        # Return error audio stream
        error_text = "कुछ समस्या आई है। कृपया फिर से कोशिश करें।"
        return await generate_premium_tts_stream(error_text)
    finally:
        if hasattr(audio, 'file'):
            audio.file.close()

async def generate_voice_response(transcription: str, conversation_handler, user_language: str = "hindi") -> str:
    """Generate concise AI response for voice calls using VOICE_CALL_SYSTEM_PROMPT."""
    try:
        # Profile extraction is handled by conversation_handler.generate_response()
        # via the fast regex-based extract_profile_from_message() — no need for
        # a separate LLM call here (saves ~400ms per turn).
        
        # Add user message to conversation history BEFORE generating response
        conversation_handler.state.add_message("user", transcription)
        
        print(f"🤖 Generating voice response for: {transcription}")
        
        # Build the voice-optimized system prompt with profile + scheme context
        from config.prompts import VOICE_CALL_SYSTEM_PROMPT
        
        profile_summary = f"CURRENT USER PROFILE:\n{conversation_handler.state.get_profile_summary()}"
        
        # Get scheme context from RAG if enough profile info is available
        scheme_context = ""
        profile = conversation_handler.state.profile
        if profile.scheme_type or profile.state:
            try:
                rag = conversation_handler.rag
                if rag:
                    results = await rag.search(transcription, top_k=3, state=profile.state)
                    if results:
                        schemes = []
                        for r in results[:2]:  # Only top 2 for voice brevity
                            name = r.get("name", r.get("scheme_name", ""))
                            benefit = r.get("benefits", r.get("details", ""))[:150]
                            schemes.append(f"- {name}: {benefit}")
                        scheme_context = "AVAILABLE SCHEMES:\n" + "\n".join(schemes)
            except Exception as e:
                print(f"⚠️ RAG search failed: {e}")
        
        system_prompt = VOICE_CALL_SYSTEM_PROMPT.format(
            profile_summary=profile_summary,
            scheme_context=scheme_context if scheme_context else "No specific schemes found yet. Ask user for more details."
        )
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        history = conversation_handler.state.get_history_for_llm()
        if history:
            messages.extend(history)
        
        # Generate response — low max_tokens for concise voice output
        response = await conversation_handler.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=150,  # Voice: short responses only
            top_p=0.9
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Validate response
        if len(ai_response) < 5:
            print(f"❌ Response too short: {ai_response}")
            return "Main Vidya hoon. Aapka naam kya hai?"
        
        # Add AI response to conversation history
        conversation_handler.state.add_message("assistant", ai_response)
        
        print(f"✅ Voice response: {len(ai_response)} chars")
        return ai_response
        
    except Exception as e:
        print(f"❌ Voice response generation failed: {e}")
        return "Main Vidya hoon. Aapka naam kya hai?"

def get_fallback_response(user_language: str) -> str:
    """Get fallback response by language"""
    fallback_responses = {
        "hindi": "मैं विद्या हूँ। आपका नाम क्या है?",
        "english": "I'm Vidya. What is your name?",
        "hinglish": "Main Vidya hoon. Aapka naam kya hai?"
    }
    return fallback_responses.get(user_language, fallback_responses["english"])

async def update_profile_from_transcription(transcription: str, conversation_handler):
    """Extract profile info from user speech using a lightweight LLM call (fast, accurate)."""
    try:
        # Skip extraction for very short or uninformative utterances
        if len(transcription.strip()) < 3:
            return
        
        # Use a fast, small model for extraction — not the main 70B model
        extraction_prompt = f"""Extract any user profile information from this message. Return ONLY valid JSON.

Message: "{transcription}"

Extract these fields if mentioned (null if not mentioned):
- "name": user's name (string or null)
- "state": Indian state name in English (e.g. "Uttar Pradesh", "Maharashtra") or null
- "category": caste category (General/OBC/SC/ST) or null  
- "scheme_type": one of "scholarship", "kisan", "business", "women", "senior_citizen" or null
- "course": education course/field or null

Return JSON only, no explanation."""

        try:
            response = await conversation_handler.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Fast, cheap model for extraction
                messages=[
                    {"role": "system", "content": "You extract structured data from Hindi/Hinglish/English text. Return ONLY valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            import json
            extracted = json.loads(response.choices[0].message.content)
            
            profile = conversation_handler.state.profile
            updated_fields = []
            
            # Update name
            if extracted.get("name") and not profile.name:
                name = extracted["name"].strip()
                if len(name) >= 2 and is_valid_name(name):
                    profile.name = name
                    updated_fields.append(f"Name={name}")
                    conversation_handler.state.add_message("system", f"User's name is {name}")
            
            # Update state
            if extracted.get("state") and not profile.state:
                profile.state = extracted["state"]
                updated_fields.append(f"State={extracted['state']}")
                conversation_handler.state.add_message("system", f"User is from {extracted['state']}")
            
            # Update category
            if extracted.get("category") and not profile.category:
                profile.category = extracted["category"]
                updated_fields.append(f"Category={extracted['category']}")
            
            # Update scheme type
            if extracted.get("scheme_type") and not profile.scheme_type:
                profile.scheme_type = extracted["scheme_type"]
                updated_fields.append(f"SchemeType={extracted['scheme_type']}")
                conversation_handler.state.add_message("system", f"User wants {extracted['scheme_type']} information")
            
            # Update course
            if extracted.get("course") and not profile.course:
                profile.course = extracted["course"]
                updated_fields.append(f"Course={extracted['course']}")
                conversation_handler.state.add_message("system", f"User is studying {extracted['course']}")
            
            if updated_fields:
                print(f"✅ LLM profile extraction: {', '.join(updated_fields)}")
            else:
                print(f"ℹ️ No new profile info extracted from: {transcription[:50]}")
                
        except Exception as e:
            print(f"⚠️ LLM extraction failed, using regex fallback: {e}")
            # Fallback to basic regex for name only
            import re
            name_patterns = [
                r'(?:mera naam|my name is|i am|main)\s+([a-zA-Z]{2,20})',
                r'(?:मेरा नाम|नाम)\s+([a-zA-Z\u0900-\u097F]{2,20})',
            ]
            for pattern in name_patterns:
                match = re.search(pattern, transcription, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    if is_valid_name(name):
                        conversation_handler.state.profile.name = name
                        conversation_handler.state.add_message("system", f"User's name is {name}")
                        print(f"✅ Regex fallback: Name={name}")
                        break
        
    except Exception as e:
        print(f"❌ Profile update failed: {e}")

def is_valid_name(name: str) -> bool:
    """Validate if extracted text is actually a name"""
    if not name or len(name.strip()) < 2:
        return False
    
    name_clean = name.strip()
    
    # Allow Hindi/Devanagari names and English names with spaces
    if re.match(r'^[a-zA-Z\u0900-\u097F\s]+$', name_clean):
        # Check against common non-name words - EXPANDED LIST
        invalid_words = {
            'hello', 'hi', 'namaste', 'main', 'mera', 'hai', 'hoon', 
            'student', 'boy', 'girl', 'sir', 'madam', 'ji', 'scholarship',
            'scheme', 'madad', 'help', 'chahiye', 'vidya', 'government',
            'aap', 'kya', 'kaise', 'kahan', 'kab', 'kyun', 'kaun', 'yeh', 'woh',
            'मैं', 'मेरा', 'है', 'हूं', 'आप', 'क्या', 'कैसे', 'कहां', 'कब', 'क्यों', 'कौन',
            'स्कॉलरशिप', 'योजना', 'मदद', 'चाहिए', 'सरकार', 'हरियाणा', 'पंजाब', 'दिल्ली',
            'अनपढ़', 'गवार', 'पढ़ा', 'लिखा', 'नहीं', 'बिजनेस', 'लोन', 'चालू', 'करना',
            'मिशन', 'नाम', 'mission', 'name'  # Added these
        }
        
        name_lower = name_clean.lower()
        
        # Accept common Indian names (both Hindi and English)
        common_names = {
            'आर्यन', 'aryan', 'राहुल', 'rahul', 'प्रिया', 'priya', 'अमित', 'amit',
            'सुनीता', 'sunita', 'राज', 'raj', 'अनिल', 'anil', 'मीरा', 'meera',
            'विकास', 'vikas', 'पूजा', 'pooja', 'संजय', 'sanjay', 'रीता', 'rita',
            'मृदुल', 'mridul', 'राघव', 'raghav', 'अर्जुन', 'arjun', 'कविता', 'kavita',
            'आर्यन सिंह', 'aryan singh', 'राहुल शर्मा', 'rahul sharma', 'कृष्णा', 'krishna'
        }
        
        if name_lower in common_names:
            return True
        
        # Reject if entire name is a common word
        if name_lower in invalid_words:
            return False
            
        # Reject if name contains only common words
        words = name_lower.split()
        if all(word in invalid_words for word in words):
            return False
        
        # Additional validation: reject phrases that are clearly not names
        if any(phrase in name_lower for phrase in ['अनपढ़ गवार', 'पढ़ा लिखा नहीं', 'बिजनेस चालू', 'मेरा मिशन', 'मेरा नाम']):
            return False
            
        return True
    
    return False
async def generate_fast_response(transcription: str, conversation_handler) -> str:
    """Generate fast AI response without RAG for browser voice chat"""
    try:
        # Simple system prompt for fast responses
        fast_system_prompt = """Tu "Vidya" hai — ek friendly government scheme assistant.

RULES:
1. Tu SIRF Hinglish (Roman script) mein bolegi
2. Tu ek ladki hai — feminine words use kar
3. Chhote 2-3 sentences max — browser voice call hai
4. FAST responses — no long explanations

CONVERSATION:
- Agar user ne naam nahi bataya, puch: "Kripya apna naam bataiye"
- Agar user scholarship/scheme chahta hai, basic info puch: "Aap kis state se hain aur kya course kar rahe hain?"
- General helpful responses de

Keep it SHORT and FAST!"""

        messages = [
            {"role": "system", "content": fast_system_prompt},
            {"role": "user", "content": transcription}
        ]

        response = await conversation_handler.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use same model for consistency
            messages=messages,
            temperature=0.1,
            max_tokens=150,  # Short but complete for browser
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Fast response generation failed: {e}")
        return "Main Vidya hoon. Aap kaise madad chahiye?"

async def generate_premium_tts_stream(text: str):
    """Generate streaming TTS using best available service"""
    try:
        # Try Cartesia first (fastest and highest quality)
        cartesia_api_key = os.getenv("CARTESIA_API_KEY", "")
        if cartesia_api_key:
            print("🎙️ Using Cartesia TTS (premium)...")
            return await generate_cartesia_stream(text, cartesia_api_key)
        
        # Try ElevenLabs second (high quality)
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if elevenlabs_api_key:
            print("🔊 Using ElevenLabs TTS (premium)...")
            return await generate_elevenlabs_stream(text, elevenlabs_api_key)
        
        # Fallback to Edge TTS (free but good quality)
        print("🔊 Using Edge TTS (free fallback)...")
        return await generate_edge_tts_stream(text)
            
    except Exception as e:
        print(f"❌ Premium TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="Audio generation failed")

async def generate_cartesia_stream(text: str, api_key: str):
    """Generate streaming audio using Cartesia (fastest, best quality)"""
    import httpx
    
    # Use a good female Hindi voice
    voice_id = "694f9389-aac1-45b6-b726-9d9369183238"  # Default Cartesia voice
    
    url = "https://api.cartesia.ai/tts/bytes"
    headers = {
        "Cartesia-Version": "2024-06-10",
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "model_id": "sonic-3",  # Latest Sonic model
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": voice_id
        },
        "output_format": {
            "container": "mp3",
            "encoding": "mp3",
            "sample_rate": 22050
        },
        "language": "hi"  # Hindi
    }
    
    async def audio_stream():
        try:
            http_client = await get_http_client()
            async with http_client.stream("POST", url, headers=headers, json=data) as response:
                if response.status_code != 200:
                    print(f"❌ Cartesia error: {response.status_code}")
                    return
                
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        yield chunk
        except Exception as e:
            print(f"❌ Cartesia streaming error: {e}")
    
    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

async def generate_elevenlabs_stream(text: str, api_key: str):
    """Generate streaming audio using ElevenLabs with premium female voice"""
    import httpx
    
    # Use a premium female voice - Rachel (natural, clear female voice)
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",  # Fastest model
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
            "style": 0.2,
            "use_speaker_boost": True
        }
    }
    
    async def audio_stream():
        try:
            http_client = await get_http_client()
            async with http_client.stream("POST", url, headers=headers, json=data) as response:
                if response.status_code != 200:
                    print(f"❌ ElevenLabs error: {response.status_code}")
                    return
                
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        yield chunk
        except Exception as e:
            print(f"❌ ElevenLabs streaming error: {e}")
    
    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

async def generate_edge_tts_stream(text: str):
    """Generate streaming audio using Edge TTS with premium Hindi female voice"""
    try:
        import edge_tts
        import io
        
        # Use the best Hindi female voice - Swara Neural (natural, clear)
        voice = "hi-IN-SwaraNeural"
        
        # Generate audio with optimized settings
        communicate = edge_tts.Communicate(
            text, 
            voice,
            rate="+10%",  # Slightly faster for responsiveness
            pitch="+0Hz"
        )
        audio_data = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        audio_data.seek(0)
        
        def audio_stream():
            while True:
                chunk = audio_data.read(1024)
                if not chunk:
                    break
                yield chunk
        
        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except ImportError:
        print("⚠️ edge-tts not installed, falling back to text response")
        raise HTTPException(status_code=503, detail="TTS service not available")
    except Exception as e:
        print(f"❌ Edge TTS error: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed")

@app.post("/reset")
async def reset_session(request: ResetRequest):
    """Reset conversation session"""
    trace_id = new_trace_id()
    started_at = time.time()
    print(f"🔄 [{trace_id}] Reset request for session: {request.session_id}")
    
    try:
        # Get conversation handler and reset its state
        session_id = request.session_id or "default"
        conversation_handler = get_conversation_handler(session_id)
        await conversation_handler.initialize()
        await conversation_handler.reset_and_persist()
        
        print(f"✅ [{trace_id}] Session reset successfully")
        metrics.record("reset", (time.time() - started_at) * 1000, success=True)
        return {
            "status": "success",
            "message": "Conversation reset successfully",
            "trace_id": trace_id,
        }
        
    except Exception as e:
        print(f"❌ [{trace_id}] Reset error: {e}")
        metrics.record("reset", (time.time() - started_at) * 1000, success=False)
        return {
            "status": "error",
            "message": f"Reset failed: {str(e)}",
            "trace_id": trace_id,
        }

@app.post("/call")
async def initiate_call(request: CallRequest):
    """Initiate outbound call via Twilio"""
    print(f"📞 Call request received: {request.phone_number}, language: {request.language}")
    
    if not twilio_client:
        print("❌ Twilio client not initialized")
        raise HTTPException(status_code=503, detail="Twilio service not configured")
    
    try:
        # Validate and format phone number
        phone = request.phone_number.strip()
        if not phone.startswith("+"):
            phone = "+" + phone
        
        print(f"📱 Formatted phone: {phone}")
        
        # Validate Indian phone number format
        if not phone.startswith("+91") or len(phone) != 13:
            print(f"❌ Invalid phone format: {phone}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid phone number format. Expected: +91XXXXXXXXXX"
            )
        
        print(f"🔗 Webhook URL: {webhook_base_url}")
        print(f"📞 From number: {twilio_from_number}")
        
        # Create call via Twilio with webhook URL
        call = twilio_client.calls.create(
            to=phone,
            from_=twilio_from_number,
            url=f"{webhook_base_url}/twilio/voice"
        )
        
        # Store call info
        call_id = call.sid
        call_status_store[call_id] = {
            "call_id": call_id,
            "status": call.status,
            "to": phone,
            "from": twilio_from_number,
            "language": "hindi"  # Always Hindi
        }
        
        # Store call session data (always Hindi)
        call_sessions[phone] = {
            "call_id": call_id,
            "language": "hindi",  # Always Hindi
            "phone": phone
        }
        
        print(f"✅ Call created successfully: {call_id} → {phone}")
        print(f"   Status: {call.status}")
        
        return {
            "call_id": call_id,
            "status": call.status,
            "message": f"Call initiated to {phone}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Call initiation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initiate call: {str(e)}")

@app.post("/twilio/call")
async def initiate_call_legacy(request: Request):
    """Legacy call endpoint kept for older frontend pages."""
    payload = await request.json()
    phone_number = payload.get("phone_number") or payload.get("phone")
    language = payload.get("language") or "en"

    if not phone_number:
        raise HTTPException(status_code=400, detail="phone_number is required")

    return await initiate_call(CallRequest(phone_number=phone_number, language=language))

@app.get("/call/status/{call_id}")
async def get_call_status(call_id: str):
    """Get call status from Twilio"""
    if not twilio_client:
        raise HTTPException(status_code=503, detail="Twilio service not configured")
    
    try:
        # Try to get from Twilio
        call = twilio_client.calls(call_id).fetch()
        
        # Update local store
        call_status_store[call_id] = {
            "call_id": call_id,
            "status": call.status,
            "duration": call.duration or 0
        }
        
        return {
            "call_id": call_id,
            "status": call.status,
            "duration": call.duration or 0
        }
        
    except Exception as e:
        # Check local store
        if call_id in call_status_store:
            return call_status_store[call_id]
        
        raise HTTPException(status_code=404, detail=f"Call not found: {call_id}")


@app.get("/sessions/active")
async def list_active_sessions(limit: int = 25):
    """Operator endpoint to inspect active sessions."""
    sessions = await session_store.list_sessions(limit=limit)
    items = []
    for session in sessions:
        profile = session.get("profile", {}) if isinstance(session.get("profile"), dict) else {}
        last_schemes = session.get("last_scholarships", [])
        items.append({
            "session_id": session.get("session_id"),
            "updated_at": session.get("updated_at"),
            "turn_count": session.get("turn_count", 0),
            "message_count": len(session.get("messages", [])),
            "last_scheme_count": len(last_schemes) if isinstance(last_schemes, list) else 0,
            "profile": profile,
        })
    return {
        "sessions": items,
        "count": len(items),
        "backend": session_store.backend,
    }


@app.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Operator endpoint for a single session."""
    conversation_handler = get_conversation_handler(session_id)
    await conversation_handler.initialize()
    session_view = conversation_handler.get_public_session_view()
    session_view["store_backend"] = session_store.backend
    return session_view


@app.get("/metrics/summary")
async def metrics_summary():
    """Operator metrics for dashboard/demo use."""
    handler = get_conversation_handler("metrics-summary")
    await handler.initialize()
    summary = metrics.summary()
    summary["active_sessions"] = await session_store.get_active_count()
    summary["session_store_backend"] = session_store.backend
    summary["cache"] = handler.rag.get_cache_stats()
    summary["twilio_configured"] = twilio_client is not None
    return summary

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    active_sessions = await session_store.get_active_count()
    return {
        "status": "ok",
        "message": "Server is running",
        "twilio_configured": twilio_client is not None,
        "session_store_backend": session_store.backend,
        "active_sessions": active_sessions,
        "frontend2_available": frontend_path.exists(),
    }

# WebSocket endpoint for real-time voice communication
@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """Real-time voice communication with streaming TTS pipeline.
    
    Optimized pipeline:
    1. STT (Groq Whisper) — blocking, ~200-400ms
    2. LLM streaming (sentence by sentence) — overlapped with TTS
    3. TTS per sentence — audio sent as binary WS frames immediately
    4. Client plays audio chunks progressively — no waiting for full response
    """
    await websocket.accept()
    print("🎙️ WebSocket voice connection established")
    
    # Connection state
    session_id = websocket.query_params.get("session_id") or f"ws-voice-{int(time.time())}"
    conversation_handler = get_conversation_handler(session_id)
    await conversation_handler.initialize()
    
    # Heartbeat: send a server ping every 20s so intermediaries (ngrok, etc.)
    # don't close the connection during long idle periods.
    async def _heartbeat():
        try:
            while True:
                await asyncio.sleep(20)
                try:
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))
                except Exception:
                    return
        except asyncio.CancelledError:
            return
    
    heartbeat_task = asyncio.create_task(_heartbeat())
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio_chunk":
                await websocket.send_text(json.dumps({
                    "type": "audio_received",
                    "session_id": session_id
                }))
            elif message["type"] == "audio_end":
                await process_complete_audio_streaming(
                    websocket, message, conversation_handler, session_id
                )
            elif message["type"] == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message["type"] == "reset":
                await conversation_handler.reset_and_persist()
                await websocket.send_text(json.dumps({
                    "type": "reset_ok", "session_id": session_id
                }))
                
    except WebSocketDisconnect:
        print("🔌 WebSocket voice connection closed")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        heartbeat_task.cancel()


async def _generate_tts_for_sentence(text: str, user_language: str = "hindi") -> bytes:
    """Generate TTS audio for a single sentence. Returns raw audio bytes."""
    chunks = await generate_realtime_tts(text, user_language)
    if chunks:
        return b"".join(chunks)
    return b""


def _sniff_audio_format(data: bytes) -> tuple:
    """Detect audio container from magic bytes. Returns (mime, filename)."""
    if not data or len(data) < 4:
        return "audio/webm", "audio.webm"
    # WAV
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav", "audio.wav"
    # MP3 (ID3 or sync frame)
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return "audio/mpeg", "audio.mp3"
    # OGG / Opus
    if data[:4] == b"OggS":
        return "audio/ogg", "audio.ogg"
    # WebM / Matroska (EBML header 0x1A 0x45 0xDF 0xA3)
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm", "audio.webm"
    # MP4 / M4A
    if data[4:8] == b"ftyp":
        return "audio/mp4", "audio.m4a"
    # FLAC
    if data[:4] == b"fLaC":
        return "audio/flac", "audio.flac"
    # Default — assume webm/opus from MediaRecorder
    return "audio/webm", "audio.webm"


# ── Streaming-TTS phrase cache ───────────────────────────────────────────
_tts_phrase_cache: Dict[str, bytes] = {}
_tts_cache_max = 60


async def _generate_tts_for_sentence_cached(text: str, user_language: str = "hindi") -> bytes:
    """TTS with an in-memory cache keyed by text. Common greetings / follow-up
    prompts hit the cache and return in < 1ms instead of re-calling Cartesia."""
    key = f"{user_language}::{text.strip()}"
    cached = _tts_phrase_cache.get(key)
    if cached is not None:
        return cached
    audio = await _generate_tts_for_sentence(text, user_language)
    if audio and len(_tts_phrase_cache) < _tts_cache_max:
        # Only cache short, reusable phrases (greetings, follow-ups)
        if len(text) < 120:
            _tts_phrase_cache[key] = audio
    return audio


async def process_complete_audio_streaming(
    websocket: WebSocket,
    message: dict,
    conversation_handler,
    session_id: str,
):
    """Process audio with a streaming pipeline for minimum latency.
    
    Pipeline: STT → LLM stream (sentence by sentence) → TTS per sentence → binary WS frames
    Each sentence's audio is sent to the client as soon as it's ready,
    so the user hears the first words while the rest is still being generated.
    """
    pipeline_start = time.time()
    
    try:
        audio_base64 = message.get("audio_data", "")
        if not audio_base64:
            return
        
        audio_data = base64.b64decode(audio_base64)
        print(f"🎤 Processing audio: {len(audio_data)} bytes")
        
        # ── Phase 1: STT ─────────────────────────────────────────────
        await websocket.send_text(json.dumps({
            "type": "processing", "message": "Transcribing..."
        }))
        
        if not conversation_handler.groq_client:
            raise Exception("Speech recognition not available")
        
        # Sniff the audio format from magic bytes so Whisper knows what it's getting
        audio_mime, audio_filename = _sniff_audio_format(audio_data)
        
        # Context prompt primes Whisper with domain vocabulary — drastically
        # improves accuracy for Indian names, states, scheme names, and Hinglish.
        stt_prompt = (
            "Namaste, main Vidya hoon. Scholarship, PM Kisan, Mudra loan, AICTE, "
            "NSP, Pragati, INSPIRE, engineering, MBBS, Maharashtra, Uttar Pradesh, "
            "Karnataka, SC, ST, OBC, General, Minority. Mera naam Aryan hai. "
            "Main BTech kar raha hoon. Haryana se hoon. Scholarship chahiye."
        )
        
        stt_start = time.time()
        try:
            # Let Whisper auto-detect language — works for Hindi, English, Hinglish
            # (forcing language="hi" was causing English words to be mis-transcribed)
            transcription_response = await conversation_handler.groq_client.audio.transcriptions.create(
                file=(audio_filename, audio_data, audio_mime),
                model="whisper-large-v3-turbo",
                prompt=stt_prompt,
                temperature=0.0,
                response_format="text",
            )
            transcription = transcription_response.strip()
        except Exception as e1:
            print(f"⚠️ STT auto-detect failed ({e1}), retrying with Hindi hint...")
            try:
                transcription_response = await conversation_handler.groq_client.audio.transcriptions.create(
                    file=(audio_filename, audio_data, audio_mime),
                    model="whisper-large-v3-turbo",
                    language="hi",
                    prompt=stt_prompt,
                    temperature=0.0,
                    response_format="text",
                )
                transcription = transcription_response.strip()
            except Exception as e2:
                print(f"❌ STT failed twice: {e2}")
                transcription = ""
        
        stt_ms = (time.time() - stt_start) * 1000
        print(f"📝 STT ({stt_ms:.0f}ms): {transcription}")
        
        # Send transcription
        await websocket.send_text(json.dumps({
            "type": "transcription",
            "text": transcription,
            "session_id": session_id,
        }))
        
        if not transcription or len(transcription.strip()) < 2:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Could not understand audio. Please speak clearly.",
            }))
            return
        
        # ── Phase 2: Streaming LLM → TTS → binary WS ──────────────
        await websocket.send_text(json.dumps({
            "type": "processing", "message": "Generating response..."
        }))
        
        llm_start = time.time()
        full_response_text = ""
        sentence_count = 0
        first_audio_sent = False
        
        async for sentence in conversation_handler.generate_response_stream(
            transcription, buffer_sentences=True
        ):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            full_response_text += (" " if full_response_text else "") + sentence
            sentence_count += 1
            
            # Send partial text so the client can show it immediately
            if sentence_count == 1:
                llm_first_ms = (time.time() - llm_start) * 1000
                print(f"⚡ LLM first sentence ({llm_first_ms:.0f}ms): {sentence[:60]}")
                # Send the full text response (will be updated as more arrives)
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "text": full_response_text,
                    "session_id": session_id,
                    "partial": True,
                }))
            
            # Generate TTS for this sentence and send audio immediately (cached)
            tts_start = time.time()
            audio_bytes = await _generate_tts_for_sentence_cached(sentence, "hindi")
            tts_ms = (time.time() - tts_start) * 1000
            
            if audio_bytes:
                if not first_audio_sent:
                    first_audio_ms = (time.time() - pipeline_start) * 1000
                    print(f"🔊 First audio chunk at {first_audio_ms:.0f}ms (TTS: {tts_ms:.0f}ms)")
                    first_audio_sent = True
                else:
                    print(f"🔊 Sentence {sentence_count} TTS: {tts_ms:.0f}ms, {len(audio_bytes)} bytes")
                
                # Send audio as binary WebSocket frame (no base64 overhead)
                await websocket.send_bytes(audio_bytes)
            else:
                print(f"⚠️ TTS failed for sentence {sentence_count}: {sentence[:40]}")
        
        # Send final complete text response
        if full_response_text:
            await websocket.send_text(json.dumps({
                "type": "response",
                "text": full_response_text,
                "session_id": session_id,
                "partial": False,
            }))
        
        # Signal audio complete
        await websocket.send_text(json.dumps({
            "type": "audio_complete",
            "session_id": session_id,
        }))
        
        total_ms = (time.time() - pipeline_start) * 1000
        print(f"✅ Streaming pipeline complete: {total_ms:.0f}ms total, {sentence_count} sentences")
    
    except Exception as e:
        print(f"❌ Streaming pipeline error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Processing failed: {str(e)}",
            }))
        except Exception:
            pass

async def generate_realtime_tts(text: str, user_language: str = "english") -> List[bytes]:
    """Generate TTS audio chunks for real-time streaming with language support"""
    try:
        print(f"🔊 Generating TTS for: {text[:50]}... (Language: {user_language})")
        
        # Try Cartesia first (best for real-time)
        cartesia_api_key = os.getenv("CARTESIA_API_KEY", "")
        if cartesia_api_key:
            print("🎙️ Using Cartesia for real-time TTS...")
            chunks = await generate_cartesia_chunks(text, cartesia_api_key, user_language)
            if chunks:
                return chunks
            print("❌ Cartesia failed, trying ElevenLabs...")
        
        # Try ElevenLabs
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if elevenlabs_api_key:
            print("🔊 Using ElevenLabs for real-time TTS...")
            chunks = await generate_elevenlabs_chunks(text, elevenlabs_api_key, user_language)
            if chunks:
                return chunks
            print("❌ ElevenLabs failed, trying Edge TTS...")
        
        # Fallback to Edge TTS
        print("🔊 Using Edge TTS for real-time...")
        chunks = await generate_edge_tts_chunks(text, user_language)
        if chunks:
            return chunks
        
        print("❌ All TTS services failed")
        return []
        
    except Exception as e:
        print(f"❌ Real-time TTS failed: {e}")
        import traceback
        traceback.print_exc()
        return []

async def generate_cartesia_chunks(text: str, api_key: str, user_language: str = "english") -> List[bytes]:
    """Generate audio chunks using Cartesia with language support"""
    try:
        import httpx
        
        # Cartesia model selection:
        # - sonic-2 is multilingual (supports Hindi, Hinglish properly)
        # - sonic-english is English-only (was mis-pronouncing Hindi)
        voice_id = os.getenv("CARTESIA_VOICE_ID", "694f9389-aac1-45b6-b726-9d9369183238")
        voice_settings = {
            "hindi":    {"voice_id": voice_id, "model": "sonic-2", "language": "hi"},
            "english":  {"voice_id": voice_id, "model": "sonic-2", "language": "en"},
            "hinglish": {"voice_id": voice_id, "model": "sonic-2", "language": "hi"},
        }
        settings = voice_settings.get(user_language, voice_settings["hindi"])
        
        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "Cartesia-Version": "2024-11-13",
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }
        
        data = {
            "model_id": settings["model"],
            "transcript": text,
            "voice": {"mode": "id", "id": settings["voice_id"]},
            "output_format": {
                "container": "mp3",
                "encoding": "mp3",
                "sample_rate": 22050,
            },
            "language": settings["language"],
            "speed": "normal",
        }
        
        cartesia_start = time.time()
        chunks = []
        http_client = await get_http_client()
        
        async with http_client.stream("POST", url, headers=headers, json=data, timeout=20.0) as response:
            if response.status_code == 200:
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        chunks.append(chunk)
            else:
                error_text = await response.atext()
                print(f"❌ Cartesia error: {response.status_code} - {error_text[:200]}")
                return []
        
        cartesia_ms = (time.time() - cartesia_start) * 1000
        total_bytes = sum(len(c) for c in chunks)
        print(f"🎙️ Cartesia ({settings['model']}/{settings['language']}) {cartesia_ms:.0f}ms, {total_bytes}B, {len(text)}ch")
        return chunks
        
    except Exception as e:
        print(f"❌ Cartesia chunks error: {e}")
        import traceback
        traceback.print_exc()
        return []

async def generate_elevenlabs_chunks(text: str, api_key: str, user_language: str = "english") -> List[bytes]:
    """Generate audio chunks using ElevenLabs with language support"""
    try:
        import httpx
        
        # Use language-appropriate voice
        voice_settings = {
            "hindi": "21m00Tcm4TlvDq8ikWAM",  # Rachel (works well with Hindi)
            "english": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "hinglish": "21m00Tcm4TlvDq8ikWAM"  # Rachel
        }
        
        voice_id = voice_settings.get(user_language, voice_settings["english"])
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.8,
                "style": 0.3,
                "use_speaker_boost": True
            }
        }
        
        print(f"🔊 ElevenLabs request for: {text[:50]}... (Voice: {voice_id})")
        
        chunks = []
        http_client = await get_http_client()
        
        async with http_client.stream("POST", url, headers=headers, json=data, timeout=30.0) as response:
            print(f"🔊 ElevenLabs response status: {response.status_code}")
            
            if response.status_code == 200:
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        chunks.append(chunk)
            else:
                error_text = await response.atext()
                print(f"❌ ElevenLabs error: {response.status_code} - {error_text}")
                return []
        
        print(f"🔊 ElevenLabs generated {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"❌ ElevenLabs chunks error: {e}")
        import traceback
        traceback.print_exc()
        return []

async def generate_edge_tts_chunks(text: str, user_language: str = "english") -> List[bytes]:
    """Generate audio chunks using Edge TTS with language support"""
    try:
        import edge_tts
        
        # Use language-appropriate voices
        voice_settings = {
            "hindi": "hi-IN-SwaraNeural",  # Female Hindi voice
            "english": "en-US-AriaNeural",  # Female English voice
            "hinglish": "hi-IN-SwaraNeural"  # Use Hindi voice for mixed content
        }
        
        voice = voice_settings.get(user_language, voice_settings["english"])
        
        print(f"🔊 Edge TTS using voice: {voice} for language: {user_language}")
        
        communicate = edge_tts.Communicate(text, voice, rate="+10%", pitch="+0Hz")
        
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        
        print(f"🔊 Edge TTS generated {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"❌ Edge TTS chunks error: {e}")
        import traceback
        traceback.print_exc()
        return []

# Twilio webhook endpoints
@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls — uses Cartesia/ElevenLabs TTS."""
    print("📞 Twilio voice webhook called")
    
    # Parse form data to get caller info
    form_data = await request.form()
    
    # For OUTBOUND calls (Direction: 'outbound-api'), the fields are reversed:
    # - 'From' = Our Twilio service number  
    # - 'To' = The user's actual phone number
    direction = form_data.get("Direction", "")
    if direction == "outbound-api":
        caller_number = form_data.get("To", "")
    else:
        caller_number = form_data.get("From", "")
    
    print(f"📱 Direction: {direction} | User: {caller_number}")
    
    # Store call session data
    call_sessions[caller_number] = {
        "language": "hindi",
        "phone": caller_number
    }

    session_id = f"phone:{caller_number}"
    conversation_handler = get_conversation_handler(session_id)
    await conversation_handler.initialize()
    await conversation_handler.reset_and_persist()
    print(f"☎️ Prepared phone session {session_id}")
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Greet the user — use pre-generated Cartesia audio or fall back to Polly
    gather = Gather(
        input='speech',
        action='/twilio/process-speech',
        language='hi-IN',
        speech_timeout='2',
        speech_model='experimental_conversations',
        bargeIn=True,
        enhanced=True,
    )
    
    if twilio_greeting_audio_id and twilio_greeting_audio_id in twilio_audio_cache:
        # ✅ Use premium Cartesia/ElevenLabs voice
        gather.play(_twilio_audio_url(twilio_greeting_audio_id))
        print("🎙️ Using Cartesia greeting audio")
    else:
        # Fallback to Polly (only if TTS pre-generation failed)
        greeting = "Namaste! Main Vidya hoon, aapki government scheme assistant. Aapka naam kya hai?"
        gather.say(greeting, voice='Polly.Aditi', language='hi-IN')
        print("⚠️ Using Polly fallback for greeting")
    
    response.append(gather)
    
    # Fallback if no speech detected
    response.say("Koi awaaz nahi sunaai di. Kripya dobara call karein.", voice='Polly.Aditi', language='hi-IN')
    response.hangup()
    
    return Response(content=str(response), media_type="text/xml")

@app.post("/twilio/process-speech")
async def twilio_process_speech(request: Request):
    """Process speech from Twilio call — uses Polly.Aditi TTS with bargeIn on all gathers."""
    print("🎤 Twilio process speech webhook called")
    
    # Parse form data
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    
    # Determine caller number (inbound vs outbound)
    direction = form_data.get("Direction", "")
    if direction == "outbound-api":
        caller_number = form_data.get("To", "")
    else:
        caller_number = form_data.get("From", "")
    
    print(f"🎤 User said: {speech_result}")
    print(f"📱 Direction: {direction} | User: {caller_number}")
    
    response = VoiceResponse()
    
    # Common Gather settings — ensures bargeIn everywhere and consistent behavior
    def _make_gather(**overrides):
        defaults = dict(
            input='speech',
            action='/twilio/process-speech',
            language='hi-IN',
            speech_timeout='2',
            speech_model='experimental_conversations',
            bargeIn=True,
            enhanced=True,
        )
        defaults.update(overrides)
        return Gather(**defaults)
    
    if not speech_result:
        # No speech detected — prompt again with bargeIn
        gather = _make_gather(speech_timeout='3', timeout=8)
        gather.say("Mujhe aapki awaaz sunaai nahi di. Kripya phir se bolein.", voice='Polly.Aditi', language='hi-IN')
        response.append(gather)
        
        # Final fallback before hangup
        response.say("Dhanyavaad! Kripya dobara call karein.", voice='Polly.Aditi', language='hi-IN')
        response.hangup()
    else:
        # Process the speech with conversation handler
        try:
            session_id = f"phone:{caller_number}"
            conversation_handler = get_conversation_handler(session_id)
            await conversation_handler.initialize()
            print(f"♻️ Using phone session {session_id}")
            
            # Set language preference
            conversation_handler.state.profile.language = "hindi"
            
            # Generate grounded response through the shared conversation pipeline
            ai_response = await conversation_handler.generate_response(speech_result, language="hindi")
            
            # ── Truncate for voice: cap response to ~300 chars for natural TTS ──
            # Long text causes Polly to drone for 30+ seconds — user loses interest.
            # We keep the first 2 complete sentences and append a follow-up offer.
            voice_response = _truncate_for_voice(ai_response)
            
            print(f"🤖 AI Response ({len(ai_response)} chars, voice: {len(voice_response)} chars): {voice_response[:100]}...")
            
            # ── Primary Gather: speak response WITH bargeIn ──
            # User can interrupt at any point by speaking — Twilio will
            # stop playback and send their speech to /twilio/process-speech.
            gather = _make_gather()
            gather.say(voice_response, voice='Polly.Aditi', language='hi-IN')
            response.append(gather)
            
            # ── Secondary silent Gather: wait for user to respond ──
            # If user didn't barge in, give them 10 seconds of silence to speak.
            gather2 = _make_gather(speech_timeout='3', timeout=10)
            response.append(gather2)
            
            # ── "Are you there?" prompt with bargeIn ──
            gather3 = _make_gather(speech_timeout='3', timeout=8)
            gather3.say("Kya aap wahan hain? Kuch aur jaanna hai to boliye.", voice='Polly.Aditi', language='hi-IN')
            response.append(gather3)
            
            # Final goodbye
            response.say("Dhanyavaad! Kripya dobara call karein.", voice='Polly.Aditi', language='hi-IN')
            response.hangup()
            
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
            import traceback
            traceback.print_exc()
            
            # Error fallback — also with bargeIn
            gather = _make_gather(speech_timeout='3')
            gather.say("Kuch samasya aayi hai. Kya aap phir se bol sakte hain?", voice='Polly.Aditi', language='hi-IN')
            response.append(gather)
    
    return Response(content=str(response), media_type="text/xml")


def _truncate_for_voice(text: str, max_chars: int = 350) -> str:
    """Truncate a response for voice TTS — keep complete sentences, cap length.
    
    Voice TTS should never exceed ~350 chars (~25 seconds of speech).
    We keep complete sentences and add a follow-up prompt if truncated.
    """
    if len(text) <= max_chars:
        return text
    
    # Find sentence boundaries within the limit
    sentence_enders = '.!?।'
    best_cut = -1
    for i, ch in enumerate(text[:max_chars]):
        if ch in sentence_enders:
            best_cut = i + 1
    
    if best_cut > 50:  # Found a reasonable sentence boundary
        truncated = text[:best_cut].strip()
    else:
        # No sentence boundary found — cut at last space
        best_cut = text[:max_chars].rfind(' ')
        truncated = text[:best_cut].strip() if best_cut > 50 else text[:max_chars].strip()
        # Append ellipsis to indicate truncation
        truncated = truncated.rstrip('.,;:') + '.'
    
    # Add follow-up prompt so user knows there's more
    truncated += " Aur jaankari chahiye to boliye."
    return truncated

# Cache statistics endpoint
@app.get("/cache/stats")
async def get_cache_stats():
    """Get semantic cache statistics"""
    try:
        handler = get_conversation_handler("cache-stats")
        await handler.initialize()
        
        stats = handler.rag.get_cache_stats()
        
        return {
            "status": "success",
            "cache_stats": stats,
            "message": f"Cache hit rate: {stats['hit_rate']}"
        }
    except Exception as e:
        print(f"❌ Cache stats error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/cache/clear")
async def clear_cache():
    """Clear semantic cache"""
    try:
        handler = get_conversation_handler("cache-clear")
        await handler.initialize()
        
        handler.rag.clear_cache()
        
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        print(f"❌ Cache clear error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# Serve Frontend2 static files - ONLY for non-API routes
frontend_path = Path(__file__).parent.parent / "frontend2" / "vidya-sathi-ui" / "dist"
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_path / "assets")), name="assets")
    
    @app.get("/")
    async def serve_frontend_root():
        """Serve Frontend2 root"""
        return FileResponse(frontend_path / "index.html")
    
    # Only serve frontend for specific routes that are NOT API routes
    @app.get("/discover")
    async def serve_discover():
        return FileResponse(frontend_path / "index.html")
    
    @app.get("/hub")
    async def serve_hub():
        return FileResponse(frontend_path / "index.html")
    
    @app.get("/dashboard")
    async def serve_dashboard():
        return FileResponse(frontend_path / "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"🚀 Starting Sarkari Mitra Backend Server on port {port}")
    print(f"📁 Frontend2 path: {frontend_path}")
    print(f"✅ Frontend2 exists: {frontend_path.exists()}")
    uvicorn.run(app, host="0.0.0.0", port=port)
