"""
Scholarship Voice Assistant - Conversation Handler
===================================================
LLM integration with Groq Llama and Gemini fallback.
Manages conversation context and RAG integration.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config import get_config
from rag.scholarship_rag import get_scholarship_rag

# Add parent for config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.prompts import get_system_prompt_with_context, format_scholarships_for_context

logger = get_logger()
config = get_config()


class ConversationFlow(Enum):
    """States in the conversation flow."""
    GREETING = "greeting"
    ASKING_USER_TYPE = "asking_user_type"
    ASKING_STATE = "asking_state"
    ASKING_CATEGORY = "asking_category"  # For students
    ASKING_DETAILS = "asking_details"    # Course/Crop/Business type
    SEARCHING = "searching"
    RECOMMENDING = "recommending"

@dataclass
class ConversationMessage:
    """Single message in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserProfile:
    """User profile extracted from conversation."""
    user_type: Optional[str] = None  # student/farmer/business/other
    state: Optional[str] = None
    category: Optional[str] = None
    course: Optional[str] = None
    marks: Optional[float] = None
    income: Optional[int] = None
    education_level: Optional[str] = None
    gender: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def is_complete(self) -> bool:
        """Check if we have enough info to search."""
        return any([self.state, self.category, self.course])


@dataclass
class ConversationState:
    """Maintains conversation state and context."""
    session_id: str = field(default_factory=lambda: str(time.time()))
    messages: List[ConversationMessage] = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
    last_scholarships: List[Dict] = field(default_factory=list)
    conversation_flow: ConversationFlow = ConversationFlow.GREETING
    
    # Legacy compatibility
    @property
    def preferred_state(self): return self.profile.state
    @property
    def preferred_category(self): return self.profile.category
    @property
    def preferred_course(self): return self.profile.course
    @property
    def marks_percentage(self): return self.profile.marks
    
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append(ConversationMessage(role=role, content=content))
        # Keep only last 12 messages to avoid context overflow
        if len(self.messages) > 12:
            self.messages = self.messages[-12:]
    
    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """Format message history for LLM."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
    
    def get_profile_summary(self) -> str:
        """Get a summary of what we know about user."""
        parts = []
        if self.profile.state: parts.append(f"State: {self.profile.state}")
        if self.profile.category: parts.append(f"Category: {self.profile.category}")
        if self.profile.course: parts.append(f"Course: {self.profile.course}")
        if self.profile.marks: parts.append(f"Marks: {self.profile.marks}%")
        if self.profile.gender: parts.append(f"Gender: {self.profile.gender}")
        return ", ".join(parts) if parts else "No profile info yet"
    
    def extract_preferences(self, user_message: str):
        """
        Extract scholarship preferences from user message.
        Simple keyword-based extraction for hackathon.
        """
        message_lower = user_message.lower()
        
        # Extract state
        states = {
            "maharashtra": "Maharashtra", "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
            "karnataka": "Karnataka", "tamil nadu": "Tamil Nadu", "rajasthan": "Rajasthan",
            "bihar": "Bihar", "west bengal": "West Bengal", "gujarat": "Gujarat",
            "madhya pradesh": "Madhya Pradesh", "mp": "Madhya Pradesh",
            "kerala": "Kerala", "delhi": "Delhi", "telangana": "Telangana",
            "andhra pradesh": "Andhra Pradesh", "ap": "Andhra Pradesh",
            "punjab": "Punjab", "haryana": "Haryana", "odisha": "Odisha",
            "jharkhand": "Jharkhand", "chhattisgarh": "Chhattisgarh", "assam": "Assam",
            "himachal pradesh": "Himachal Pradesh", "hp": "Himachal Pradesh",
            "uttarakhand": "Uttarakhand", "uk": "Uttarakhand",
            "jammu": "Jammu", "kashmir": "Jammu"
        }
        
        for key, value in states.items():
            # Check for exact word match to avoid substrings (e.g., 'up' in 'support')
            if f" {key} " in f" {message_lower} " or message_lower.startswith(f"{key} ") or message_lower.endswith(f" {key}") or message_lower == key:
                self.profile.state = value
                break
        
        # Extract category
        categories = {
            "sc": "SC",
            "scheduled caste": "SC",
            "st": "ST",
            "scheduled tribe": "ST",
            "obc": "OBC",
            "other backward": "OBC",
            "general": "General",
            "minority": "Minority",
            "muslim": "Minority",
            "christian": "Minority",
            "sikh": "Minority"
        }
        for keyword, category in categories.items():
            if keyword in message_lower:
                self.profile.category = category
                break
        
        # Extract course type
        courses = {
            "engineering": "Engineering",
            "btech": "Engineering",
            "b.tech": "Engineering",
            "medical": "Medical",
            "mbbs": "Medical",
            "mba": "Management",
            "law": "Law",
            "science": "Science",
            "bsc": "Science",
            "arts": "Arts",
            "commerce": "Commerce"
        }
        for keyword, course in courses.items():
            if keyword in message_lower:
                self.profile.course = course
                break
        
        # Extract gender
        if any(w in message_lower for w in ["ladki", "girl", "female", "women"]):
            self.profile.gender = "Female"
        elif any(w in message_lower for w in ["ladka", "boy", "male"]):
            self.profile.gender = "Male"
        
        # Extract marks (simple pattern matching)
        import re
        marks_match = re.search(r'(\d{1,3})\s*%', message_lower)
        if marks_match:
            self.profile.marks = float(marks_match.group(1))


class ConversationHandler:
    """
    Handles LLM conversation with RAG integration.
    Uses Groq Llama with Gemini fallback.
    """
    
    def __init__(self):
        """Initialize the conversation handler."""
        self.groq_client = None
        self.gemini_model = None
        self.rag = get_scholarship_rag()
        self.state = ConversationState()
        self._initialized = False
    
    async def initialize(self):
        """Initialize LLM clients."""
        if self._initialized:
            return
        
        # Initialize Groq client
        if config.groq.is_configured():
            try:
                from groq import AsyncGroq
                self.groq_client = AsyncGroq(api_key=config.groq.api_key)
                logger.info("✅ Groq LLM client initialized")
            except ImportError:
                logger.warning("⚠️ groq package not installed")
        
        # Initialize Gemini as fallback
        if config.google.is_gemini_configured():
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.google.api_key)
                self.gemini_model = genai.GenerativeModel(config.google.gemini_model)
                logger.info("✅ Gemini LLM initialized as fallback")
            except ImportError:
                logger.warning("⚠️ google-generativeai package not installed")
        
        # Initialize RAG
        if not self.rag.is_ready:
            self.rag.build_index()
        
        self._initialized = True
    
    async def generate_response(
        self,
        user_message: str,
        stream: bool = False
    ) -> str:
        """
        Generate a response to user message.
        
        Args:
            user_message: User's transcribed speech
            stream: Whether to stream response (for faster TTS start)
            
        Returns:
            Assistant's response text
        """
        await self.initialize()
        
        start_time = time.time()
        
        # Extract preferences from message
        self.state.extract_preferences(user_message)
        
        # Check if this is a profile/conversational question (skip RAG for speed)
        is_profile_question = self._is_profile_question(user_message)
        
        if is_profile_question:
            # Skip RAG for profile questions - just use base prompt
            scholarship_context = "No specific schemes yet - gather user info first."
            logger.info("⚡ Skipping RAG for profile question")
        else:
            # Search for relevant scholarships
            filters = {}
            if self.state.preferred_category:
                filters['category'] = self.state.preferred_category
            if self.state.preferred_state:
                filters['state'] = self.state.preferred_state
            
            # Build search query combining user message with preferences
            search_query = user_message
            if self.state.preferred_course:
                search_query += f" {self.state.preferred_course}"
            if self.state.preferred_category:
                search_query += f" {self.state.preferred_category}"
            
            # Get relevant scholarships (use config top_k)
            results = self.rag.search(search_query, top_k=config.data.top_k_results, filters=filters)
            scholarship_context = format_scholarships_for_context([s for s, _ in results])
        
        # Build system prompt with context
        system_prompt = get_system_prompt_with_context(scholarship_context)
        
        # Add user message to history
        self.state.add_message("user", user_message)
        
        # Generate response using Groq or Gemini
        response = None
        
        if self.groq_client:
            response = await self._generate_groq(system_prompt)
        
        if response is None and self.gemini_model:
            response = await self._generate_gemini(system_prompt, user_message)
        
        if response is None:
            response = "Sorry, mujhe thoda problem aa gaya. Kya aap phir se bol sakte hain?"
            logger.error("❌ All LLM providers failed")
        
        # Add response to history
        self.state.add_message("assistant", response)
        
        elapsed = (time.time() - start_time) * 1000
        logger.latency("LLM Response", elapsed)
        logger.assistant_response(response[:100] + "..." if len(response) > 100 else response)
        
        return response
    
    async def _generate_groq(self, system_prompt: str) -> Optional[str]:
        """Generate response using Groq Llama."""
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ] + self.state.get_history_for_llm()
            
            logger.api_call("Groq", "Llama 3.3 70B")
            
            response = await self.groq_client.chat.completions.create(
                model=config.groq.llm_model,
                messages=messages,
                temperature=config.groq.llm_temperature,
                max_tokens=config.groq.llm_max_tokens,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error_with_context("Groq LLM", e)
            return None
    
    async def _generate_gemini(
        self, 
        system_prompt: str, 
        user_message: str
    ) -> Optional[str]:
        """Generate response using Gemini."""
        try:
            # Gemini uses a slightly different format
            # Combine system prompt with conversation
            full_prompt = f"{system_prompt}\n\n"
            
            for msg in self.state.messages:
                role = "User" if msg.role == "user" else "Assistant"
                full_prompt += f"{role}: {msg.content}\n"
            
            full_prompt += "Assistant:"
            
            logger.api_call("Google", "Gemini Flash")
            
            # Run in executor since genai is sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 256,
                    }
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error_with_context("Gemini LLM", e)
            return None
    
    async def generate_response_stream(
        self,
        user_message: str,
        buffer_sentences: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Generate streamed response with sentence buffering for TTS.
        
        Args:
            user_message: User's transcribed speech
            buffer_sentences: If True, yield complete sentences (better for TTS)
            
        Yields:
            Response chunks (sentences if buffer_sentences=True)
        """
        await self.initialize()
        start_time = time.time()
        
        # Extract preferences
        self.state.extract_preferences(user_message)
        
        # Build search query with full profile
        search_query = user_message
        if self.state.profile.course:
            search_query += f" {self.state.profile.course}"
        if self.state.profile.category:
            search_query += f" {self.state.profile.category}"
        
        # Build filters from profile
        filters = {}
        if self.state.profile.category:
            filters['category'] = self.state.profile.category
        if self.state.profile.state:
            filters['state'] = self.state.profile.state
        
        results = self.rag.search(search_query, top_k=5, filters=filters)
        self.state.last_scholarships = [s for s, _ in results]
        scholarship_context = format_scholarships_for_context(self.state.last_scholarships)
        
        # Add profile context to system prompt
        profile_info = self.state.get_profile_summary()
        system_prompt = get_system_prompt_with_context(scholarship_context)
        if profile_info != "No profile info yet":
            system_prompt += f"\n\nUser profile known so far: {profile_info}"
        
        self.state.add_message("user", user_message)
        
        # Stream from Groq with sentence buffering
        if self.groq_client:
            full_response = ""
            sentence_buffer = ""
            first_chunk_time = None
            
            try:
                messages = [
                    {"role": "system", "content": system_prompt}
                ] + self.state.get_history_for_llm()
                
                stream = await self.groq_client.chat.completions.create(
                    model=config.groq.llm_model,
                    messages=messages,
                    temperature=config.groq.llm_temperature,
                    max_tokens=config.groq.llm_max_tokens,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            logger.latency("First Token", (first_chunk_time - start_time) * 1000)
                        
                        if buffer_sentences:
                            sentence_buffer += content
                            
                            # FAST PATH: Yield short acknowledgments quickly (<50 chars)
                            # These are common conversational starters that should be spoken ASAP
                            if len(sentence_buffer) < 60:
                                lower_buffer = sentence_buffer.lower()
                                short_ack_phrases = [
                                    'ji haan', 'zaroor', 'bilkul', 'theek hai', 'haan ji',
                                    'main batata', 'main batati', 'dekhiye', 'suniye',
                                    'aapke liye', 'aap ke liye', 'hello', 'namaste'
                                ]
                                for phrase in short_ack_phrases:
                                    if phrase in lower_buffer:
                                        # Check if we have a natural break after the phrase
                                        if any(d in sentence_buffer for d in '.!?।,'):
                                            yield sentence_buffer.strip()
                                            sentence_buffer = ""
                                            break
                            
                            # NORMAL: Split on sentence delimiters
                            for delim in '.!?।':
                                if delim in sentence_buffer:
                                    # Split on first occurrence
                                    idx = sentence_buffer.index(delim)
                                    sentence = sentence_buffer[:idx + 1].strip()
                                    sentence_buffer = sentence_buffer[idx + 1:]
                                    if sentence:
                                        yield sentence
                                    break
                        else:
                            yield content
                
                # Yield remaining buffer
                if buffer_sentences and sentence_buffer.strip():
                    yield sentence_buffer.strip()
                
                self.state.add_message("assistant", full_response)
                logger.latency("Full Stream", (time.time() - start_time) * 1000)
                return
                
            except Exception as e:
                logger.error_with_context("Groq Stream", e)
        
        # Fallback to non-streaming
        response = await self.generate_response(user_message)
        yield response
    
    def _is_profile_question(self, user_message: str) -> bool:
        """
        Detect if message is a profile/conversational question that doesn't need RAG.
        
        Profile questions are simple responses to questions like:
        - "Kis state se ho?" -> "UP"
        - "Aap kaun hain?" -> "Student"
        - "Hello" / "Hi" / "Namaste"
        
        Returns:
            True if this is a profile question (skip RAG), False otherwise
        """
        msg_lower = user_message.lower().strip()
        
        # Very short messages (likely one-word answers)
        if len(msg_lower) <= 15:
            # Common profile answer patterns
            profile_patterns = [
                # States
                'up', 'delhi', 'punjab', 'bihar', 'mp', 'maharashtra', 'karnataka',
                # User types
                'student', 'kisan', 'farmer', 'business', 'businessman',
                # Categories
                'general', 'obc', 'sc', 'st', 'ews',
                # Courses (short forms)
                'btech', 'mtech', 'bsc', 'msc', 'ba', 'ma', 'bcom', 'mcom',
                # Greetings
                'hi', 'hello', 'namaste', 'haan', 'nahi', 'yes', 'no',  
            ]
            if any(pattern in msg_lower for pattern in profile_patterns):
                return True
        
        # If message is asking for scheme info (contains scheme keywords)
        scheme_keywords = ['scheme', 'yojana', 'scholarship', 'benefit', 'money', 'paisa', 'apply']
        if any(keyword in msg_lower for keyword in scheme_keywords):
            return False  # Need RAG for this
        
        # If we have no profile yet, first few messages are likely profile questions
        if not self.state.profile.state and not self.state.profile.user_type:
            return True
        
        return False
    
    def reset_conversation(self):
        """Reset conversation state for new session."""
        self.state = ConversationState()
        logger.info("🔄 Conversation reset")


# Singleton instance
_conversation_handler: Optional[ConversationHandler] = None

def get_conversation_handler() -> ConversationHandler:
    """Get the global conversation handler instance."""
    global _conversation_handler
    if _conversation_handler is None:
        _conversation_handler = ConversationHandler()
    return _conversation_handler
