"""
Government Schemes Assistant conversation handler.

This module owns:
- user profile extraction from messages
- session state and conversation history
- RAG triggering / search query building
- grounded, deterministic responses for common paths
- LLM fallback for anything more open-ended
"""

import asyncio
import json as _json
import re
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import get_logger
from rag.scholarship_rag import get_scholarship_rag
from session_store import get_session_store

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.prompts import get_system_prompt_with_context, format_scholarships_for_context

logger = get_logger()
config = get_config()


@dataclass
class ConversationMessage:
    """Single message in the conversation."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConversationMessage":
        return cls(
            role=str(payload.get("role", "assistant")),
            content=str(payload.get("content", "")),
            timestamp=float(payload.get("timestamp", time.time())),
        )


@dataclass
class UserProfile:
    """Profile fields extracted from conversation."""

    name: Optional[str] = None
    language: Optional[str] = None
    scheme_type: Optional[str] = None
    state: Optional[str] = None
    category: Optional[str] = None
    course: Optional[str] = None
    marks: Optional[float] = None
    income: Optional[int] = None
    gender: Optional[str] = None
    # Stores the citizen's original problem description (e.g. "meri fasal kharab hogyi")
    # so it persists across turns and enriches RAG queries.
    problem_description: Optional[str] = None
    # LLM-generated search terms for the problem (persists across turns)
    problem_search_query: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "UserProfile":
        if not isinstance(payload, dict):
            return cls()
        allowed = {
            "name", "language", "scheme_type", "state", "category",
            "course", "marks", "income", "gender",
            "problem_description", "problem_search_query",
        }
        return cls(**{key: payload.get(key) for key in allowed})


@dataclass
class ConversationState:
    """Conversation history plus extracted state."""

    session_id: str = field(default_factory=lambda: str(time.time()))
    messages: List[ConversationMessage] = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
    last_scholarships: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0

    @property
    def preferred_state(self):
        return self.profile.state

    @property
    def preferred_category(self):
        return self.profile.category

    @property
    def preferred_course(self):
        return self.profile.course

    def add_message(self, role: str, content: str):
        self.messages.append(ConversationMessage(role=role, content=content))
        if role == "user":
            self.turn_count += 1
        if len(self.messages) > config.session.history_limit:
            self.messages = self.messages[-config.session.history_limit:]

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def get_profile_summary(self) -> str:
        parts: List[str] = []
        if self.profile.name:
            parts.append(f"Name: {self.profile.name}")
        if self.profile.language:
            parts.append(f"Language: {self.profile.language}")
        if self.profile.state:
            parts.append(f"State: {self.profile.state}")
        if self.profile.category:
            parts.append(f"Category: {self.profile.category}")
        if self.profile.scheme_type:
            parts.append(f"Scheme type: {self.profile.scheme_type}")
        if self.profile.course:
            parts.append(f"Course: {self.profile.course}")
        if self.profile.marks is not None:
            parts.append(f"Marks: {self.profile.marks}%")
        if self.profile.income is not None:
            parts.append(f"Income: {self.profile.income}")
        if self.profile.gender:
            parts.append(f"Gender: {self.profile.gender}")
        return ", ".join(parts) if parts else "No profile info yet"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [message.to_dict() for message in self.messages],
            "profile": self.profile.to_dict(),
            "last_scholarships": self.last_scholarships,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]], session_id: str) -> "ConversationState":
        if not isinstance(payload, dict):
            return cls(session_id=session_id)
        state = cls(session_id=str(payload.get("session_id") or session_id))
        state.messages = [
            ConversationMessage.from_dict(item)
            for item in payload.get("messages", [])
            if isinstance(item, dict)
        ]
        state.profile = UserProfile.from_dict(payload.get("profile"))
        state.last_scholarships = payload.get("last_scholarships", []) if isinstance(payload.get("last_scholarships", []), list) else []
        state.turn_count = int(payload.get("turn_count", sum(1 for item in state.messages if item.role == "user")))
        return state


def _word_match(keyword: str, text: str) -> bool:
    """Whole-word-ish keyword matching with punctuation stripped."""

    cleaned = text.translate(str.maketrans("", "", string.punctuation + "।"))
    padded = f" {cleaned} "
    return f" {keyword} " in padded


STATES_MAP = {
    "maharashtra": "Maharashtra",
    "महाराष्ट्र": "Maharashtra",
    "uttar pradesh": "Uttar Pradesh",
    "उत्तर प्रदेश": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "यूपी": "Uttar Pradesh",
    "karnataka": "Karnataka",
    "कर्नाटक": "Karnataka",
    "tamil nadu": "Tamil Nadu",
    "तमिलनाडु": "Tamil Nadu",
    "rajasthan": "Rajasthan",
    "राजस्थान": "Rajasthan",
    "bihar": "Bihar",
    "बिहार": "Bihar",
    "west bengal": "West Bengal",
    "पश्चिम बंगाल": "West Bengal",
    "gujarat": "Gujarat",
    "गुजरात": "Gujarat",
    "madhya pradesh": "Madhya Pradesh",
    "मध्य प्रदेश": "Madhya Pradesh",
    "mp": "Madhya Pradesh",
    "एमपी": "Madhya Pradesh",
    "kerala": "Kerala",
    "केरल": "Kerala",
    "delhi": "Delhi",
    "दिल्ली": "Delhi",
    "telangana": "Telangana",
    "तेलंगाना": "Telangana",
    "andhra pradesh": "Andhra Pradesh",
    "आंध्र प्रदेश": "Andhra Pradesh",
    "ap": "Andhra Pradesh",
    "punjab": "Punjab",
    "पंजाब": "Punjab",
    "haryana": "Haryana",
    "हरियाणा": "Haryana",
    "odisha": "Odisha",
    "ओडिशा": "Odisha",
    "jharkhand": "Jharkhand",
    "झारखंड": "Jharkhand",
    "chhattisgarh": "Chhattisgarh",
    "छत्तीसगढ़": "Chhattisgarh",
    "assam": "Assam",
    "असम": "Assam",
    "himachal pradesh": "Himachal Pradesh",
    "हिमाचल प्रदेश": "Himachal Pradesh",
    "hp": "Himachal Pradesh",
    "uttarakhand": "Uttarakhand",
    "उत्तराखंड": "Uttarakhand",
    "goa": "Goa",
    "गोवा": "Goa",
    "jammu and kashmir": "Jammu and Kashmir",
    "जम्मू और कश्मीर": "Jammu and Kashmir",
    "jammu": "Jammu and Kashmir",
    "kashmir": "Jammu and Kashmir",
    "gujarat state": "Gujarat",
}

CATEGORIES_MAP: List[Tuple[str, str]] = [
    ("scheduled caste", "SC"),
    ("scheduled tribe", "ST"),
    ("other backward", "OBC"),
    ("obc", "OBC"),
    ("ओबीसी", "OBC"),
    ("sc", "SC"),
    ("एससी", "SC"),
    ("अनुसूचित जाति", "SC"),
    ("st", "ST"),
    ("एसटी", "ST"),
    ("अनुसूचित जनजाति", "ST"),
    ("general", "General"),
    ("जनरल", "General"),
    ("सामान्य", "General"),
    ("minority", "Minority"),
    ("अल्पसंख्यक", "Minority"),
    ("muslim", "Minority"),
    ("christian", "Minority"),
    ("sikh", "Minority"),
]

SCHEME_TYPES_MAP: List[Tuple[str, str]] = [
    ("scholarship", "scholarship"),
    ("student", "scholarship"),
    ("education", "scholarship"),
    ("padhai", "scholarship"),
    ("fees", "scholarship"),
    ("छात्रवृत्ति", "scholarship"),
    ("स्कॉलरशिप", "scholarship"),
    ("स्टूडेंट", "scholarship"),
    ("पढ़ाई", "scholarship"),
    ("शिक्षा", "scholarship"),
    ("kisan", "kisan"),
    ("farmer", "kisan"),
    ("farming", "kisan"),
    ("agriculture", "kisan"),
    ("fasal", "kisan"),
    ("crop", "kisan"),
    ("kheti", "kisan"),
    ("किसान", "kisan"),
    ("खेती", "kisan"),
    ("कृषि", "kisan"),
    ("फसल", "kisan"),
    ("फ़सल", "kisan"),
    ("बीमा", "kisan"),
    ("business", "business"),
    ("startup", "business"),
    ("loan", "business"),
    ("udyam", "business"),
    ("dukaan", "business"),
    ("बिजनेस", "business"),
    ("व्यापार", "business"),
    ("लोन", "business"),
    ("उद्यम", "business"),
    ("दुकान", "business"),
    ("pension", "pension"),
    ("senior citizen", "pension"),
    ("budhaapa", "pension"),
    ("पेंशन", "pension"),
    ("वृद्ध", "pension"),
    ("बुढ़ापा", "pension"),
    ("women", "women"),
    ("mahila", "women"),
    ("beti", "women"),
    ("shaadi", "women"),
    ("महिला", "women"),
    ("औरत", "women"),
    ("बेटी", "women"),
    ("शादी", "women"),
    ("health", "health"),
    ("hospital", "health"),
    ("bimaar", "health"),
    ("ilaaj", "health"),
    ("अस्पताल", "health"),
    ("बीमार", "health"),
    ("इलाज", "health"),
    ("दवाई", "health"),
    ("housing", "housing"),
    ("ghar", "housing"),
    ("makaan", "housing"),
    ("awas", "housing"),
    ("घर", "housing"),
    ("मकान", "housing"),
    ("आवास", "housing"),
]

COURSES_MAP: List[Tuple[str, str]] = [
    ("engineering", "Engineering"),
    ("btech", "Engineering"),
    ("b.tech", "Engineering"),
    ("इंजीनियरिंग", "Engineering"),
    ("बीटेक", "Engineering"),
    ("medical", "Medical"),
    ("mbbs", "Medical"),
    ("मेडिकल", "Medical"),
    ("एमबीबीएस", "Medical"),
    ("phd", "PhD"),
    ("पीएचडी", "PhD"),
    ("mba", "Management"),
    ("एमबीए", "Management"),
    ("law", "Law"),
    ("कानून", "Law"),
    ("science", "Science"),
    ("bsc", "Science"),
    ("साइंस", "Science"),
    ("विज्ञान", "Science"),
    ("बीएससी", "Science"),
    ("arts", "Arts"),
    ("कला", "Arts"),
    ("आर्ट्स", "Arts"),
    ("commerce", "Commerce"),
    ("कॉमर्स", "Commerce"),
    ("वाणिज्य", "Commerce"),
    ("बीए", "Arts"),
    ("बीकॉम", "Commerce"),
    ("नर्सिंग", "Medical"),
    ("आईटीआई", "Engineering"),
    ("पॉलिटेक्निक", "Engineering"),
    ("polytechnic", "Engineering"),
]

NAME_PATTERNS = [
    r"mera naam\s+([a-zA-Z\u0900-\u097F]{2,20}(?:\s[a-zA-Z\u0900-\u097F]{2,20})?)\s+hai",
    r"my name is\s+([a-zA-Z\u0900-\u097F]{2,20}(?:\s[a-zA-Z\u0900-\u097F]{2,20})?)",
    r"i am\s+(?!from\b|looking\b|need\b|interested\b|studying\b)([a-zA-Z\u0900-\u097F]{2,20}(?:\s[a-zA-Z\u0900-\u097F]{2,20})?)(?:\s+(?:from|and|$))",
    # Hindi Devanagari patterns (Whisper outputs)
    r"(?:मेरा\s+नाम|नाम)\s+([\u0900-\u097F]{2,20}(?:\s[\u0900-\u097F]{2,20})?)\s+है",
    r"(?:जी\s+)?(?:मेरा\s+नाम|नाम)\s+([\u0900-\u097F]{2,20}(?:\s[\u0900-\u097F]{2,20})?)",
    r"(?:मैं|मेरा नाम)\s+([\u0900-\u097F]{2,20}(?:\s[\u0900-\u097F]{2,20})?)\s+(?:हूं|हूँ|हैं|बोल रहा|बोल रही)",
]

INVALID_NAME_WORDS = {
    "scholarship", "student", "scheme", "help", "madad", "chahiye", "loan",
    "business", "mission", "name", "mera", "main", "kisan", "farmer", "apply",
    "from", "maharashtra", "karnataka", "uttar", "pradesh", "need", "looking",
    # Hindi words that are NOT names
    "स्कॉलरशिप", "स्टूडेंट", "योजना", "मदद", "चाहिए", "लोन", "बिजनेस",
    "किसान", "मेरा", "मैं", "नाम", "है", "हूं", "हूँ", "जी", "हां", "नहीं",
    "स्कीम", "सरकार", "सरकारी", "पेंशन", "महिला",
}

DIRECT_LOOKUP_TERMS = [
    "pm-kisan", "pm kisan", "mudra", "pragati", "inspire", "nsp",
    "pmmsy", "aicte", "ayushman", "post matric", "kisan samman nidhi",
    # Hindi variants
    "पीएम किसान", "मुद्रा", "प्रगति", "इंस्पायर", "आयुष्मान",
    "पोस्ट मैट्रिक", "किसान सम्मान निधि",
]

DIRECT_SCHEME_ALIASES = {
    "pm-kisan": [
        "pm-kisan",
        "pm kisan",
        "kisan samman nidhi",
        "pradhan mantri kisan samman nidhi",
    ],
    "mudra": [
        "mudra",
        "pradhan mantri mudra yojana",
        "pmmy",
    ],
    "pragati": [
        "pragati",
        "pragati scholarship",
        "aicte pragati",
    ],
    "inspire": [
        "inspire",
        "inspire scholarship",
    ],
    "ayushman": [
        "ayushman",
        "ayushman bharat",
        "pmjay",
    ],
    "post matric": [
        "post matric",
        "post-matric",
    ],
}

FOLLOW_UP_MARKERS = [
    "first", "second", "third", "1st", "2nd", "3rd", "eligibility", "documents",
    "apply", "application", "deadline", "benefit", "amount", "how to apply",
    "iske", "uski", "inmein", "isme", "tell me more", "more about", "details",
    # Hindi follow-up markers
    "पहला", "पहली", "दूसरा", "दूसरी", "तीसरा", "तीसरी",
    "पात्रता", "दस्तावेज़", "दस्तावेज", "आवेदन", "फायदा", "राशि",
    "इसकी", "इसके", "उसकी", "उसके", "और बताओ", "और बताइए", "विवरण",
]


def _coerce_text(value: Any) -> str:
    """Convert nested scheme fields into compact plain text."""

    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            label = key.replace("_", " ").strip().title()
            parts.append(f"{label}: {item}")
        return "; ".join(parts)
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return re.sub(r"\s+", " ", str(value)).strip()


def _short_text(value: Any, limit: int = 160) -> str:
    """Trim long fields for conversational replies and clean for voice."""
    from utils.voice_normalizer import normalize_for_voice

    text = _coerce_text(value)
    if not text:
        return ""
    # Clean for voice first (removes URLs, special chars, normalizes numbers)
    text = normalize_for_voice(text)
    if len(text) <= limit:
        return text
    truncated = text[:limit].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def _extract_income_value(message: str) -> Optional[int]:
    """Parse income expressions like '1.5 lakh' or 'income 120000'."""

    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|लाख)", message)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)

    thousand_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:thousand|hazaar|हज़ार|हजार)", message)
    if thousand_match:
        return int(float(thousand_match.group(1)) * 1000)

    rupee_match = re.search(r"(?:family income|annual income|income|aamdani|आमदनी|आय|इनकम)\D{0,12}(\d{4,8})", message)
    if rupee_match:
        return int(rupee_match.group(1))

    return None


def _extract_name(message: str) -> Optional[str]:
    """Tight name extraction with rejection of generic words."""

    for pattern in NAME_PATTERNS:
        match = re.search(pattern, message, re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip()
        lowered = candidate.lower()
        if (
            2 <= len(candidate) <= 25
            and not candidate.isdigit()
            and not any(word in lowered for word in INVALID_NAME_WORDS)
            and not lowered.startswith("from ")
        ):
            return candidate
    return None


_GREETING_WORDS = frozenset([
    "hi", "hello", "namaste", "namaskar", "haan", "ha", "yes", "no", "nahi",
    "ok", "okay", "theek", "thik", "accha", "acha", "ji", "shukriya",
    "dhanyavaad", "thank", "thanks", "bye", "alvida",
    "नमस्ते", "नमस्कार", "हां", "हाँ", "नहीं", "ठीक", "अच्छा", "शुक्रिया", "धन्यवाद",
])


def _is_just_name_or_greeting(msg: str) -> bool:
    """Return True if the message is ONLY a greeting, yes/no, or just a name.

    Important: long messages (>3 words) are NEVER just a name/greeting,
    even if they contain a name. 'Mera naam Aryan hai aur meri fasal
    kharab hogyi' should NOT be classified as a greeting.
    """
    lowered = msg.lower().strip()
    words = lowered.split()
    # Long messages are never just greetings
    if len(words) > 3:
        return False
    if all(w in _GREETING_WORDS for w in words):
        return True
    if _extract_name(lowered):
        return True
    return False


def extract_profile_from_message(profile: UserProfile, user_message: str):
    """
    Extract profile fields from a user message.

    Overwrites fields on explicit corrections instead of freezing the first value.
    """

    msg = user_message.lower().strip()

    for key, value in sorted(STATES_MAP.items(), key=lambda item: -len(item[0])):
        if _word_match(key, msg):
            profile.state = value
            break

    for keyword, category in CATEGORIES_MAP:
        if _word_match(keyword, msg):
            profile.category = category
            break

    for keyword, scheme_type in SCHEME_TYPES_MAP:
        if keyword in msg:
            profile.scheme_type = scheme_type
            break

    for keyword, course in COURSES_MAP:
        if keyword in msg:
            profile.course = course
            break

    if any(_word_match(word, msg) for word in ["ladki", "girl", "female", "women"]):
        profile.gender = "Female"
    elif any(_word_match(word, msg) for word in ["ladka", "boy", "male"]):
        profile.gender = "Male"

    extracted_name = _extract_name(msg)
    if extracted_name:
        profile.name = extracted_name

    # \b after % fails because both '%' and a following space are non-word
    # characters — no word boundary exists between them. Use a lookahead
    # that accepts %, end-of-string, or a non-digit instead.
    marks_match = re.search(
        r"\b(\d{1,3})\s*(?:%|percent(?:age)?)(?=\s|$|[^\w%])",
        msg,
    )
    if marks_match:
        marks = float(marks_match.group(1))
        if 0 <= marks <= 100:
            profile.marks = marks

    income_value = _extract_income_value(msg)
    if income_value:
        profile.income = income_value


class ConversationHandler:
    """Session-scoped conversation controller."""

    def __init__(self, session_id: str = "default"):
        self.groq_client = None
        self.gemini_model = None
        self.rag = get_scholarship_rag()
        self.session_id = session_id
        self.state = ConversationState(session_id=session_id)
        self._initialized = False
        self._state_loaded = False
        # LLM intent resolver state (per-turn, cleared after use)
        self._intent_search_query: str = ""

    async def initialize(self):
        """Lazy init external clients plus the RAG index."""

        if self._initialized:
            return

        if config.groq.is_configured():
            try:
                from groq import AsyncGroq

                self.groq_client = AsyncGroq(api_key=config.groq.api_key)
                logger.info("Groq LLM client initialized")
            except ImportError:
                logger.warning("groq package not installed")

        if config.google.is_gemini_configured():
            try:
                import google.generativeai as genai

                genai.configure(api_key=config.google.api_key)
                self.gemini_model = genai.GenerativeModel(config.google.gemini_model)
                logger.info("Gemini fallback initialized")
            except ImportError:
                logger.warning("google-generativeai package not installed")

        if not self.rag.is_ready:
            self.rag.build_index()

        await self._load_persisted_state()

        self._initialized = True

    async def _load_persisted_state(self):
        if self._state_loaded:
            return
        # If another (recently-evicted) handler for this session has a
        # pending persist task, wait for it to finish first so we don't
        # race past it and load stale data.
        pending = _pending_persists.get(self.session_id)
        if pending is not None and not pending.done():
            try:
                await pending
            except Exception:
                pass
        store = get_session_store()
        payload = await store.load_session(self.session_id)
        self.state = ConversationState.from_dict(payload, self.session_id)
        self._state_loaded = True

    async def _persist_state(self):
        store = get_session_store()
        await store.save_session(self.session_id, self.state.to_dict())

    async def _resolve_citizen_intent(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Use fast 8b LLM to understand a citizen's natural problem description.

        Only called when keyword-based extraction fails to identify a scheme_type
        AND the message looks like a real problem (not a greeting/name).
        Adds ~100ms latency via Groq 8b-instant.

        Returns:
            {"scheme_type": "kisan", "search_query": "...", "needs_scheme": true}
            or None on failure.
        """
        if not self.groq_client:
            return None

        prompt = (
            'A citizen called a government scheme helpline and said:\n'
            f'"{user_message}"\n\n'
            'Understand their problem and return JSON with these fields:\n'
            '- "scheme_type": one of "scholarship", "kisan", "business", '
            '"health", "housing", "women", "pension", "employment", '
            '"disability", "welfare", or "unknown"\n'
            '- "search_query": 4-6 Hindi+English keywords to search '
            'for relevant Indian government schemes\n'
            '- "needs_scheme": true if they need scheme info, '
            'false if just greeting/name/chitchat\n\n'
            'Return ONLY valid JSON, no explanation.'
        )

        try:
            response = await self.groq_client.chat.completions.create(
                model=config.groq.voice_llm_model,  # 8b-instant — fastest
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            result = _json.loads(response.choices[0].message.content)
            logger.info(
                "LLM intent resolved: type=%s, needs=%s, query=%s",
                result.get("scheme_type"),
                result.get("needs_scheme"),
                str(result.get("search_query", ""))[:60],
            )
            return result
        except Exception as err:
            logger.warning("LLM intent resolution failed: %s", err)
            return None

    def _is_english(self, language: str) -> bool:
        return language.lower() in {"en", "english"}

    def _has_direct_lookup_intent(self, user_message: str) -> bool:
        lowered = user_message.lower()
        return any(term in lowered for term in DIRECT_LOOKUP_TERMS)

    def _get_direct_lookup_aliases(self, user_message: str) -> List[str]:
        lowered = user_message.lower()
        for aliases in DIRECT_SCHEME_ALIASES.values():
            if any(alias in lowered for alias in aliases):
                return aliases
        return []

    def _is_follow_up_message(self, user_message: str) -> bool:
        lowered = user_message.lower()
        # Strong follow-up: contains an explicit follow-up marker
        if any(marker in lowered for marker in FOLLOW_UP_MARKERS):
            return True
        # Weak follow-up: very short utterance after schemes were just shown,
        # AND it doesn't introduce a fresh scheme intent or new state name.
        if not self.state.last_scholarships:
            return False
        if any(
            keyword in lowered
            for keyword in [
                "scholarship", "scheme", "yojana", "kisan", "loan", "business", "women", "pension",
                "स्कॉलरशिप", "योजना", "किसान", "लोन", "बिजनेस", "महिला", "पेंशन",
                "छात्रवृत्ति", "स्टूडेंट", "कृषि", "व्यापार",
            ]
        ):
            return False
        # If the user mentioned a NEW state, treat as a fresh query (not a follow-up).
        for state_key in STATES_MAP:
            if _word_match(state_key, lowered):
                return False
        # Only count as follow-up if the message is very short (likely "first one", "eligibility")
        return len(lowered.split()) <= 4

    def _should_apply_profile_context(self, user_message: str) -> bool:
        lowered = user_message.lower().strip()
        if self._has_direct_lookup_intent(lowered):
            return False
        if self._is_follow_up_message(lowered):
            return True
        strong_fresh_query = any(keyword in lowered for keyword in [
            "scholarship", "scheme", "yojana", "kisan", "loan", "business", "women",
            "स्कॉलरशिप", "योजना", "किसान", "लोन", "बिजनेस", "महिला", "छात्रवृत्ति",
        ])
        if strong_fresh_query and len(lowered.split()) >= 5:
            return False
        return True

    def _needs_rag(self, current_user_message: Optional[str] = None) -> bool:
        """
        Decide whether retrieval should run for the current turn.

        The previous implementation inspected only the last stored message,
        which frequently skipped RAG for the exact message being processed.
        """

        profile = self.state.profile
        candidate_message = (current_user_message or "").lower().strip()
        if not candidate_message:
            for message in reversed(self.state.messages):
                if message.role == "user":
                    candidate_message = message.content.lower()
                    break

        scheme_keywords = [
            "scholarship", "scheme", "yojana", "kisan", "farmer", "business", "loan",
            "help", "benefit", "amount", "apply", "eligibility", "documents",
            "pm-kisan", "pm kisan", "pmmsy", "mudra", "pragati", "inspire",
            "nsp", "pension", "women", "mahila",
            # Hindi keywords
            "स्कॉलरशिप", "छात्रवृत्ति", "योजना", "किसान", "बिजनेस", "लोन",
            "मदद", "फायदा", "राशि", "आवेदन", "पात्रता", "दस्तावेज",
            "पीएम किसान", "मुद्रा", "प्रगति", "इंस्पायर",
            "पेंशन", "महिला", "स्टूडेंट",
        ]
        direct_lookup_terms = [
            "pm-kisan", "pm kisan", "mudra", "pragati", "inspire", "nsp",
            "pmmsy", "aicte", "pension", "farmer scheme", "business loan",
            # Hindi variants
            "पीएम किसान", "मुद्रा", "प्रगति", "इंस्पायर", "आयुष्मान",
        ]

        if any(term in candidate_message for term in direct_lookup_terms):
            return True

        asking_for_schemes = any(keyword in candidate_message for keyword in scheme_keywords)
        context_score = sum(
            bool(value)
            for value in [profile.state, profile.scheme_type, profile.category, profile.course]
        )
        if asking_for_schemes and context_score >= 2:
            return True

        if profile.state and profile.scheme_type:
            return True

        high_intent = [
            "kaun si", "konsi", "kaunsi", "which", "recommend", "suggest", "batao", "bataiye",
            # Hindi
            "कौन सी", "कौनसी", "बताओ", "बताइए", "बताइये", "सुझाव", "दिखाओ",
        ]
        if any(marker in candidate_message for marker in high_intent) and (
            profile.state or profile.scheme_type or profile.category or profile.course
        ):
            return True

        if self.state.turn_count >= 3 and (profile.state or profile.scheme_type):
            return True

        # If LLM intent resolver provided search terms, always do RAG
        if getattr(self, '_intent_search_query', ''):
            return True

        return False

    def _build_rag_query(self, user_message: str) -> str:
        """Build a search query from current utterance plus extracted profile."""

        if not self._should_apply_profile_context(user_message):
            return user_message

        parts = []
        if self.state.profile.state and self.state.profile.state.lower() not in user_message.lower():
            parts.append(self.state.profile.state)
        if self.state.profile.category and self.state.profile.category.lower() not in user_message.lower():
            parts.append(self.state.profile.category)
        if self.state.profile.scheme_type and self.state.profile.scheme_type.lower() not in user_message.lower():
            parts.append(self.state.profile.scheme_type)
        if self.state.profile.course and self.state.profile.course.lower() not in user_message.lower():
            parts.append(self.state.profile.course)
        # Append LLM-generated search terms if available (from _resolve_citizen_intent)
        intent_query = getattr(self, '_intent_search_query', '')
        if intent_query:
            parts.append(intent_query)
        # Also include stored problem search query from previous turns
        elif self.state.profile.problem_search_query:
            parts.append(self.state.profile.problem_search_query)
        # Include the raw problem description words for better embedding match
        elif self.state.profile.problem_description:
            parts.append(self.state.profile.problem_description)
        parts.append(user_message)
        return " ".join(parts)

    def _build_search_filters(self) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if self.state.profile.state:
            filters["state"] = self.state.profile.state
        return filters

    def _prefer_explicit_scheme_matches(
        self,
        user_message: str,
        results: List[Tuple[Dict[str, Any], float]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        lowered = user_message.lower()
        exact_terms = [term for term in DIRECT_LOOKUP_TERMS if term in lowered]
        if not exact_terms:
            return results

        def score(item: Tuple[Dict[str, Any], float]) -> Tuple[int, float]:
            scheme, base_score = item
            searchable_text = " ".join([
                str(scheme.get("name", "")),
                str(scheme.get("details", "")),
                str(scheme.get("benefits", "")),
                " ".join(scheme.get("tags", [])) if isinstance(scheme.get("tags"), list) else str(scheme.get("tags", "")),
            ]).lower()
            exact_hits = sum(term in searchable_text for term in exact_terms)
            name_hits = sum(term in str(scheme.get("name", "")).lower() for term in exact_terms)
            return (name_hits * 2 + exact_hits, base_score)

        ranked = sorted(results, key=score, reverse=True)
        return ranked

    def _find_exact_scheme_matches(
        self,
        user_message: str,
        limit: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        aliases = self._get_direct_lookup_aliases(user_message)
        if not aliases:
            return []

        matches: List[Tuple[Dict[str, Any], float]] = []
        document_pool: List[Dict[str, Any]] = []
        scholarships = getattr(self.rag, "scholarships", None) or []
        if isinstance(scholarships, list):
            document_pool.extend(item for item in scholarships if isinstance(item, dict))

        vectorstore = getattr(self.rag, "vectorstore", None)
        vector_documents = getattr(vectorstore, "documents", None) or []
        if isinstance(vector_documents, list):
            document_pool.extend(item for item in vector_documents if isinstance(item, dict))

        seen_ids = set()
        for scheme in document_pool:
            scheme_id = str(scheme.get("id", ""))
            dedupe_key = scheme_id or str(scheme.get("name", "")).lower()
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)

            name = str(scheme.get("name", "")).lower()
            searchable_text = " ".join([
                name,
                str(scheme.get("details", "")),
                str(scheme.get("benefits", "")),
                str(scheme.get("eligibility", "")),
                " ".join(scheme.get("tags", [])) if isinstance(scheme.get("tags"), list) else str(scheme.get("tags", "")),
            ]).lower()

            name_hits = sum(alias in name for alias in aliases)
            text_hits = sum(alias in searchable_text for alias in aliases)
            if not name_hits and not text_hits:
                continue

            level_bonus = 1 if str(scheme.get("level", "")).lower() == "central" else 0
            exact_score = float(name_hits * 10 + text_hits * 3 + level_bonus)
            matches.append((scheme, exact_score))

        matches.sort(
            key=lambda item: (
                item[1],
                len(str(item[0].get("name", ""))),
            ),
            reverse=True,
        )
        return matches[:limit]

    async def _search_for_schemes(
        self,
        user_message: str,
        language: str = "hinglish",
        streaming: bool = False,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Run retrieval and store the latest candidate schemes."""

        exact_matches = self._find_exact_scheme_matches(user_message, limit=5)
        if exact_matches:
            self.state.last_scholarships = [scheme for scheme, _ in exact_matches]
            logger.info("Using exact scheme alias match for '%s' with %d results", user_message[:80], len(exact_matches))
            return exact_matches

        search_query = self._build_rag_query(user_message)
        filters = self._build_search_filters()
        is_voice_mode = streaming or language.lower() in {"hi", "hinglish", "hindi"}

        start_time = time.time()
        results = await self.rag.search_parallel(
            search_query,
            top_k=5,
            filters=filters,
            rerank=not is_voice_mode,
        )
        results = self._prefer_explicit_scheme_matches(user_message, results)
        self.state.last_scholarships = [scheme for scheme, _ in results]
        logger.info(
            "RAG completed in %.0fms with %d results (rerank=%s)",
            (time.time() - start_time) * 1000,
            len(results),
            not is_voice_mode,
        )
        return results

    def _select_referenced_scheme(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Resolve 'first / second / third scheme' style follow-ups."""

        if not self.state.last_scholarships:
            return None

        msg = user_message.lower()
        index = 0
        ordinal_map = {
            0: ["first", "1st", "pehla", "pehli", "pahla", "पहला", "पहली"],
            1: ["second", "2nd", "dusra", "dusri", "doosra", "doosri", "दूसरा", "दूसरी"],
            2: ["third", "3rd", "teesra", "teesri", "तीसरा", "तीसरी"],
        }
        for candidate_index, markers in ordinal_map.items():
            if any(marker in msg for marker in markers):
                index = candidate_index
                break

        return self.state.last_scholarships[min(index, len(self.state.last_scholarships) - 1)]

    def _build_detail_response(self, user_message: str, language: str) -> Optional[str]:
        """Answer detail follow-ups from cached retrieved schemes."""
        from utils.voice_normalizer import (
            voice_friendly_name, spoken_amount, simplify_eligibility, normalize_for_voice,
        )

        if not self.state.last_scholarships:
            return None

        msg = user_message.lower()
        detail_patterns = {
            "eligibility": r"\beligib(?:ility|le)\b|पात्रता|एलिजिबिलिटी|योग्यता",
            "documents": r"\bdocuments?\b|दस्तावेज़|दस्तावेज|कागजात",
            "application": r"\bapply\b|\bapplication\b|\bform\b|\blast date\b|\bdeadline\b|आवेदन|अप्लाई|फॉर्म|अंतिम तिथि",
            "benefit": r"\bbenefits?\b|\bamount\b|\bpaisa\b|फायदा|राशि|पैसा|लाभ",
        }
        matched_detail_keys = {
            key for key, pattern in detail_patterns.items() if re.search(pattern, msg)
        }
        if not matched_detail_keys:
            return None

        scheme = self._select_referenced_scheme(user_message)
        if not scheme:
            return None

        lang = self.state.profile.language or language
        name = voice_friendly_name(_short_text(scheme.get("name"), 90)) or "this scheme"
        benefit = spoken_amount(_short_text(scheme.get("benefits") or scheme.get("award_amount"), 150), lang)
        eligibility = simplify_eligibility(_short_text(scheme.get("eligibility"), 170), lang)
        application = normalize_for_voice(_short_text(scheme.get("application_process"), 170), lang)
        documents = normalize_for_voice(_short_text(scheme.get("documents"), 140), lang)

        parts = []
        if "eligibility" in matched_detail_keys:
            parts.append(f"Eligibility: {eligibility or 'Official portal pe eligibility details available hain.'}")
        if "documents" in matched_detail_keys:
            parts.append(f"Documents: {documents or 'Required documents official application page par listed hote hain.'}")
        if "application" in matched_detail_keys:
            parts.append(f"Application: {application or 'Official portal pe application process diya gaya hai.'}")
        if "benefit" in matched_detail_keys:
            parts.append(f"Benefit: {benefit or 'Benefit amount official source se confirm karna hoga.'}")

        if not parts:
            parts.append(f"Benefit: {benefit or 'Official source dekhna hoga.'}")
            parts.append(f"Eligibility: {eligibility or 'Official portal pe available hai.'}")

        if self._is_english(language):
            return f"{name}: " + " ".join(parts[:2]) + " If you want, I can also summarise the next step to apply."
        return f"{name}: " + " ".join(parts[:2]) + " Chahein to main next step bhi short mein bata sakti hoon."

    def _next_profile_question(self, language: str) -> Optional[str]:
        """Ask for the next missing field deterministically."""

        profile = self.state.profile

        if not profile.scheme_type:
            if self._is_english(profile.language or language):
                return "Please tell me, what problem are you facing? Or what kind of help do you need?"
            return "Bataiye, aapko kis cheez mein madad chahiye? Apni problem bataiye."

        if profile.scheme_type == "scholarship":
            if not profile.state:
                return "Which state are you from?" if self._is_english(language) else "Aap kis state se hain?"
            if not profile.category:
                return (
                    "What is your category: General, OBC, SC, ST, or Minority?"
                    if self._is_english(language)
                    else "Aapki category kya hai: General, OBC, SC, ST, ya Minority?"
                )
            if not profile.course:
                return "Which course are you studying?" if self._is_english(language) else "Aap kaun sa course kar rahe hain?"

        if profile.scheme_type == "kisan" and not profile.state:
            return "Which state are you from?" if self._is_english(language) else "Aap kis state se hain?"

        if profile.scheme_type == "business" and not profile.state:
            return "Which state is your business in?" if self._is_english(language) else "Aapka business kis state mein hai?"

        return None

    def _build_grounded_scheme_response(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        language: str,
    ) -> Optional[str]:
        """Summarise retrieved schemes in conversational Hindi/English.

        Key improvements over the old version:
        - Uses Hindi labels ("Fayda:", "Patrta:") instead of English on Hindi calls
        - References the citizen's original problem in the intro
        - Conversational connectors instead of numbered list dump
        """
        from utils.voice_normalizer import voice_friendly_name, spoken_amount, simplify_eligibility

        if not results:
            return None

        lang = self.state.profile.language or language
        is_eng = self._is_english(language)
        problem = self.state.profile.problem_description

        summaries = []
        for idx, (scheme, _) in enumerate(results[:2], start=1):
            name = voice_friendly_name(_short_text(scheme.get("name"), 80)) or f"Scheme {idx}"
            benefit = spoken_amount(
                _short_text(scheme.get("benefits") or scheme.get("award_amount"), 80), lang
            )
            eligibility = simplify_eligibility(
                _short_text(scheme.get("eligibility"), 80), lang
            )

            if is_eng:
                parts = [name]
                if benefit:
                    parts.append(f"You can get {benefit}")
                if eligibility:
                    parts.append(f"For this, {eligibility}")
                summaries.append(". ".join(parts))
            else:
                parts = [name]
                if benefit:
                    parts.append(f"Isme aapko {benefit} mil sakta hai")
                if eligibility:
                    parts.append(f"Iske liye {eligibility}")
                summaries.append(". ".join(parts))

        # Build intro that references the user's actual problem
        if is_eng:
            if problem:
                intro = f"For your problem, I found these schemes."
            else:
                intro = "I found these relevant schemes."
            connector = "Also, " if len(summaries) > 1 else ""
            tail = "Should I tell you more about any of these?"
        else:
            if problem:
                intro = f"Aapki samasya ke liye ye schemes hain."
            else:
                intro = "Aapke liye ye schemes mili hain."
            connector = "Doosri scheme hai " if len(summaries) > 1 else ""
            tail = "Kya aap kisi scheme ke baare mein aur jaanna chahenge?"

        if len(summaries) == 1:
            return f"{intro} {summaries[0]}. {tail}"
        return f"{intro} Pehli scheme hai {summaries[0]}. {connector}{summaries[1]}. {tail}"

    def _build_direct_response(
        self,
        user_message: str,
        results: List[Tuple[Dict[str, Any], float]],
        language: str,
    ) -> Optional[str]:
        """Use deterministic replies for the high-volume grounded paths.

        Order:
        1. Detail follow-up about a previously listed scheme.
        2. Direct lookup of a named scheme (PM-KISAN, MUDRA, etc) → skip
           profile gathering, return grounded results immediately.
        3. Profile question if any required field is missing AND the user did
           not name a specific scheme. ``_next_profile_question`` returns None
           the moment the user has provided enough context, so a single
           multi-fact utterance like "Maharashtra SC engineering scholarship"
           short-circuits straight to results.
        4. Grounded results from RAG.
        5. Fallback prompt for more info.
        """

        detail_response = self._build_detail_response(user_message, language)
        if detail_response:
            return detail_response

        specific_lookup = any(
            term in user_message.lower()
            for term in ["pm-kisan", "pm kisan", "mudra", "pragati", "inspire", "nsp", "pmmsy", "aicte"]
        )

        # When the user names a specific scheme, jump straight to grounded
        # results — don't ask them for profile fields.
        if specific_lookup:
            grounded_results = self._build_grounded_scheme_response(results, language)
            if grounded_results:
                return grounded_results

        # Profile question only fires when fields are still missing.
        follow_up = self._next_profile_question(language)
        if follow_up and not specific_lookup:
            return follow_up

        grounded_results = self._build_grounded_scheme_response(results, language)
        if grounded_results:
            return grounded_results

        if self._needs_rag(user_message):
            if self._is_english(language):
                return "I could not find a confident match yet. Please share your state, category, or course so I can narrow it down."
            return "Mujhe abhi confident match nahi mila. Aap apna state, category, ya course batayein, main search aur narrow kar dungi."

        return None

    async def _finalize_response(self, response: str, start_time: float):
        """Store and log responses in one place.
        
        Persistence is fire-and-forget so it never blocks the user-visible reply,
        but a reference is kept in ``_pending_persists`` so a freshly-recreated
        handler can wait on the in-flight write before reading state.
        """

        self.state.add_message("assistant", response)
        # Schedule persist as a background task; don't await it here.
        try:
            task = asyncio.create_task(self._persist_state())
            _pending_persists[self.session_id] = task
            # Drop the reference once it's done so the dict doesn't leak.
            task.add_done_callback(lambda t, sid=self.session_id: _pending_persists.pop(sid, None) if _pending_persists.get(sid) is t else None)
        except RuntimeError:
            # No running loop (rare — e.g. test harness). Persist synchronously.
            await self._persist_state()
        logger.info("Profile: %s", self.state.get_profile_summary())
        logger.latency("Conversation Turn", (time.time() - start_time) * 1000)
        logger.assistant_response(response[:100] + "..." if len(response) > 100 else response)

    async def generate_response(self, user_message: str, language: str = "hinglish") -> str:
        """Generate a response for a text or non-streaming audio turn."""

        await self.initialize()
        start_time = time.time()

        extract_profile_from_message(self.state.profile, user_message)
        if not self.state.profile.language:
            self.state.profile.language = language

        # LLM intent resolution: if keywords didn't find scheme_type, ask LLM
        self._intent_search_query = ""
        if (not self.state.profile.scheme_type
                and len(user_message.split()) >= 4
                and not _is_just_name_or_greeting(user_message)):
            intent = await self._resolve_citizen_intent(user_message)
            if intent and intent.get("needs_scheme"):
                stype = intent.get("scheme_type", "unknown")
                if stype and stype != "unknown":
                    self.state.profile.scheme_type = stype
                self._intent_search_query = intent.get("search_query", "")
                # Persist problem context so it survives across turns
                self.state.profile.problem_description = user_message
                self.state.profile.problem_search_query = self._intent_search_query
        # Also store problem when user describes one even if keywords matched
        elif (self.state.profile.scheme_type
                and not self.state.profile.problem_description
                and len(user_message.split()) >= 6):
            self.state.profile.problem_description = user_message

        self.state.add_message("user", user_message)

        results: List[Tuple[Dict[str, Any], float]] = []
        if self._needs_rag(user_message):
            results = await self._search_for_schemes(user_message, language=language, streaming=False)
            scholarship_context = format_scholarships_for_context(self.state.last_scholarships)
        else:
            scholarship_context = "No specific schemes yet - gather user profile info first."
            logger.info("Skipping RAG while gathering profile")

        direct_response = self._build_direct_response(user_message, results, language)
        if direct_response:
            await self._finalize_response(direct_response, start_time)
            return direct_response

        system_prompt = get_system_prompt_with_context(scholarship_context, language)
        profile_info = self.state.get_profile_summary()
        if profile_info != "No profile info yet":
            system_prompt += f"\n\nUser profile: {profile_info}"

        response = None
        if self.groq_client:
            response = await self._generate_groq(system_prompt)
        if response is None and self.gemini_model:
            response = await self._generate_gemini(system_prompt)
        if response is None:
            response = "Thodi problem aa gayi. Kya aap phir se bolenge?"
            logger.error("All LLM providers failed")

        await self._finalize_response(response, start_time)
        return response

    async def _generate_groq(self, system_prompt: str) -> Optional[str]:
        """LLM fallback via Groq."""

        try:
            messages = [{"role": "system", "content": system_prompt}] + self.state.get_history_for_llm()
            response = await self.groq_client.chat.completions.create(
                model=config.groq.llm_model,
                messages=messages,
                temperature=config.groq.llm_temperature,
                max_tokens=config.groq.llm_max_tokens,
            )
            return response.choices[0].message.content
        except Exception as error:
            logger.error_with_context("Groq LLM", error)
            return None

    async def _generate_gemini(self, system_prompt: str) -> Optional[str]:
        """LLM fallback via Gemini."""

        try:
            full_prompt = f"{system_prompt}\n\n"
            for message in self.state.messages:
                role = "User" if message.role == "user" else "Assistant"
                full_prompt += f"{role}: {message.content}\n"
            full_prompt += "Assistant:"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    full_prompt,
                    generation_config={"temperature": 0.4, "max_output_tokens": 220},
                ),
            )
            return response.text
        except Exception as error:
            logger.error_with_context("Gemini LLM", error)
            return None

    async def generate_response_stream(
        self,
        user_message: str,
        buffer_sentences: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate a streamed response for websocket / SSE style callers."""

        await self.initialize()
        start_time = time.time()

        extract_profile_from_message(self.state.profile, user_message)
        if not self.state.profile.language:
            self.state.profile.language = "hinglish"

        # LLM intent resolution: if keywords didn't find scheme_type, ask LLM
        self._intent_search_query = ""
        if (not self.state.profile.scheme_type
                and len(user_message.split()) >= 4
                and not _is_just_name_or_greeting(user_message)):
            intent = await self._resolve_citizen_intent(user_message)
            if intent and intent.get("needs_scheme"):
                stype = intent.get("scheme_type", "unknown")
                if stype and stype != "unknown":
                    self.state.profile.scheme_type = stype
                self._intent_search_query = intent.get("search_query", "")
                # Persist problem context so it survives across turns
                self.state.profile.problem_description = user_message
                self.state.profile.problem_search_query = self._intent_search_query
        # Also store problem when user describes one even if keywords matched
        elif (self.state.profile.scheme_type
                and not self.state.profile.problem_description
                and len(user_message.split()) >= 6):
            self.state.profile.problem_description = user_message

        self.state.add_message("user", user_message)

        results: List[Tuple[Dict[str, Any], float]] = []
        if self._needs_rag(user_message):
            results = await self._search_for_schemes(user_message, language=self.state.profile.language, streaming=True)
            scholarship_context = format_scholarships_for_context(self.state.last_scholarships)
        else:
            scholarship_context = "No specific schemes yet - gather user profile info first."
            logger.info("Skipping RAG while gathering profile")

        direct_response = self._build_direct_response(user_message, results, self.state.profile.language or "hinglish")
        if direct_response:
            await self._finalize_response(direct_response, start_time)
            yield direct_response
            return

        profile_info = self.state.get_profile_summary()
        system_prompt = get_system_prompt_with_context(
            scholarship_context,
            self.state.profile.language or "hinglish",
        )
        if profile_info != "No profile info yet":
            system_prompt += f"\n\nUser profile: {profile_info}"

        if self.groq_client:
            full_response = ""
            sentence_buffer = ""
            finish_reason: Optional[str] = None
            # Soft breakers (commas, colons, danda) flush early once the
            # buffer has at least MIN_CLAUSE_LEN chars — this lets the first
            # TTS chunk play 300-500ms sooner.
            HARD_BREAKERS = ".!?।"
            SOFT_BREAKERS = ",;:"
            MIN_CLAUSE_LEN = 28
            try:
                messages = [{"role": "system", "content": system_prompt}] + self.state.get_history_for_llm()
                # Voice path uses faster, smaller model + lower max_tokens.
                stream = await self.groq_client.chat.completions.create(
                    model=config.groq.voice_llm_model,
                    messages=messages,
                    temperature=config.groq.voice_llm_temperature,
                    max_tokens=config.groq.voice_llm_max_tokens,
                    stream=True,
                )

                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta.content if choice.delta else None
                        if getattr(choice, "finish_reason", None):
                            finish_reason = choice.finish_reason
                    else:
                        delta = None
                    if not delta:
                        continue
                    full_response += delta

                    if not buffer_sentences:
                        yield delta
                        continue

                    sentence_buffer += delta
                    flushed = True
                    while flushed:
                        flushed = False
                        # Hard breakers always flush
                        for delimiter in HARD_BREAKERS:
                            if delimiter in sentence_buffer:
                                idx = sentence_buffer.index(delimiter)
                                sentence = sentence_buffer[: idx + 1].strip()
                                sentence_buffer = sentence_buffer[idx + 1 :].strip()
                                if sentence:
                                    yield sentence
                                flushed = True
                                break
                        if flushed:
                            continue
                        # Soft breakers flush only if the clause is long enough
                        for delimiter in SOFT_BREAKERS:
                            idx = sentence_buffer.find(delimiter)
                            if idx >= MIN_CLAUSE_LEN:
                                sentence = sentence_buffer[: idx + 1].strip()
                                sentence_buffer = sentence_buffer[idx + 1 :].strip()
                                if sentence:
                                    yield sentence
                                flushed = True
                                break

                # Stream finished. Decide what to do with whatever is left in
                # the sentence buffer:
                # - If the model finished cleanly (finish_reason='stop'), the
                #   leftover is a real (final) sentence — yield it.
                # - If the model was cut off by max_tokens (finish_reason=
                #   'length'), the leftover is mid-word; don't yield it as-is
                #   because TTS would speak a clipped fragment. Instead append
                #   a graceful closer so the user hears a complete thought.
                trailing = sentence_buffer.strip()
                if buffer_sentences and trailing:
                    if finish_reason == "length":
                        # Drop trailing partial; emit a polite closer instead.
                        if self._is_english(self.state.profile.language or "hinglish"):
                            closer = (
                                "Let me know if you want full details on any one of these."
                            )
                        else:
                            closer = (
                                "Aap chahein to main kisi ek scheme ki puri detail bata sakti hoon."
                            )
                        full_response = full_response.rstrip() + " " + closer
                        yield closer
                        logger.warning(
                            "Voice LLM hit max_tokens (%s) — dropped %d-char partial trailing buffer",
                            config.groq.voice_llm_max_tokens,
                            len(trailing),
                        )
                    else:
                        # Clean stop. Only yield trailing buffer if it looks
                        # like a real fragment (ends with punctuation, OR has
                        # at least 12 chars + 3 words). This avoids speaking
                        # stray '...' or 'Up to' kind of fragments.
                        words = trailing.split()
                        ends_punct = trailing[-1] in ".!?।,:;"
                        looks_complete = ends_punct or (len(trailing) >= 12 and len(words) >= 3)
                        if looks_complete:
                            yield trailing
                        else:
                            logger.info(
                                "Discarded short trailing fragment from clean stream: %r",
                                trailing,
                            )

                await self._finalize_response(full_response, start_time)
                return
            except Exception as error:
                logger.error_with_context("Groq Stream", error)

        response = "Thodi problem aa gayi."
        if self.gemini_model:
            fallback = await self._generate_gemini(system_prompt)
            if fallback:
                response = fallback

        await self._finalize_response(response, start_time)
        yield response

    def reset_conversation(self):
        """Reset session state for a new conversation."""

        self.state = ConversationState(session_id=self.session_id)
        logger.info("Conversation reset")

    async def reset_and_persist(self):
        self.reset_conversation()
        await self._persist_state()

    def get_public_session_view(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "profile_summary": self.state.get_profile_summary(),
            "profile": self.state.profile.to_dict(),
            "turn_count": self.state.turn_count,
            "message_count": len(self.state.messages),
            "messages": [message.to_dict() for message in self.state.messages[-6:]],
            "last_schemes": self.state.last_scholarships[:3],
            "last_scheme_count": len(self.state.last_scholarships),
        }


_conversation_handlers: Dict[str, ConversationHandler] = {}
_handler_timestamps: Dict[str, float] = {}
_pending_persists: Dict[str, asyncio.Task] = {}
_SESSION_TTL = 3600


def get_conversation_handler(session_id: str = "default") -> ConversationHandler:
    """Get or create a conversation handler for a session ID."""

    now = time.time()
    expired = [sid for sid, timestamp in _handler_timestamps.items() if now - timestamp > _SESSION_TTL]
    for sid in expired:
        _conversation_handlers.pop(sid, None)
        _handler_timestamps.pop(sid, None)

    if session_id not in _conversation_handlers:
        _conversation_handlers[session_id] = ConversationHandler(session_id=session_id)

    _handler_timestamps[session_id] = now
    return _conversation_handlers[session_id]
