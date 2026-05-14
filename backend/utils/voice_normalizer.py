"""
Voice Text Normalizer
=====================
Converts raw database text into natural spoken Hinglish/English for TTS.
Pure regex + dictionary lookups — no LLM calls, ~0ms latency.
"""

import re
from typing import Optional


# ── Number-to-spoken-word mappings ────────────────────────────────────────

_HINDI_ONES = {
    0: "", 1: "ek", 2: "do", 3: "teen", 4: "chaar", 5: "paanch",
    6: "chhah", 7: "saat", 8: "aath", 9: "nau", 10: "das",
    11: "gyaarah", 12: "baarah", 13: "terah", 14: "chaudah", 15: "pandrah",
    16: "solah", 17: "satrah", 18: "atharah", 19: "unees", 20: "bees",
    25: "pacchees", 30: "tees", 40: "chaalis", 50: "pachaas",
    60: "saath", 70: "sattar", 80: "assi", 90: "nabbe",
}

_ENGLISH_ONES = {
    0: "", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    25: "twenty five", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


def spoken_number(n: int, lang: str = "hinglish") -> str:
    """
    Convert an integer to natural spoken form using Indian numbering.

    Examples (hinglish):
        100      → "sau"
        1000     → "ek hazaar"
        12000    → "baarah hazaar"
        100000   → "ek lakh"
        250000   → "dhai lakh"
        500000   → "paanch lakh"
        1000000  → "das lakh"
        10000000 → "ek crore"
    """
    if n <= 0:
        return "zero" if lang == "en" else "zero"

    is_english = lang in ("en", "english")
    ones = _ENGLISH_ONES if is_english else _HINDI_ONES

    # Special cases
    if n in ones:
        return ones[n]

    parts = []

    # Crore (10,000,000)
    if n >= 10000000:
        crore = n // 10000000
        n %= 10000000
        label = "crore" if is_english else "crore"
        parts.append(f"{ones.get(crore, str(crore))} {label}")

    # Lakh (100,000)
    if n >= 100000:
        lakh = n // 100000
        n %= 100000
        # Special: 2.5 lakh = "dhai lakh"
        if not is_english and lakh == 2 and n == 50000:
            parts.append("dhai lakh")
            n = 0
        elif not is_english and lakh == 1 and n == 50000:
            parts.append("dedh lakh")
            n = 0
        else:
            label = "lakh" if is_english else "lakh"
            parts.append(f"{ones.get(lakh, str(lakh))} {label}")

    # Hazaar / Thousand (1,000)
    if n >= 1000:
        hazaar = n // 1000
        n %= 1000
        label = "thousand" if is_english else "hazaar"
        parts.append(f"{ones.get(hazaar, str(hazaar))} {label}")

    # Sau / Hundred (100)
    if n >= 100:
        sau = n // 100
        n %= 100
        label = "hundred" if is_english else "sau"
        if sau == 1:
            parts.append(label)
        else:
            parts.append(f"{ones.get(sau, str(sau))} {label}")

    # Remaining (< 100)
    if n > 0:
        if n in ones:
            parts.append(ones[n])
        else:
            parts.append(str(n))

    return " ".join(parts).strip()


# ── Currency / amount normalization ───────────────────────────────────────

# Regex to extract numbers from currency strings
_CURRENCY_RE = re.compile(
    r'[₹$Rs\.]*\s*([\d,]+(?:\.\d+)?)\s*(?:/[-–])?'
)
_LAKH_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|लाख)', re.IGNORECASE)
_CRORE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:crore|करोड़)', re.IGNORECASE)

# Per-period phrases to humanize
_PERIOD_MAP_HI = {
    "per annum": "saalana", "per year": "saalana", "annually": "saalana",
    "per month": "har mahine", "monthly": "har mahine",
    "per semester": "har semester", "one time": "ek baar",
    "one-time": "ek baar", "lump sum": "ek saath",
}
_PERIOD_MAP_EN = {
    "per annum": "per year", "annually": "per year",
    "per month": "per month", "monthly": "per month",
    "one time": "one time", "one-time": "one time",
    "lump sum": "as a lump sum",
}


def spoken_amount(text: str, lang: str = "hinglish") -> str:
    """
    Convert a monetary amount string into natural spoken form.

    Examples:
        "₹12,000/- per annum"   → "baarah hazaar rupaye saalana"
        "Rs. 2,50,000"          → "dhai lakh rupaye"
        "Up to 5 lakh"          → "paanch lakh rupaye tak"
        "₹75,000 one time"      → "pachattar hazaar rupaye ek baar"
    """
    if not text or not text.strip():
        return ""

    is_english = lang in ("en", "english")
    result = text.strip()

    # Try lakh/crore expressions first
    crore_m = _CRORE_RE.search(result)
    if crore_m:
        val = float(crore_m.group(1))
        n = int(val * 10000000)
        spoken = spoken_number(n, lang)
        suffix = "rupees" if is_english else "rupaye"
        result = f"{spoken} {suffix}"
        # Preserve period info
        remainder = result[crore_m.end():].strip() if crore_m.end() < len(text) else ""
        if remainder:
            result += f" {remainder}"
        return _humanize_period(result, lang)

    lakh_m = _LAKH_RE.search(result)
    if lakh_m:
        val = float(lakh_m.group(1))
        n = int(val * 100000)
        spoken = spoken_number(n, lang)
        suffix = "rupees" if is_english else "rupaye"
        result = f"{spoken} {suffix}"
        remainder = text[lakh_m.end():].strip()
        if remainder:
            result += f" {remainder}"
        return _humanize_period(result, lang)

    # Try raw number extraction
    currency_m = _CURRENCY_RE.search(result)
    if currency_m:
        raw = currency_m.group(1).replace(",", "")
        try:
            n = int(float(raw))
            if n > 0:
                spoken = spoken_number(n, lang)
                suffix = "rupees" if is_english else "rupaye"
                remainder = text[currency_m.end():].strip()
                result = f"{spoken} {suffix}"
                if remainder:
                    result += f" {remainder}"
                return _humanize_period(result, lang)
        except ValueError:
            pass

    # If nothing matched, just clean up symbols
    result = re.sub(r'[₹$]', '', result)
    result = re.sub(r'/[-–]', '', result)
    return _humanize_period(result.strip(), lang)


def _humanize_period(text: str, lang: str) -> str:
    """Replace 'per annum' etc with spoken equivalents."""
    period_map = _PERIOD_MAP_EN if lang in ("en", "english") else _PERIOD_MAP_HI
    lowered = text.lower()
    for formal, spoken in period_map.items():
        if formal in lowered:
            text = re.sub(re.escape(formal), spoken, text, flags=re.IGNORECASE)
    return text.strip()


# ── Eligibility simplification ────────────────────────────────────────────

def simplify_eligibility(text: str, lang: str = "hinglish") -> str:
    """
    Convert technical eligibility text to simple conversational form.

    Input:  "Education Level: Post Matric; Marks Criteria: 60%; Category: SC, ST; Income Limit: 250000"
    Output: "10th pass hona chahiye, 60% marks, SC ya ST category, aur family income dhai lakh se kam"
    """
    if not text or not text.strip():
        return ""

    is_english = lang in ("en", "english")

    # Handle dict-style text (from _coerce_text)
    # "Education Level: Post Matric; Marks Criteria: 60%; ..."
    parts = []
    segments = re.split(r'[;|]', text)

    for segment in segments[:3]:  # Max 3 criteria for voice
        segment = segment.strip()
        if not segment:
            continue

        # Parse "Key: Value" format
        kv_match = re.match(r'^([^:]+):\s*(.+)$', segment)
        if kv_match:
            key = kv_match.group(1).strip().lower()
            value = kv_match.group(2).strip()
        else:
            key = ""
            value = segment

        # Normalize specific fields
        if "education" in key or "class" in key or "level" in key:
            if is_english:
                parts.append(f"{value} education required")
            else:
                parts.append(f"{value} pass hona chahiye")

        elif "marks" in key or "percentage" in key or "grade" in key:
            if is_english:
                parts.append(f"minimum {value} marks")
            else:
                parts.append(f"kam se kam {value} marks")

        elif "income" in key:
            # Try to convert number
            num_match = re.search(r'(\d[\d,]*)', value)
            if num_match:
                n = int(num_match.group(1).replace(",", ""))
                spoken = spoken_number(n, lang)
                if is_english:
                    parts.append(f"family income below {spoken} rupees")
                else:
                    parts.append(f"family income {spoken} rupaye se kam")
            else:
                parts.append(value)

        elif "category" in key or "caste" in key:
            if is_english:
                parts.append(f"{value} category")
            else:
                parts.append(f"{value} category ke liye")

        elif "age" in key:
            if is_english:
                parts.append(f"age {value}")
            else:
                parts.append(f"umar {value}")

        else:
            # Generic — just include value, trimmed
            if len(value) < 60:
                parts.append(value)

    if not parts:
        # Fallback: just return first 120 chars cleaned
        return _clean_for_voice(text[:120])

    joiner = ", and " if is_english else ", aur "
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + joiner + parts[-1]


# ── Scheme name shortening ────────────────────────────────────────────────

_NAME_STRIP_SUFFIXES = [
    r'\s*under\s+CSS\b', r'\s*under\s+\w+\s+scheme\b',
    r'\s*[-–]\s*Government of India\b', r'\s*[-–]\s*Ministry of\s+\w+',
    r'\s*\(Central\s+Sector\s+Scheme\)', r'\s*\(CSS\)',
    r'\s*[-–]\s*Centrally\s+Sponsored\b',
]
_NAME_STRIP_RE = re.compile('|'.join(_NAME_STRIP_SUFFIXES), re.IGNORECASE)


def voice_friendly_name(name: str) -> str:
    """
    Shorten bureaucratic scheme names for voice.

    "Post Matric Scholarship for SC Students under CSS" → "Post Matric SC Scholarship"
    "Pradhan Mantri Kisan Samman Nidhi Yojana" → "PM Kisan Samman Nidhi"
    """
    if not name:
        return ""

    result = name.strip()

    # Strip bureaucratic suffixes
    result = _NAME_STRIP_RE.sub('', result).strip()

    # Common abbreviations
    result = re.sub(r'\bPradhan Mantri\b', 'PM', result)
    result = re.sub(r'\bGovernment of India\b', '', result, flags=re.IGNORECASE)

    # If still too long (>8 words), truncate gracefully
    words = result.split()
    if len(words) > 8:
        result = ' '.join(words[:7])

    return result.strip()


# ── Master cleanup ────────────────────────────────────────────────────────

_URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)
_SPECIAL_CHAR_RE = re.compile(r'[*#_\[\](){}|<>~`]')
_MULTI_SPACE_RE = re.compile(r'\s{2,}')


def _clean_for_voice(text: str) -> str:
    """Remove URLs, special chars, and excess whitespace."""
    text = _URL_RE.sub('', text)
    text = _SPECIAL_CHAR_RE.sub(' ', text)
    text = text.replace('/-', '')
    text = text.replace('₹', '')
    text = text.replace('Rs.', '')
    text = text.replace('Rs', '')
    text = _MULTI_SPACE_RE.sub(' ', text)
    return text.strip()


def normalize_for_voice(text: str, lang: str = "hinglish") -> str:
    """
    Master normalization: apply all cleanup + spoken conversions.
    Safe to call on any text — returns cleaned, TTS-friendly string.
    """
    if not text or not text.strip():
        return ""

    result = _clean_for_voice(text)

    # Convert inline numbers that look like amounts
    def _replace_amount(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        try:
            n = int(float(raw))
            if n >= 1000:
                return spoken_number(n, lang) + " rupaye"
            return spoken_number(n, lang)
        except ValueError:
            return m.group(0)

    result = re.sub(r'[\d,]{4,}', _replace_amount, result)

    return result.strip()
