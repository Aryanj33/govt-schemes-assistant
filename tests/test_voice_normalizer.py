"""Tests for voice_normalizer and intent resolution integration."""

import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from utils.voice_normalizer import (
    spoken_number,
    spoken_amount,
    simplify_eligibility,
    voice_friendly_name,
    normalize_for_voice,
)


def test_spoken_number():
    """Test Indian number system conversion."""
    cases = [
        (100, "hinglish", "sau"),
        (1000, "hinglish", "ek hazaar"),
        (12000, "hinglish", "baarah hazaar"),
        (100000, "hinglish", "ek lakh"),
        (250000, "hinglish", "dhai lakh"),
        (150000, "hinglish", "dedh lakh"),
        (500000, "hinglish", "paanch lakh"),
        (10000000, "hinglish", "ek crore"),
        (5000, "en", "five thousand"),
        (100000, "en", "one lakh"),
    ]
    passed = 0
    for n, lang, expected in cases:
        result = spoken_number(n, lang)
        status = "✅" if result == expected else "❌"
        if result != expected:
            print(f"  {status} spoken_number({n}, {lang}) = '{result}' (expected '{expected}')")
        else:
            passed += 1
    print(f"spoken_number: {passed}/{len(cases)} passed")


def test_spoken_amount():
    """Test currency amount conversion."""
    cases = [
        ("₹12,000/- per annum", "hinglish"),
        ("Rs. 2,50,000", "hinglish"),
        ("₹75,000 one time", "hinglish"),
        ("5 lakh", "hinglish"),
        ("Up to ₹10,000 per month", "hinglish"),
    ]
    print("\nspoken_amount results:")
    for text, lang in cases:
        result = spoken_amount(text, lang)
        print(f"  '{text}' → '{result}'")


def test_simplify_eligibility():
    """Test eligibility text simplification."""
    cases = [
        "Education Level: Post Matric; Marks Criteria: 60%; Category: SC, ST; Income Limit: 250000",
        "Must be a student of Class 9-12. Family income should not exceed 2.5 lakh per annum.",
        "Applicant must be an Indian citizen. Age between 18-35 years.",
    ]
    print("\nsimplify_eligibility results:")
    for text in cases:
        result = simplify_eligibility(text, "hinglish")
        print(f"  INPUT:  '{text[:80]}...'")
        print(f"  OUTPUT: '{result}'")
        print()


def test_voice_friendly_name():
    """Test scheme name shortening."""
    cases = [
        ("Post Matric Scholarship for SC Students under CSS", "Post Matric Scholarship for SC Students"),
        ("Pradhan Mantri Kisan Samman Nidhi Yojana", "PM Kisan Samman Nidhi Yojana"),
        ("PM Mudra Yojana", "PM Mudra Yojana"),
    ]
    print("voice_friendly_name results:")
    for name, _ in cases:
        result = voice_friendly_name(name)
        print(f"  '{name}' → '{result}'")


def test_normalize_for_voice():
    """Test master normalization."""
    cases = [
        "Apply at https://scholarships.gov.in before 31st Dec. Amount: ₹25,000/year.",
        "Benefit: **₹12,000** per annum for _eligible_ students",
        "Education Level: Post Matric; Income: 250000; Category: SC",
    ]
    print("\nnormalize_for_voice results:")
    for text in cases:
        result = normalize_for_voice(text)
        print(f"  INPUT:  '{text}'")
        print(f"  OUTPUT: '{result}'")
        print()


if __name__ == "__main__":
    test_spoken_number()
    test_spoken_amount()
    test_simplify_eligibility()
    test_voice_friendly_name()
    test_normalize_for_voice()
    print("\n✅ All voice normalizer tests complete!")
