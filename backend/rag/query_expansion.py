"""
Query Expansion for Hindi/Hinglish Scholarship Search
====================================================
Expands user queries with synonyms, related terms, and Hindi variations.
"""

from typing import List, Set, Dict, Optional
import re


class QueryExpander:
    """
    Expands search queries with Hindi synonyms and related terms.
    Improves recall for voice-based queries with transcription variations.
    """
    
    def __init__(self):
        """Initialize with comprehensive synonym mappings."""
        
        # Core scholarship/scheme terms
        self.scheme_synonyms = {
            'scholarship': ['छात्रवृत्ति', 'स्कॉलरशिप', 'स्कालरशिप', 'student aid', 'financial aid', 'education grant'],
            'छात्रवृत्ति': ['scholarship', 'स्कॉलरशिप', 'student aid', 'education support'],
            'scheme': ['योजना', 'yojana', 'program', 'initiative'],
            'योजना': ['scheme', 'yojana', 'program', 'plan'],
            'yojana': ['योजना', 'scheme', 'program'],
        }
        
        # Education levels and courses
        self.education_synonyms = {
            'engineering': ['बीटेक', 'बी.टेक', 'btech', 'b.tech', 'इंजीनियरिंग', 'technical', 'technology'],
            'btech': ['बीटेक', 'बी.टेक', 'engineering', 'इंजीनियरिंग', 'b.tech', 'bachelor technology'],
            'बीटेक': ['btech', 'b.tech', 'engineering', 'इंजीनियरिंग'],
            'medical': ['एमबीबीएस', 'mbbs', 'doctor', 'medicine', 'चिकित्सा', 'डॉक्टर'],
            'mbbs': ['medical', 'एमबीबीएस', 'doctor', 'medicine', 'चिकित्सा'],
            'graduation': ['स्नातक', 'graduate', 'degree', 'bachelor'],
            'post graduation': ['स्नातकोत्तर', 'postgraduate', 'masters', 'pg'],
            '10th': ['दसवीं', 'tenth', 'matriculation', 'high school'],
            '12th': ['बारहवीं', 'twelfth', 'intermediate', 'senior secondary'],
        }
        
        # Categories and demographics
        self.category_synonyms = {
            'sc': ['scheduled caste', 'अनुसूचित जाति', 'dalit'],
            'st': ['scheduled tribe', 'अनुसूचित जनजाति', 'tribal'],
            'obc': ['other backward class', 'अन्य पिछड़ा वर्ग', 'backward class'],
            'general': ['सामान्य', 'unreserved', 'open category'],
            'minority': ['अल्पसंख्यक', 'muslim', 'christian', 'sikh', 'buddhist', 'parsi'],
            'women': ['महिला', 'girl', 'female', 'लड़की', 'बेटी'],
            'girls': ['लड़की', 'महिला', 'female', 'women', 'बेटी'],
        }
        
        # States (Hindi/English variations)
        self.state_synonyms = {
            'uttar pradesh': ['यूपी', 'up', 'उत्तर प्रदेश'],
            'up': ['uttar pradesh', 'यूपी', 'उत्तर प्रदेश'],
            'यूपी': ['up', 'uttar pradesh', 'उत्तर प्रदेश'],
            'maharashtra': ['महाराष्ट्र', 'mh'],
            'bihar': ['बिहार', 'br'],
            'west bengal': ['पश्चिम बंगाल', 'wb', 'bengal'],
            'rajasthan': ['राजस्थान', 'rj'],
            'gujarat': ['गुजरात', 'gj'],
            'karnataka': ['कर्नाटक', 'ka'],
            'tamil nadu': ['तमिल नाडु', 'tn'],
            'kerala': ['केरल', 'kl'],
            'punjab': ['पंजाब', 'pb'],
            'haryana': ['हरियाणा', 'hr'],
            'delhi': ['दिल्ली', 'dl', 'new delhi'],
            'madhya pradesh': ['मध्य प्रदेश', 'mp'],
        }
        
        # Farmer/Agriculture terms
        self.farmer_synonyms = {
            'farmer': ['किसान', 'kisan', 'agriculture', 'खेती', 'farming'],
            'किसान': ['farmer', 'kisan', 'agriculture', 'खेती'],
            'kisan': ['किसान', 'farmer', 'agriculture', 'खेती'],
            'agriculture': ['कृषि', 'खेती', 'farming', 'किसान', 'kisan'],
            'farming': ['खेती', 'कृषि', 'agriculture', 'किसान'],
        }
        
        # Business/Loan terms
        self.business_synonyms = {
            'business': ['व्यापार', 'व्यवसाय', 'धंधा', 'काम', 'udyam', 'enterprise'],
            'loan': ['ऋण', 'कर्ज', 'लोन', 'credit', 'finance'],
            'startup': ['स्टार्टअप', 'new business', 'नया व्यापार'],
            'entrepreneur': ['उद्यमी', 'व्यापारी', 'businessman'],
        }
        
        # Common Hindi words that need expansion
        self.hindi_synonyms = {
            'मदद': ['help', 'assistance', 'support', 'aid'],
            'चाहिए': ['need', 'want', 'require', 'chahiye'],
            'पैसा': ['money', 'paisa', 'rupees', 'amount', 'fund'],
            'सहायता': ['help', 'assistance', 'support', 'aid'],
            'आवेदन': ['application', 'apply', 'form'],
            'पात्रता': ['eligibility', 'qualification', 'criteria'],
        }
        
        # Combine all synonym dictionaries
        self.all_synonyms = {}
        for syn_dict in [
            self.scheme_synonyms, self.education_synonyms, self.category_synonyms,
            self.state_synonyms, self.farmer_synonyms, self.business_synonyms, self.hindi_synonyms
        ]:
            self.all_synonyms.update(syn_dict)
    
    def expand_query(self, query: str, max_expansions: int = 3) -> str:
        """
        Expand a query with synonyms and related terms.
        
        Args:
            query: Original search query
            max_expansions: Maximum number of synonym expansions per term
            
        Returns:
            Expanded query string with synonyms
        """
        if not query or len(query.strip()) < 2:
            return query
        
        # Normalize query
        query_lower = query.lower().strip()
        words = re.findall(r'\b\w+\b', query_lower)
        
        expanded_terms = set([query_lower])  # Always include original
        
        # Expand individual words
        for word in words:
            if word in self.all_synonyms:
                synonyms = self.all_synonyms[word][:max_expansions]
                expanded_terms.update(synonyms)
        
        # Expand common phrases
        expanded_terms.update(self._expand_phrases(query_lower))
        
        # Join expanded terms
        return " ".join(expanded_terms)
    
    def _expand_phrases(self, query: str) -> Set[str]:
        """Expand common multi-word phrases."""
        expansions = set()
        
        # Common phrase patterns
        phrase_patterns = {
            r'\b(engineering|btech|b\.tech)\s+(scholarship|छात्रवृत्ति)\b': 
                ['इंजीनियरिंग छात्रवृत्ति', 'technical scholarship', 'बीटेक स्कॉलरशिप'],
            
            r'\b(medical|mbbs)\s+(scholarship|छात्रवृत्ति)\b':
                ['चिकित्सा छात्रवृत्ति', 'doctor scholarship', 'एमबीबीएस स्कॉलरशिप'],
            
            r'\b(sc|st|obc)\s+(scholarship|छात्रवृत्ति)\b':
                ['आरक्षित वर्ग छात्रवृत्ति', 'reserved category scholarship'],
            
            r'\b(girl|girls|women|महिला)\s+(scholarship|छात्रवृत्ति)\b':
                ['महिला छात्रवृत्ति', 'बेटी बचाओ', 'girl child scholarship'],
            
            r'\b(farmer|किसान|kisan)\s+(scheme|योजना)\b':
                ['किसान योजना', 'कृषि योजना', 'agriculture scheme'],
            
            r'\b(business|व्यापार)\s+(loan|ऋण)\b':
                ['व्यापारिक ऋण', 'उद्यम लोन', 'startup loan'],
        }
        
        for pattern, synonyms in phrase_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                expansions.update(synonyms)
        
        return expansions
    
    def expand_for_voice_search(self, query: str) -> str:
        """
        Special expansion for voice queries with transcription errors.
        
        Args:
            query: Voice-transcribed query (may have errors)
            
        Returns:
            Expanded query optimized for voice search
        """
        # Common voice transcription corrections
        voice_corrections = {
            'स्कॉलरशिप': 'scholarship छात्रवृत्ति',
            'स्कालरशिप': 'scholarship छात्रवृत्ति',
            'इंजीनियरिंग': 'engineering btech',
            'एमबीबीएस': 'mbbs medical doctor',
            'यूपी': 'uttar pradesh up',
            'बीटेक': 'btech engineering',
        }
        
        expanded = query
        for incorrect, correct in voice_corrections.items():
            if incorrect in query:
                expanded += f" {correct}"
        
        # Apply regular expansion
        return self.expand_query(expanded, max_expansions=2)
    
    def get_related_terms(self, term: str) -> List[str]:
        """Get all related terms for a specific word."""
        term_lower = term.lower()
        if term_lower in self.all_synonyms:
            return self.all_synonyms[term_lower]
        return []


# Singleton instance
_query_expander: Optional[QueryExpander] = None

def get_query_expander() -> QueryExpander:
    """Get the global query expander instance."""
    global _query_expander
    if _query_expander is None:
        _query_expander = QueryExpander()
    return _query_expander