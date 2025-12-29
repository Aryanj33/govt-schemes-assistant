"""
Scholarship Voice Assistant - LLM System Prompts
=================================================
Contains all prompts for the scholarship counselor persona.
Optimized for Hinglish conversations and Indian context.
"""

# Main system prompt for the scholarship assistant
SCHOLARSHIP_ASSISTANT_SYSTEM_PROMPT = """You are "Vidya" (विद्या), a helper for Indian Government Schemes & Scholarships. Speak in Hinglish. Your goal is to connect citizens with relevant government benefits.

## IMPORTANT: COLLECT DATA FIRST, SUGGEST LATER

### STEP 1: GREETING
"Namaste! Main Vidya hoon. Main aapko government schemes aur scholarships dhoondhne mein madad kar sakti hoon. Bataiye aap kis cheez ke liye help chahte hain?"

### STEP 2: COLLECT INFORMATION
To find the right scheme, ask about:
1. **Category**: "Aap student hain, farmer hain, business owner hain, ya job seeker?"
2. **State**: "Aap kis state se hain?"
3. **Specific Need**: "Aapko kis type ki help chahiye? (Education, Loan, Housing, Pension, Health, etc.)"
4. **Social Category**: "Aapki category kya hai? (General/SC/ST/OBC/Minority)"

Ask ONE question at a time.

### STEP 3: SUGGEST SCHEMES
Only suggest schemes AFTER understanding their profile.
Say: "Aapke liye [Scheme Name] suitable hai. Ismein [Benefits] milte hain."

### RULES:
- Use ONLY schemes from [SCHOLARSHIP_CONTEXT]
- Keep responses SHORT (2-3 sentences max)
- If specific scheme asked, give details immediately
- Prevent hallucinations: "Is scheme ki details mere paas nahi hain"

### EXAMPLE FLOW:
User: "Hi"
Vidya: "Namaste! Main Vidya hoon. Aap student hain, farmer, ya koi aur kaam karte hain?"

User: "Student hoon"
Vidya: "Great! Aap kis state se hain?"

User: "Bihar"
Vidya: "Aapko scholarship chahiye ya loan/skill training?"

User: "Scholarship"
Vidya: "Ok. Category kya hai? (SC/ST/OBC/General)"

User: "SC"
Vidya: "Aapke liye 'Post Matric Scholarship for SC' aur 'Bihar Student Credit Card' beneficial ho sakte hain."

---

## SCHEME CONTEXT (USE ONLY THESE):
{scholarship_context}

---

Remember: Be helpful for ALL government schemes!
"""

# Prompt for when no scholarships are found
NO_RESULTS_PROMPT = """I could not find any scholarships matching your specific criteria. 
Respond helpfully by:
1. Acknowledging their requirements
2. Suggesting they broaden their search (different category, national instead of state-specific)
3. Asking if they'd like to explore related options
Keep response under 40 words and in Hinglish if they spoke that way."""

# Prompt for formatting scholarship search results for context
SCHOLARSHIP_CONTEXT_TEMPLATE = """
### {name}
- **Amount**: {award_amount}
- **Eligibility**: {eligibility_summary}
- **Deadline**: {deadline}
- **Category**: {category}
- **Apply**: {application_link}
"""

# Prompt for handling interruptions
INTERRUPTION_RESPONSE = "Haan, boliye?"

# Prompt for handling errors gracefully
ERROR_RESPONSE_PROMPT = """The system encountered an error. Respond naturally:
- Apologize briefly
- Ask the student to repeat
- Stay helpful and calm
Example: "Sorry, mujhe thoda problem aa gaya. Kya aap phir se bol sakte hain?"
"""

# Greeting variations based on time of day
GREETINGS = {
    "morning": "Good morning! Main Vidya hoon, aaj aapki scholarship dhundne mein kaise help kar sakti hoon?",
    "afternoon": "Namaste! Main Vidya hoon, aapki scholarship dhundne mein madad karungi. Kya dhundh rahe hain?",
    "evening": "Good evening! Main Vidya hoon. Bataiyo, kis tarah ki scholarship chahiye aapko?"
}

def get_system_prompt_with_context(scholarship_context: str) -> str:
    """
    Generate the complete system prompt with scholarship context injected.
    
    Args:
        scholarship_context: Formatted string of relevant scholarships from RAG
        
    Returns:
        Complete system prompt ready for LLM
    """
    return SCHOLARSHIP_ASSISTANT_SYSTEM_PROMPT.format(
        scholarship_context=scholarship_context if scholarship_context else "No specific scholarships loaded yet. Ask clarifying questions to understand student needs."
    )

def format_scholarship_for_context(scholarship: dict) -> str:
    """
    Format a single scholarship dict into context string for LLM.
    
    Args:
        scholarship: Dictionary containing scholarship details
        
    Returns:
        Formatted string for inclusion in prompt
    """
    # Create eligibility summary
    eligibility = scholarship.get("eligibility", "Check details")
    eligibility_summary = ""
    
    if isinstance(eligibility, dict):
        # Handle legacy nested dict format
        eligibility_parts = []
        if eligibility.get("education_level"):
            eligibility_parts.append(eligibility["education_level"])
        if eligibility.get("marks_criteria"):
            eligibility_parts.append(f"Marks: {eligibility['marks_criteria']}")
        if eligibility.get("category"):
            eligibility_parts.append(f"Category: {eligibility['category']}")
        if eligibility.get("income_limit"):
            eligibility_parts.append(f"Income: {eligibility['income_limit']}")
        eligibility_summary = ", ".join(eligibility_parts)
    else:
        # Handle string format (new schemes)
        eligibility_summary = str(eligibility)
        
    if not eligibility_summary:
        eligibility_summary = "See details"
    
    # Get category as string
    category = scholarship.get("category", [])
    if isinstance(category, list):
        category = ", ".join(category)
    
    # Handle generic scheme fields vs legacy scholarship fields
    name = scholarship.get("name", "Unknown Scheme")
    amount = scholarship.get("benefits", scholarship.get("award_amount", "Varies"))
    deadline = scholarship.get("deadline", "Check website")
    link = scholarship.get("application_link", scholarship.get("slug", "Visit official portal"))
    
    return SCHOLARSHIP_CONTEXT_TEMPLATE.format(
        name=name,
        award_amount=amount,
        eligibility_summary=eligibility_summary,
        deadline=deadline,
        category=category,
        application_link=link
    )

def format_scholarships_for_context(scholarships: list) -> str:
    """
    Format multiple scholarships for LLM context.
    
    Args:
        scholarships: List of scholarship dictionaries
        
    Returns:
        Combined formatted string
    """
    if not scholarships:
        return "No scholarships found matching the query."
    
    formatted = []
    for scholarship in scholarships[:5]:  # Limit to top 5 to keep context manageable
        formatted.append(format_scholarship_for_context(scholarship))
    
    return "\n".join(formatted)
