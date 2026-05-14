# Demo Guide - Scholarship Voice Assistant

Complete guide for demonstrating the voice assistant to hackathon judges.

---

## 🎯 Pre-Demo Checklist

### 30 Minutes Before

- [ ] Laptop charged to 100%
- [ ] All API keys configured in `.env`
- [ ] Backend running: `python backend/main.py`
- [ ] Frontend accessible: `http://localhost:3000`
- [ ] Test microphone working
- [ ] Run one test conversation successfully
- [ ] Backup demo video ready (if live fails)

### Environment Check

```bash
# Test backend health
curl http://localhost:8080/health

# Expected: {"status": "healthy", "scholarships_loaded": 20}
```

---

## 🎤 3 Demo Conversations (Memorize These!)

### Demo 1: Simple Query (30 seconds)
**Shows:** Natural Hinglish, relevant results

```
YOU: "Hello, mujhe engineering scholarship chahiye"

VIDYA: "Namaste! Engineering ke liye kai scholarships hain. 
        Aap kis state se hain? Aur general category ya reserved?"

YOU: "Maharashtra, OBC category"

VIDYA: "Perfect! Aapke liye Post Matric Scholarship for OBC students 
        suitable hai - ₹20,000 per year. Iske baare mein detail chahiye?"
```

### Demo 2: Multi-Filter Search (45 seconds)
**Shows:** Complex filtering, accurate data

```
YOU: "I'm an SC student from UP, scored 85% in 12th, 
      want to study engineering"

VIDYA: "Excellent marks! UP se SC students ke liye main 3 options 
        suggest karungi:
        1. UP Post Matric Scholarship - full tuition fees
        2. Central Sector Scheme - ₹12,000 per year
        3. INSPIRE Scholarship agar science mein interest hai
        
        Kis scholarship ke details chahiye?"

YOU: "INSPIRE ke baare mein batao"

VIDYA: "INSPIRE Scholarship DST ki taraf se hai. Top 1% board ranks 
        ya JEE qualifiers ke liye. Amount ₹80,000 per year for 5 years.
        Apply online at online-inspire.gov.in"
```

### Demo 3: Conversation Context (60 seconds)
**Shows:** Memory, clarifying questions, no hallucination

```
YOU: "Scholarship chahiye"

VIDYA: "Zaroor! Pehle batayein - aap kis level pe padh rahe hain? 
        12th, graduation, ya higher studies?"

YOU: "12th complete kiya, 60% marks"

VIDYA: "Okay! Kya aap general category mein hain ya SC/ST/OBC?"

YOU: "General category"

VIDYA: "General category mein 60% ke saath options limited hain, 
        but Central Sector Scheme check kar sakte hain agar 
        80th percentile aaye the boards mein. 
        Koi specific field mein interest hai - engineering, medical?"

YOU: "Medical"

VIDYA: "Medical ke liye NEET qualify karna padega. AIIMS scholarships 
        aur state medical colleges mein fee waivers milte hain merit pe.
        Kya aapne NEET attempt kiya hai?"
```

---

## 💡 Key Demo Points to Highlight

| Feature | What to Say |
|---------|-------------|
| **Latency** | "Notice response comes in under 500ms - that's Groq LPU optimization" |
| **Accuracy** | "It only mentions real scholarships from our database - zero hallucination" |
| **Hinglish** | "Naturally switches between Hindi and English based on how user speaks" |
| **Context** | "Remembers I said Maharashtra earlier - no need to repeat" |
| **Inclusivity** | "Designed for first-generation college students who may not speak fluent English" |

---

## 🛡️ Handling Demo Failures

### If Backend Crashes
```
"Let me restart the server quickly..."
python backend/main.py
```

### If STT Fails
Use text fallback:
```bash
curl -X POST http://localhost:8080/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I need scholarship for engineering"}'
```

### If Internet Fails
"Our RAG system works offline with cached scholarships. 
Only the TTS requires internet, but I can show the text responses."

### Complete Backup
Play pre-recorded 2-minute demo video showing full conversation.

---

## 📊 Judging Criteria Alignment

| Criteria | Our Solution |
|----------|--------------|
| **Innovation** | First Hinglish voice assistant for scholarships |
| **Technical Complexity** | RAG + STT + LLM + TTS pipeline |
| **Impact** | Helps rural students find scholarships without English literacy |
| **Feasibility** | 100% free tier, works on any laptop |
| **Demo Quality** | Real-time voice conversation, not slides |

---

## 🎬 Recording Backup Demo

If you want to pre-record:

1. Use OBS Studio (free)
2. Record desktop with audio
3. Run through all 3 demo conversations
4. Keep under 3 minutes
5. Save as MP4
6. Test playback before demo day

---

## ⚡ Speed Tips for Judges

- Start with the WOW moment: Voice conversation in Hinglish
- Keep each demo interaction short (2-3 exchanges max)
- Highlight latency: "Notice how fast that response came"
- End with impact: "Imagine a student in rural Bihar using this"

---

## 📝 Q&A Preparation

**Q: How do you handle hallucinations?**
> "Our RAG only allows responses from indexed scholarships. The LLM is constrained to never invent names or deadlines."

**Q: What about data privacy?**
> "Audio is processed in real-time through Groq's API - we don't store any voice recordings. Session data is cleared after conversation ends."

**Q: How scalable is this?**
> "Free tier handles 14,400 requests/day. For production, we'd upgrade to Groq enterprise. The architecture is stateless and horizontally scalable."

**Q: Why not use ChatGPT?**
> "Groq gives us 300+ tokens/second - that's what enables the natural conversation flow. GPT-4 would have 2-3 second delays."

---

Good luck! 🎓✨
