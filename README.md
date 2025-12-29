# 🎓 Scholarship Voice Assistant

> AI-powered voice assistant helping Indian students discover scholarships through natural conversations in Hindi, English, and Hinglish.

![Demo](https://img.shields.io/badge/Demo-Live-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **🎤 Natural Voice Conversations** - Speak in Hindi, English, or Hinglish
- **🔍 Smart Scholarship Search** - RAG-powered semantic search across 20+ scholarships
- **🤖 AI Counselor "Vidya"** - Helpful persona that understands Indian education context
- **⚡ Ultra-Low Latency** - < 500ms response time with Groq LPU
- **🌐 Zero Cost** - Built entirely on free-tier APIs

## 🏗️ Architecture

```
User (Browser) ──WebRTC──> LiveKit Server
                              │
                              ▼
                      LiveKit Agent (Python)
                              │
        ┌───────────┬─────────┼─────────┬───────────┐
        ▼           ▼         ▼         ▼           ▼
    Silero VAD  Groq Whisper  FAISS   Groq Llama  Bhashini
    (Voice      (STT)        (RAG)    (LLM)       (TTS)
     Detection)
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
cd c:\Users\alexu\Desktop\PBL\Voice
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
copy ..\config\.env.example ..\config\.env
# Edit .env with your GROQ_API_KEY
```

See [API_KEYS.md](docs/API_KEYS.md) for detailed instructions.

### 3. Start Backend

```bash
python main.py
```

### 4. Start Frontend

```bash
cd ..\frontend
python -m http.server 3000
```

Open **http://localhost:3000** and start talking!

## 🎯 Try These Queries

- "Mujhe engineering scholarship chahiye"
- "I'm from SC category, UP state"
- "What scholarships for 85% marks in 12th?"
- "AICTE Pragati scholarship ke baare mein batao"

## 📁 Project Structure

```
scholarship-voice-assistant/
├── backend/
│   ├── agent/           # Voice pipeline & orchestration
│   ├── rag/             # FAISS search & embeddings
│   └── utils/           # Config & logging
├── frontend/
│   ├── css/             # Modern dark theme
│   └── js/              # Audio recording & playback
├── data/
│   └── processed/       # 20+ Indian scholarships
├── config/
│   └── prompts.py       # Hinglish counselor persona
└── docs/
    ├── SETUP.md
    ├── API_KEYS.md
    └── DEMO.md
```

## 📊 Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| STT | Groq Whisper | 216x faster, Indian accent support |
| LLM | Groq Llama 3.3 70B | 300+ tokens/sec, free tier |
| TTS | Bhashini / Google | Natural Hindi voices |
| RAG | FAISS + sentence-transformers | Local, fast, free |
| Frontend | Vanilla JS | No build step, works offline |

## 🎓 Scholarships Included

- PM Scholarship for CAPF
- Post Matric for SC/ST/OBC
- AICTE Pragati & Saksham
- INSPIRE Scholarship
- NTSE Fellowship
- State-specific (Maharashtra, UP)
- And 14+ more...

## 🏆 Hackathon Demo

See [DEMO.md](docs/DEMO.md) for:
- 3 memorized demo conversations
- Judging criteria alignment
- Failure recovery strategies
- Q&A preparation

## 📜 License

MIT License - Free to use and modify.

## 👥 Team

Built with ❤️ for Indian Students at Hackathon 2025

---

**🌟 Star this repo if it helped you find a scholarship!**
