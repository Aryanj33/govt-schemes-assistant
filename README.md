# 🏛️ Government Schemes Voice Assistant

> AI-powered voice assistant helping Indian citizens discover government schemes through natural conversations in Hindi, English, and Hinglish.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **🎤 Natural Voice Conversations** - Speak in Hindi, English, or Hinglish
- **🔍 Smart Scheme Search** - Hybrid RAG (FAISS + BM25 + Cross-Encoder) for semantic search
- **🤖 AI Counselor "Vidya"** - Bureaucratic insider persona with practical government advice
- **⚡ Ultra-Low Latency** - Groq LPU for fast STT/LLM inference
- **📱 Modern React UI** - Dashboard, scheme discovery, and voice interaction hub
- **📞 Telephony Support** - Twilio integration for phone-based assistance
- **🏥 Health Camps** - Nearby health camp discovery with geolocation

## 🎯 Who It Helps

- **Students** - Scholarships, hostels, books
- **Farmers/Kisan** - PM-KISAN, crop insurance
- **Fishermen** - PMMSY, boat loans
- **MSME/Businessmen** - Mudra loans, MSME schemes
- **Divyang (Disabled)** - Disability pensions, ADIP scheme
- **Women, Senior Citizens, Laborers** - Various welfare schemes

## 🏗️ Architecture

```
User (Browser) ──WebSocket──> FastAPI Backend
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
    LiveKit Agent              RAG Pipeline              Telephony
          │                         │                         │
    ┌─────┴─────┐           ┌───────┴───────┐                 │
    ▼           ▼           ▼       ▼       ▼                 ▼
  Silero     Groq        FAISS   BM25    Cross-          Twilio
   VAD      Whisper     (Vector) (Kwd)   Encoder         (SMS/Call)
                                    
    ▼                               
  Groq Llama 3.3 70B ──> Edge TTS / Bhashini (TTS)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Clone & Setup Backend

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
# Edit .env with your API keys (GROQ_API_KEY, LIVEKIT_*, TWILIO_*)
```

See [API_KEYS.md](docs/API_KEYS.md) for detailed instructions.

### 3. Start Backend

```bash
python main.py
```

### 4. Start Frontend

```bash
cd ..\frontend
npm install
npm run dev
```

Open **http://localhost:5173** and start exploring!

## 🎯 Try These Queries

- "Mujhe engineering scholarship chahiye"
- "PM-KISAN mein kitna paisa milta hai?"
- "Main SC category se hoon, UP state"
- "Divyang pension ke baare mein batao"
- "MSME loan kaise milega?"

## 📁 Project Structure

```
govt-schemes-assistant/
├── backend/
│   ├── agent/              # LiveKit voice pipeline & conversation handler
│   │   ├── livekit_agent.py
│   │   ├── conversation_handler.py
│   │   └── voice_pipeline.py
│   ├── rag/                # Hybrid RAG (FAISS + BM25 + Cross-Encoder)
│   │   ├── scholarship_rag.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   ├── data/               # Scrapers & preprocessors
│   │   ├── advanced_scraper.py
│   │   ├── preprocessor.py
│   │   └── health_camps.json
│   ├── database/           # SQLite call records
│   ├── telephony/          # Twilio SMS & voice calls
│   └── utils/              # Config, logging, geocoding
│
├── frontend/               # React + TypeScript + Vite
│   ├── src/
│   │   ├── pages/          # Dashboard, Discover, Login, Signup, VidyaHub
│   │   ├── components/     # Header, SchemeCard, VidyaOrb, UI (shadcn)
│   │   ├── lib/            # API client, utilities
│   │   └── contexts/       # React contexts
│   └── package.json
│
├── config/
│   ├── .env                # API keys & configuration
│   └── prompts.py          # Vidya persona & LLM prompts
│
├── data/
│   ├── processed/          # Scheme JSONs
│   └── embeddings/         # FAISS & BM25 indices
│
└── docs/
    ├── SETUP.md
    ├── API_KEYS.md
    └── DEMO.md
```

## 📊 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | React 18 + TypeScript + Vite | Modern SPA with shadcn/ui |
| **Styling** | TailwindCSS | Utility-first CSS framework |
| **Backend** | Python + FastAPI | API server & agent orchestration |
| **STT** | Groq Whisper | 216x faster, Indian accent support |
| **LLM** | Groq Llama 3.3 70B | 300+ tokens/sec, free tier |
| **TTS** | Edge TTS / Bhashini | Natural Hindi voices |
| **RAG** | FAISS + BM25 + Cross-Encoder | Hybrid semantic + keyword search |
| **Voice** | LiveKit + Silero VAD | Real-time voice streaming |
| **Telephony** | Twilio | SMS notifications & voice calls |
| **Database** | SQLite | Call records & user sessions |

## 🎓 Schemes Covered

- **Scholarships**: PM Scholarship, Post Matric SC/ST/OBC, AICTE Pragati & Saksham, INSPIRE, NTSE
- **Agriculture**: PM-KISAN, PM Fasal Bima Yojana
- **Fisheries**: PMMSY, Blue Revolution
- **MSME**: Mudra Loans, MSME Registration
- **Disability**: IGNDPS, ADIP Scheme, National Trust schemes
- **Women**: PM Kaushal Vikas Yojana, Mahila Shakti Kendra
- **State-specific**: Maharashtra, UP, and more

## 🖥️ Frontend Pages

| Page | Description |
|------|-------------|
| **Index** | Landing page with hero section |
| **Dashboard** | User profile & scheme recommendations |
| **Discover** | Browse & filter 50+ schemes with infinite scroll |
| **VidyaHub** | Voice conversation interface with Vidya |
| **Login/Signup** | User authentication |

## 📞 Telephony Features

- **SMS Notifications** - Scheme details sent via Twilio
- **Voice Calls** - Phone-based assistance (Twilio voice)
- **Call Records** - SQLite database for tracking

## 🏆 Demo

See [DEMO.md](docs/DEMO.md) for:
- Demo conversations
- Judging criteria alignment
- Failure recovery strategies
- Q&A preparation

## 📄 License

MIT License - see LICENSE file for details.
