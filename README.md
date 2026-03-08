# 🏛️ Government Schemes Voice Assistant

> AI-powered assistant helping Indian citizens discover government schemes through natural language search and voice conversations in Hindi, English, and Hinglish.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **🔍 AI Scheme Search** — Natural language search over 3,400+ government schemes (FAISS + BM25 + Cross-Encoder)
- **🎤 Voice Conversations** — Speak in Hindi, English, or Hinglish with AI counselor "Vidya"
- **⚡ Ultra-Low Latency** — Groq LPU for fast STT/LLM inference (<500ms first audio chunk)
- **📱 Modern React UI** — Discover page, Dashboard, and VidyaHub voice interface
- **📞 Phone Support** — Twilio integration for phone-based scheme assistance
- **🤖 Smart Ranking** — Reciprocal Rank Fusion + Cross-Encoder re-ranking for best results

## 🎯 Who It Helps

| Citizen | Schemes |
|---------|---------|
| **Students** | PM Scholarship, Post Matric SC/ST/OBC, AICTE Pragati, INSPIRE |
| **Farmers** | PM-KISAN, PM Fasal Bima Yojana, crop insurance |
| **Fishermen** | PMMSY, Blue Revolution |
| **MSME / Businesses** | Mudra Loans, MSME Registration |
| **Divyang (Disabled)** | IGNDPS, ADIP Scheme, National Trust |
| **Women** | PM Kaushal Vikas, Mahila Shakti Kendra |
| **Senior Citizens** | Pension schemes, elder welfare |

## 🏗️ Architecture

```
Browser ──HTTP──> aiohttp Backend (port 8080)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  RAG Pipeline    Voice Agent    Telephony
        │              │              │
  FAISS + BM25    Groq Whisper    Twilio
  Cross-Encoder   Llama 3.3 70B   (Calls)
                  Cartesia TTS
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Clone & Setup

```bash
git clone https://github.com/Aryanj33/govt-schemes-assistant.git
cd govt-schemes-assistant
```

### 2. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# .\venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Configure API Keys

Create `backend/.env`:

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
CARTESIA_API_KEY=your_cartesia_key
ELEVENLABS_API_KEY=your_elevenlabs_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=your_secret
PORT=8080
```

See [docs/API_KEYS.md](docs/API_KEYS.md) for how to get each key.

### 4. Start Backend

```bash
cd backend
source venv/bin/activate
python main.py
```

Backend runs at **http://localhost:8080**

### 5. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:5173** — open it in your browser!

## 🎯 Try These Searches

- `Engineering scholarships for SC students in Maharashtra`
- `PM-KISAN mein kitna paisa milta hai?`
- `Mudra loan for new business`
- `Pension schemes for senior citizens`
- `Financial help for pregnant women`

## 📁 Project Structure

```
govt-schemes-assistant/
├── backend/
│   ├── agent/
│   │   ├── livekit_agent.py       # HTTP server + all API endpoints
│   │   ├── conversation_handler.py # Vidya AI persona + LLM
│   │   └── voice_pipeline.py      # STT → LLM → TTS pipeline
│   ├── rag/
│   │   ├── scholarship_rag.py     # Hybrid search (FAISS + BM25 + Cross-Encoder)
│   │   ├── embeddings.py          # Sentence Transformer embeddings
│   │   └── vectorstore.py         # FAISS vector store
│   ├── telephony/
│   │   └── twilio_handler.py      # Phone call handling
│   ├── utils/                     # Config, logging
│   ├── data/                      # Scheme JSONs + FAISS index
│   ├── make_call.py               # Script to initiate outbound calls
│   ├── main.py                    # Entry point
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── DiscoverPage.tsx   # Scheme search UI (calls POST /search)
│       │   ├── Dashboard.tsx      # User dashboard
│       │   └── VidyaHub.tsx       # Voice conversation interface
│       └── components/            # Reusable UI components (shadcn/ui)
│
└── docs/
    ├── API_KEYS.md
    └── DEMO.md
```

## 📊 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + TypeScript + Vite | Modern SPA |
| **UI Components** | shadcn/ui + Framer Motion | Polished, animated UI |
| **Backend** | Python + aiohttp | Async HTTP server |
| **STT** | Groq Whisper | Fast, Indian-accent-aware transcription |
| **LLM** | Groq Llama 3.3 70B | 300+ tok/s responses |
| **TTS** | Cartesia Sonic / ElevenLabs / Edge TTS | Natural voice output |
| **RAG** | FAISS + BM25 + Cross-Encoder | Hybrid semantic + keyword search |
| **Voice** | LiveKit + Silero VAD | Real-time voice streaming |
| **Telephony** | Twilio | Outbound voice calls |

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search` | Search schemes by natural language query |
| `POST` | `/text` | Send text, get AI text response |
| `POST` | `/audio` | Send audio bytes, get voice response |
| `POST` | `/audio/stream` | Streaming voice response (<500ms first chunk) |
| `POST` | `/token` | Get LiveKit access token |
| `GET`  | `/health` | Health check + scheme count |
| `POST` | `/reset` | Reset conversation session |

## 📞 Making a Phone Call

```bash
cd backend
source venv/bin/activate
python -c "from make_call import make_call; make_call('+91XXXXXXXXXX')"
```

> Requires Twilio credentials in `.env` and ngrok running for the webhook.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
