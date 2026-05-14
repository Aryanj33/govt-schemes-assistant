# Government Schemes Voice Assistant (Vidya)

An AI assistant that helps Indian citizens find and understand government schemes through voice and text — in Hindi, English, or Hinglish. Built with a hybrid RAG pipeline and low-latency voice infrastructure.

---

## What it does

Finding the right government scheme is genuinely hard. There are thousands of them, spread across different ministries, with eligibility criteria buried in PDFs. Vidya tries to fix that — you describe your situation in plain language (or just speak), and it finds what applies to you.

A few things it handles:
- Searching 3,400+ schemes using a combination of semantic search (FAISS) and keyword search (BM25), with a Cross-Encoder re-ranker on top
- Voice conversations with an AI counselor persona called "Vidya" — STT via Groq Whisper, LLM via Llama 3.3 70B, TTS via Cartesia
- Phone support through Twilio for people without smartphones
- Sub-500ms first audio chunk latency using Groq's LPU inference

---

## Who it's for

| Group | Example schemes covered |
|---|---|
| Students | PM Scholarship, Post Matric SC/ST/OBC, AICTE Pragati, INSPIRE |
| Farmers | PM-KISAN, PM Fasal Bima Yojana, crop insurance |
| Fishermen | PMMSY, Blue Revolution |
| MSME / Small businesses | Mudra Loans, MSME Registration |
| Divyang (Disabled persons) | IGNDPS, ADIP Scheme, National Trust |
| Women | PM Kaushal Vikas, Mahila Shakti Kendra |
| Senior citizens | Pension schemes, elder welfare |

---

## Architecture

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

The backend is a single aiohttp server handling all three pipelines. The frontend is a React + TypeScript SPA with three main pages: Discover (search), Dashboard, and VidyaHub (voice).

---

## Getting started

**You'll need:** Python 3.9+ and Node.js 18+

### Clone the repo

```bash
git clone https://github.com/Aryanj33/govt-schemes-assistant.git
cd govt-schemes-assistant
```

### Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# .\venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### API keys

Create a `backend/.env` file:

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

Where to get each key: [docs/API_KEYS.md](docs/API_KEYS.md)

### Run the backend

```bash
cd backend
source venv/bin/activate
python main.py
```

Runs at `http://localhost:8080`

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:5173`

---

## Example queries to try

```
Engineering scholarships for SC students in Maharashtra
PM-KISAN mein kitna paisa milta hai?
Mudra loan for new business
Pension schemes for senior citizens
Financial help for pregnant women
```

---

## Project structure

```
govt-schemes-assistant/
├── backend/
│   ├── agent/
│   │   ├── livekit_agent.py         # HTTP server + API endpoints
│   │   ├── conversation_handler.py  # Vidya persona + LLM logic
│   │   └── voice_pipeline.py        # STT → LLM → TTS pipeline
│   ├── rag/
│   │   ├── scholarship_rag.py       # Hybrid search (FAISS + BM25 + Cross-Encoder)
│   │   ├── embeddings.py            # Sentence Transformer embeddings
│   │   └── vectorstore.py           # FAISS vector store
│   ├── telephony/
│   │   └── twilio_handler.py        # Phone call handling
│   ├── utils/                       # Config, logging
│   ├── data/                        # Scheme JSONs + FAISS index
│   ├── make_call.py                 # Outbound call script
│   ├── main.py                      # Entry point
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── DiscoverPage.tsx     # Search UI
│       │   ├── Dashboard.tsx        # User dashboard
│       │   └── VidyaHub.tsx         # Voice interface
│       └── components/              # shadcn/ui components
│
└── docs/
    ├── API_KEYS.md
    └── DEMO.md
```

---

## System Architecture
<img width="1396" height="712" alt="image" src="https://github.com/user-attachments/assets/a46583d5-ddc6-4d03-9d91-2d4e5a5b5b14" />


## Tech stack

| Layer | Tech | Notes |
|---|---|---|
| Frontend | React 18 + TypeScript + Vite | |
| UI | shadcn/ui + Framer Motion | |
| Backend | Python + aiohttp | Async, single-server setup |
| STT | Groq Whisper | Works well with Indian accents |
| LLM | Groq Llama 3.3 70B | ~300 tok/s |
| TTS | Cartesia Sonic / ElevenLabs / Edge TTS | Configurable |
| Search | FAISS + BM25 + Cross-Encoder | Reciprocal Rank Fusion |
| Voice streaming | LiveKit + Silero VAD | |
| Telephony | Twilio | |

---

## API endpoints

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/search` | Natural language scheme search |
| `POST` | `/text` | Text in, text response out |
| `POST` | `/audio` | Audio in, voice response out |
| `POST` | `/audio/stream` | Streaming voice response |
| `POST` | `/token` | LiveKit access token |
| `GET` | `/health` | Health check + scheme count |
| `POST` | `/reset` | Reset conversation session |

---

## Making a phone call

```bash
cd backend
source venv/bin/activate
python -c "from make_call import make_call; make_call('+91XXXXXXXXXX')"
```

Requires Twilio credentials in `.env` and ngrok running for the webhook URL.

---

## License

MIT — see [LICENSE](LICENSE).
