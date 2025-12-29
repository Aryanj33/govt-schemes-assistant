# Setup Guide - Scholarship Voice Assistant

Complete installation guide for the AI-powered voice assistant.

## Prerequisites

- **Python 3.10+** - [Download](https://python.org/downloads)
- **Node.js 18+** (optional, for advanced frontend serving)
- **Git** - [Download](https://git-scm.com)
- **FFmpeg** (for audio processing) - [Download](https://ffmpeg.org/download.html)

### Windows FFmpeg Setup
```powershell
# Using winget
winget install ffmpeg

# Or download from https://ffmpeg.org and add to PATH
```

## Quick Start (5 minutes)

### Step 1: Clone and Navigate
```bash
cd c:\Users\alexu\Desktop\PBL\Voice
```

### Step 2: Create Virtual Environment
```bash
cd backend
python -m venv venv

# Windows activation
.\venv\Scripts\activate

# Linux/Mac activation
# source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Copy the example config:
```bash
copy ..\config\.env.example ..\config\.env
```

Edit `config/.env` with your API keys (see [API_KEYS.md](API_KEYS.md) for how to get them):

```env
# Required - Get from https://console.groq.com
GROQ_API_KEY=gsk_xxxxxxxxxxxx

# Optional - For Gemini fallback
GOOGLE_API_KEY=AIzaxxxxxxxxxxxx
```

### Step 5: Build the RAG Index

Run once to create the scholarship search index:
```bash
python -c "from rag import get_scholarship_rag; rag = get_scholarship_rag(); rag.build_index()"
```

### Step 6: Start the Backend
```bash
python main.py
```

You should see:
```
🚀 Scholarship Voice Assistant - Logging initialized
📋 CONFIGURATION STATUS
✅ Groq STT/LLM: ✅ Configured
🌐 Simple server running on http://0.0.0.0:8080
```

### Step 7: Start the Frontend

In a new terminal:
```bash
cd c:\Users\alexu\Desktop\PBL\Voice\frontend
python -m http.server 3000
```

### Step 8: Open in Browser

Navigate to: **http://localhost:3000**

Click the microphone and start speaking!

---

## Verification Checklist

- [ ] Backend shows "Connected" with scholarship count
- [ ] Frontend loads with saffron/green theme
- [ ] Microphone permission granted
- [ ] Health check works: `curl http://localhost:8080/health`

## Troubleshooting

### "GROQ_API_KEY not set"
- Ensure `.env` file is in the `config/` directory
- Check the key starts with `gsk_`

### "Microphone access denied"
- Allow microphone in browser settings
- Use HTTPS in production (localhost is allowed)

### "Connection refused"
- Ensure backend is running on port 8080
- Check firewall settings

### Audio not playing
- Check browser allows autoplay
- Verify TTS is configured (Bhashini or Google)

---

## Production Deployment

For hackathon demo on a local network:

1. Find your IP: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
2. Update backend to bind to `0.0.0.0`
3. Update frontend `apiUrl` to your IP
4. Share `http://YOUR_IP:3000` with judges
