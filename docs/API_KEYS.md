# API Keys Guide

How to obtain all required API keys for the Scholarship Voice Assistant.

---

## 🎤 Groq API (Required)

**Used for:** Whisper STT + Llama 3.3 70B LLM  
**Free Tier:** 14,400 requests/day  
**Signup Time:** ~2 minutes

### Steps:

1. Go to **https://console.groq.com**

2. Click **"Sign Up"** (GitHub auth is fastest)

3. Verify email if required

4. Navigate to **API Keys** in the left sidebar

5. Click **"Create API Key"**

6. Name it: `scholarship-assistant`

7. Copy the key (starts with `gsk_`)

8. Add to `.env`:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

> ⚠️ **Important:** Copy immediately! Key is only shown once.

---

## 🤖 Google AI Studio - Gemini (Recommended Fallback)

**Used for:** LLM fallback when Groq is rate-limited  
**Free Tier:** 1,500 requests/day  
**Signup Time:** ~1 minute

### Steps:

1. Go to **https://aistudio.google.com/app/apikey**

2. Sign in with Google account

3. Click **"Create API Key"**

4. Select or create a Google Cloud project

5. Copy the API key

6. Add to `.env`:
   ```
   GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

---

## 🗣️ Bhashini TTS (Optional - Best Hindi Voice)

**Used for:** Natural Hindi text-to-speech  
**Free Tier:** Free for educational projects  
**Signup Time:** ~24-48 hours (approval required)

### Steps:

1. Go to **https://bhashini.gov.in/ulca/user/register**

2. Register with:
   - College/university email (faster approval)
   - Phone number for OTP

3. In the registration form:
   - **Purpose:** Educational/Research
   - **Project:** "Scholarship Voice Assistant for Students"

4. Submit and wait for approval email

5. Once approved, log in and navigate to **API Access**

6. Create a pipeline for TTS:
   - Source Language: Hindi
   - Task Type: TTS
   
7. Copy the credentials:
   ```
   BHASHINI_USER_ID=your_user_id
   BHASHINI_API_KEY=your_api_key
   BHASHINI_PIPELINE_ID=your_pipeline_id
   ```

> 💡 **Tip:** If Bhashini approval is slow, use Google Cloud TTS as fallback.

---

## ☁️ Google Cloud TTS (Fallback)

**Used for:** TTS when Bhashini unavailable  
**Free Tier:** 1 million characters/month  
**Signup Time:** ~5 minutes

### Steps:

1. Go to **https://console.cloud.google.com**

2. Create a new project or select existing

3. Enable **Cloud Text-to-Speech API**:
   - Search "Text-to-Speech" in top bar
   - Click **Enable**

4. Create Service Account:
   - Go to **IAM & Admin > Service Accounts**
   - Click **Create Service Account**
   - Name: `scholarship-tts`
   - Grant role: **Cloud Text-to-Speech User**

5. Create JSON key:
   - Click on the service account
   - Go to **Keys** tab
   - Add Key > Create new key > JSON

6. Save the JSON file to project root

7. Add to `.env`:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
   ```

---

## 🎙️ LiveKit (Optional - For WebRTC)

**Used for:** Real-time WebRTC audio streaming  
**Free Tier:** 100 monthly active users  
**Signup Time:** ~3 minutes

### Option 1: LiveKit Cloud (Easier)

1. Go to **https://cloud.livekit.io**

2. Sign up with GitHub

3. Create a new project

4. Copy credentials from dashboard:
   ```
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=APIxxxxxxxx
   LIVEKIT_API_SECRET=xxxxxxxxxxxxx
   ```

### Option 2: Self-Hosted (for offline demo)

```bash
cd docker
docker-compose up -d
```

Use these credentials:
```
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

---

## ✅ Minimum Required for Demo

For a basic working demo, you only need:

| API | Required? | Purpose |
|-----|-----------|---------|
| Groq | ✅ Yes | STT + LLM |
| Google Gemini | ⚡ Recommended | LLM fallback |
| Bhashini TTS | 🎯 Best | Hindi voice |
| Google Cloud TTS | ⚡ Alternative | TTS fallback |
| LiveKit | ❌ Optional | WebRTC (HTTP works fine) |

---

## 🔐 Security Reminders

1. **Never commit `.env` to Git** - It's in `.gitignore`
2. **Rotate keys** after hackathon if shared publicly
3. **Use environment variables** in production
4. **Monitor usage** on API dashboards

---

## 📊 Free Tier Limits Summary

| Service | Daily Limit | Notes |
|---------|-------------|-------|
| Groq STT | 14,400 requests | ~400 audio minutes |
| Groq LLM | 14,400 requests | Shared with STT |
| Gemini | 1,500 requests | Per day |
| Google TTS | 1M chars/month | ~15 hours audio |
| Bhashini | Unlimited | Educational use |

For a 48-hour hackathon, these limits are more than sufficient!
