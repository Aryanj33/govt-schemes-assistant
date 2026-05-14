import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Keyboard, Mic, RotateCcw, Zap, Volume2, Phone, PhoneCall, PhoneOff, ArrowRight, Shield, Globe, Headphones } from "lucide-react";
import Header from "@/components/Header";
import AshokaChakra from "@/components/AshokaChakra";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { api } from "@/lib/api";
import { useLocation } from "react-router-dom";

type OrbState = "idle" | "listening" | "processing" | "speaking";
type CallStatus = "idle" | "initiating" | "ringing" | "connected" | "ended" | "error";

interface Message {
  role: "user" | "vidya";
  text: string;
  timestamp: Date;
}

// WebSocket Voice Communication Class — Streaming Audio Pipeline
class VoiceWebSocket {
  private ws: WebSocket | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  // Audio playback queue — plays binary audio chunks in sequence
  private audioPlaybackQueue: ArrayBuffer[] = [];
  private isPlayingAudio = false;

  constructor(
    private onMessage: (message: any) => void,
    private onConnectionChange: (connected: boolean) => void,
    private sessionId: string,
  ) {}

  connect() {
    try {
      // Derive WebSocket URL from current page origin (works in dev & prod)
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname;
      const port = '8080'; // backend port
      const wsUrl = `${protocol}//${host}:${port}/ws/voice?session_id=${encodeURIComponent(this.sessionId)}`;
      console.log('🔌 Connecting to', wsUrl);

      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer'; // receive binary frames as ArrayBuffer

      this.ws.onopen = () => {
        console.log('🔌 WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.onConnectionChange(true);
      };

      this.ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Binary frame = raw audio chunk from streaming TTS
          this.enqueueAudioChunk(event.data);
        } else {
          // Text frame = JSON control message
          const message = JSON.parse(event.data);
          this.onMessage(message);
        }
      };

      this.ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        this.isConnected = false;
        this.onConnectionChange(false);
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          setTimeout(() => this.connect(), 3000);
        }
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };
    } catch (error) {
      console.error('❌ Connection failed:', error);
      this.onConnectionChange(false);
    }
  }

  disconnect() {
    this.clearAudioQueue();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
  }

  // ── VAD state ───────────────────────────────────────────────────
  private vadAudioCtx: AudioContext | null = null;
  private vadAnalyser: AnalyserNode | null = null;
  private vadRafId: number | null = null;
  private vadHasSpoken = false;
  private vadSilenceStart = 0;
  // VAD thresholds — tuned for soft-spoken users on laptop mics
  // 0.022 catches normal indoor speech (was 0.04, missed soft speakers)
  // 800ms silence — snappier turn-end (was 1200ms)
  // 4s leading-silence — if user never speaks, stop early instead of 15s wait
  private static VAD_SPEECH_THRESHOLD = 0.022;
  private static VAD_SILENCE_MS       = 800;
  private static VAD_LEADING_SILENCE_MS = 4000;
  private static VAD_MAX_DURATION_MS  = 15000;
  private vadStartedAt = 0;
  private onTooQuiet?: () => void;
  setOnTooQuiet(cb: () => void) { this.onTooQuiet = cb; }

  async startListening() {
    if (!this.isConnected) return false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,         // Whisper prefers 16kHz
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'audio/mp4';
          if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'audio/wav';
        }
      }

      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType,
        audioBitsPerSecond: 64000,
      });
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = () => {
        this.stopVad();
        stream.getTracks().forEach((t) => t.stop());
        this.processRecording();
      };

      this.mediaRecorder.start(100);

      // ── Start VAD loop ──
      this.startVad(stream);
      return true;
    } catch (error) {
      console.error('❌ Microphone access failed:', error);
      return false;
    }
  }

  /** Analyse microphone input for speech/silence and auto-stop when the user
   * finishes speaking. This removes the 1-2s delay of the user tapping stop. */
  private startVad(stream: MediaStream) {
    try {
      const AnyAudioCtx: any = (window as any).AudioContext || (window as any).webkitAudioContext;
      this.vadAudioCtx = new AnyAudioCtx();
      const source = this.vadAudioCtx!.createMediaStreamSource(stream);
      const analyser = this.vadAudioCtx!.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.4;
      source.connect(analyser);
      this.vadAnalyser = analyser;

      this.vadHasSpoken = false;
      this.vadSilenceStart = 0;
      this.vadStartedAt = performance.now();

      const buf = new Uint8Array(analyser.fftSize);
      const tick = () => {
        if (!this.vadAnalyser || !this.mediaRecorder) return;
        analyser.getByteTimeDomainData(buf);
        // Compute normalized RMS (0..1)
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / buf.length);
        const now = performance.now();

        if (rms > VoiceWebSocket.VAD_SPEECH_THRESHOLD) {
          this.vadHasSpoken = true;
          this.vadSilenceStart = now;
        } else if (this.vadHasSpoken) {
          if (now - this.vadSilenceStart > VoiceWebSocket.VAD_SILENCE_MS) {
            console.log('🔇 VAD: silence detected → auto-stopping');
            this.stopListening();
            return;
          }
        } else {
          // Leading silence: user has not spoken yet. Stop early after 4s
          // so we don't sit on an open mic capturing room noise for 15s.
          if (now - this.vadStartedAt > VoiceWebSocket.VAD_LEADING_SILENCE_MS) {
            console.log('🤫 VAD: no speech detected (leading silence) → auto-stopping');
            this.onTooQuiet?.();
            this.stopListening();
            return;
          }
        }
        // Hard cap — prevent runaway recording
        if (now - this.vadStartedAt > VoiceWebSocket.VAD_MAX_DURATION_MS) {
          console.log('⏱️ VAD: max duration reached → auto-stopping');
          this.stopListening();
          return;
        }
        this.vadRafId = requestAnimationFrame(tick);
      };
      this.vadRafId = requestAnimationFrame(tick);
    } catch (e) {
      console.warn('⚠️ VAD setup failed:', e);
    }
  }

  private stopVad() {
    if (this.vadRafId !== null) {
      cancelAnimationFrame(this.vadRafId);
      this.vadRafId = null;
    }
    if (this.vadAudioCtx) {
      try { this.vadAudioCtx.close(); } catch {}
      this.vadAudioCtx = null;
    }
    this.vadAnalyser = null;
  }

  stopListening() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
    }
  }

  private async processRecording() {
    if (this.audioChunks.length === 0 || !this.isConnected) {
      this.onTooQuiet?.();
      return;
    }

    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
    // 2KB threshold catches truly empty / pure-silence recordings.
    if (audioBlob.size < 2000) {
      console.warn('🔇 Recording too small (likely silence):', audioBlob.size, 'bytes');
      this.onTooQuiet?.();
      return;
    }

    const arrayBuffer = await audioBlob.arrayBuffer();
    // Use chunked base64 to avoid "Maximum call stack" on long recordings
    const bytes = new Uint8Array(arrayBuffer);
    let binary = '';
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(
        null,
        Array.from(bytes.subarray(i, i + chunkSize)) as unknown as number[],
      );
    }
    const base64Audio = btoa(binary);

    // Clear playback queue before new turn
    this.clearAudioQueue();

    this.ws?.send(
      JSON.stringify({
        type: 'audio_end',
        audio_data: base64Audio,
        language: 'hindi',
      })
    );
  }

  // ── Streaming audio playback (binary chunks from server) ──────────

  /** Queue a raw audio ArrayBuffer for sequential playback. */
  private enqueueAudioChunk(data: ArrayBuffer) {
    this.audioPlaybackQueue.push(data);
    if (!this.isPlayingAudio) {
      this.playNextChunk();
    }
  }

  /** Play queued audio chunks one after another using HTML5 Audio. */
  private async playNextChunk() {
    if (this.audioPlaybackQueue.length === 0) {
      this.isPlayingAudio = false;
      return;
    }
    this.isPlayingAudio = true;
    const chunk = this.audioPlaybackQueue.shift()!;
    try {
      await this.playBinaryAudio(chunk);
    } catch (err) {
      console.warn('⚠️ Audio chunk playback failed, skipping:', err);
    }
    // Play next chunk (if any)
    this.playNextChunk();
  }

  /** Play raw audio bytes (mp3/wav) via HTML5 Audio element. */
  private playBinaryAudio(buffer: ArrayBuffer): Promise<void> {
    return new Promise((resolve, reject) => {
      const blob = new Blob([buffer], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.volume = 1.0;
      audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
      audio.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
      audio.play().catch(reject);
    });
  }

  clearAudioQueue() {
    this.audioPlaybackQueue = [];
    this.isPlayingAudio = false;
  }
}

const stateConfig: Record<OrbState, { borderColor: string; label: string; labelColor: string }> = {
  idle: { borderColor: "rgba(255,153,51,0.4)", label: "IDLE", labelColor: "text-muted-foreground" },
  listening: { borderColor: "#FF9933", label: "LISTENING", labelColor: "text-primary" },
  processing: { borderColor: "#1A6BD4", label: "PROCESSING...", labelColor: "text-accent" },
  speaking: { borderColor: "#138808", label: "SPEAKING", labelColor: "text-secondary" },
};

const callStatusLabels: Record<CallStatus, { label: string; color: string }> = {
  idle: { label: "", color: "" },
  initiating: { label: "Initiating call...", color: "text-primary" },
  ringing: { label: "Ringing... Please pick up your phone", color: "text-primary" },
  connected: { label: "Connected — Vidya is speaking with you", color: "text-secondary" },
  ended: { label: "Call ended", color: "text-muted-foreground" },
  error: { label: "Could not connect the call", color: "text-destructive" },
};

const Hub = () => {
  const location = useLocation();
  const [orbState, setOrbState] = useState<OrbState>("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const [textInput, setTextInput] = useState("");
  const [showTextInput, setShowTextInput] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [textSessionId, setTextSessionId] = useState<string | null>(() => localStorage.getItem("vidya-text-session"));
  const [voiceSessionId] = useState<string>(() => {
    const existing = localStorage.getItem("vidya-voice-session");
    if (existing) return existing;
    const sessionId = `voice-console-${crypto.randomUUID()}`;
    localStorage.setItem("vidya-voice-session", sessionId);
    return sessionId;
  });

  // Telephony state
  const [phoneNumber, setPhoneNumber] = useState("");
  const [callStatus, setCallStatus] = useState<CallStatus>("idle");
  const [callId, setCallId] = useState<string | null>(null);
  const [callDuration, setCallDuration] = useState(0);
  const [activeTab, setActiveTab] = useState<"voice" | "call">("voice");
  const callTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // WebSocket voice communication state
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [isRealTimeListening, setIsRealTimeListening] = useState(false);
  const voiceWSRef = useRef<VoiceWebSocket | null>(null);
  const responseStartTimeRef = useRef<number>(0);

  const transcriptRef = useRef<HTMLDivElement>(null);
  const { isRecording, audioLevel, toggleRecording } = useAudioRecorder();

  // Handle scheme context from navigation state
  useEffect(() => {
    const state = location.state as { askAboutScheme?: any } | null;
    if (state?.askAboutScheme) {
      const scheme = state.askAboutScheme;
      const schemeQuestion = `Tell me more about the ${scheme.name} scheme. I want to know about eligibility, application process, and benefits.`;
      
      // Add the question to messages and trigger text chat
      setMessages(prev => [...prev, { role: "user", text: schemeQuestion, timestamp: new Date() }]);
      setShowTextInput(true);
      setTextInput(schemeQuestion);
      
      // Auto-submit the question
      setTimeout(() => {
        handleTextSubmitWithMessage(schemeQuestion);
      }, 500);
      
      // Clear the navigation state to prevent re-triggering
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  // Initialize WebSocket connection
  useEffect(() => {
    const handleWebSocketMessage = (message: any) => {
      console.log('📨 WS message:', message.type);

      switch (message.type) {
        case 'transcription':
          setMessages(prev => [
            ...prev,
            { role: "user", text: message.text, timestamp: new Date() }
          ]);
          break;

        case 'response':
          if (message.partial) {
            // First partial response — add a new Vidya message
            setMessages(prev => [
              ...prev,
              { role: "vidya", text: message.text, timestamp: new Date() }
            ]);
          } else {
            // Final complete response — update the last Vidya message in place
            setMessages(prev => {
              const copy = [...prev];
              let lastVidyaIdx = -1;
              for (let i = copy.length - 1; i >= 0; i--) {
                if (copy[i].role === "vidya") { lastVidyaIdx = i; break; }
              }
              if (lastVidyaIdx >= 0) {
                copy[lastVidyaIdx] = { ...copy[lastVidyaIdx], text: message.text };
              } else {
                copy.push({ role: "vidya", text: message.text, timestamp: new Date() });
              }
              return copy;
            });
          }
          setOrbState("speaking");
          break;

        case 'processing':
          if (message.message === 'Transcribing...') {
            setOrbState("processing");
          } else if (message.message === 'Generating response...') {
            responseStartTimeRef.current = Date.now();
          }
          break;

        // NOTE: audio chunks now arrive as binary WS frames and are handled
        // directly by VoiceWebSocket.enqueueAudioChunk — no JSON message needed.

        case 'audio_complete':
          setOrbState("idle");
          setIsRealTimeListening(false);
          const responseTime = Date.now() - responseStartTimeRef.current;
          setLatency(responseTime);
          break;

        case 'error':
          console.error('WebSocket error:', message.message);
          setOrbState("idle");
          setIsRealTimeListening(false);
          setMessages(prev => [
            ...prev,
            { role: "vidya", text: "Sorry, I couldn't process that. Please try again.", timestamp: new Date() }
          ]);
          break;
      }
    };

    const handleConnectionChange = (connected: boolean) => {
      setIsWebSocketConnected(connected);
      if (!connected) {
        setOrbState("idle");
        setIsRealTimeListening(false);
      }
    };

    voiceWSRef.current = new VoiceWebSocket(handleWebSocketMessage, handleConnectionChange, voiceSessionId);
    voiceWSRef.current.setOnTooQuiet(() => {
      // VAD detected leading silence or empty recording — surface a hint and reset orb.
      setOrbState("idle");
      setIsRealTimeListening(false);
      setMessages(prev => [
        ...prev,
        {
          role: "vidya",
          text: "Mujhe aapki awaaz nahi sunaai di. Mic ke paas thoda paas aakar dobara boliye.",
          timestamp: new Date(),
        },
      ]);
    });
    voiceWSRef.current.connect();

    return () => {
      voiceWSRef.current?.disconnect();
    };
  }, [voiceSessionId]);

  useEffect(() => {
    if (transcriptRef.current) transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if (isRecording) setOrbState("listening");
  }, [isRecording]);


  // Poll call status
  useEffect(() => {
    if (!callId || callStatus === "ended" || callStatus === "error" || callStatus === "idle") return;
    const interval = setInterval(async () => {
      try {
        const res = await api.callStatus(callId);
        if (res.status === "connected") {
          setCallStatus("connected");
          if (!callTimerRef.current) {
            callTimerRef.current = setInterval(() => setCallDuration(d => d + 1), 1000);
          }
        } else if (res.status === "ended" || res.status === "completed") {
          setCallStatus("ended");
          clearInterval(interval);
          if (callTimerRef.current) { clearInterval(callTimerRef.current); callTimerRef.current = null; }
        } else if (res.status === "ringing") {
          setCallStatus("ringing");
        }
      } catch {
        // silently continue polling
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [callId, callStatus]);

  const formatPhone = (value: string) => {
    const digits = value.replace(/\D/g, "");
    if (digits.length <= 5) return digits;
    return digits.slice(0, 5) + " " + digits.slice(5, 10);
  };

  const handlePhoneChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value.replace(/\D/g, "");
    if (raw.length <= 10) setPhoneNumber(raw);
  };

  const handleInitiateCall = async () => {
    if (phoneNumber.length !== 10) return;
    setCallStatus("initiating");
    setCallDuration(0);
    try {
      const fullNumber = `+91${phoneNumber}`;
      // Always use Hindi
      const res = await api.initiateCall(fullNumber, "hi");
      setCallId(res.call_id || "demo-call");
      setCallStatus("ringing");
    } catch {
      setCallStatus("error");
      setTimeout(() => setCallStatus("idle"), 4000);
    }
  };

  const handleEndCall = () => {
    setCallStatus("ended");
    setCallId(null);
    if (callTimerRef.current) { clearInterval(callTimerRef.current); callTimerRef.current = null; }
    setTimeout(() => setCallStatus("idle"), 3000);
  };

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  // Real-time voice communication handler
  const handleVoiceTap = async () => {
    if (!isWebSocketConnected) {
      setMessages(prev => [
        ...prev,
        { role: "vidya", text: "Connection lost. Reconnecting...", timestamp: new Date() }
      ]);
      return;
    }

    if (isRealTimeListening) {
      // Stop listening
      voiceWSRef.current?.stopListening();
      setIsRealTimeListening(false);
      setOrbState("processing");
    } else {
      // Start listening
      const success = await voiceWSRef.current?.startListening();
      if (success) {
        setIsRealTimeListening(true);
        setOrbState("listening");
      } else {
        setMessages(prev => [
          ...prev,
          { role: "vidya", text: "Microphone access denied. Please allow microphone access and try again.", timestamp: new Date() }
        ]);
      }
    }
  };

  const handleTextSubmit = async () => {
    if (!textInput.trim()) return;
    const userMsg = textInput;
    setTextInput("");
    await handleTextSubmitWithMessage(userMsg);
  };

  const handleTextSubmitWithMessage = async (userMsg: string) => {
    setMessages(prev => [...prev, { role: "user", text: userMsg, timestamp: new Date() }]);
    setOrbState("processing");
    const start = Date.now();
    try {
      const response = await api.textChat(userMsg, textSessionId ?? undefined);
      if (response.session_id) {
        setTextSessionId(response.session_id);
        localStorage.setItem("vidya-text-session", response.session_id);
      }
      setLatency(Date.now() - start);
      setMessages(prev => [...prev, { role: "vidya", text: response.response || "Here are the schemes I found.", timestamp: new Date() }]);
      setOrbState("speaking");
      setTimeout(() => setOrbState("idle"), 2000);
    } catch {
      setMessages(prev => [...prev, { role: "vidya", text: "Couldn't connect to Vidya server. Check if backend is running on port 8080.", timestamp: new Date() }]);
      setOrbState("idle");
    }
  };

  const turns = messages.filter(m => m.role === "user").length;
  const cfg = stateConfig[orbState];
  const scale = orbState === "listening" ? 1 + audioLevel * 0.04 : 1;
  const waveformBars = [0.8, 1.1, 0.6, 0.9, 0.7, 1.0, 0.85];
  const isCallActive = callStatus === "ringing" || callStatus === "connected";

  // Update orb state label for real-time communication
  const getOrbLabel = () => {
    if (!isWebSocketConnected) return "CONNECTING...";
    if (isRealTimeListening) return "LISTENING";
    return cfg.label;
  };

  const getOrbLabelColor = () => {
    if (!isWebSocketConnected) return "text-yellow-500";
    if (isRealTimeListening) return "text-primary";
    return cfg.labelColor;
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Header />

      <div className="flex flex-1 pt-16">
        <main className="flex flex-1 flex-col items-center px-4 py-8">
          {/* Background */}
          <div className="pointer-events-none fixed inset-0 z-0" style={{ background: "radial-gradient(circle at 50% 40%, rgba(255,153,51,0.05), transparent 40%)" }} />
          <div className="pointer-events-none fixed inset-0 z-0 grid-lines" />

          {/* Tab switcher: Voice Chat / Call Me */}
          <div className="relative z-10 mb-8 flex rounded-lg border border-border bg-card/50 backdrop-blur p-1">
            {(["voice", "call"] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex items-center gap-2 rounded-md px-5 py-2.5 font-display text-sm font-bold transition-all duration-200 ${
                  activeTab === tab
                    ? "bg-primary/15 text-primary shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {tab === "voice" ? <Mic size={16} /> : <Phone size={16} />}
                {tab === "voice" ? "Voice Chat" : "Call Me"}
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            {/* ==================== VOICE CHAT TAB ==================== */}
            {activeTab === "voice" && (
              <motion.div
                key="voice"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex flex-1 flex-col items-center"
              >
                {/* Orb */}
                <div className="relative" style={{ width: 280, height: 280 }}>
                  <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" style={{ animation: "orb-rotate 8s linear infinite" }}>
                    <circle cx="50" cy="50" r="48" fill="none" stroke={cfg.borderColor} strokeWidth="0.6" strokeDasharray="8 12" opacity={orbState === "idle" ? 0.5 : 0.8} />
                  </svg>
                  <button
                    onClick={handleVoiceTap}
                    className="absolute inset-2 rounded-full border-2 transition-all duration-300 flex items-center justify-center"
                    style={{
                      borderColor: cfg.borderColor,
                      background: "radial-gradient(circle, #0D2444 0%, #050810 100%)",
                      boxShadow: orbState === "listening" ? "0 0 0 8px rgba(255,153,51,0.1), 0 0 0 16px rgba(255,153,51,0.05), 0 0 40px rgba(255,153,51,0.1) inset" : orbState === "speaking" ? "0 0 40px rgba(19,136,8,0.1) inset" : "0 0 40px rgba(255,153,51,0.1) inset",
                      transform: `scale(${scale})`,
                      animation: orbState === "listening" ? "breathe 1.5s ease-in-out infinite" : undefined,
                    }}
                    aria-label="Voice input"
                  >
                    {orbState === "idle" && <AshokaChakra size={80} className="text-primary opacity-60" />}
                    {orbState === "listening" && (
                      <div className="flex items-end gap-1" style={{ height: 64 }}>
                        {waveformBars.map((dur, i) => (
                          <div key={i} className="w-1.5 rounded-full bg-primary origin-bottom" style={{ animation: `eq-pulse ${dur}s ease-in-out ${i * 0.08}s infinite`, height: `${20 + Math.random() * 40}px` }} />
                        ))}
                      </div>
                    )}
                    {orbState === "processing" && (
                      <div className="flex items-center gap-1.5">
                        {[0, 1, 2].map(i => (
                          <div key={i} className="h-2.5 w-2.5 rounded-full bg-accent" style={{ animation: `eq-pulse 0.8s ease-in-out ${i * 0.15}s infinite`, position: "absolute", transform: `rotate(${i * 120}deg) translateX(120px)` }} />
                        ))}
                        <span className="font-mono text-sm text-accent-foreground animate-pulse">...</span>
                      </div>
                    )}
                    {orbState === "speaking" && (
                      <>
                        <div className="absolute inset-0 rounded-full border border-secondary/20" style={{ animation: "ringExpand 2s ease-out infinite" }} />
                        <div className="absolute inset-0 rounded-full border border-secondary/15" style={{ animation: "ringExpand 2s ease-out 0.4s infinite" }} />
                        <div className="absolute inset-0 rounded-full border border-secondary/10" style={{ animation: "ringExpand 2s ease-out 0.8s infinite" }} />
                        <Volume2 size={32} className="text-secondary animate-pulse" />
                      </>
                    )}
                  </button>
                </div>

                <span className={`mt-6 font-mono text-xs tracking-[0.2em] ${getOrbLabelColor()}`}>{getOrbLabel()}</span>

                <div className="mt-6 flex items-center gap-4">
                  <button 
                    onClick={handleVoiceTap} 
                    className="flex items-center justify-center rounded-full transition-all duration-200" 
                    style={{ 
                      width: 72, 
                      height: 72, 
                      backgroundColor: isRealTimeListening ? "hsl(var(--primary))" : "transparent", 
                      borderWidth: 2, 
                      borderColor: isWebSocketConnected ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))", 
                      boxShadow: isRealTimeListening ? "0 0 30px rgba(255,153,51,0.3)" : undefined 
                    }}
                    disabled={!isWebSocketConnected}
                  >
                    <Mic size={24} className={isRealTimeListening ? "text-primary-foreground" : isWebSocketConnected ? "text-primary" : "text-muted-foreground"} />
                  </button>
                  <button onClick={() => setShowTextInput(!showTextInput)} className="flex h-11 w-11 items-center justify-center rounded-full border border-border text-muted-foreground transition-colors hover:text-foreground hover:border-primary/40">
                    <Keyboard size={18} />
                  </button>
                </div>

                {/* Connection status indicator */}
                {!isWebSocketConnected && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 rounded-lg border border-yellow-500/30 bg-yellow-500/10 px-4 py-2"
                  >
                    <p className="font-body text-sm text-yellow-600">
                      🔄 Connecting to voice server...
                    </p>
                  </motion.div>
                )}

                {/* Real-time voice instructions - Hindi only */}
                {isWebSocketConnected && messages.length === 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 rounded-lg border border-primary/30 bg-primary/10 px-4 py-2 max-w-lg text-center"
                  >
                    <p className="font-body text-sm text-primary">
                      🎙️ माइक्रोफोन दबाकर विद्या से हिंदी में बात करें!
                    </p>
                  </motion.div>
                )}

                <AnimatePresence>
                  {showTextInput && (
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} className="mt-4 w-full max-w-lg">
                      <form onSubmit={(e) => { e.preventDefault(); handleTextSubmit(); }} className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2">
                        <input type="text" value={textInput} onChange={(e) => setTextInput(e.target.value)} placeholder="यहाँ टाइप करें..." className="flex-1 bg-transparent font-body text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none" autoFocus />
                        <button type="submit" className="rounded bg-primary px-4 py-1.5 font-display text-sm font-bold text-primary-foreground">Send</button>
                      </form>
                    </motion.div>
                  )}
                </AnimatePresence>

                {messages.length > 0 && (
                  <div ref={transcriptRef} className="mt-8 w-full max-w-2xl max-h-[50vh] overflow-y-auto space-y-4 px-4 py-4 bg-card/30 backdrop-blur rounded-xl border border-border/50">
                    <div className="flex items-center gap-2 mb-4 pb-2 border-b border-border/30">
                      <AshokaChakra size={16} className="text-primary" />
                      <h3 className="font-display text-sm font-bold text-foreground">Conversation</h3>
                    </div>
                    {messages.map((msg, i) => (
                      <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                        <div className={`max-w-[85%] rounded-xl px-4 py-3 ${
                          msg.role === "user" 
                            ? "bg-primary/15 border border-primary/30 text-foreground" 
                            : "bg-card border border-border/50 text-foreground"
                        }`}>
                          {msg.role === "vidya" && (
                            <div className="mb-2 flex items-center gap-2">
                              <AshokaChakra size={14} className="text-primary" />
                              <span className="font-mono text-[11px] text-primary font-bold">VIDYA</span>
                            </div>
                          )}
                          <p className="font-body text-sm leading-relaxed">{msg.text}</p>
                          <p className="mt-2 font-mono text-[10px] text-muted-foreground opacity-70">
                            {msg.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </motion.div>
            )}

            {/* ==================== CALL ME TAB ==================== */}
            {activeTab === "call" && (
              <motion.div
                key="call"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex flex-1 flex-col items-center w-full max-w-xl"
              >
                {/* Hero section for call */}
                <div className="text-center mb-8">
                  <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full border-2 border-primary/40" style={{ background: "radial-gradient(circle, rgba(255,153,51,0.15) 0%, transparent 70%)" }}>
                    <Phone size={32} className="text-primary" />
                  </div>
                  <h2 className="font-display text-3xl font-bold text-foreground">
                    Vidya Will Call You
                  </h2>
                  <p className="mt-2 font-body text-sm text-muted-foreground max-w-md mx-auto">
                    अपना मोबाइल नंबर डालें और विद्या आपको कॉल करेगी। हिंदी में बात करें।
                  </p>
                </div>

                {/* Phone input card */}
                <div className="w-full rounded-xl border border-border bg-card p-6 mb-6">
                  <label className="mb-3 block font-mono text-[11px] tracking-[0.15em] text-muted-foreground uppercase">
                    Mobile Number
                  </label>
                  <div className="flex items-center gap-3">
                    {/* Country code */}
                    <div className="flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-3 font-mono text-sm text-foreground">
                      <span className="text-base">🇮🇳</span>
                      <span>+91</span>
                    </div>
                    {/* Number input */}
                    <div className="relative flex-1">
                      <input
                        type="tel"
                        value={formatPhone(phoneNumber)}
                        onChange={handlePhoneChange}
                        placeholder="XXXXX XXXXX"
                        disabled={isCallActive}
                        className="w-full rounded-lg border border-border bg-background px-4 py-3 font-mono text-lg tracking-widest text-foreground placeholder:text-muted-foreground/30 focus:outline-none focus:border-primary/60 transition-colors disabled:opacity-50"
                        maxLength={11}
                      />
                      {phoneNumber.length === 10 && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 rounded-full bg-secondary/20 flex items-center justify-center"
                        >
                          <span className="text-secondary text-xs">✓</span>
                        </motion.div>
                      )}
                    </div>
                  </div>

                  {/* Language preference - Hindi only */}
                  <div className="mt-4">
                    <label className="mb-2 block font-mono text-[10px] tracking-[0.1em] text-muted-foreground uppercase">
                      Call Language
                    </label>
                    <div className="flex gap-2">
                      <button
                        className="rounded-md px-4 py-2 font-body text-xs bg-primary/15 text-primary border border-primary/40"
                      >
                        हिंदी (Hindi Only)
                      </button>
                    </div>
                  </div>

                  {/* Call button */}
                  <div className="mt-6">
                    {!isCallActive ? (
                      <button
                        onClick={handleInitiateCall}
                        disabled={phoneNumber.length !== 10 || callStatus === "initiating"}
                        className="group w-full flex items-center justify-center gap-3 rounded-lg bg-primary py-4 font-display text-lg font-bold text-primary-foreground transition-all duration-200 hover:shadow-lg disabled:opacity-40 disabled:cursor-not-allowed"
                        style={{ boxShadow: phoneNumber.length === 10 ? "0 0 30px rgba(255,153,51,0.2)" : undefined }}
                      >
                        {callStatus === "initiating" ? (
                          <>
                            <div className="h-5 w-5 rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground animate-spin" />
                            Connecting...
                          </>
                        ) : (
                          <>
                            <PhoneCall size={20} />
                            Call Me Now
                            <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
                          </>
                        )}
                      </button>
                    ) : (
                      <div className="space-y-3">
                        {/* Active call card */}
                        <div className="rounded-lg border border-secondary/30 p-4" style={{ backgroundColor: "rgba(19,136,8,0.08)" }}>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <div className="relative">
                                <div className="h-3 w-3 rounded-full bg-secondary" />
                                <div className="absolute inset-0 h-3 w-3 rounded-full bg-secondary animate-ping" />
                              </div>
                              <div>
                                <p className="font-display text-sm font-bold text-foreground">
                                  {callStatus === "ringing" ? "Ringing..." : "Call Connected"}
                                </p>
                                <p className="font-mono text-xs text-muted-foreground">
                                  +91 {formatPhone(phoneNumber)}
                                </p>
                              </div>
                            </div>
                            {callStatus === "connected" && (
                              <span className="font-mono text-lg text-secondary font-bold">
                                {formatDuration(callDuration)}
                              </span>
                            )}
                          </div>

                          {callStatus === "ringing" && (
                            <div className="mt-3 flex items-center gap-2">
                              <div className="flex gap-0.5">
                                {[0, 1, 2, 3, 4].map(i => (
                                  <div
                                    key={i}
                                    className="w-1 rounded-full bg-primary"
                                    style={{
                                      animation: `eq-pulse 0.8s ease-in-out ${i * 0.12}s infinite`,
                                      height: 16,
                                    }}
                                  />
                                ))}
                              </div>
                              <span className="font-body text-xs text-muted-foreground">Pick up your phone...</span>
                            </div>
                          )}
                        </div>

                        <button
                          onClick={handleEndCall}
                          className="w-full flex items-center justify-center gap-2 rounded-lg bg-destructive py-3 font-display text-sm font-bold text-destructive-foreground transition-all hover:bg-destructive/90"
                        >
                          <PhoneOff size={16} />
                          End Call
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Call status message */}
                  <AnimatePresence>
                    {callStatus !== "idle" && callStatusLabels[callStatus].label && (
                      <motion.p
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className={`mt-3 text-center font-mono text-xs ${callStatusLabels[callStatus].color}`}
                      >
                        {callStatusLabels[callStatus].label}
                      </motion.p>
                    )}
                  </AnimatePresence>
                </div>

                {/* How it works */}
                <div className="w-full rounded-xl border border-border bg-card/50 p-5 mb-6">
                  <h3 className="font-mono text-[11px] tracking-[0.15em] text-muted-foreground uppercase mb-4">
                    How Voice Call Works
                  </h3>
                  <div className="space-y-3">
                    {[
                      { icon: Phone, title: "Vidya calls your number", desc: "You'll receive a call from Vidya within seconds" },
                      { icon: Headphones, title: "Speak naturally", desc: "Ask about any government scheme in your language" },
                      { icon: Globe, title: "Get instant answers", desc: "Vidya searches 3,400+ schemes and responds in real-time" },
                    ].map((step, i) => (
                      <div key={i} className="flex items-start gap-3">
                        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary/30 bg-primary/10">
                          <step.icon size={14} className="text-primary" />
                        </div>
                        <div>
                          <p className="font-display text-sm font-bold text-foreground">{step.title}</p>
                          <p className="font-body text-xs text-muted-foreground">{step.desc}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Trust signals */}
                <div className="flex items-center gap-4 text-muted-foreground">
                  <div className="flex items-center gap-1.5">
                    <Shield size={12} className="text-secondary" />
                    <span className="font-mono text-[10px]">Encrypted</span>
                  </div>
                  <div className="h-3 w-px bg-border" />
                  <span className="font-mono text-[10px]">No data stored</span>
                  <div className="h-3 w-px bg-border" />
                  <span className="font-mono text-[10px]">Free of charge</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Bottom bar - Hindi only */}
          <div className="fixed bottom-0 left-0 right-0 z-20 border-t border-border bg-background/90 backdrop-blur px-4 py-3">
            <div className="container mx-auto flex flex-wrap items-center justify-between gap-3 max-w-lg">
              <div className="flex gap-1">
                <button className="rounded px-3 py-1 font-body text-xs bg-primary/15 text-primary border-b-2 border-primary">
                  हिंदी
                </button>
              </div>
              <div className="flex items-center gap-4">
                {latency !== null && (
                  <span className="flex items-center gap-1 font-mono text-xs text-secondary">
                    <Zap size={12} />{latency}ms
                  </span>
                )}
                <span className="font-mono text-[10px] text-muted-foreground">Session: {turns} turns</span>
                <button onClick={async () => { try { await api.reset(textSessionId ?? undefined); } catch {} setMessages([]); setOrbState("idle"); setTextSessionId(null); }} className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground hover:text-primary transition-colors">
                  <RotateCcw size={10} /> Reset
                </button>
              </div>
            </div>
          </div>
        </main>


      </div>
    </div>
  );
};

export default Hub;
