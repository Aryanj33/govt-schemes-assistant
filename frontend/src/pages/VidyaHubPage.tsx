import { useState } from "react";
import {
    LiveKitRoom,
    RoomAudioRenderer,
    VoiceAssistantControlBar,
    AudioVisualizer,
    useVoiceAssistant,
} from "@livekit/components-react";
import { Loader2, Mic, PhoneOff, Sparkles, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

/* ─── Suggested queries ──────────────────────────────────────────────── */
const SAMPLE_QUERIES = [
    { hi: "मुझे इंजीनियरिंग की पढ़ाई के लिए छात्रवृत्ति चाहिए", en: "Scholarship for engineering" },
    { hi: "PM-KISAN में कितना पैसा मिलता है?", en: "PM-KISAN amount" },
    { hi: "मुद्रा लोन कैसे मिलेगा?", en: "Mudra loan process" },
    { hi: "दिव्यांग पेंशन के बारे में बताओ", en: "Divyang pension info" },
];

/* ─── State → visual config map ─────────────────────────────────────── */
const STATE_CONFIG: Record<string, { color: string; glow: string; label: string; pulse: boolean }> = {
    connecting: { color: "#6366f1", glow: "rgba(99,102,241,0.4)", label: "Connecting…", pulse: true },
    idle: { color: "#8b5cf6", glow: "rgba(139,92,246,0.3)", label: "Ready", pulse: false },
    listening: { color: "#3b82f6", glow: "rgba(59,130,246,0.6)", label: "Listening", pulse: true },
    thinking: { color: "#a855f7", glow: "rgba(168,85,247,0.6)", label: "Thinking…", pulse: true },
    speaking: { color: "#22c55e", glow: "rgba(34,197,94,0.6)", label: "Speaking", pulse: true },
    disconnected: { color: "#ef4444", glow: "rgba(239,68,68,0.3)", label: "Disconnected", pulse: false },
};

/* ═══════════════════════════════════════════════════════════════════════
   Main page component
═══════════════════════════════════════════════════════════════════════ */
export default function VidyaHubPage() {
    const [token, setToken] = useState<string>("");
    const [serverUrl, setServerUrl] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const [inRoom, setInRoom] = useState<boolean>(false);

    const connectToVoice = async () => {
        setLoading(true);
        try {
            const res = await fetch("/token", { method: "POST" });
            if (res.ok) {
                const data = await res.json();
                if (data.error) { alert(`Backend: ${data.error}`); return; }
                setToken(data.token);
                setServerUrl(data.url);
                setInRoom(true);
            } else {
                alert("Could not get token — is the Python backend running?");
            }
        } catch (e) {
            console.error(e);
            alert("Failed to reach backend server.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div
            className="min-h-screen w-full flex flex-col items-center"
            style={{
                background: "linear-gradient(135deg, #0f0c29 0%, #1a1040 40%, #0f172a 100%)",
            }}
        >
            {/* ── Ambient background blobs ── */}
            <div className="pointer-events-none fixed inset-0 overflow-hidden">
                <div style={{
                    position: "absolute", top: "10%", left: "15%",
                    width: 400, height: 400, borderRadius: "50%",
                    background: "radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%)",
                    filter: "blur(60px)",
                }} />
                <div style={{
                    position: "absolute", bottom: "15%", right: "10%",
                    width: 350, height: 350, borderRadius: "50%",
                    background: "radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%)",
                    filter: "blur(60px)",
                }} />
            </div>

            {/* ── Header ── */}
            <div className="relative z-10 text-center pt-16 pb-8 px-4">
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <span style={{
                        display: "inline-flex", alignItems: "center", gap: 6,
                        padding: "6px 16px", borderRadius: 999,
                        background: "rgba(99,102,241,0.15)",
                        border: "1px solid rgba(99,102,241,0.3)",
                        color: "#a5b4fc", fontSize: 13, fontWeight: 600,
                        letterSpacing: "0.06em", marginBottom: 20,
                    }}>
                        <Zap size={13} /> LIVE VOICE ASSISTANT
                    </span>

                    <h1 style={{
                        fontSize: "clamp(2rem, 5vw, 3.5rem)", fontWeight: 800,
                        background: "linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 50%, #818cf8 100%)",
                        WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                        marginBottom: 16, lineHeight: 1.1,
                    }}>
                        Talk to Vidya
                    </h1>

                    <p style={{ color: "rgba(165,180,252,0.7)", fontSize: 16, maxWidth: 480, margin: "0 auto" }}>
                        Your AI government counselor — speak in Hindi, English, or Hinglish to discover
                        benefits across <strong style={{ color: "#a5b4fc" }}>3,400+ schemes</strong>.
                    </p>
                </motion.div>
            </div>

            {/* ── Main card ── */}
            <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                style={{
                    position: "relative", zIndex: 10,
                    width: "100%", maxWidth: 640, margin: "0 16px",
                    borderRadius: 28,
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    backdropFilter: "blur(24px)",
                    boxShadow: "0 32px 64px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)",
                    overflow: "hidden",
                    minHeight: 460,
                }}
            >
                <AnimatePresence mode="wait">
                    {!inRoom ? (
                        <motion.div
                            key="idle"
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "60px 40px", textAlign: "center" }}
                        >
                            {/* Orb */}
                            <div style={{ position: "relative", marginBottom: 40 }}>
                                <motion.div
                                    animate={{ scale: [1, 1.08, 1], opacity: [0.4, 0.7, 0.4] }}
                                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                                    style={{
                                        position: "absolute", inset: -24, borderRadius: "50%",
                                        background: "radial-gradient(circle, rgba(99,102,241,0.25) 0%, transparent 70%)",
                                        filter: "blur(12px)",
                                    }}
                                />
                                <div style={{
                                    width: 120, height: 120, borderRadius: "50%",
                                    background: "linear-gradient(135deg, rgba(99,102,241,0.3) 0%, rgba(139,92,246,0.2) 100%)",
                                    border: "1px solid rgba(99,102,241,0.4)",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    boxShadow: "0 0 40px rgba(99,102,241,0.3)",
                                }}>
                                    <span style={{ fontSize: 48 }}>🤖</span>
                                </div>
                            </div>

                            <h2 style={{ color: "#e0e7ff", fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
                                Vidya is ready
                            </h2>
                            <p style={{ color: "rgba(165,180,252,0.6)", fontSize: 14, marginBottom: 36, maxWidth: 340 }}>
                                Click below to start a voice session. Allow microphone access when prompted.
                            </p>

                            <motion.button
                                whileHover={{ scale: 1.04 }}
                                whileTap={{ scale: 0.97 }}
                                onClick={connectToVoice}
                                disabled={loading}
                                style={{
                                    display: "inline-flex", alignItems: "center", gap: 10,
                                    padding: "14px 36px", borderRadius: 999,
                                    background: loading
                                        ? "rgba(99,102,241,0.4)"
                                        : "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                                    color: "#fff", fontWeight: 700, fontSize: 15,
                                    border: "none", cursor: loading ? "not-allowed" : "pointer",
                                    boxShadow: "0 8px 32px rgba(99,102,241,0.4)",
                                    transition: "all 0.2s",
                                }}
                            >
                                {loading
                                    ? <><Loader2 size={18} className="animate-spin" /> Connecting…</>
                                    : <><Mic size={18} /> Connect to Voice</>
                                }
                            </motion.button>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="in-room"
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            style={{ width: "100%", height: "100%" }}
                        >
                            <LiveKitRoom
                                key={token}
                                token={token}
                                serverUrl={serverUrl}
                                connect={true}
                                audio={true}
                                video={false}
                                onDisconnected={() => setInRoom(false)}
                                onError={(err) => {
                                    console.error("LiveKit error:", err);
                                    alert(`Voice error: ${err.message}`);
                                    setInRoom(false);
                                }}
                            >
                                <ActiveVoiceSession onDisconnect={() => setInRoom(false)} />
                                <RoomAudioRenderer />
                            </LiveKitRoom>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>

            {/* ── Sample queries ── */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
                style={{ position: "relative", zIndex: 10, width: "100%", maxWidth: 640, padding: "32px 16px 48px" }}
            >
                <p style={{
                    display: "flex", alignItems: "center", gap: 6,
                    color: "rgba(165,180,252,0.6)", fontSize: 13, fontWeight: 600,
                    letterSpacing: "0.05em", marginBottom: 16, justifyContent: "center",
                }}>
                    <Sparkles size={13} /> TRY ASKING VIDYA
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    {SAMPLE_QUERIES.map((q, i) => (
                        <motion.div
                            key={i}
                            whileHover={{ scale: 1.02, borderColor: "rgba(99,102,241,0.5)" }}
                            style={{
                                padding: "14px 16px", borderRadius: 14,
                                background: "rgba(255,255,255,0.04)",
                                border: "1px solid rgba(255,255,255,0.08)",
                                cursor: "default", transition: "all 0.2s",
                            }}
                        >
                            <p style={{ color: "#e0e7ff", fontSize: 13, marginBottom: 4, fontWeight: 500 }}>
                                {q.hi}
                            </p>
                            <p style={{ color: "rgba(165,180,252,0.45)", fontSize: 11 }}>{q.en}</p>
                        </motion.div>
                    ))}
                </div>
            </motion.div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════
   Inner voice session — uses LiveKit hooks
═══════════════════════════════════════════════════════════════════════ */
function ActiveVoiceSession({ onDisconnect }: { onDisconnect: () => void }) {
    const { state, audioTrack } = useVoiceAssistant();
    const cfg = STATE_CONFIG[state] ?? STATE_CONFIG.idle;

    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "50px 24px 32px" }}>
            {/* ── Animated orb ── */}
            <div style={{ position: "relative", marginBottom: 32 }}>
                {/* Outer pulse ring */}
                {cfg.pulse && (
                    <motion.div
                        animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                        transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut" }}
                        style={{
                            position: "absolute", inset: -20, borderRadius: "50%",
                            border: `2px solid ${cfg.color}`,
                            pointerEvents: "none",
                        }}
                    />
                )}
                {/* Glow */}
                <div style={{
                    position: "absolute", inset: -16, borderRadius: "50%",
                    background: `radial-gradient(circle, ${cfg.glow} 0%, transparent 70%)`,
                    filter: "blur(8px)", transition: "all 0.5s",
                }} />
                {/* Orb body */}
                <motion.div
                    animate={cfg.pulse ? { scale: [1, 1.04, 1] } : { scale: 1 }}
                    transition={{ duration: 1.2, repeat: cfg.pulse ? Infinity : 0, ease: "easeInOut" }}
                    style={{
                        width: 160, height: 160, borderRadius: "50%",
                        background: `radial-gradient(circle at 35% 35%, ${cfg.color}55 0%, ${cfg.color}22 60%, transparent 100%)`,
                        border: `2px solid ${cfg.color}66`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        boxShadow: `0 0 48px ${cfg.glow}, inset 0 0 32px ${cfg.color}22`,
                        transition: "border-color 0.5s, box-shadow 0.5s",
                        position: "relative",
                        overflow: "hidden",
                    }}
                >
                    {/* AudioVisualizer when speaking */}
                    {state === "speaking" && audioTrack && (
                        <div style={{ position: "absolute", inset: 0, opacity: 0.7 }}>
                            <AudioVisualizer trackRef={audioTrack} style={{ width: "100%", height: "100%", color: cfg.color }} />
                        </div>
                    )}
                    {/* Static inner emoji/icon */}
                    <AnimatePresence mode="wait">
                        <motion.span
                            key={state}
                            initial={{ opacity: 0, scale: 0.6 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.6 }}
                            style={{ fontSize: 52, zIndex: 2, position: "relative" }}
                        >
                            {state === "listening" ? "🎤"
                                : state === "thinking" ? "💭"
                                    : state === "speaking" ? "🔊"
                                        : state === "disconnected" ? "📵"
                                            : "🤖"}
                        </motion.span>
                    </AnimatePresence>
                </motion.div>
            </div>

            {/* State badge */}
            <AnimatePresence mode="wait">
                <motion.div
                    key={state}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    style={{
                        padding: "6px 20px", borderRadius: 999,
                        background: `${cfg.color}22`,
                        border: `1px solid ${cfg.color}55`,
                        color: cfg.color, fontWeight: 700, fontSize: 13,
                        letterSpacing: "0.1em", marginBottom: 36,
                        transition: "all 0.3s",
                    }}
                >
                    {cfg.label.toUpperCase()}
                </motion.div>
            </AnimatePresence>

            {/* Controls */}
            <div style={{
                display: "flex", alignItems: "center", gap: 12,
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 999, padding: "10px 16px",
                backdropFilter: "blur(12px)",
            }}>
                <VoiceAssistantControlBar controls={{ leave: false }} />
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={onDisconnect}
                    title="End call"
                    style={{
                        width: 40, height: 40, borderRadius: "50%",
                        background: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
                        border: "none", cursor: "pointer", color: "#fff",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        boxShadow: "0 4px 16px rgba(239,68,68,0.4)",
                    }}
                >
                    <PhoneOff size={16} />
                </motion.button>
            </div>
        </div>
    );
}
