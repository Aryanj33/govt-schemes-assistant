import { useState } from "react";
import {
    LiveKitRoom,
    RoomAudioRenderer,
    VoiceAssistantControlBar,
    AudioVisualizer,
    useVoiceAssistant,
} from "@livekit/components-react";
import { Loader2, Mic, PhoneOff, Sparkles, Zap, GraduationCap, Leaf, DollarSign, Heart, ArrowRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

/* ─── Suggested queries ──────────────────────────────────────────────── */
const SAMPLE_QUERIES = [
    { icon: <GraduationCap size={14} />, hi: "मुझे इंजीनियरिंग की पढ़ाई के लिए छात्रवृत्ति चाहिए", en: "Scholarship for engineering" },
    { icon: <Leaf size={14} />, hi: "PM-KISAN में कितना पैसा मिलता है?", en: "PM-KISAN amount" },
    { icon: <DollarSign size={14} />, hi: "मुद्रा लोन कैसे मिलेगा?", en: "Mudra loan process" },
    { icon: <Heart size={14} />, hi: "दिव्यांग पेंशन के बारे में बताओ", en: "Divyang pension info" },
];

const LANGUAGES = ["English", "हिंदी", "Hinglish"];

/* ─── State → visual config map ─────────────────────────────────────── */
const STATE_CONFIG: Record<string, { color: string; glow: string; label: string; sublabel: string; emoji: string; pulse: boolean }> = {
    connecting: { color: "#6366f1", glow: "rgba(99,102,241,0.4)", label: "Connecting…", sublabel: "Please wait", emoji: "⏳", pulse: true },
    idle: { color: "#8b5cf6", glow: "rgba(139,92,246,0.3)", label: "Vidya is ready", sublabel: "Click to start voice", emoji: "🤖", pulse: false },
    listening: { color: "#3b82f6", glow: "rgba(59,130,246,0.6)", label: "Listening…", sublabel: "Speak now", emoji: "🎤", pulse: true },
    thinking: { color: "#a855f7", glow: "rgba(168,85,247,0.6)", label: "Thinking…", sublabel: "Finding schemes", emoji: "💭", pulse: true },
    speaking: { color: "#22c55e", glow: "rgba(34,197,94,0.6)", label: "Vidya is speaking", sublabel: "Listen carefully", emoji: "🔊", pulse: true },
    disconnected: { color: "#ef4444", glow: "rgba(239,68,68,0.3)", label: "Disconnected", sublabel: "Session ended", emoji: "📵", pulse: false },
};




/* ─── AI Voice Visualizer — the centrepiece ────────────────────────────── */
const WAVE_BASE = [8, 14, 22, 30, 40, 28, 36, 20, 44, 26, 18, 38, 24, 42, 16, 32, 20, 40, 28, 36, 10, 24, 34, 18, 42, 22, 30, 16, 28, 12];

function VoiceVisualizer({ active = false, color = "#818cf8" }: { active?: boolean; color?: string }) {
    return (
        <div style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            gap: 4, height: 80, width: "100%", maxWidth: 380, margin: "0 auto",
        }}>
            {WAVE_BASE.map((h, i) => (
                <motion.div
                    key={i}
                    style={{
                        width: active ? 4 : 3,
                        borderRadius: 999,
                        background: active
                            ? `linear-gradient(to top, ${color}55, ${color})`
                            : `linear-gradient(to top, rgba(129,140,248,0.15), rgba(129,140,248,0.55))`,
                    }}
                    animate={{
                        height: active
                            ? [h * 0.6, h * 1.6, h * 0.4, h * 1.8, h * 0.7]
                            : [h * 0.3, h * 0.6, h * 0.2, h * 0.5, h * 0.3],
                        opacity: active ? [0.8, 1, 0.6, 1, 0.8] : [0.25, 0.45, 0.2, 0.4, 0.25],
                    }}
                    transition={{
                        duration: active ? 0.9 + (i % 5) * 0.12 : 2.2 + (i % 7) * 0.18,
                        repeat: Infinity,
                        delay: i * 0.04,
                        ease: "easeInOut",
                    }}
                />
            ))}
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════
   Main page component
═══════════════════════════════════════════════════════════════════════ */
export default function VidyaHubPage() {
    const [token, setToken] = useState<string>("");
    const [serverUrl, setServerUrl] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const [inRoom, setInRoom] = useState<boolean>(false);
    const [activeLang, setActiveLang] = useState("English");

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
            style={{ background: "linear-gradient(135deg, #0f0c29 0%, #1a1040 40%, #0f172a 100%)" }}
        >
            {/* ── Animated mesh background blobs ── */}
            <div className="pointer-events-none fixed inset-0 overflow-hidden">
                {/* Large primary blob */}
                <motion.div
                    animate={{ scale: [1, 1.15, 1], x: [0, 30, 0], y: [0, -20, 0] }}
                    transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
                    style={{
                        position: "absolute", top: "5%", left: "10%",
                        width: 500, height: 500, borderRadius: "50%",
                        background: "radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%)",
                        filter: "blur(70px)",
                    }}
                />
                {/* Purple blob */}
                <motion.div
                    animate={{ scale: [1, 1.2, 1], x: [0, -25, 0], y: [0, 30, 0] }}
                    transition={{ duration: 15, repeat: Infinity, ease: "easeInOut", delay: 2 }}
                    style={{
                        position: "absolute", bottom: "10%", right: "8%",
                        width: 420, height: 420, borderRadius: "50%",
                        background: "radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%)",
                        filter: "blur(60px)",
                    }}
                />
                {/* Blue accent blob */}
                <motion.div
                    animate={{ scale: [1, 1.1, 1], x: [0, 20, 0], y: [0, 15, 0] }}
                    transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 4 }}
                    style={{
                        position: "absolute", top: "40%", right: "25%",
                        width: 300, height: 300, borderRadius: "50%",
                        background: "radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%)",
                        filter: "blur(50px)",
                    }}
                />
                {/* Subtle grid overlay */}
                <div style={{
                    position: "absolute", inset: 0,
                    backgroundImage: "linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px)",
                    backgroundSize: "40px 40px",
                }} />
            </div>

            {/* ── Header ── */}
            <div className="relative z-10 text-center pt-14 pb-4 px-4 w-full max-w-2xl">
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
                        letterSpacing: "0.06em", marginBottom: 16,
                    }}>
                        <Zap size={13} /> LIVE VOICE ASSISTANT
                    </span>

                    <h1 style={{
                        fontSize: "clamp(2rem, 5vw, 3.5rem)", fontWeight: 800,
                        background: "linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 50%, #818cf8 100%)",
                        WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                        marginBottom: 12, lineHeight: 1.1,
                    }}>
                        Talk to Vidya
                    </h1>

                    <p style={{ color: "rgba(165,180,252,0.7)", fontSize: 15, maxWidth: 480, margin: "0 auto 20px" }}>
                        Your AI government counselor — speak in Hindi, English, or Hinglish to discover
                        benefits across <strong style={{ color: "#a5b4fc" }}>3,400+ schemes</strong>.
                    </p>

                    {/* Language selector */}
                    <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 20 }}>
                        <span style={{ color: "rgba(165,180,252,0.5)", fontSize: 13, alignSelf: "center" }}>🌐</span>
                        {LANGUAGES.map((lang) => (
                            <motion.button
                                key={lang}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => setActiveLang(lang)}
                                style={{
                                    padding: "5px 14px", borderRadius: 999, fontSize: 13, fontWeight: 600,
                                    cursor: "pointer", border: "1px solid",
                                    borderColor: activeLang === lang ? "rgba(99,102,241,0.6)" : "rgba(255,255,255,0.1)",
                                    background: activeLang === lang ? "rgba(99,102,241,0.25)" : "rgba(255,255,255,0.04)",
                                    color: activeLang === lang ? "#a5b4fc" : "rgba(165,180,252,0.45)",
                                    transition: "all 0.2s",
                                }}
                            >
                                {lang}
                            </motion.button>
                        ))}
                    </div>
                </motion.div>
            </div>

            {/* ── Main card ── */}
            <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                style={{
                    position: "relative", zIndex: 10,
                    width: "100%", maxWidth: 640, margin: "16px 16px 0",
                    borderRadius: 28,
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    backdropFilter: "blur(24px)",
                    boxShadow: "0 32px 64px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)",
                    overflow: "hidden",
                    minHeight: 420,
                }}
            >
                <AnimatePresence mode="wait">
                    {!inRoom ? (
                        <motion.div
                            key="idle"
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "52px 40px", textAlign: "center" }}
                        >
                            {/* ── Premium Mic Orb ── */}
                            <div style={{ position: "relative", marginBottom: 28 }}>

                                {/* Far outer glow */}
                                <motion.div
                                    animate={{ scale: [1, 1.12, 1], opacity: [0.3, 0.55, 0.3] }}
                                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                                    style={{
                                        position: "absolute", top: "50%", left: "50%",
                                        transform: "translate(-50%, -50%)",
                                        width: 280, height: 280, borderRadius: "50%",
                                        background: "radial-gradient(circle, rgba(99,102,241,0.22) 0%, transparent 70%)",
                                        filter: "blur(18px)", pointerEvents: "none",
                                    }}
                                />

                                {/* Expanding pulse rings */}
                                {[0, 1, 2].map((i) => (
                                    <motion.div
                                        key={i}
                                        animate={{ scale: [1, 2.0], opacity: [0.4, 0] }}
                                        transition={{ duration: 3, repeat: Infinity, delay: i * 1.0, ease: "easeOut" }}
                                        style={{
                                            position: "absolute", top: "50%", left: "50%",
                                            transform: "translate(-50%, -50%)",
                                            width: 160, height: 160, borderRadius: "50%",
                                            border: "1.5px solid rgba(99,102,241,0.5)",
                                            pointerEvents: "none",
                                        }}
                                    />
                                ))}

                                {/* Main orb */}
                                <motion.div
                                    animate={{ scale: [1, 1.035, 1] }}
                                    transition={{ duration: 2.8, repeat: Infinity, ease: "easeInOut" }}
                                    style={{
                                        position: "relative",
                                        width: 160, height: 160, borderRadius: "50%",
                                        background: "radial-gradient(circle at 38% 32%, rgba(139,92,246,0.55) 0%, rgba(99,102,241,0.45) 40%, rgba(67,56,202,0.3) 100%)",
                                        border: "1.5px solid rgba(129,140,248,0.5)",
                                        display: "flex", flexDirection: "column",
                                        alignItems: "center", justifyContent: "center",
                                        boxShadow: "0 0 60px rgba(99,102,241,0.5), 0 0 120px rgba(99,102,241,0.2), inset 0 1px 0 rgba(255,255,255,0.12), inset 0 0 40px rgba(99,102,241,0.15)",
                                        overflow: "hidden",
                                    }}
                                >
                                    {/* Decorative inner arc */}
                                    <div style={{
                                        position: "absolute", top: 10, left: 10, right: 10, bottom: 10,
                                        borderRadius: "50%",
                                        border: "1px solid rgba(165,180,252,0.12)",
                                        pointerEvents: "none",
                                    }} />

                                    {/* Inner mini waveform */}
                                    <div style={{ display: "flex", alignItems: "center", gap: 2.5, marginBottom: 10, zIndex: 2 }}>
                                        {[5, 9, 15, 11, 19, 13, 7, 17, 9, 5].map((h, i) => (
                                            <motion.div
                                                key={i}
                                                style={{ width: 2.5, borderRadius: 999, background: "rgba(255,255,255,0.7)" }}
                                                animate={{ height: [h, h * 1.8, h * 0.4, h * 2, h] }}
                                                transition={{ duration: 1.4, repeat: Infinity, delay: i * 0.1, ease: "easeInOut" }}
                                            />
                                        ))}
                                    </div>

                                    {/* Mic icon */}
                                    <div style={{
                                        width: 44, height: 44, borderRadius: "50%",
                                        background: "rgba(255,255,255,0.12)",
                                        border: "1px solid rgba(255,255,255,0.2)",
                                        display: "flex", alignItems: "center", justifyContent: "center",
                                        backdropFilter: "blur(4px)", zIndex: 2,
                                    }}>
                                        <Mic size={22} color="rgba(255,255,255,0.95)" strokeWidth={2} />
                                    </div>
                                </motion.div>
                            </div>

                            <h2 style={{ color: "#e0e7ff", fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Vidya is ready</h2>
                            <p style={{ color: "rgba(165,180,252,0.6)", fontSize: 14, marginBottom: 20, maxWidth: 320 }}>
                                Speak in <strong style={{ color: "#a5b4fc" }}>{activeLang}</strong> — ask about any government scheme, scholarship, or benefit.
                            </p>

                            {/* ── VOICE VISUALIZER ── */}
                            <div style={{
                                width: "100%", padding: "14px 20px 10px", borderRadius: 18, marginBottom: 24,
                                background: "rgba(99,102,241,0.05)",
                                border: "1px solid rgba(99,102,241,0.13)",
                            }}>
                                <p style={{ color: "rgba(165,180,252,0.3)", fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textAlign: "center", marginBottom: 8 }}>IDLE · WAITING FOR VOICE</p>
                                <VoiceVisualizer active={false} />
                            </div>

                            <motion.button
                                whileHover={{ scale: 1.05, boxShadow: "0 0 32px rgba(99,102,241,0.7)" }}
                                whileTap={{ scale: 0.97 }}
                                onClick={connectToVoice}
                                disabled={loading}
                                style={{
                                    display: "inline-flex", alignItems: "center", gap: 10,
                                    padding: "14px 40px", borderRadius: 999,
                                    background: loading
                                        ? "rgba(99,102,241,0.4)"
                                        : "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                                    color: "#fff", fontWeight: 700, fontSize: 16,
                                    border: "none", cursor: loading ? "not-allowed" : "pointer",
                                    boxShadow: "0 8px 32px rgba(99,102,241,0.45)",
                                    transition: "all 0.25s",
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
                style={{ position: "relative", zIndex: 10, width: "100%", maxWidth: 640, padding: "28px 16px 0" }}
            >
                <p style={{
                    display: "flex", alignItems: "center", gap: 6,
                    color: "rgba(165,180,252,0.6)", fontSize: 13, fontWeight: 600,
                    letterSpacing: "0.05em", marginBottom: 12, justifyContent: "center",
                }}>
                    <Sparkles size={13} /> TRY ASKING VIDYA
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    {SAMPLE_QUERIES.map((q, i) => (
                        <motion.div
                            key={i}
                            whileHover={{ scale: 1.03, borderColor: "rgba(99,102,241,0.5)", background: "rgba(99,102,241,0.12)" }}
                            whileTap={{ scale: 0.97 }}
                            onClick={connectToVoice}
                            style={{
                                padding: "14px 16px", borderRadius: 14,
                                background: "rgba(255,255,255,0.04)",
                                border: "1px solid rgba(255,255,255,0.08)",
                                cursor: "pointer", transition: "all 0.2s",
                            }}
                        >
                            <p style={{ color: "#a5b4fc", fontSize: 12, marginBottom: 6, fontWeight: 600, display: "flex", alignItems: "center", gap: 5 }}>
                                {q.icon} {q.en}
                            </p>
                            <p style={{ color: "rgba(165,180,252,0.45)", fontSize: 11 }}>{q.hi}</p>
                        </motion.div>
                    ))}
                </div>
            </motion.div>

            {/* ── How Vidya Works ── */}
            <motion.div
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
                style={{ position: "relative", zIndex: 10, width: "100%", maxWidth: 640, padding: "48px 16px 8px" }}
            >
                <p style={{
                    color: "rgba(165,180,252,0.6)", fontSize: 13, fontWeight: 600,
                    letterSpacing: "0.08em", textAlign: "center", marginBottom: 24,
                }}>
                    HOW VIDYA WORKS
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                    {[
                        { emoji: "🎤", title: "Speak your question", desc: "Ask in Hindi, English, or Hinglish" },
                        { emoji: "🤖", title: "AI understands", desc: "Vidya analyses 3,400+ schemes" },
                        { emoji: "📋", title: "Get results", desc: "Eligible schemes delivered instantly" },
                    ].map((step, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 16 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.4, delay: i * 0.12 }}
                            style={{
                                padding: "20px 14px", borderRadius: 18, textAlign: "center",
                                background: "rgba(255,255,255,0.03)",
                                border: "1px solid rgba(255,255,255,0.07)",
                            }}
                        >
                            <div style={{ fontSize: 28, marginBottom: 10 }}>{step.emoji}</div>
                            <p style={{ color: "#e0e7ff", fontSize: 13, fontWeight: 700, marginBottom: 6 }}>{step.title}</p>
                            <p style={{ color: "rgba(165,180,252,0.45)", fontSize: 11, lineHeight: 1.5 }}>{step.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </motion.div>

            {/* ── Sample Scheme Cards ── */}
            <motion.div
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.1 }}
                style={{ position: "relative", zIndex: 10, width: "100%", maxWidth: 640, padding: "36px 16px 56px" }}
            >
                <p style={{
                    color: "rgba(165,180,252,0.6)", fontSize: 13, fontWeight: 600,
                    letterSpacing: "0.08em", textAlign: "center", marginBottom: 20,
                }}>
                    SUGGESTED SCHEMES
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    {[
                        { name: "PM-KISAN Samman Nidhi", benefit: "₹6,000 yearly support", color: "rgba(34,197,94,0.15)", border: "rgba(34,197,94,0.2)", tag: "Farmers" },
                        { name: "National Scholarship Portal", benefit: "Up to ₹50,000 scholarship", color: "rgba(59,130,246,0.15)", border: "rgba(59,130,246,0.2)", tag: "Students" },
                    ].map((scheme, i) => (
                        <motion.div
                            key={i}
                            whileHover={{ scale: 1.02, borderColor: "rgba(99,102,241,0.4)" }}
                            style={{
                                padding: "18px", borderRadius: 16,
                                background: scheme.color,
                                border: `1px solid ${scheme.border}`,
                                cursor: "pointer", transition: "all 0.2s",
                            }}
                        >
                            <span style={{
                                fontSize: 10, fontWeight: 700, color: "rgba(165,180,252,0.6)",
                                letterSpacing: "0.06em", display: "block", marginBottom: 8,
                            }}>
                                {scheme.tag.toUpperCase()}
                            </span>
                            <p style={{ color: "#e0e7ff", fontSize: 13, fontWeight: 700, marginBottom: 8, lineHeight: 1.4 }}>{scheme.name}</p>
                            <p style={{ color: "rgba(165,180,252,0.7)", fontSize: 12, marginBottom: 14 }}>{scheme.benefit}</p>
                            <a
                                href="#"
                                style={{ color: "#818cf8", fontSize: 12, fontWeight: 600, display: "flex", alignItems: "center", gap: 4, textDecoration: "none" }}
                            >
                                Apply <ArrowRight size={12} />
                            </a>
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
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "44px 24px 32px" }}>
            {/* ── Animated orb ── */}
            <div style={{ position: "relative", marginBottom: 28 }}>
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
                <div style={{
                    position: "absolute", inset: -16, borderRadius: "50%",
                    background: `radial-gradient(circle, ${cfg.glow} 0%, transparent 70%)`,
                    filter: "blur(8px)", transition: "all 0.5s",
                }} />
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
                        position: "relative", overflow: "hidden",
                    }}
                >
                    {state === "speaking" && audioTrack && (
                        <div style={{ position: "absolute", inset: 0, opacity: 0.7 }}>
                            <AudioVisualizer trackRef={audioTrack} style={{ width: "100%", height: "100%", color: cfg.color }} />
                        </div>
                    )}
                    <AnimatePresence mode="wait">
                        <motion.span
                            key={state}
                            initial={{ opacity: 0, scale: 0.6 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.6 }}
                            style={{ fontSize: 52, zIndex: 2, position: "relative" }}
                        >
                            {cfg.emoji}
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
                    style={{ textAlign: "center", marginBottom: 28 }}
                >
                    <div style={{
                        display: "inline-flex", padding: "6px 20px", borderRadius: 999, marginBottom: 6,
                        background: `${cfg.color}22`,
                        border: `1px solid ${cfg.color}55`,
                        color: cfg.color, fontWeight: 700, fontSize: 13,
                        letterSpacing: "0.1em",
                    }}>
                        {cfg.label.toUpperCase()}
                    </div>
                    <p style={{ color: "rgba(165,180,252,0.5)", fontSize: 12 }}>{cfg.sublabel}</p>
                </motion.div>
            </AnimatePresence>

            {/* ── LIVE VOICE VISUALIZER ── */}
            <AnimatePresence mode="wait">
                <motion.div
                    key={state}
                    initial={{ opacity: 0, scaleY: 0.6 }}
                    animate={{ opacity: 1, scaleY: 1 }}
                    exit={{ opacity: 0, scaleY: 0.6 }}
                    style={{
                        width: "100%", padding: "14px 20px", borderRadius: 18, marginBottom: 24,
                        background: `${cfg.color}0d`,
                        border: `1px solid ${cfg.color}33`,
                    }}
                >
                    <VoiceVisualizer active={cfg.pulse} color={cfg.color} />
                </motion.div>
            </AnimatePresence>
            <div style={{
                display: "flex", alignItems: "center", gap: 12,
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 999, padding: "10px 16px",
                backdropFilter: "blur(12px)",
            }}>
                <VoiceAssistantControlBar controls={{ leave: false }} />
                <motion.button
                    whileHover={{ scale: 1.08, boxShadow: "0 0 20px rgba(239,68,68,0.5)" }}
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
