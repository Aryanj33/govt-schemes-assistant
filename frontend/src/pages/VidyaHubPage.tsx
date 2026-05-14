import { useState } from "react";
import {
    Phone, Loader2, Sparkles, Zap, GraduationCap, Leaf,
    DollarSign, Heart, CheckCircle, AlertCircle, ArrowRight,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

/* ─── Suggested queries ──────────────────────────────────────────────── */
const SAMPLE_QUERIES = [
    { icon: <GraduationCap size={14} />, hi: "मुझे इंजीनियरिंग की पढ़ाई के लिए छात्रवृत्ति चाहिए", en: "Scholarship for engineering" },
    { icon: <Leaf size={14} />, hi: "PM-KISAN में कितना पैसा मिलता है?", en: "PM-KISAN amount" },
    { icon: <DollarSign size={14} />, hi: "मुद्रा लोन कैसे मिलेगा?", en: "Mudra loan process" },
    { icon: <Heart size={14} />, hi: "दिव्यांग पेंशन के बारे में बताओ", en: "Divyang pension info" },
];

type CallStatus = "idle" | "calling" | "success" | "error";

/* ─── Animated waveform decoration ──────────────────────────────────── */
const WAVE_BASE = [6, 12, 20, 14, 28, 10, 22, 8, 18, 26, 12, 16, 24, 8, 20, 14, 6];
function AmbientWave({ active }: { active: boolean }) {
    return (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 4, height: 48 }}>
            {WAVE_BASE.map((h, i) => (
                <motion.div
                    key={i}
                    style={{ width: 3, borderRadius: 999, background: "rgba(165,180,252,0.7)" }}
                    animate={active
                        ? { height: [h * 0.6, h * 1.8, h * 0.4, h * 2, h * 0.7], opacity: [0.6, 1, 0.4, 1, 0.6] }
                        : { height: [h * 0.3, h * 0.6, h * 0.2, h * 0.5, h * 0.3], opacity: [0.2, 0.4, 0.15, 0.35, 0.2] }
                    }
                    transition={{ duration: active ? 0.9 + (i % 5) * 0.12 : 2.4 + (i % 7) * 0.2, repeat: Infinity, delay: i * 0.06, ease: "easeInOut" }}
                />
            ))}
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════
   Main page
═══════════════════════════════════════════════════════════════════════ */
export default function VidyaHubPage() {
    const [phone, setPhone]         = useState("");
    const [status, setStatus]       = useState<CallStatus>("idle");
    const [errorMsg, setErrorMsg]   = useState("");

    const isValidPhone = phone.replace(/\s|-/g, "").length >= 10;

    const handleCall = async () => {
        if (!isValidPhone || status === "calling") return;
        setStatus("calling");
        setErrorMsg("");
        try {
            const res = await fetch("/call", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ phone_number: phone.trim() }),
            });
            const data = await res.json();
            if (!res.ok || data.error) {
                setErrorMsg(data.error || "Something went wrong. Please try again.");
                setStatus("error");
            } else {
                setStatus("success");
            }
        } catch {
            setErrorMsg("Could not reach server. Is the backend running?");
            setStatus("error");
        }
    };

    const reset = () => { setStatus("idle"); setPhone(""); setErrorMsg(""); };

    return (
        <div
            className="min-h-screen w-full flex flex-col items-center"
            style={{ background: "linear-gradient(135deg, #0f0c29 0%, #1a1040 40%, #0f172a 100%)" }}
        >
            {/* ── Animated background blobs ── */}
            <div className="pointer-events-none fixed inset-0 overflow-hidden">
                <motion.div
                    animate={{ scale: [1, 1.15, 1], x: [0, 30, 0], y: [0, -20, 0] }}
                    transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
                    style={{ position: "absolute", top: "5%", left: "10%", width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%)", filter: "blur(70px)" }}
                />
                <motion.div
                    animate={{ scale: [1, 1.2, 1], x: [0, -25, 0], y: [0, 30, 0] }}
                    transition={{ duration: 15, repeat: Infinity, ease: "easeInOut", delay: 2 }}
                    style={{ position: "absolute", bottom: "10%", right: "8%", width: 420, height: 420, borderRadius: "50%", background: "radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%)", filter: "blur(60px)" }}
                />
                <div style={{ position: "absolute", inset: 0, backgroundImage: "linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px)", backgroundSize: "40px 40px" }} />
            </div>

            {/* ── Header ── */}
            <div className="relative z-10 text-center pt-14 pb-4 px-4 w-full max-w-2xl">
                <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 16px", borderRadius: 999, background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.3)", color: "#a5b4fc", fontSize: 13, fontWeight: 600, letterSpacing: "0.06em", marginBottom: 16 }}>
                        <Zap size={13} /> AI PHONE ASSISTANT
                    </span>
                    <h1 style={{ fontSize: "clamp(2rem, 5vw, 3.5rem)", fontWeight: 800, background: "linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 50%, #818cf8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: 12, lineHeight: 1.1 }}>
                        Talk to Vidya
                    </h1>
                    <p style={{ color: "rgba(165,180,252,0.7)", fontSize: 15, maxWidth: 480, margin: "0 auto" }}>
                        Enter your number and Vidya will call you — ask about any government scheme in Hindi, English, or Hinglish across <strong style={{ color: "#a5b4fc" }}>3,400+ schemes</strong>.
                    </p>
                </motion.div>
            </div>

            {/* ── Main card ── */}
            <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                style={{ position: "relative", zIndex: 10, width: "100%", maxWidth: 520, margin: "24px 16px 0", borderRadius: 28, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)", backdropFilter: "blur(24px)", boxShadow: "0 32px 64px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)", overflow: "hidden" }}
            >
                <AnimatePresence mode="wait">

                    {/* ── Idle / input state ── */}
                    {(status === "idle" || status === "error") && (
                        <motion.div key="input" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "44px 36px 40px", textAlign: "center" }}
                        >
                            {/* Orb */}
                            <div style={{ position: "relative", marginBottom: 28 }}>
                                {[0, 1, 2].map(i => (
                                    <motion.div key={i}
                                        animate={{ scale: [1, 1.9], opacity: [0.35, 0] }}
                                        transition={{ duration: 3, repeat: Infinity, delay: i * 1.0, ease: "easeOut" }}
                                        style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", width: 140, height: 140, borderRadius: "50%", border: "1.5px solid rgba(99,102,241,0.45)", pointerEvents: "none" }}
                                    />
                                ))}
                                <motion.div
                                    animate={{ scale: [1, 1.04, 1] }}
                                    transition={{ duration: 2.8, repeat: Infinity, ease: "easeInOut" }}
                                    style={{ width: 140, height: 140, borderRadius: "50%", background: "radial-gradient(circle at 38% 32%, rgba(139,92,246,0.55) 0%, rgba(99,102,241,0.45) 40%, rgba(67,56,202,0.3) 100%)", border: "1.5px solid rgba(129,140,248,0.5)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 60px rgba(99,102,241,0.5), inset 0 1px 0 rgba(255,255,255,0.12)" }}
                                >
                                    <Phone size={44} color="rgba(255,255,255,0.92)" strokeWidth={1.8} />
                                </motion.div>
                            </div>

                            <h2 style={{ color: "#e0e7ff", fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Get a call from Vidya</h2>
                            <p style={{ color: "rgba(165,180,252,0.55)", fontSize: 13, marginBottom: 28, maxWidth: 300 }}>
                                Enter your phone number with country code and Vidya will call you right away.
                            </p>

                            {/* Phone input */}
                            <div style={{ width: "100%", marginBottom: 12 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 0, borderRadius: 14, overflow: "hidden", border: `1.5px solid ${status === "error" ? "rgba(239,68,68,0.5)" : "rgba(99,102,241,0.35)"}`, background: "rgba(255,255,255,0.04)", transition: "border-color 0.2s" }}>
                                    <span style={{ padding: "0 14px", color: "rgba(165,180,252,0.5)", fontSize: 18, borderRight: "1px solid rgba(255,255,255,0.08)", display: "flex", alignItems: "center", height: 52 }}>
                                        <Phone size={16} />
                                    </span>
                                    <input
                                        type="tel"
                                        value={phone}
                                        onChange={e => { setPhone(e.target.value); if (status === "error") setStatus("idle"); }}
                                        onKeyDown={e => e.key === "Enter" && handleCall()}
                                        placeholder="+91 98765 43210"
                                        style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#e0e7ff", fontSize: 17, fontWeight: 500, padding: "0 16px", height: 52, letterSpacing: "0.03em" }}
                                    />
                                </div>
                                {status === "error" && errorMsg && (
                                    <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
                                        style={{ color: "#f87171", fontSize: 12, marginTop: 8, display: "flex", alignItems: "center", gap: 5 }}>
                                        <AlertCircle size={12} /> {errorMsg}
                                    </motion.p>
                                )}
                            </div>

                            {/* Call button */}
                            <motion.button
                                whileHover={isValidPhone ? { scale: 1.04, boxShadow: "0 0 36px rgba(99,102,241,0.7)" } : {}}
                                whileTap={isValidPhone ? { scale: 0.97 } : {}}
                                onClick={handleCall}
                                disabled={!isValidPhone}
                                style={{ display: "inline-flex", alignItems: "center", gap: 10, padding: "14px 40px", borderRadius: 999, background: isValidPhone ? "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)" : "rgba(99,102,241,0.25)", color: isValidPhone ? "#fff" : "rgba(255,255,255,0.35)", fontWeight: 700, fontSize: 16, border: "none", cursor: isValidPhone ? "pointer" : "not-allowed", boxShadow: isValidPhone ? "0 8px 32px rgba(99,102,241,0.45)" : "none", transition: "all 0.25s", width: "100%" , justifyContent: "center" }}
                            >
                                <Phone size={18} /> Call Me Now
                            </motion.button>

                            <p style={{ color: "rgba(165,180,252,0.3)", fontSize: 11, marginTop: 14 }}>
                                Vidya will call within seconds. Standard call rates apply.
                            </p>
                        </motion.div>
                    )}

                    {/* ── Calling state ── */}
                    {status === "calling" && (
                        <motion.div key="calling" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "56px 36px", textAlign: "center" }}
                        >
                            <div style={{ position: "relative", marginBottom: 28 }}>
                                {[0, 1].map(i => (
                                    <motion.div key={i}
                                        animate={{ scale: [1, 2.2], opacity: [0.5, 0] }}
                                        transition={{ duration: 1.6, repeat: Infinity, delay: i * 0.8, ease: "easeOut" }}
                                        style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", width: 100, height: 100, borderRadius: "50%", border: "2px solid rgba(99,102,241,0.6)", pointerEvents: "none" }}
                                    />
                                ))}
                                <div style={{ width: 100, height: 100, borderRadius: "50%", background: "radial-gradient(circle, rgba(99,102,241,0.5) 0%, rgba(67,56,202,0.3) 100%)", border: "1.5px solid rgba(129,140,248,0.5)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 40px rgba(99,102,241,0.5)" }}>
                                    <Loader2 size={36} color="#a5b4fc" className="animate-spin" />
                                </div>
                            </div>
                            <h2 style={{ color: "#e0e7ff", fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Calling {phone}…</h2>
                            <p style={{ color: "rgba(165,180,252,0.55)", fontSize: 13, marginBottom: 24 }}>Vidya is dialling your number. Please pick up!</p>
                            <AmbientWave active={true} />
                        </motion.div>
                    )}

                    {/* ── Success state ── */}
                    {status === "success" && (
                        <motion.div key="success" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                            style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "56px 36px", textAlign: "center" }}
                        >
                            <motion.div
                                initial={{ scale: 0 }} animate={{ scale: 1 }}
                                transition={{ type: "spring", stiffness: 200, damping: 14 }}
                                style={{ width: 100, height: 100, borderRadius: "50%", background: "radial-gradient(circle, rgba(34,197,94,0.35) 0%, rgba(16,185,129,0.2) 100%)", border: "1.5px solid rgba(34,197,94,0.5)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 24, boxShadow: "0 0 40px rgba(34,197,94,0.35)" }}
                            >
                                <CheckCircle size={44} color="#4ade80" />
                            </motion.div>
                            <h2 style={{ color: "#e0e7ff", fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Call incoming!</h2>
                            <p style={{ color: "rgba(165,180,252,0.6)", fontSize: 14, marginBottom: 28, maxWidth: 300 }}>
                                Vidya is calling <strong style={{ color: "#a5b4fc" }}>{phone}</strong>. Pick up and start speaking — in Hindi, English, or Hinglish.
                            </p>
                            <AmbientWave active={true} />
                            <motion.button
                                whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
                                onClick={reset}
                                style={{ marginTop: 28, padding: "10px 28px", borderRadius: 999, background: "rgba(99,102,241,0.2)", border: "1px solid rgba(99,102,241,0.35)", color: "#a5b4fc", fontWeight: 600, fontSize: 14, cursor: "pointer" }}
                            >
                                Call another number
                            </motion.button>
                        </motion.div>
                    )}

                </AnimatePresence>
            </motion.div>

            {/* ── How it works ── */}
            <motion.div
                initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.35 }}
                style={{ position: "relative", zIndex: 10, width: "100%", maxWidth: 520, padding: "40px 16px 8px" }}
            >
                <p style={{ color: "rgba(165,180,252,0.5)", fontSize: 12, fontWeight: 700, letterSpacing: "0.08em", textAlign: "center", marginBottom: 20 }}>HOW IT WORKS</p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                    {[
                        { emoji: "📱", title: "Enter number", desc: "Type your mobile number with country code" },
                        { emoji: "📞", title: "Vidya calls you", desc: "Our server calls you within seconds" },
                        { emoji: "🤖", title: "Ask anything", desc: "Speak about any scheme, in any language" },
                    ].map((step, i) => (
                        <motion.div key={i}
                            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.4 + i * 0.1 }}
                            style={{ padding: "18px 12px", borderRadius: 18, textAlign: "center", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}
                        >
                            <div style={{ fontSize: 26, marginBottom: 8 }}>{step.emoji}</div>
                            <p style={{ color: "#e0e7ff", fontSize: 12, fontWeight: 700, marginBottom: 5 }}>{step.title}</p>
                            <p style={{ color: "rgba(165,180,252,0.4)", fontSize: 11, lineHeight: 1.5 }}>{step.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </motion.div>

            {/* ── Sample queries ── */}
            <motion.div
                initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.45 }}
                style={{ position: "relative", zIndex: 10, width: "100%", maxWidth: 520, padding: "32px 16px 56px" }}
            >
                <p style={{ color: "rgba(165,180,252,0.5)", fontSize: 12, fontWeight: 700, letterSpacing: "0.08em", textAlign: "center", marginBottom: 16 }}>
                    <Sparkles size={12} style={{ display: "inline", marginRight: 5 }} />TRY ASKING VIDYA
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    {SAMPLE_QUERIES.map((q, i) => (
                        <motion.div key={i}
                            whileHover={{ scale: 1.03, borderColor: "rgba(99,102,241,0.5)", background: "rgba(99,102,241,0.1)" }}
                            whileTap={{ scale: 0.97 }}
                            style={{ padding: "14px 16px", borderRadius: 14, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", cursor: "pointer", transition: "all 0.2s" }}
                        >
                            <p style={{ color: "#a5b4fc", fontSize: 12, marginBottom: 5, fontWeight: 600, display: "flex", alignItems: "center", gap: 5 }}>
                                {q.icon} {q.en} <ArrowRight size={10} style={{ marginLeft: "auto" }} />
                            </p>
                            <p style={{ color: "rgba(165,180,252,0.4)", fontSize: 11 }}>{q.hi}</p>
                        </motion.div>
                    ))}
                </div>
            </motion.div>
        </div>
    );
}
