import { useState, useRef, useEffect, useCallback } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Search, Mic, X, Loader2, MapPin, Tag, ArrowRight, ChevronDown, BookOpen, GraduationCap, Leaf, Briefcase, Heart, Users, Baby, Wheat } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

/* ─── Types ──────────────────────────────────────────────────────────── */
interface Scheme {
    id: string;
    name: string;
    details: string;
    benefits: string;
    eligibility: string;
    application_process: string;
    state: string;
    category: string;
    source: string;
}

/* ─── Data ───────────────────────────────────────────────────────────── */
const CATEGORIES = [
    { label: "All", icon: <BookOpen size={13} />, color: "#6366f1" },
    { label: "Scholarships", icon: <GraduationCap size={13} />, color: "#8b5cf6" },
    { label: "Agriculture", icon: <Leaf size={13} />, color: "#22c55e" },
    { label: "Business", icon: <Briefcase size={13} />, color: "#f59e0b" },
    { label: "Women", icon: <Users size={13} />, color: "#ec4899" },
    { label: "Health", icon: <Heart size={13} />, color: "#ef4444" },
    { label: "Senior Citizen", icon: <Users size={13} />, color: "#a78bfa" },
    { label: "Farmer", icon: <Wheat size={13} />, color: "#84cc16" },
    { label: "Child Welfare", icon: <Baby size={13} />, color: "#06b6d4" },
];

const SUGGESTIONS = [
    { icon: <GraduationCap size={14} />, text: "Scholarships for SC students in Maharashtra", color: "#8b5cf6" },
    { icon: <Leaf size={14} />, text: "Subsidies for farmers growing wheat", color: "#22c55e" },
    { icon: <Briefcase size={14} />, text: "Mudra loan for new business", color: "#f59e0b" },
    { icon: <Baby size={14} />, text: "Financial help for pregnant women", color: "#ec4899" },
    { icon: <Users size={14} />, text: "Pension schemes for senior citizens", color: "#a78bfa" },
];

const SORT_OPTIONS = ["Relevance", "Latest", "Most Popular"];

const CAT_COLORS: Record<string, { bg: string; text: string; border: string }> = {
    Scholarships: { bg: "rgba(139,92,246,0.1)", text: "#a78bfa", border: "rgba(139,92,246,0.25)" },
    Agriculture: { bg: "rgba(34,197,94,0.1)", text: "#4ade80", border: "rgba(34,197,94,0.25)" },
    Business: { bg: "rgba(245,158,11,0.1)", text: "#fbbf24", border: "rgba(245,158,11,0.25)" },
    Women: { bg: "rgba(236,72,153,0.1)", text: "#f472b6", border: "rgba(236,72,153,0.25)" },
    Health: { bg: "rgba(239,68,68,0.1)", text: "#f87171", border: "rgba(239,68,68,0.25)" },
    default: { bg: "rgba(99,102,241,0.1)", text: "#a5b4fc", border: "rgba(99,102,241,0.25)" },
};

function catStyle(cat: string) {
    return CAT_COLORS[cat] ?? CAT_COLORS.default;
}

/* ─── Skeleton card ──────────────────────────────────────────────────── */
function SkeletonCard() {
    return (
        <div style={{
            borderRadius: 20, padding: 24, height: 220,
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.07)",
        }}>
            {[60, 40, 80, 50].map((w, i) => (
                <motion.div
                    key={i}
                    animate={{ opacity: [0.3, 0.6, 0.3] }}
                    transition={{ duration: 1.4, repeat: Infinity, delay: i * 0.15 }}
                    style={{
                        height: i === 0 ? 18 : 12,
                        width: `${w}%`,
                        borderRadius: 6,
                        background: "rgba(165,180,252,0.12)",
                        marginBottom: i === 0 ? 20 : 10,
                    }}
                />
            ))}
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════
   Main page
═══════════════════════════════════════════════════════════════════════ */
export default function DiscoverPage() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<Scheme[]>([]);
    const [loading, setLoading] = useState(false);
    const [selectedScheme, setSelectedScheme] = useState<Scheme | null>(null);
    const [activeCategory, setActiveCategory] = useState("All");
    const [sortBy, setSortBy] = useState("Relevance");
    const [showSort, setShowSort] = useState(false);
    const [listening, setListening] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    /* ── Voice search ── */
    const startVoice = useCallback(() => {
        const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SR) { alert("Voice search isn't supported in this browser."); return; }
        const rec = new SR();
        rec.lang = "en-IN";
        rec.onstart = () => setListening(true);
        rec.onend = () => setListening(false);
        rec.onresult = (e: any) => {
            const t = e.results[0][0].transcript;
            setQuery(t);
            setTimeout(() => handleSearch(undefined, t), 100);
        };
        rec.start();
    }, []);

    /* ── Search ── */
    const handleSearch = async (e?: React.FormEvent, overrideQuery?: string) => {
        if (e) e.preventDefault();
        const q = (overrideQuery ?? query).trim();
        if (!q) return;
        setLoading(true);
        setResults([]);
        try {
            const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";
            const res = await fetch(`${API_URL}/search`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: q, limit: 12 }),
            });
            if (res.ok) {
                const data = await res.json();
                setResults(Array.isArray(data.results) ? data.results.map((r: any) => r[0]) : []);
            } else {
                setResults([]);
            }
        } catch { setResults([]); }
        finally { setLoading(false); }
    };

    /* ── Filtered results ── */
    const filtered = activeCategory === "All"
        ? results
        : results.filter(s => String(s.category ?? '').toLowerCase().includes(activeCategory.toLowerCase()));

    /* ── Close sort dropdown on outside click ── */
    useEffect(() => {
        const h = () => setShowSort(false);
        document.addEventListener("click", h);
        return () => document.removeEventListener("click", h);
    }, []);


    const hasQuery = query.trim().length > 0;

    return (
        <div style={{
            minHeight: "100vh",
            background: "#000000",
            paddingBottom: 80,
        }}>
            {/* ── Background blobs ── */}
            <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
                <motion.div
                    animate={{ scale: [1, 1.1, 1], x: [0, 20, 0] }}
                    transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
                    style={{
                        position: "absolute", top: "0%", left: "5%",
                        width: 500, height: 500, borderRadius: "50%",
                        background: "radial-gradient(circle, rgba(99,102,241,0.14) 0%, transparent 70%)",
                        filter: "blur(60px)",
                    }}
                />
                <motion.div
                    animate={{ scale: [1, 1.15, 1], y: [0, 30, 0] }}
                    transition={{ duration: 18, repeat: Infinity, ease: "easeInOut", delay: 3 }}
                    style={{
                        position: "absolute", bottom: "5%", right: "5%",
                        width: 400, height: 400, borderRadius: "50%",
                        background: "radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%)",
                        filter: "blur(55px)",
                    }}
                />
                <div style={{
                    position: "absolute", inset: 0,
                    backgroundImage: "linear-gradient(rgba(99,102,241,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.025) 1px, transparent 1px)",
                    backgroundSize: "42px 42px",
                }} />
            </div>

            <div style={{ position: "relative", zIndex: 1, maxWidth: 900, margin: "0 auto", padding: "0 20px" }}>

                {/* ── Hero / Stats ── */}
                <motion.div
                    initial={{ opacity: 0, y: -16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.55 }}
                    style={{ textAlign: "center", paddingTop: 64, paddingBottom: 40 }}
                >
                    <span style={{
                        display: "inline-flex", alignItems: "center", gap: 6,
                        padding: "5px 14px", borderRadius: 999,
                        background: "rgba(99,102,241,0.15)",
                        border: "1px solid rgba(99,102,241,0.3)",
                        color: "#a5b4fc", fontSize: 12, fontWeight: 700,
                        letterSpacing: "0.07em", marginBottom: 18,
                    }}>
                        <Search size={11} /> AI-POWERED SEARCH
                    </span>

                    <h1 style={{
                        fontSize: "clamp(2rem, 5vw, 3.2rem)", fontWeight: 800, lineHeight: 1.1, marginBottom: 12,
                        background: "linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 55%, #818cf8 100%)",
                        WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                    }}>
                        Discover Schemes
                    </h1>
                    <p style={{ color: "rgba(165,180,252,0.6)", fontSize: 16, marginBottom: 28, maxWidth: 520, margin: "0 auto 28px" }}>
                        Search through <strong style={{ color: "#a5b4fc" }}>3,400+</strong> government schemes using natural language.
                    </p>

                    {/* Stats row */}
                    <div style={{ display: "flex", justifyContent: "center", gap: 32, marginBottom: 36 }}>
                        {[
                            { n: "3,400+", l: "Schemes" },
                            { n: "28", l: "States" },
                            { n: "15+", l: "Categories" },
                        ].map((s, i) => (
                            <div key={i} style={{ textAlign: "center" }}>
                                <div style={{ color: "#e0e7ff", fontWeight: 800, fontSize: 20 }}>{s.n}</div>
                                <div style={{ color: "rgba(165,180,252,0.45)", fontSize: 12 }}>{s.l}</div>
                            </div>
                        ))}
                    </div>

                    {/* ── Search bar ── */}
                    <form onSubmit={handleSearch} id="search-form" style={{ position: "relative", marginBottom: 20 }}>
                        <div style={{
                            display: "flex", alignItems: "center",
                            background: "rgba(255,255,255,0.07)",
                            border: "1.5px solid rgba(99,102,241,0.35)",
                            borderRadius: 999,
                            padding: "6px 6px 6px 22px",
                            backdropFilter: "blur(16px)",
                            boxShadow: "0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px rgba(99,102,241,0.1) inset",
                            transition: "border-color 0.2s, box-shadow 0.2s",
                        }}
                            onFocus={() => { }}
                        >
                            <Search size={18} color="rgba(165,180,252,0.5)" style={{ flexShrink: 0 }} />
                            <input
                                ref={inputRef}
                                value={query}
                                onChange={e => setQuery(e.target.value)}
                                placeholder="E.g. Engineering scholarships for SC students in Maharashtra…"
                                style={{
                                    flex: 1, background: "transparent", border: "none", outline: "none",
                                    color: "#e0e7ff", fontSize: 15, padding: "10px 12px",
                                    caretColor: "#818cf8",
                                }}
                            />

                            {/* Clear */}
                            <AnimatePresence>
                                {query && (
                                    <motion.button
                                        initial={{ opacity: 0, scale: 0.7 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.7 }}
                                        type="button"
                                        onClick={() => { setQuery(""); setResults([]); inputRef.current?.focus(); }}
                                        style={{
                                            background: "none", border: "none", cursor: "pointer",
                                            color: "rgba(165,180,252,0.5)", padding: "0 6px",
                                            display: "flex", alignItems: "center",
                                        }}
                                    >
                                        <X size={16} />
                                    </motion.button>
                                )}
                            </AnimatePresence>

                            {/* Voice */}
                            <motion.button
                                type="button"
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                                onClick={startVoice}
                                style={{
                                    width: 40, height: 40, borderRadius: "50%", flexShrink: 0,
                                    background: listening ? "rgba(239,68,68,0.3)" : "rgba(99,102,241,0.2)",
                                    border: `1px solid ${listening ? "rgba(239,68,68,0.5)" : "rgba(99,102,241,0.35)"}`,
                                    color: listening ? "#f87171" : "#a5b4fc",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    cursor: "pointer", marginRight: 6, transition: "all 0.2s",
                                }}
                            >
                                {listening
                                    ? <motion.div animate={{ scale: [1, 1.3, 1] }} transition={{ duration: 0.7, repeat: Infinity }}><Mic size={16} /></motion.div>
                                    : <Mic size={16} />
                                }
                            </motion.button>

                            {/* Search button */}
                            <motion.button
                                type="submit"
                                whileHover={{ scale: 1.04, boxShadow: "0 0 24px rgba(99,102,241,0.6)" }}
                                whileTap={{ scale: 0.97 }}
                                disabled={loading}
                                style={{
                                    height: 44, padding: "0 28px", borderRadius: 999,
                                    background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                                    color: "#fff", fontWeight: 700, fontSize: 14,
                                    border: "none", cursor: loading ? "not-allowed" : "pointer",
                                    display: "flex", alignItems: "center", gap: 7,
                                    boxShadow: "0 4px 20px rgba(99,102,241,0.4)",
                                    flexShrink: 0, transition: "all 0.25s",
                                }}
                            >
                                {loading ? <Loader2 size={16} className="animate-spin" /> : <><Search size={15} /> Search</>}
                            </motion.button>
                        </div>
                    </form>

                    {/* ── Category filter chips ── */}
                    <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 8 }}>
                        {CATEGORIES.map(cat => (
                            <motion.button
                                key={cat.label}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => setActiveCategory(cat.label)}
                                style={{
                                    display: "inline-flex", alignItems: "center", gap: 5,
                                    padding: "6px 14px", borderRadius: 999, fontSize: 13, fontWeight: 600,
                                    cursor: "pointer", border: "1px solid",
                                    borderColor: activeCategory === cat.label ? cat.color : "rgba(255,255,255,0.1)",
                                    background: activeCategory === cat.label ? `${cat.color}22` : "rgba(255,255,255,0.04)",
                                    color: activeCategory === cat.label ? cat.color : "rgba(165,180,252,0.5)",
                                    transition: "all 0.2s",
                                }}
                            >
                                {cat.icon} {cat.label}
                            </motion.button>
                        ))}
                    </div>
                </motion.div>

                {/* ── Content area ── */}
                <AnimatePresence mode="wait">

                    {/* Loading skeletons */}
                    {loading && (
                        <motion.div key="loading"
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        >
                            <p style={{ color: "rgba(165,180,252,0.5)", textAlign: "center", marginBottom: 28, fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                                <Loader2 size={14} className="animate-spin" /> Searching government schemes…
                            </p>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 16 }}>
                                {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
                            </div>
                        </motion.div>
                    )}

                    {/* Results */}
                    {!loading && filtered.length > 0 && (
                        <motion.div key="results"
                            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                            transition={{ duration: 0.35 }}
                        >
                            {/* Sort + count row */}
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                                <span style={{ color: "rgba(165,180,252,0.5)", fontSize: 13 }}>
                                    <strong style={{ color: "#e0e7ff" }}>{filtered.length}</strong> results found
                                </span>
                                <div style={{ position: "relative" }} onClick={e => e.stopPropagation()}>
                                    <motion.button
                                        whileHover={{ scale: 1.03 }}
                                        onClick={() => setShowSort(v => !v)}
                                        style={{
                                            display: "flex", alignItems: "center", gap: 6,
                                            padding: "6px 14px", borderRadius: 999, fontSize: 13,
                                            background: "rgba(255,255,255,0.05)",
                                            border: "1px solid rgba(255,255,255,0.1)",
                                            color: "#a5b4fc", cursor: "pointer",
                                        }}
                                    >
                                        Sort: {sortBy} <ChevronDown size={13} />
                                    </motion.button>
                                    <AnimatePresence>
                                        {showSort && (
                                            <motion.div
                                                initial={{ opacity: 0, y: -8, scale: 0.95 }}
                                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                                exit={{ opacity: 0, y: -8, scale: 0.95 }}
                                                style={{
                                                    position: "absolute", right: 0, top: "calc(100% + 8px)",
                                                    background: "rgba(15,12,41,0.98)", backdropFilter: "blur(12px)",
                                                    border: "1px solid rgba(99,102,241,0.3)", borderRadius: 14,
                                                    padding: 6, zIndex: 50, minWidth: 150,
                                                    boxShadow: "0 16px 40px rgba(0,0,0,0.4)",
                                                }}
                                            >
                                                {SORT_OPTIONS.map(s => (
                                                    <div
                                                        key={s}
                                                        onClick={() => { setSortBy(s); setShowSort(false); }}
                                                        style={{
                                                            padding: "8px 14px", borderRadius: 8, cursor: "pointer",
                                                            fontSize: 13, fontWeight: sortBy === s ? 700 : 400,
                                                            color: sortBy === s ? "#a5b4fc" : "rgba(165,180,252,0.6)",
                                                            background: sortBy === s ? "rgba(99,102,241,0.15)" : "transparent",
                                                            transition: "all 0.15s",
                                                        }}
                                                    >
                                                        {s}
                                                    </div>
                                                ))}
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            </div>

                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 16 }}>
                                {filtered.map((scheme, idx) => {
                                    const cs = catStyle(scheme.category);
                                    return (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.35, delay: idx * 0.05 }}
                                            whileHover={{ y: -4, boxShadow: "0 20px 48px rgba(0,0,0,0.4)" }}
                                            style={{
                                                borderRadius: 20, padding: 22,
                                                background: "rgba(255,255,255,0.04)",
                                                border: "1px solid rgba(255,255,255,0.08)",
                                                backdropFilter: "blur(16px)",
                                                display: "flex", flexDirection: "column",
                                                transition: "all 0.25s",
                                                cursor: "pointer",
                                            }}
                                            onClick={() => setSelectedScheme(scheme)}
                                        >
                                            {/* Tags row */}
                                            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
                                                {scheme.state && scheme.state !== "NaN" && (
                                                    <span style={{
                                                        fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 999,
                                                        background: "rgba(99,102,241,0.12)",
                                                        border: "1px solid rgba(99,102,241,0.25)",
                                                        color: "#a5b4fc",
                                                        display: "flex", alignItems: "center", gap: 4,
                                                    }}>
                                                        <MapPin size={10} /> {scheme.state}
                                                    </span>
                                                )}
                                                {scheme.category && scheme.category !== "NaN" && (
                                                    <span style={{
                                                        fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 999,
                                                        background: cs.bg, border: `1px solid ${cs.border}`, color: cs.text,
                                                        display: "flex", alignItems: "center", gap: 4,
                                                    }}>
                                                        <Tag size={10} />
                                                        {scheme.category.length > 18 ? scheme.category.substring(0, 18) + "…" : scheme.category}
                                                    </span>
                                                )}
                                            </div>

                                            <h3 style={{
                                                color: "#e0e7ff", fontSize: 15, fontWeight: 700,
                                                lineHeight: 1.4, marginBottom: 10,
                                                display: "-webkit-box", WebkitLineClamp: 2,
                                                WebkitBoxOrient: "vertical", overflow: "hidden",
                                            }}>
                                                {scheme.name}
                                            </h3>

                                            <p style={{
                                                color: "rgba(165,180,252,0.55)", fontSize: 13, lineHeight: 1.6,
                                                display: "-webkit-box", WebkitLineClamp: 3,
                                                WebkitBoxOrient: "vertical", overflow: "hidden",
                                                flex: 1, marginBottom: 16,
                                            }}>
                                                {scheme.details}
                                            </p>

                                            <div style={{
                                                display: "flex", justifyContent: "space-between", alignItems: "center",
                                                paddingTop: 14,
                                                borderTop: "1px solid rgba(255,255,255,0.06)",
                                            }}>
                                                <span style={{ color: "rgba(165,180,252,0.4)", fontSize: 11 }}>Click to view details</span>
                                                <motion.span
                                                    whileHover={{ x: 3 }}
                                                    style={{ color: "#818cf8", display: "flex", alignItems: "center", gap: 4, fontSize: 12, fontWeight: 600 }}
                                                >
                                                    Details <ArrowRight size={12} />
                                                </motion.span>
                                            </div>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        </motion.div>
                    )}

                    {/* No results */}
                    {!loading && hasQuery && !loading && results.length === 0 && (
                        <motion.div key="noresults"
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            style={{ textAlign: "center", padding: "60px 20px" }}
                        >
                            <div style={{ fontSize: 48, marginBottom: 16 }}>🔍</div>
                            <h3 style={{ color: "#e0e7ff", fontSize: 20, fontWeight: 700, marginBottom: 8 }}>No schemes found</h3>
                            <p style={{ color: "rgba(165,180,252,0.5)", fontSize: 14 }}>Try rephrasing or using fewer keywords.</p>
                        </motion.div>
                    )}

                    {/* Empty state */}
                    {!loading && !hasQuery && (
                        <motion.div key="empty"
                            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                            transition={{ duration: 0.4 }}
                        >
                            <p style={{
                                color: "rgba(165,180,252,0.45)", fontSize: 12, fontWeight: 700,
                                letterSpacing: "0.08em", textAlign: "center", marginBottom: 16,
                            }}>
                                ✨ POPULAR SEARCHES
                            </p>
                            <div style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: 560, margin: "0 auto 48px" }}>
                                {SUGGESTIONS.map((s, i) => (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, x: -12 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.07 }}
                                        whileHover={{ scale: 1.02, borderColor: s.color, background: `${s.color}14` }}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={() => {
                                            setQuery(s.text);
                                            setTimeout(() => {
                                                const form = document.getElementById("search-form");
                                                if (form) form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
                                            }, 50);
                                        }}
                                        style={{
                                            display: "flex", alignItems: "center", gap: 12,
                                            padding: "14px 18px", borderRadius: 14, cursor: "pointer",
                                            background: "rgba(255,255,255,0.04)",
                                            border: "1px solid rgba(255,255,255,0.08)",
                                            transition: "all 0.2s",
                                        }}
                                    >
                                        <span style={{
                                            width: 32, height: 32, borderRadius: 8, flexShrink: 0,
                                            background: `${s.color}22`, border: `1px solid ${s.color}44`,
                                            color: s.color, display: "flex", alignItems: "center", justifyContent: "center",
                                        }}>
                                            {s.icon}
                                        </span>
                                        <span style={{ color: "#c7d2fe", fontSize: 14, fontWeight: 500 }}>{s.text}</span>
                                        <ArrowRight size={14} color="rgba(165,180,252,0.3)" style={{ marginLeft: "auto", flexShrink: 0 }} />
                                    </motion.div>
                                ))}
                            </div>

                            {/* Browse categories */}
                            <p style={{
                                color: "rgba(165,180,252,0.45)", fontSize: 12, fontWeight: 700,
                                letterSpacing: "0.08em", textAlign: "center", marginBottom: 16,
                            }}>
                                BROWSE BY CATEGORY
                            </p>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))", gap: 12 }}>
                                {CATEGORIES.slice(1).map(cat => (
                                    <motion.div
                                        key={cat.label}
                                        whileHover={{ scale: 1.04, borderColor: cat.color }}
                                        whileTap={{ scale: 0.97 }}
                                        onClick={() => { setActiveCategory(cat.label); setQuery(cat.label); setTimeout(() => { const f = document.getElementById("search-form"); if (f) f.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true })); }, 50); }}
                                        style={{
                                            padding: "18px 14px", borderRadius: 16, textAlign: "center",
                                            background: `${cat.color}0d`,
                                            border: `1px solid ${cat.color}2a`,
                                            cursor: "pointer", transition: "all 0.2s",
                                        }}
                                    >
                                        <div style={{
                                            width: 36, height: 36, borderRadius: 10, margin: "0 auto 10px",
                                            background: `${cat.color}22`, border: `1px solid ${cat.color}44`,
                                            display: "flex", alignItems: "center", justifyContent: "center",
                                            color: cat.color,
                                        }}>
                                            {cat.icon}
                                        </div>
                                        <p style={{ color: "#c7d2fe", fontSize: 12, fontWeight: 600 }}>{cat.label}</p>
                                    </motion.div>
                                ))}
                            </div>
                        </motion.div>
                    )}

                </AnimatePresence>
            </div>

            {/* ── Full-detail dialog ── */}
            <Dialog open={!!selectedScheme} onOpenChange={open => !open && setSelectedScheme(null)}>
                <DialogContent style={{ maxWidth: 700, maxHeight: "90vh", display: "flex", flexDirection: "column", padding: 0, overflow: "hidden", background: "#0f0c29", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 24 }}>
                    {selectedScheme && (
                        <>
                            <DialogHeader style={{ padding: "24px 28px 20px", borderBottom: "1px solid rgba(255,255,255,0.08)", background: "rgba(99,102,241,0.06)" }}>
                                <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                                    {selectedScheme.state !== "NaN" && (
                                        <span style={{ fontSize: 12, padding: "3px 12px", borderRadius: 999, background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.3)", color: "#a5b4fc", display: "flex", alignItems: "center", gap: 4 }}>
                                            <MapPin size={11} /> {selectedScheme.state}
                                        </span>
                                    )}
                                    {selectedScheme.category !== "NaN" && (
                                        <span style={{ fontSize: 12, padding: "3px 12px", borderRadius: 999, background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.3)", color: "#c4b5fd", display: "flex", alignItems: "center", gap: 4 }}>
                                            <Tag size={11} /> {selectedScheme.category}
                                        </span>
                                    )}
                                </div>
                                <DialogTitle style={{ color: "#e0e7ff", fontSize: 22, fontWeight: 700 }}>{selectedScheme.name}</DialogTitle>
                                <DialogDescription style={{ color: "rgba(165,180,252,0.5)", fontSize: 12, marginTop: 4 }}>
                                    Source: <a href={selectedScheme.source} target="_blank" rel="noreferrer" style={{ color: "#818cf8", textDecoration: "underline" }}>{selectedScheme.source.substring(0, 55)}…</a>
                                </DialogDescription>
                            </DialogHeader>
                            <ScrollArea style={{ flex: 1 }}>
                                <div style={{ padding: "24px 28px", display: "flex", flexDirection: "column", gap: 20 }}>
                                    <div>
                                        <h3 style={{ color: "#e0e7ff", fontSize: 14, fontWeight: 700, marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}><Tag size={14} color="#818cf8" /> Overview</h3>
                                        <p style={{ color: "rgba(165,180,252,0.65)", fontSize: 14, lineHeight: 1.7 }}>{selectedScheme.details}</p>
                                    </div>
                                    {selectedScheme.benefits && selectedScheme.benefits !== "NaN" && (
                                        <div style={{ padding: "16px 20px", borderRadius: 14, background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.15)" }}>
                                            <h3 style={{ color: "#4ade80", fontSize: 14, fontWeight: 700, marginBottom: 8 }}>✅ Benefits</h3>
                                            <p style={{ color: "rgba(165,180,252,0.65)", fontSize: 14, lineHeight: 1.7, whiteSpace: "pre-line" }}>{selectedScheme.benefits}</p>
                                        </div>
                                    )}
                                    {selectedScheme.eligibility && selectedScheme.eligibility !== "NaN" && (
                                        <div>
                                            <h3 style={{ color: "#e0e7ff", fontSize: 14, fontWeight: 700, marginBottom: 8 }}>👤 Eligibility Criteria</h3>
                                            <p style={{ color: "rgba(165,180,252,0.65)", fontSize: 14, lineHeight: 1.7, whiteSpace: "pre-line" }}>{selectedScheme.eligibility}</p>
                                        </div>
                                    )}
                                    {selectedScheme.application_process && selectedScheme.application_process !== "NaN" && (
                                        <div style={{ padding: "16px 20px", borderRadius: 14, background: "rgba(59,130,246,0.06)", border: "1px solid rgba(59,130,246,0.15)" }}>
                                            <h3 style={{ color: "#60a5fa", fontSize: 14, fontWeight: 700, marginBottom: 8 }}>📋 How to Apply</h3>
                                            <p style={{ color: "rgba(165,180,252,0.65)", fontSize: 14, lineHeight: 1.7, whiteSpace: "pre-line" }}>{selectedScheme.application_process}</p>
                                        </div>
                                    )}
                                </div>
                            </ScrollArea>
                            <div style={{ padding: "16px 28px", borderTop: "1px solid rgba(255,255,255,0.08)", display: "flex", gap: 10, justifyContent: "flex-end" }}>
                                <motion.a
                                    href={selectedScheme.source} target="_blank" rel="noreferrer"
                                    whileHover={{ scale: 1.04, boxShadow: "0 0 20px rgba(99,102,241,0.5)" }}
                                    style={{
                                        padding: "10px 24px", borderRadius: 999, fontSize: 13, fontWeight: 700,
                                        background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                                        color: "#fff", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 6,
                                        boxShadow: "0 4px 16px rgba(99,102,241,0.35)",
                                    }}
                                >
                                    Apply Now <ArrowRight size={13} />
                                </motion.a>
                                <motion.button
                                    whileHover={{ scale: 1.03 }}
                                    onClick={() => setSelectedScheme(null)}
                                    style={{
                                        padding: "10px 20px", borderRadius: 999, fontSize: 13, fontWeight: 600,
                                        background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)",
                                        color: "rgba(165,180,252,0.7)", cursor: "pointer",
                                    }}
                                >
                                    Close
                                </motion.button>
                            </div>
                        </>
                    )}
                </DialogContent>
            </Dialog>
        </div>
    );
}
