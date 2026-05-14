import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Search, Mic, X, Loader2, MapPin, Tag, ArrowRight,
    ChevronDown, BookOpen, GraduationCap, Leaf, Briefcase,
    Heart, Users, Baby, Wheat, IndianRupee, ExternalLink, Sparkles,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

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

/* ─── Benefit amount extractor ───────────────────────────────────────── */
function extractBenefitAmount(benefits: string): string | null {
    if (!benefits || benefits === "NaN") return null;
    const patterns = [
        /₹\s*[\d,]+(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores)?(?:\s*(?:per\s+year|\/year|p\.a\.|per annum|annually|per\s+month|\/month))?/i,
        /Rs\.?\s*[\d,]+(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores)?(?:\s*(?:per\s+year|\/year|p\.a\.|per annum|annually|per\s+month|\/month))?/i,
        /[\d,]+(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores)(?:\s*(?:per\s+year|\/year|per annum|annually))?/i,
        /INR\s*[\d,]+(?:\.\d+)?/i,
    ];
    for (const pat of patterns) {
        const m = benefits.match(pat);
        if (m) {
            let val = m[0].trim().replace(/\s+/g, " ");
            if (!val.startsWith("₹") && !val.toLowerCase().startsWith("rs") && !val.toLowerCase().startsWith("inr")) {
                val = "₹" + val;
            }
            return val;
        }
    }
    return null;
}

/* ─── Data ───────────────────────────────────────────────────────────── */
const CATEGORIES = [
    { label: "All",           icon: BookOpen,      color: "indigo"  },
    { label: "Scholarships",  icon: GraduationCap, color: "violet"  },
    { label: "Agriculture",   icon: Leaf,          color: "emerald" },
    { label: "Business",      icon: Briefcase,     color: "amber"   },
    { label: "Women",         icon: Users,         color: "pink"    },
    { label: "Health",        icon: Heart,         color: "rose"    },
    { label: "Senior Citizen",icon: Users,         color: "purple"  },
    { label: "Farmer",        icon: Wheat,         color: "lime"    },
    { label: "Child Welfare", icon: Baby,          color: "cyan"    },
] as const;

const SUGGESTIONS = [
    { icon: GraduationCap, text: "Scholarships for SC students in Maharashtra", color: "violet" },
    { icon: Leaf,          text: "Subsidies for farmers growing wheat",          color: "emerald"},
    { icon: Briefcase,     text: "Mudra loan for new business",                  color: "amber"  },
    { icon: Baby,          text: "Financial help for pregnant women",            color: "pink"   },
    { icon: Users,         text: "Pension schemes for senior citizens",          color: "purple" },
] as const;

const CAT_BADGE_MAP: Record<string, string> = {
    Scholarships:   "bg-violet-500/10 text-violet-300 border-violet-500/20",
    Agriculture:    "bg-emerald-500/10 text-emerald-300 border-emerald-500/20",
    Business:       "bg-amber-500/10 text-amber-300 border-amber-500/20",
    Women:          "bg-pink-500/10 text-pink-300 border-pink-500/20",
    Health:         "bg-rose-500/10 text-rose-300 border-rose-500/20",
    "Senior Citizen":"bg-purple-500/10 text-purple-300 border-purple-500/20",
    Farmer:         "bg-lime-500/10 text-lime-300 border-lime-500/20",
    "Child Welfare":"bg-cyan-500/10 text-cyan-300 border-cyan-500/20",
};

const COLOR_MAP: Record<string, string> = {
    indigo: "bg-indigo-500/10 text-indigo-300 border-indigo-500/20 hover:bg-indigo-500/20",
    violet: "bg-violet-500/10 text-violet-300 border-violet-500/20 hover:bg-violet-500/20",
    emerald:"bg-emerald-500/10 text-emerald-300 border-emerald-500/20 hover:bg-emerald-500/20",
    amber:  "bg-amber-500/10 text-amber-300 border-amber-500/20 hover:bg-amber-500/20",
    pink:   "bg-pink-500/10 text-pink-300 border-pink-500/20 hover:bg-pink-500/20",
    rose:   "bg-rose-500/10 text-rose-300 border-rose-500/20 hover:bg-rose-500/20",
    purple: "bg-purple-500/10 text-purple-300 border-purple-500/20 hover:bg-purple-500/20",
    lime:   "bg-lime-500/10 text-lime-300 border-lime-500/20 hover:bg-lime-500/20",
    cyan:   "bg-cyan-500/10 text-cyan-300 border-cyan-500/20 hover:bg-cyan-500/20",
};

const SORT_OPTIONS = ["Relevance", "Latest", "Most Popular"];

/* ─── Skeleton card ──────────────────────────────────────────────────── */
function SkeletonCard() {
    return (
        <Card className="border-white/5 bg-white/[0.02] overflow-hidden">
            <CardContent className="p-5 space-y-3">
                {[60, 40, 80, 50].map((w, i) => (
                    <motion.div
                        key={i}
                        animate={{ opacity: [0.2, 0.5, 0.2] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.12 }}
                        className="rounded-md bg-white/5"
                        style={{ height: i === 0 ? 18 : 11, width: `${w}%` }}
                    />
                ))}
            </CardContent>
        </Card>
    );
}

/* ═══════════════════════════════════════════════════════════════════════
   Main page
═══════════════════════════════════════════════════════════════════════ */
export default function DiscoverPage() {
    const [query, setQuery]               = useState("");
    const [results, setResults]           = useState<Scheme[]>([]);
    const [loading, setLoading]           = useState(false);
    const [selectedScheme, setSelected]   = useState<Scheme | null>(null);
    const [activeCategory, setActiveCat]  = useState("All");
    const [sortBy, setSortBy]             = useState("Relevance");
    const [showSort, setShowSort]         = useState(false);
    const [listening, setListening]       = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    /* ── Voice search ── */
    const startVoice = useCallback(() => {
        const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SR) { alert("Voice search isn't supported in this browser."); return; }
        const rec = new SR();
        rec.lang = "en-IN";
        rec.onstart = () => setListening(true);
        rec.onend   = () => setListening(false);
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
            const res = await fetch("/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: q, limit: 24 }),
            });
            if (res.ok) {
                const data = await res.json();
                setResults(Array.isArray(data.results) ? data.results.map((r: any) => r[0]) : []);
            } else { setResults([]); }
        } catch { setResults([]); }
        finally { setLoading(false); }
    };

    const triggerSearch = (text: string) => {
        setQuery(text);
        setTimeout(() => {
            const f = document.getElementById("search-form");
            if (f) f.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
        }, 50);
    };

    const filtered = activeCategory === "All"
        ? results
        : results.filter(s => String(s.category ?? "").toLowerCase().includes(activeCategory.toLowerCase()));

    useEffect(() => {
        const h = () => setShowSort(false);
        document.addEventListener("click", h);
        return () => document.removeEventListener("click", h);
    }, []);

    const hasQuery = query.trim().length > 0;

    return (
        <div className="min-h-screen bg-[hsl(var(--background))] pb-24">
            {/* ── Ambient background ── */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
                <motion.div
                    animate={{ scale: [1, 1.12, 1], x: [0, 24, 0] }}
                    transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
                    className="absolute top-0 left-[5%] w-[500px] h-[500px] rounded-full"
                    style={{ background: "radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)", filter: "blur(72px)" }}
                />
                <motion.div
                    animate={{ scale: [1, 1.18, 1], y: [0, 32, 0] }}
                    transition={{ duration: 18, repeat: Infinity, ease: "easeInOut", delay: 3 }}
                    className="absolute bottom-[5%] right-[5%] w-[420px] h-[420px] rounded-full"
                    style={{ background: "radial-gradient(circle, rgba(139,92,246,0.10) 0%, transparent 70%)", filter: "blur(60px)" }}
                />
                <div className="absolute inset-0 opacity-[0.015]"
                    style={{ backgroundImage: "linear-gradient(rgba(165,180,252,1) 1px, transparent 1px), linear-gradient(90deg, rgba(165,180,252,1) 1px, transparent 1px)", backgroundSize: "44px 44px" }} />
            </div>

            <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6">

                {/* ── Hero ── */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="text-center pt-16 pb-10"
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-bold tracking-widest mb-5">
                        <Sparkles size={11} /> AI-POWERED SEARCH
                    </div>

                    <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight mb-3 bg-gradient-to-br from-slate-100 via-indigo-200 to-violet-300 bg-clip-text text-transparent">
                        Discover Schemes
                    </h1>
                    <p className="text-muted-foreground text-base mb-8 max-w-md mx-auto">
                        Search through <span className="text-indigo-300 font-semibold">3,400+</span> government schemes using natural language
                    </p>

                    {/* Stats */}
                    <div className="flex justify-center gap-10 mb-8">
                        {[{ n: "3,400+", l: "Schemes" }, { n: "28", l: "States" }, { n: "15+", l: "Categories" }].map((s, i) => (
                            <div key={i} className="text-center">
                                <div className="text-slate-100 font-bold text-xl">{s.n}</div>
                                <div className="text-muted-foreground text-xs mt-0.5">{s.l}</div>
                            </div>
                        ))}
                    </div>

                    {/* ── Search bar ── */}
                    <form id="search-form" onSubmit={handleSearch} className="mb-6">
                        <div className="flex items-center gap-2 bg-white/[0.04] border border-indigo-500/25 rounded-2xl px-4 py-2 backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.35)] focus-within:border-indigo-500/50 transition-all duration-200">
                            <Search size={17} className="text-muted-foreground shrink-0" />
                            <input
                                ref={inputRef}
                                value={query}
                                onChange={e => setQuery(e.target.value)}
                                placeholder="E.g. Engineering scholarships for SC students in Maharashtra…"
                                className="flex-1 bg-transparent border-none outline-none text-slate-100 text-sm placeholder:text-muted-foreground/60 py-2.5 caret-indigo-400"
                            />
                            <AnimatePresence>
                                {query && (
                                    <motion.button
                                        initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.6 }}
                                        type="button"
                                        onClick={() => { setQuery(""); setResults([]); inputRef.current?.focus(); }}
                                        className="text-muted-foreground hover:text-slate-200 transition-colors p-1"
                                    >
                                        <X size={15} />
                                    </motion.button>
                                )}
                            </AnimatePresence>
                            <Button
                                type="button" variant="ghost" size="icon"
                                onClick={startVoice}
                                className={cn(
                                    "rounded-xl w-9 h-9 shrink-0 transition-all",
                                    listening
                                        ? "bg-red-500/20 border border-red-500/40 text-red-400"
                                        : "bg-white/5 border border-white/10 text-muted-foreground hover:text-indigo-300 hover:border-indigo-500/30"
                                )}
                            >
                                {listening
                                    ? <motion.div animate={{ scale: [1, 1.35, 1] }} transition={{ duration: 0.65, repeat: Infinity }}><Mic size={15} /></motion.div>
                                    : <Mic size={15} />
                                }
                            </Button>
                            <Button
                                type="submit" disabled={loading}
                                className="rounded-xl h-9 px-5 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 shrink-0 transition-all"
                            >
                                {loading ? <Loader2 size={15} className="animate-spin" /> : <><Search size={14} className="mr-1.5" />Search</>}
                            </Button>
                        </div>
                    </form>

                    {/* ── Category chips ── */}
                    <div className="flex flex-wrap justify-center gap-2">
                        {CATEGORIES.map(cat => {
                            const Icon = cat.icon;
                            const active = activeCategory === cat.label;
                            return (
                                <motion.button
                                    key={cat.label}
                                    whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.95 }}
                                    onClick={() => setActiveCat(cat.label)}
                                    className={cn(
                                        "inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold border transition-all duration-200",
                                        active ? COLOR_MAP[cat.color] : "bg-white/[0.03] border-white/[0.07] text-muted-foreground hover:border-white/20 hover:text-slate-300"
                                    )}
                                >
                                    <Icon size={12} /> {cat.label}
                                </motion.button>
                            );
                        })}
                    </div>
                </motion.div>

                {/* ── Content ── */}
                <AnimatePresence mode="wait">

                    {/* Loading */}
                    {loading && (
                        <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                            <p className="text-muted-foreground text-sm text-center flex items-center justify-center gap-2 mb-6">
                                <Loader2 size={13} className="animate-spin" /> Searching government schemes…
                            </p>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
                            </div>
                        </motion.div>
                    )}

                    {/* Results */}
                    {!loading && filtered.length > 0 && (
                        <motion.div key="results" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
                            {/* Sort row */}
                            <div className="flex items-center justify-between mb-5">
                                <p className="text-muted-foreground text-sm">
                                    <span className="text-slate-200 font-semibold">{filtered.length}</span> results found
                                </p>
                                <div className="relative" onClick={e => e.stopPropagation()}>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setShowSort(v => !v)}
                                        className="bg-white/[0.04] border-white/10 text-muted-foreground hover:text-slate-200 hover:bg-white/[0.07] rounded-xl text-xs gap-1.5"
                                    >
                                        Sort: {sortBy} <ChevronDown size={12} />
                                    </Button>
                                    <AnimatePresence>
                                        {showSort && (
                                            <motion.div
                                                initial={{ opacity: 0, y: -6, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -6, scale: 0.95 }}
                                                className="absolute right-0 top-[calc(100%+8px)] bg-[#0d1117] border border-white/10 rounded-2xl shadow-2xl p-1.5 z-50 min-w-[160px] backdrop-blur-xl"
                                            >
                                                {SORT_OPTIONS.map(s => (
                                                    <div
                                                        key={s}
                                                        onClick={() => { setSortBy(s); setShowSort(false); }}
                                                        className={cn(
                                                            "px-4 py-2 rounded-xl text-sm cursor-pointer transition-all",
                                                            sortBy === s ? "bg-indigo-500/15 text-indigo-300 font-semibold" : "text-muted-foreground hover:bg-white/5 hover:text-slate-200"
                                                        )}
                                                    >
                                                        {s}
                                                    </div>
                                                ))}
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                {filtered.map((scheme, idx) => {
                                    const catClass = CAT_BADGE_MAP[scheme.category] ?? "bg-indigo-500/10 text-indigo-300 border-indigo-500/20";
                                    const amount = extractBenefitAmount(scheme.benefits);
                                    return (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0, y: 18 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.3, delay: idx * 0.04 }}
                                        >
                                            <Card
                                                className="group relative bg-white/[0.03] border-white/[0.07] hover:border-indigo-500/30 hover:-translate-y-1 hover:shadow-[0_20px_50px_rgba(0,0,0,0.5)] transition-all duration-300 cursor-pointer h-full flex flex-col overflow-hidden rounded-2xl"
                                                onClick={() => setSelected(scheme)}
                                            >
                                                {/* Subtle gradient overlay on hover */}
                                                <div className="absolute inset-0 bg-gradient-to-br from-indigo-600/[0.04] to-violet-600/[0.03] opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none rounded-2xl" />

                                                <CardHeader className="p-5 pb-3 space-y-0">
                                                    {/* Top badges */}
                                                    <div className="flex flex-wrap gap-1.5 mb-3">
                                                        {scheme.state && scheme.state !== "NaN" && (
                                                            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                                                                <MapPin size={9} /> {scheme.state}
                                                            </span>
                                                        )}
                                                        {scheme.category && scheme.category !== "NaN" && (
                                                            <span className={cn("inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium border", catClass)}>
                                                                <Tag size={9} />
                                                                {scheme.category.length > 20 ? scheme.category.slice(0, 20) + "…" : scheme.category}
                                                            </span>
                                                        )}
                                                    </div>

                                                    <h3 className="text-slate-100 font-bold text-[15px] leading-snug line-clamp-2">
                                                        {scheme.name}
                                                    </h3>
                                                </CardHeader>

                                                <CardContent className="px-5 pb-4 flex-1">
                                                    <p className="text-muted-foreground text-[13px] leading-relaxed line-clamp-3">
                                                        {scheme.details}
                                                    </p>
                                                </CardContent>

                                                <CardFooter className="px-5 pb-5 flex-col items-stretch gap-3">
                                                    {/* Benefit Amount */}
                                                    {amount && (
                                                        <div className="rounded-xl bg-amber-950/50 border border-amber-500/20 px-4 py-2.5">
                                                            <div className="text-[10px] font-bold tracking-wider text-amber-500/70 uppercase mb-0.5">Benefit Amount</div>
                                                            <div className="text-amber-400 font-extrabold text-base flex items-center gap-1">
                                                                <IndianRupee size={14} className="shrink-0" />
                                                                {amount.replace(/^₹/, "")}
                                                            </div>
                                                        </div>
                                                    )}

                                                    <div className="flex items-center justify-between pt-1 border-t border-white/[0.05]">
                                                        <span className="text-[11px] text-muted-foreground/50">View details</span>
                                                        <motion.span whileHover={{ x: 3 }} className="text-indigo-400 flex items-center gap-1 text-xs font-semibold">
                                                            Details <ArrowRight size={12} />
                                                        </motion.span>
                                                    </div>
                                                </CardFooter>
                                            </Card>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        </motion.div>
                    )}

                    {/* No results */}
                    {!loading && hasQuery && results.length === 0 && (
                        <motion.div key="noresults" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="text-center py-20">
                            <div className="text-5xl mb-4">🔍</div>
                            <h3 className="text-slate-200 text-xl font-bold mb-2">No schemes found</h3>
                            <p className="text-muted-foreground text-sm">Try rephrasing or using fewer keywords.</p>
                        </motion.div>
                    )}

                    {/* Empty state */}
                    {!loading && !hasQuery && (
                        <motion.div key="empty" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }}>
                            <p className="text-muted-foreground/60 text-xs font-bold tracking-widest text-center mb-4 uppercase">
                                ✨ Popular Searches
                            </p>
                            <div className="flex flex-col gap-2.5 max-w-xl mx-auto mb-12">
                                {SUGGESTIONS.map((s, i) => {
                                    const Icon = s.icon;
                                    return (
                                        <motion.div
                                            key={i}
                                            initial={{ opacity: 0, x: -14 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: i * 0.07 }}
                                            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                                            onClick={() => triggerSearch(s.text)}
                                            className={cn(
                                                "flex items-center gap-3 px-4 py-3.5 rounded-2xl cursor-pointer transition-all duration-200",
                                                "bg-white/[0.03] border border-white/[0.07] hover:border-white/20 hover:bg-white/[0.06]"
                                            )}
                                        >
                                            <span className={cn("w-8 h-8 rounded-xl flex items-center justify-center shrink-0", COLOR_MAP[s.color])}>
                                                <Icon size={14} />
                                            </span>
                                            <span className="text-slate-300 text-sm font-medium">{s.text}</span>
                                            <ArrowRight size={13} className="text-muted-foreground/40 ml-auto shrink-0" />
                                        </motion.div>
                                    );
                                })}
                            </div>

                            <p className="text-muted-foreground/60 text-xs font-bold tracking-widest text-center mb-4 uppercase">
                                Browse by Category
                            </p>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                {CATEGORIES.slice(1).map(cat => {
                                    const Icon = cat.icon;
                                    return (
                                        <motion.div
                                            key={cat.label}
                                            whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
                                            onClick={() => { setActiveCat(cat.label); triggerSearch(cat.label); }}
                                            className={cn(
                                                "flex flex-col items-center gap-2.5 p-5 rounded-2xl cursor-pointer border transition-all duration-200",
                                                "bg-white/[0.02] border-white/[0.06] hover:border-white/20 hover:bg-white/[0.05]"
                                            )}
                                        >
                                            <span className={cn("w-10 h-10 rounded-xl flex items-center justify-center", COLOR_MAP[cat.color])}>
                                                <Icon size={16} />
                                            </span>
                                            <span className="text-slate-300 text-xs font-semibold text-center">{cat.label}</span>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        </motion.div>
                    )}

                </AnimatePresence>
            </div>

            {/* ── Detail dialog ── */}
            <Dialog open={!!selectedScheme} onOpenChange={open => !open && setSelected(null)}>
                <DialogContent className="max-w-2xl max-h-[88vh] flex flex-col p-0 overflow-hidden bg-[#0a0d14] border-white/10 rounded-3xl shadow-2xl">
                    {selectedScheme && (
                        <>
                            <DialogHeader className="px-7 pt-6 pb-5 border-b border-white/[0.07] bg-gradient-to-br from-indigo-600/[0.07] to-transparent shrink-0">
                                <div className="flex flex-wrap gap-2 mb-3">
                                    {selectedScheme.state && selectedScheme.state !== "NaN" && (
                                        <Badge className="bg-indigo-500/10 text-indigo-300 border-indigo-500/20 text-xs px-2.5 py-0.5 rounded-full">
                                            <MapPin size={10} className="mr-1" /> {selectedScheme.state}
                                        </Badge>
                                    )}
                                    {selectedScheme.category && selectedScheme.category !== "NaN" && (
                                        <Badge className={cn("text-xs px-2.5 py-0.5 rounded-full", CAT_BADGE_MAP[selectedScheme.category] ?? "bg-indigo-500/10 text-indigo-300 border-indigo-500/20")}>
                                            <Tag size={10} className="mr-1" /> {selectedScheme.category}
                                        </Badge>
                                    )}
                                </div>
                                <DialogTitle className="text-slate-100 text-xl font-bold leading-snug">{selectedScheme.name}</DialogTitle>
                                <DialogDescription className="text-muted-foreground/60 text-xs mt-1">
                                    Source:{" "}
                                    <a href={selectedScheme.source} target="_blank" rel="noreferrer" className="text-indigo-400 underline underline-offset-2 hover:text-indigo-300">
                                        {selectedScheme.source.slice(0, 55)}…
                                    </a>
                                </DialogDescription>
                            </DialogHeader>

                            <ScrollArea className="flex-1 overflow-auto">
                                <div className="px-7 py-5 space-y-5">
                                    <div>
                                        <h4 className="text-slate-200 text-sm font-bold mb-2 flex items-center gap-1.5"><Tag size={13} className="text-indigo-400" /> Overview</h4>
                                        <p className="text-muted-foreground text-sm leading-relaxed">{selectedScheme.details}</p>
                                    </div>
                                    {selectedScheme.benefits && selectedScheme.benefits !== "NaN" && (
                                        <div className="rounded-2xl bg-emerald-950/40 border border-emerald-500/15 px-5 py-4">
                                            <h4 className="text-emerald-400 text-sm font-bold mb-2">✅ Benefits</h4>
                                            <p className="text-muted-foreground text-sm leading-relaxed whitespace-pre-line">{selectedScheme.benefits}</p>
                                        </div>
                                    )}
                                    {selectedScheme.eligibility && selectedScheme.eligibility !== "NaN" && (
                                        <div>
                                            <h4 className="text-slate-200 text-sm font-bold mb-2">👤 Eligibility</h4>
                                            <p className="text-muted-foreground text-sm leading-relaxed whitespace-pre-line">{selectedScheme.eligibility}</p>
                                        </div>
                                    )}
                                    {selectedScheme.application_process && selectedScheme.application_process !== "NaN" && (
                                        <div className="rounded-2xl bg-blue-950/40 border border-blue-500/15 px-5 py-4">
                                            <h4 className="text-blue-400 text-sm font-bold mb-2">📋 How to Apply</h4>
                                            <p className="text-muted-foreground text-sm leading-relaxed whitespace-pre-line">{selectedScheme.application_process}</p>
                                        </div>
                                    )}
                                </div>
                            </ScrollArea>

                            <div className="px-7 py-4 border-t border-white/[0.07] flex gap-2.5 justify-end shrink-0">
                                <Button
                                    onClick={() => setSelected(null)}
                                    variant="ghost"
                                    className="rounded-xl text-muted-foreground hover:text-slate-200 hover:bg-white/5 text-sm"
                                >
                                    Close
                                </Button>
                                <Button asChild className="rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold shadow-lg shadow-indigo-500/25 text-sm">
                                    <a href={selectedScheme.source} target="_blank" rel="noreferrer" className="flex items-center gap-2">
                                        Apply Now <ExternalLink size={13} />
                                    </a>
                                </Button>
                            </div>
                        </>
                    )}
                </DialogContent>
            </Dialog>
        </div>
    );
}
