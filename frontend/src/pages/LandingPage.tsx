import { Link } from "react-router-dom";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Mic, Search, Shield, BookOpen, Stethoscope, Briefcase, MessageSquare, Target, GraduationCap, Zap, ArrowRight, ExternalLink } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Animated voice waveform bars
function VoiceWaveform() {
    const bars = [3, 6, 9, 12, 9, 7, 11, 5, 8, 10, 6, 4, 9, 7, 5];
    return (
        <div className="flex items-center justify-center gap-[3px] h-8">
            {bars.map((h, i) => (
                <motion.div
                    key={i}
                    className="w-1 rounded-full bg-primary"
                    animate={{
                        scaleY: [1, h / 6, 1, h / 10, 1],
                    }}
                    transition={{
                        duration: 1.4,
                        repeat: Infinity,
                        delay: i * 0.07,
                        ease: "easeInOut",
                    }}
                    style={{ height: `${h}px`, originY: "center" }}
                />
            ))}
        </div>
    );
}

// Pulsing mic button visual

// Animated loading dots
function LoadingDots() {
    return (
        <span className="inline-flex items-center gap-1 ml-2">
            {[0, 1, 2].map((i) => (
                <motion.span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-primary"
                    animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.2, 0.8] }}
                    transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                />
            ))}
        </span>
    );
}

export default function LandingPage() {
    const [searchQuery, setSearchQuery] = useState("");
    const [searching, setSearching] = useState(false);
    const [searchDone, setSearchDone] = useState(false);

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (!searchQuery.trim()) return;
        setSearching(true);
        setSearchDone(false);
        setTimeout(() => {
            setSearching(false);
            setSearchDone(true);
        }, 2000);
    };

    const containerVariants = {
        hidden: { opacity: 0 },
        show: {
            opacity: 1,
            transition: { staggerChildren: 0.1 },
        },
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0 },
    };

    return (
        <div className="flex flex-col min-h-screen">
            {/* Hero Section */}
            <section className="relative pt-20 pb-16 md:pt-28 md:pb-24 overflow-hidden">
                {/* Background glows */}
                <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-background to-background pointer-events-none" />
                <div className="absolute top-20 left-1/4 w-96 h-96 rounded-full bg-primary/10 blur-3xl pointer-events-none" />
                <div className="absolute bottom-0 right-1/4 w-72 h-72 rounded-full bg-blue-500/10 blur-3xl pointer-events-none" />

                <div className="container px-6 mx-auto relative z-10">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">

                        {/* ── LEFT: Text & CTAs ── */}
                        <div className="flex flex-col items-start text-left">
                            <motion.div
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ duration: 0.5 }}
                            >
                                <Badge
                                    variant="secondary"
                                    className="mb-6 px-4 py-1.5 text-sm font-medium rounded-full border bg-background/50 backdrop-blur-sm"
                                >
                                    ✨ AI-Powered · Voice-First · India's Own
                                </Badge>
                            </motion.div>

                            <motion.h1
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5, delay: 0.1 }}
                                className="text-5xl md:text-6xl font-extrabold tracking-tight mb-4 leading-[1.08]"
                            >
                                Find Government{" "}
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-blue-400 to-blue-600">
                                    Schemes
                                </span>
                                <br />Instantly
                            </motion.h1>

                            <motion.p
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5, delay: 0.18 }}
                                className="text-2xl font-semibold text-foreground/80 mb-4"
                            >
                                Just Speak.{" "}
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-400">Vidya Will Help.</span>
                            </motion.p>

                            <motion.p
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5, delay: 0.25 }}
                                className="text-base text-muted-foreground mb-8 max-w-lg leading-relaxed"
                            >
                                Ask Vidya in <span className="text-foreground font-medium">Hindi</span>, <span className="text-foreground font-medium">English</span>, or <span className="text-foreground font-medium">Hinglish</span> — she'll find every scheme, scholarship, and benefit you qualify for across <span className="text-foreground font-semibold">3,400+ government programmes</span>.
                            </motion.p>

                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5, delay: 0.3 }}
                                className="flex flex-col sm:flex-row items-start gap-4 w-full sm:w-auto"
                            >
                                <Link to="/vidya-hub">
                                    <motion.div
                                        whileHover={{ scale: 1.06 }}
                                        whileTap={{ scale: 0.97 }}
                                        transition={{ type: "spring", stiffness: 400, damping: 15 }}
                                        style={{ borderRadius: 9999 }}
                                    >
                                        <Button
                                            size="lg"
                                            className="h-14 px-8 rounded-full text-base gap-2 font-semibold shadow-[0_4px_24px_rgba(59,130,246,0.35)] hover:shadow-[0_0_28px_rgba(59,130,246,0.65)] transition-shadow duration-300"
                                        >
                                            <Mic className="h-5 w-5" />
                                            Talk to Vidya Now 🎤
                                        </Button>
                                    </motion.div>
                                </Link>
                                <Link to="/discover">
                                    <motion.div
                                        whileHover={{ scale: 1.05 }}
                                        whileTap={{ scale: 0.97 }}
                                        transition={{ type: "spring", stiffness: 400, damping: 15 }}
                                        style={{ borderRadius: 9999 }}
                                    >
                                        <Button
                                            size="lg"
                                            variant="outline"
                                            className="h-14 px-8 rounded-full text-base bg-background/50 backdrop-blur-sm gap-2 font-semibold hover:shadow-[0_0_22px_rgba(255,255,255,0.15)] transition-shadow duration-300"
                                        >
                                            <Search className="h-5 w-5" />
                                            Browse 3,000+ Schemes
                                        </Button>
                                    </motion.div>
                                </Link>
                            </motion.div>

                            {/* Trust badges */}
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 0.6, delay: 0.5 }}
                                className="mt-10 flex items-center gap-6 text-sm text-muted-foreground"
                            >
                                <div className="flex items-center gap-2">
                                    <span className="text-green-500 font-bold text-base">✓</span> Free to use
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-green-500 font-bold text-base">✓</span> 3,400+ schemes
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-green-500 font-bold text-base">✓</span> Multilingual
                                </div>
                            </motion.div>
                        </div>

                        {/* ── RIGHT: Animated Voice Assistant ── */}
                        <motion.div
                            initial={{ opacity: 0, x: 40 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.7, delay: 0.2, ease: "easeOut" }}
                            className="relative flex items-center justify-center"
                        >
                            <div className="relative flex items-center justify-center w-full" style={{ minHeight: 420 }}>

                                {/* Outer ambient glow */}
                                <div className="absolute inset-0 rounded-full bg-primary/5 blur-3xl" />

                                {/* Rotating orbit ring */}
                                <motion.div
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 18, repeat: Infinity, ease: "linear" }}
                                    className="absolute"
                                    style={{ width: 340, height: 340 }}
                                >
                                    {[0, 60, 120, 180, 240, 300].map((deg, i) => (
                                        <div
                                            key={i}
                                            className="absolute w-2.5 h-2.5 rounded-full bg-primary/40"
                                            style={{
                                                top: "50%", left: "50%",
                                                transform: `rotate(${deg}deg) translateX(165px) translateY(-50%)`,
                                            }}
                                        />
                                    ))}
                                </motion.div>

                                {/* Counter-rotating inner orbit */}
                                <motion.div
                                    animate={{ rotate: -360 }}
                                    transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
                                    className="absolute"
                                    style={{ width: 240, height: 240 }}
                                >
                                    {[0, 90, 180, 270].map((deg, i) => (
                                        <div
                                            key={i}
                                            className="absolute w-1.5 h-1.5 rounded-full bg-blue-400/50"
                                            style={{
                                                top: "50%", left: "50%",
                                                transform: `rotate(${deg}deg) translateX(117px) translateY(-50%)`,
                                            }}
                                        />
                                    ))}
                                </motion.div>

                                {/* Pulse rings */}
                                {[1, 2, 3].map((ring) => (
                                    <motion.div
                                        key={ring}
                                        className="absolute rounded-full border border-primary/20"
                                        animate={{ scale: [1, 1.5 + ring * 0.15], opacity: [0.5, 0] }}
                                        transition={{ duration: 2.5, repeat: Infinity, delay: ring * 0.7, ease: "easeOut" }}
                                        style={{ width: 180, height: 180 }}
                                    />
                                ))}

                                {/* Main orb */}
                                <motion.div
                                    animate={{ scale: [1, 1.04, 1] }}
                                    transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                                    className="relative z-10 flex items-center justify-center rounded-full"
                                    style={{
                                        width: 180, height: 180,
                                        background: "radial-gradient(circle at 35% 30%, rgba(99,102,241,0.5) 0%, rgba(59,130,246,0.3) 50%, rgba(99,102,241,0.1) 100%)",
                                        border: "2px solid rgba(99,102,241,0.4)",
                                        boxShadow: "0 0 60px rgba(99,102,241,0.35), 0 0 120px rgba(59,130,246,0.15), inset 0 0 40px rgba(99,102,241,0.1)",
                                    }}
                                >
                                    {/* Inner waveform */}
                                    <div className="flex items-center gap-[3px]">
                                        {[14, 22, 32, 44, 36, 28, 40, 20, 30, 24, 16].map((h, i) => (
                                            <motion.div
                                                key={i}
                                                className="rounded-full"
                                                style={{ width: 3, height: h, background: "rgba(255,255,255,0.85)" }}
                                                animate={{ scaleY: [1, h > 30 ? 1.6 : 2, 1, 0.5, 1] }}
                                                transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.08, ease: "easeInOut" }}
                                            />
                                        ))}
                                    </div>
                                </motion.div>

                                {/* "Vidya is listening" status card */}
                                <motion.div
                                    animate={{ y: [0, -6, 0] }}
                                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                                    className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-background/80 backdrop-blur-md border border-border/60 rounded-2xl px-5 py-3 shadow-xl"
                                >
                                    <motion.div
                                        animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }}
                                        transition={{ duration: 1.2, repeat: Infinity }}
                                        className="w-2.5 h-2.5 rounded-full bg-green-500"
                                    />
                                    <span className="text-sm font-medium">Vidya is listening…</span>
                                    <VoiceWaveform />
                                </motion.div>

                                {/* Language chips floating around */}
                                {[
                                    { label: "हिंदी में बोलें", top: "8%", left: "-5%", delay: 0 },
                                    { label: "English", top: "8%", right: "-5%", delay: 0.4 },
                                    { label: "Hinglish 🤌", bottom: "28%", right: "-8%", delay: 0.8 },
                                ].map((chip, i) => (
                                    <motion.div
                                        key={i}
                                        animate={{ y: [0, -8, 0] }}
                                        transition={{ duration: 3 + i * 0.5, repeat: Infinity, ease: "easeInOut", delay: chip.delay }}
                                        className="absolute bg-background/80 backdrop-blur border border-border/50 rounded-full px-3 py-1.5 text-xs font-medium shadow-lg whitespace-nowrap"
                                        style={{ top: chip.top, left: (chip as any).left, right: (chip as any).right, bottom: (chip as any).bottom }}
                                    >
                                        {chip.label}
                                    </motion.div>
                                ))}

                            </div>
                        </motion.div>

                    </div>
                </div>
            </section>

            {/* ── Search Bar Section ── */}
            <section className="py-10 border-b border-border/50">
                <div className="container px-6 mx-auto max-w-2xl">
                    <motion.form
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.4 }}
                        onSubmit={handleSearch}
                        className="relative"
                    >
                        <div className="relative flex items-center">
                            <Search className="absolute left-4 h-5 w-5 text-muted-foreground pointer-events-none" />
                            <input
                                type="text"
                                placeholder="Search schemes by keyword, e.g. scholarship, farmer, health…"
                                value={searchQuery}
                                onChange={(e) => { setSearchQuery(e.target.value); setSearchDone(false); }}
                                className="w-full h-14 pl-12 pr-36 rounded-2xl border border-border bg-background/60 backdrop-blur-sm text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all"
                            />
                            <button
                                type="submit"
                                disabled={searching}
                                className="absolute right-2 h-10 px-5 rounded-xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 active:scale-95 transition-all disabled:opacity-60"
                            >
                                {searching ? "Searching…" : "Search"}
                            </button>
                        </div>

                        <AnimatePresence>
                            {searching && (
                                <motion.p
                                    initial={{ opacity: 0, y: 6 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -6 }}
                                    className="mt-3 text-sm text-muted-foreground flex items-center"
                                >
                                    Searching government schemes<LoadingDots />
                                </motion.p>
                            )}
                            {searchDone && (
                                <motion.p
                                    initial={{ opacity: 0, y: 6 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0 }}
                                    className="mt-3 text-sm text-muted-foreground"
                                >
                                    ✅ Showing results for <span className="text-foreground font-medium">"{searchQuery}"</span> — or{" "}
                                    <Link to="/discover" className="text-primary hover:underline">browse all schemes →</Link>
                                </motion.p>
                            )}
                        </AnimatePresence>
                    </motion.form>
                </div>
            </section>

            {/* ── Scheme Cards ── */}
            <section className="py-16">
                <div className="container px-6 mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 16 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5 }}
                        className="flex items-center justify-between mb-8"
                    >
                        <div>
                            <h2 className="text-2xl font-extrabold tracking-tight">Popular Schemes</h2>
                            <p className="text-muted-foreground text-sm mt-1">Widely availed government schemes across India</p>
                        </div>
                        <Link to="/discover" className="hidden sm:flex items-center gap-1.5 text-sm text-primary font-medium hover:underline">
                            View all <ArrowRight className="h-4 w-4" />
                        </Link>
                    </motion.div>

                    <motion.div
                        variants={containerVariants}
                        initial="hidden"
                        whileInView="show"
                        viewport={{ once: true, margin: "-60px" }}
                        className="grid grid-cols-1 md:grid-cols-3 gap-5"
                    >
                        {[
                            {
                                name: "PM Scholarship Scheme",
                                category: "Students",
                                categoryColor: "text-blue-400 bg-blue-500/10 border-blue-500/20",
                                desc: "Scholarship for children of ex-servicemen and ex-coast guard personnel pursuing professional courses.",
                                eligible: "Ward of Ex-Servicemen",
                                benefit: "₹25,000 per year",
                                icon: <GraduationCap className="h-5 w-5 text-blue-400" />,
                                href: "https://scholarships.gov.in",
                            },
                            {
                                name: "PM-KISAN Samman Nidhi",
                                category: "Farmers",
                                categoryColor: "text-green-400 bg-green-500/10 border-green-500/20",
                                desc: "Direct income support to all farmer families with cultivable landholding across the country.",
                                eligible: "All Farmers",
                                benefit: "₹6,000 per year",
                                icon: <Shield className="h-5 w-5 text-green-400" />,
                                href: "https://pmkisan.gov.in",
                            },
                            {
                                name: "Ayushman Bharat Yojana",
                                category: "Health",
                                categoryColor: "text-red-400 bg-red-500/10 border-red-500/20",
                                desc: "World's largest health insurance scheme providing coverage for secondary and tertiary care hospitalisation.",
                                eligible: "BPL & Low-income Families",
                                benefit: "₹5 lakh per family / year",
                                icon: <Stethoscope className="h-5 w-5 text-red-400" />,
                                href: "https://pmjay.gov.in",
                            },
                        ].map((scheme, i) => (
                            <motion.div key={i} variants={itemVariants}>
                                <Card className="h-full border bg-background/60 backdrop-blur-sm hover:-translate-y-1 hover:shadow-lg transition-all duration-300 group">
                                    <CardContent className="p-5 flex flex-col gap-4 h-full">
                                        <div className="flex items-start justify-between gap-3">
                                            <div className="w-10 h-10 rounded-xl bg-muted flex items-center justify-center flex-shrink-0">
                                                {scheme.icon}
                                            </div>
                                            <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${scheme.categoryColor}`}>
                                                {scheme.category}
                                            </span>
                                        </div>
                                        <div className="flex-1">
                                            <h3 className="font-bold text-base mb-1.5">{scheme.name}</h3>
                                            <p className="text-xs text-muted-foreground leading-relaxed">{scheme.desc}</p>
                                        </div>
                                        <div className="space-y-1.5 text-xs">
                                            <div className="flex items-center gap-2 text-muted-foreground">
                                                <span className="font-semibold text-foreground/70">Eligible for:</span> {scheme.eligible}
                                            </div>
                                            <div className="flex items-center gap-2 text-muted-foreground">
                                                <span className="font-semibold text-foreground/70">Benefit:</span>
                                                <span className="text-green-500 font-semibold">{scheme.benefit}</span>
                                            </div>
                                        </div>
                                        <a
                                            href={scheme.href}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex items-center gap-1.5 text-xs font-semibold text-primary hover:underline mt-auto"
                                        >
                                            Apply now <ExternalLink className="h-3.5 w-3.5" />
                                        </a>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>

            {/* ── Features Section ── */}
            <section className="py-20">
                <div className="container px-6 mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5 }}
                        className="text-center mb-14"
                    >
                        <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight mb-3">
                            Everything you need, in one place
                        </h2>
                        <p className="text-muted-foreground text-lg max-w-xl mx-auto">
                            Vidya makes it effortless to discover, understand, and apply for government schemes.
                        </p>
                    </motion.div>

                    <motion.div
                        variants={containerVariants}
                        initial="hidden"
                        whileInView="show"
                        viewport={{ once: true, margin: "-80px" }}
                        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6"
                    >
                        {[
                            {
                                icon: <MessageSquare className="h-6 w-6" />,
                                color: "from-blue-500/20 to-blue-600/10 border-blue-500/20",
                                iconColor: "text-blue-400",
                                glow: "hover:shadow-[0_0_30px_rgba(59,130,246,0.15)]",
                                title: "Voice Based Search",
                                desc: "Find schemes by just speaking — in Hindi, English, or Hinglish.",
                            },
                            {
                                icon: <Target className="h-6 w-6" />,
                                color: "from-purple-500/20 to-purple-600/10 border-purple-500/20",
                                iconColor: "text-purple-400",
                                glow: "hover:shadow-[0_0_30px_rgba(168,85,247,0.15)]",
                                title: "Personalized Recommendations",
                                desc: "AI suggests the exact schemes you qualify for based on your profile.",
                            },
                            {
                                icon: <GraduationCap className="h-6 w-6" />,
                                color: "from-emerald-500/20 to-emerald-600/10 border-emerald-500/20",
                                iconColor: "text-emerald-400",
                                glow: "hover:shadow-[0_0_30px_rgba(16,185,129,0.15)]",
                                title: "Scholarship Finder",
                                desc: "Discover hundreds of scholarships for students across all categories.",
                            },
                            {
                                icon: <Zap className="h-6 w-6" />,
                                color: "from-amber-500/20 to-amber-600/10 border-amber-500/20",
                                iconColor: "text-amber-400",
                                glow: "hover:shadow-[0_0_30px_rgba(245,158,11,0.15)]",
                                title: "Instant Results",
                                desc: "Get eligibility answers and scheme details in seconds, not hours.",
                            },
                        ].map((feat, i) => (
                            <motion.div key={i} variants={itemVariants}>
                                <Card className={`h-full border bg-background/60 backdrop-blur-sm transition-all duration-300 ${feat.glow} hover:-translate-y-1`}>
                                    <CardContent className="p-6 flex flex-col gap-4">
                                        <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${feat.color} border flex items-center justify-center ${feat.iconColor}`}>
                                            {feat.icon}
                                        </div>
                                        <div>
                                            <h3 className="font-semibold text-base mb-1.5">{feat.title}</h3>
                                            <p className="text-sm text-muted-foreground leading-relaxed">{feat.desc}</p>
                                        </div>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>

            {/* Categories Grid */}
            <section className="py-20 bg-muted/30">
                <div className="container px-4 mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold tracking-tight mb-4">Who is it for?</h2>
                        <p className="text-muted-foreground">Schemes categorized to help every citizen of India.</p>
                    </div>
                    <motion.div
                        variants={containerVariants}
                        initial="hidden"
                        whileInView="show"
                        viewport={{ once: true, margin: "-100px" }}
                        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto"
                    >
                        {[
                            { title: "Students", icon: <BookOpen className="h-8 w-8 text-blue-500" />, desc: "Scholarships & Hostels" },
                            { title: "Farmers", icon: <Shield className="h-8 w-8 text-green-500" />, desc: "PM-KISAN & Insurance" },
                            { title: "MSME & Business", icon: <Briefcase className="h-8 w-8 text-purple-500" />, desc: "Mudra Loans & Registration" },
                            { title: "Health", icon: <Stethoscope className="h-8 w-8 text-red-500" />, desc: "Ayushman Bharat & Camps" },
                        ].map((cat, i) => (
                            <motion.div key={i} variants={itemVariants}>
                                <Card className="hover:shadow-md transition-shadow border-none bg-background/60 backdrop-blur supports-[backdrop-filter]:bg-background/40">
                                    <CardContent className="p-6 flex flex-col items-center text-center">
                                        <div className="p-4 bg-muted rounded-full mb-4">
                                            {cat.icon}
                                        </div>
                                        <h3 className="font-semibold text-lg mb-2">{cat.title}</h3>
                                        <p className="text-sm text-muted-foreground">{cat.desc}</p>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>
        </div >
    );
}
