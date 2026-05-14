import { motion } from "framer-motion";
import { Mic, Search, Cpu, FileText } from "lucide-react";
import { Link } from "react-router-dom";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import AudioWaveform from "@/components/AudioWaveform";
import Marquee from "@/components/Marquee";
import StatCounter from "@/components/StatCounter";
import AshokaChakra from "@/components/AshokaChakra";

const beneficiaries = [
  { icon: "🎓", label: "Students" },
  { icon: "🌾", label: "Farmers" },
  { icon: "👩", label: "Women" },
  { icon: "🏭", label: "MSME" },
  { icon: "♿", label: "Divyang" },
  { icon: "👴", label: "Senior Citizens" },
];

const steps = [
  { icon: Mic, title: "Speak or Type", subtitle: "In Hindi, English, or Hinglish" },
  { icon: Cpu, title: "AI Understands", subtitle: "RAG + Llama 3.3 70B processes your query" },
  { icon: FileText, title: "Get Schemes", subtitle: "Exact matches from 3,400+ government schemes" },
];

const Landing = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />

      {/* Hero */}
      <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-4 pt-16">
        {/* Layer 2: Radial glow */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background: "radial-gradient(circle at center, rgba(255,153,51,0.06) 0%, transparent 60%)",
          }}
        />

        {/* Layer 3: Dot grid */}
        <div className="pointer-events-none absolute inset-0 grid-dots" />

        {/* Layer 4: Ashoka Chakra watermark */}
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2" style={{ animation: "spin-slow 120s linear infinite" }}>
          <AshokaChakra size={600} className="text-primary opacity-[0.04]" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="relative z-10 flex max-w-4xl flex-col items-center text-center"
        >
          <h1
            className="font-display font-bold leading-tight text-foreground"
            style={{
              fontSize: "clamp(48px, 8vw, 96px)",
              textShadow: "0 0 80px rgba(255,153,51,0.3)",
            }}
          >
            आपकी सरकार, आपकी भाषा
          </h1>

          {/* Divider line */}
          <div
            className="mt-6 h-px w-[200px]"
            style={{
              background: "linear-gradient(90deg, transparent, rgba(255,153,51,0.4), transparent)",
            }}
          />

          <p className="mt-6 max-w-2xl font-body text-lg text-muted-foreground md:text-xl">
            AI-powered voice assistant for 3,400+ government schemes — speak in
            Hindi, English, or Hinglish
          </p>

          {/* Waveform */}
          <div className="mt-8">
            <AudioWaveform active />
          </div>

          {/* CTAs */}
          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link
              to="/hub"
              className="glow-saffron inline-flex items-center gap-2 rounded-lg bg-primary px-8 py-3 font-display text-base font-bold tracking-wider text-primary-foreground transition-all hover:scale-[1.02] hover:brightness-110"
            >
              <Mic size={18} />
              Start Speaking
            </Link>
            <Link
              to="/discover"
              className="inline-flex items-center gap-2 rounded-lg border border-primary/50 bg-transparent px-8 py-3 font-display text-base font-bold tracking-wider text-primary transition-all hover:scale-[1.02] hover:bg-primary/10"
            >
              <Search size={18} />
              Search Schemes
            </Link>
          </div>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-1 gap-4 sm:grid-cols-3 w-full max-w-3xl">
            <StatCounter value={3400} suffix="+" label="schemes indexed" />
            <StatCounter value={500} prefix="< " suffix="ms" label="response latency" />
            <StatCounter value={3} suffix="" label="languages supported" />
          </div>
        </motion.div>
      </section>

      {/* Marquee */}
      <Marquee />

      {/* How It Works */}
      <section className="py-20 px-4">
        <div className="container mx-auto max-w-4xl">
          <h2 className="text-center font-display text-3xl font-bold text-foreground mb-16">
            How It Works
          </h2>
          <div className="relative flex flex-col md:flex-row items-center md:items-start justify-between gap-12 md:gap-0">
            {/* Connecting line (desktop) */}
            <div
              className="hidden md:block absolute top-6 left-[calc(16.67%+24px)] right-[calc(16.67%+24px)] h-px"
              style={{
                borderTop: "2px dashed rgba(255,153,51,0.3)",
              }}
            />

            {steps.map((step, i) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.15 }}
                viewport={{ once: true }}
                className="flex flex-col items-center text-center flex-1 relative z-10"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-full border-2 border-primary bg-card">
                  <step.icon size={20} className="text-primary" />
                </div>
                <h3 className="mt-4 font-display text-lg font-bold text-foreground">{step.title}</h3>
                <p className="mt-1 font-body text-[13px] text-muted-foreground max-w-[200px]">{step.subtitle}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Target Beneficiaries */}
      <section className="py-20 px-4 border-t border-border/30">
        <div className="container mx-auto max-w-4xl">
          <h2 className="text-center font-display text-4xl font-bold text-foreground mb-12">
            Who Can Benefit?
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {beneficiaries.map((b, i) => (
              <motion.div
                key={b.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                viewport={{ once: true }}
                className="flex flex-col items-center gap-3 rounded-lg border border-border/50 bg-card p-6 transition-all duration-300 hover:-translate-y-1 hover:border-primary/30 hover:shadow-[0_0_30px_rgba(255,153,51,0.08)]"
              >
                <span className="text-4xl">{b.icon}</span>
                <span className="font-display text-base font-bold text-foreground">{b.label}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Landing;
