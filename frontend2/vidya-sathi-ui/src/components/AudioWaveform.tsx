const bars = [
  { duration: "0.8s", saffron: true },
  { duration: "1.1s", saffron: false },
  { duration: "0.6s", saffron: true },
  { duration: "0.9s", saffron: false },
  { duration: "0.7s", saffron: true },
  { duration: "1.0s", saffron: false },
  { duration: "0.85s", saffron: true },
];

const AudioWaveform = ({ active = true }: { active?: boolean }) => {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-end justify-center gap-1.5" style={{ height: 48 }}>
        {bars.map((bar, i) => (
          <div
            key={i}
            className="w-1.5 rounded-full origin-bottom"
            style={{
              backgroundColor: bar.saffron ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.5)",
              minHeight: 8,
              maxHeight: 48,
              height: "100%",
              animation: active
                ? `eq-pulse ${bar.duration} ease-in-out ${i * 0.08}s infinite`
                : "none",
              transform: active ? undefined : "scaleY(0.15)",
              transition: "transform 0.3s ease",
            }}
          />
        ))}
      </div>
      {/* Glow line below bars */}
      {active && (
        <div
          className="h-px w-20 rounded-full"
          style={{
            background: "linear-gradient(90deg, transparent, rgba(255,153,51,0.4), transparent)",
          }}
        />
      )}
    </div>
  );
};

export default AudioWaveform;
