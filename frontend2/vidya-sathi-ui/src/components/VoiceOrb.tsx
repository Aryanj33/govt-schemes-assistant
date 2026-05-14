import { useState } from "react";

export type OrbState = "idle" | "listening" | "processing" | "speaking";

const stateColors: Record<OrbState, { border: string; glow: string; label: string }> = {
  idle: { border: "border-primary/50", glow: "", label: "IDLE" },
  listening: { border: "border-primary", glow: "glow-saffron-intense", label: "LISTENING" },
  processing: { border: "border-accent", glow: "", label: "PROCESSING..." },
  speaking: { border: "border-secondary", glow: "glow-green", label: "SPEAKING" },
};

const VoiceOrb = ({
  state = "idle",
  audioLevel = 0,
  onTap,
}: {
  state?: OrbState;
  audioLevel?: number;
  onTap?: () => void;
}) => {
  const { border, glow, label } = stateColors[state];
  const scale = state === "listening" ? 1 + audioLevel * 0.08 : 1;

  return (
    <div className="flex flex-col items-center gap-6">
      <button
        onClick={onTap}
        className={`relative flex items-center justify-center rounded-full transition-all duration-300 ${border} ${glow} border-2 bg-surface`}
        style={{
          width: 280,
          height: 280,
          transform: `scale(${scale})`,
        }}
        aria-label="Voice input"
      >
        {/* Rotating dashed ring */}
        <svg
          className={`absolute inset-0 h-full w-full ${state === "idle" ? "orb-idle" : state === "processing" ? "orb-processing" : ""}`}
          viewBox="0 0 100 100"
        >
          <circle
            cx="50"
            cy="50"
            r="48"
            fill="none"
            stroke="hsl(var(--primary))"
            strokeWidth="0.5"
            strokeDasharray="4 6"
            opacity={state === "idle" ? 0.3 : state === "processing" ? 0.8 : 0.5}
          />
        </svg>

        {/* Inner waveform for listening state */}
        {state === "listening" && (
          <div className="flex items-end gap-1 h-16">
            {Array.from({ length: 7 }).map((_, i) => (
              <div
                key={i}
                className="w-1 rounded-full bg-primary origin-bottom"
                style={{
                  animation: `eq-pulse ${0.5 + Math.random() * 0.5}s ease-in-out ${i * 0.08}s infinite`,
                  height: `${20 + Math.random() * 40}px`,
                }}
              />
            ))}
          </div>
        )}

        {/* Processing spinner */}
        {state === "processing" && (
          <div className="font-mono text-sm text-accent-foreground animate-pulse">
            Processing...
          </div>
        )}

        {/* Speaking ripple */}
        {state === "speaking" && (
          <>
            <div className="absolute inset-0 rounded-full border border-secondary/20 animate-ping" />
            <div
              className="absolute rounded-full border border-secondary/10"
              style={{
                inset: -20,
                animation: "ping-dot 2s cubic-bezier(0,0,0.2,1) 0.3s infinite",
              }}
            />
            <div className="font-mono text-sm text-secondary">♪ Speaking</div>
          </>
        )}

        {/* Idle center dot */}
        {state === "idle" && (
          <div className="h-3 w-3 rounded-full bg-primary/60" />
        )}
      </button>

      <span className="font-mono text-xs tracking-[0.2em] text-muted-foreground">
        {label}
      </span>
    </div>
  );
};

export default VoiceOrb;
