import { motion } from "framer-motion";
import { type SchemeData } from "@/data/useSchemes";

const SchemeCard = ({
  scheme,
  index = 0,
  onClick,
}: {
  scheme: SchemeData;
  index?: number;
  onClick?: () => void;
}) => {
  const circumference = 2 * Math.PI * 14;
  const offset = circumference - (scheme.relevance / 100) * circumference;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.4 }}
      onClick={onClick}
      className="group cursor-pointer rounded-lg border border-border/50 bg-card p-5 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_8px_32px_rgba(0,0,0,0.4)]"
      style={{
        borderLeftWidth: 3,
        borderLeftColor: scheme.level === "Central" ? "hsl(var(--primary))" : "hsl(var(--secondary))",
      }}
    >
      {/* Top row: badges + score */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex flex-wrap gap-1.5">
          <span
            className={`rounded px-2 py-0.5 font-mono text-[11px] font-medium uppercase border ${
              scheme.level === "Central"
                ? "border-primary/30 bg-primary/10 text-primary"
                : "border-secondary/30 bg-secondary/10 text-secondary"
            }`}
          >
            {scheme.level}
          </span>
          {scheme.categories.slice(0, 1).map((cat) => (
            <span
              key={cat}
              className="rounded border border-border bg-muted/30 px-2 py-0.5 font-mono text-[11px] text-muted-foreground uppercase"
            >
              {cat}
            </span>
          ))}
        </div>
        <div className="relative h-9 w-9 shrink-0">
          <svg className="h-9 w-9 -rotate-90" viewBox="0 0 32 32">
            <circle cx="16" cy="16" r="14" fill="none" stroke="hsl(var(--border))" strokeWidth="2" />
            <circle
              cx="16" cy="16" r="14" fill="none"
              stroke="hsl(var(--primary))"
              strokeWidth="2"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              strokeLinecap="round"
            />
          </svg>
          <span className="absolute inset-0 flex items-center justify-center font-mono text-[9px] text-primary font-medium">
            {scheme.relevance}%
          </span>
        </div>
      </div>

      {/* Name */}
      <h3 className="font-display text-lg font-bold text-foreground mb-1 line-clamp-1">
        {scheme.name}
      </h3>

      {/* Description */}
      <p className="font-body text-sm text-muted-foreground line-clamp-2 mb-4 leading-relaxed">
        {scheme.description}
      </p>

      {/* Divider */}
      <div className="border-t border-border/50 mb-3" />

      {/* Stat chips */}
      <div className="flex items-center gap-0 flex-wrap">
        {scheme.amount && (
          <>
            <span className="font-mono text-xs text-foreground px-1">{scheme.amount}</span>
            <span className="text-border mx-1">│</span>
          </>
        )}
        {scheme.categories.slice(0, 2).map((cat, i) => (
          <span key={cat}>
            <span className="font-mono text-xs text-muted-foreground px-1">{cat}</span>
            {i < 1 && <span className="text-border mx-1">│</span>}
          </span>
        ))}
        {scheme.state && (
          <>
            <span className="text-border mx-1">│</span>
            <span className="font-mono text-xs text-muted-foreground px-1">{scheme.state === "All India" ? "All India" : scheme.state}</span>
          </>
        )}
      </div>

      {/* Details link */}
      <div className="mt-3 flex items-center gap-1 text-sm text-primary opacity-0 transition-opacity group-hover:opacity-100">
        <span className="font-display font-semibold">Details</span>
        <span>→</span>
      </div>
    </motion.div>
  );
};

export default SchemeCard;
