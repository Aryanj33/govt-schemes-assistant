import { useCountUp } from "@/hooks/useCountUp";

const StatCounter = ({
  value,
  label,
  suffix = "",
  prefix = "",
}: {
  value: number;
  label: string;
  suffix?: string;
  prefix?: string;
}) => {
  const { count, ref } = useCountUp(value);

  return (
    <div
      ref={ref}
      className="group relative border-l-[3px] border-l-primary rounded-lg border border-primary/20 bg-card px-6 py-5 transition-all duration-300 hover:border-primary/50 hover:shadow-[0_0_30px_rgba(255,153,51,0.1)]"
    >
      <div className="font-display text-4xl font-bold text-primary md:text-[56px] leading-none">
        {prefix}
        {count.toLocaleString()}
        {suffix}
      </div>
      <div className="mt-2 font-body text-[13px] uppercase tracking-[0.1em] text-muted-foreground">
        {label}
      </div>
    </div>
  );
};

export default StatCounter;
