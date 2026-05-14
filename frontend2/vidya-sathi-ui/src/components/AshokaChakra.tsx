const AshokaChakra = ({ className = "", size = 200 }: { className?: string; size?: number }) => (
  <svg
    viewBox="0 0 200 200"
    width={size}
    height={size}
    className={className}
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <circle cx="100" cy="100" r="90" stroke="currentColor" strokeWidth="1.5" />
    <circle cx="100" cy="100" r="20" stroke="currentColor" strokeWidth="1.5" />
    {Array.from({ length: 24 }).map((_, i) => {
      const angle = (i * 15 * Math.PI) / 180;
      const x1 = 100 + 22 * Math.cos(angle);
      const y1 = 100 + 22 * Math.sin(angle);
      const x2 = 100 + 88 * Math.cos(angle);
      const y2 = 100 + 88 * Math.sin(angle);
      return (
        <line
          key={i}
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke="currentColor"
          strokeWidth="0.8"
        />
      );
    })}
  </svg>
);

export default AshokaChakra;
