import { useState } from "react";

// Hexagonal grid representation of Indian states
const stateData = [
  { id: "JK", name: "Jammu & Kashmir", schemes: 145, row: 0, col: 2 },
  { id: "HP", name: "Himachal Pradesh", schemes: 156, row: 1, col: 1 },
  { id: "PB", name: "Punjab", schemes: 178, row: 1, col: 2 },
  { id: "UK", name: "Uttarakhand", schemes: 134, row: 1, col: 3 },
  { id: "HR", name: "Haryana", schemes: 189, row: 2, col: 1 },
  { id: "DL", name: "Delhi", schemes: 312, row: 2, col: 2 },
  { id: "UP", name: "Uttar Pradesh", schemes: 287, row: 2, col: 3 },
  { id: "BR", name: "Bihar", schemes: 176, row: 2, col: 4 },
  { id: "SK", name: "Sikkim", schemes: 89, row: 2, col: 5 },
  { id: "RJ", name: "Rajasthan", schemes: 198, row: 3, col: 0 },
  { id: "MP", name: "Madhya Pradesh", schemes: 213, row: 3, col: 1 },
  { id: "GJ", name: "Gujarat", schemes: 234, row: 3, col: 2 },
  { id: "JH", name: "Jharkhand", schemes: 145, row: 3, col: 3 },
  { id: "WB", name: "West Bengal", schemes: 203, row: 3, col: 4 },
  { id: "AS", name: "Assam", schemes: 167, row: 3, col: 5 },
  { id: "MH", name: "Maharashtra", schemes: 267, row: 4, col: 1 },
  { id: "CG", name: "Chhattisgarh", schemes: 156, row: 4, col: 2 },
  { id: "OD", name: "Odisha", schemes: 167, row: 4, col: 3 },
  { id: "MN", name: "Manipur", schemes: 78, row: 4, col: 5 },
  { id: "GA", name: "Goa", schemes: 112, row: 5, col: 0 },
  { id: "KA", name: "Karnataka", schemes: 221, row: 5, col: 1 },
  { id: "TS", name: "Telangana", schemes: 189, row: 5, col: 2 },
  { id: "AP", name: "Andhra Pradesh", schemes: 198, row: 5, col: 3 },
  { id: "TN", name: "Tamil Nadu", schemes: 245, row: 6, col: 2 },
  { id: "KL", name: "Kerala", schemes: 198, row: 6, col: 1 },
];

const maxSchemes = Math.max(...stateData.map(s => s.schemes));

function getColor(schemes: number): string {
  const ratio = schemes / maxSchemes;
  if (ratio < 0.2) return "#0D2444";
  if (ratio < 0.4) return "#1A3A6B";
  if (ratio < 0.6) return "#B35900";
  if (ratio < 0.8) return "#FF7700";
  return "#FF9933";
}

const IndiaMap = () => {
  const [hovered, setHovered] = useState<string | null>(null);
  const cellSize = 48;
  const gap = 4;

  return (
    <div className="relative w-full max-w-lg">
      {/* Legend */}
      <div className="flex items-center justify-center gap-2 mb-4">
        <span className="font-mono text-[10px] text-muted-foreground">Low</span>
        {["#0D2444", "#1A3A6B", "#B35900", "#FF7700", "#FF9933"].map(c => (
          <div key={c} className="h-3 w-6 rounded-sm" style={{ backgroundColor: c }} />
        ))}
        <span className="font-mono text-[10px] text-muted-foreground">High</span>
      </div>

      <div className="relative" style={{ minHeight: 7 * (cellSize + gap) + 20 }}>
        {stateData.map(state => {
          const x = state.col * (cellSize + gap) + (state.row % 2 === 1 ? (cellSize + gap) / 2 : 0);
          const y = state.row * (cellSize + gap - 8);
          const isHovered = hovered === state.id;

          return (
            <div
              key={state.id}
              className="absolute rounded-md transition-all duration-200 cursor-pointer flex flex-col items-center justify-center"
              style={{
                left: x,
                top: y,
                width: cellSize,
                height: cellSize,
                backgroundColor: getColor(state.schemes),
                border: `1px solid rgba(255,255,255,${isHovered ? 0.3 : 0.1})`,
                filter: isHovered ? "brightness(1.3)" : undefined,
                zIndex: isHovered ? 10 : 1,
              }}
              onMouseEnter={() => setHovered(state.id)}
              onMouseLeave={() => setHovered(null)}
            >
              <span className="font-mono text-[10px] text-foreground font-bold">{state.id}</span>
              <span className="font-mono text-[8px] text-foreground/70">{state.schemes}</span>

              {/* Tooltip */}
              {isHovered && (
                <div className="absolute -top-10 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-card border border-border px-3 py-1.5 shadow-lg z-20">
                  <span className="font-body text-xs text-foreground">{state.name}</span>
                  <span className="font-mono text-xs text-primary ml-2">— {state.schemes} schemes</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default IndiaMap;
