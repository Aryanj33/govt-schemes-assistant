import { Link, useLocation } from "react-router-dom";
import { useState } from "react";
import { Menu, X } from "lucide-react";
import AshokaChakra from "./AshokaChakra";

const navLinks = [
  { label: "Discover", path: "/discover" },
  { label: "Voice Hub", path: "/hub" },
  { label: "Dashboard", path: "/dashboard" },
];

const Header = () => {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background/70 backdrop-blur-[12px]">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link to="/" className="flex items-center gap-2">
          <div style={{ animation: "spin-slow 20s linear infinite" }}>
            <AshokaChakra size={24} className="text-primary" />
          </div>
          <span className="font-display text-xl font-bold tracking-[0.3em] text-primary">
            VIDYA
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-8 md:flex">
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`relative font-display text-sm font-semibold tracking-wider transition-colors hover:text-primary ${
                location.pathname === link.path
                  ? "text-primary"
                  : "text-muted-foreground"
              }`}
            >
              {link.label}
              {location.pathname === link.path && (
                <span className="absolute -bottom-1 left-0 right-0 h-0.5 bg-primary rounded-full" />
              )}
            </Link>
          ))}
          <div className="flex items-center gap-2 rounded-full border border-secondary/40 bg-secondary/10 px-3 py-1">
            <span className="relative flex h-2 w-2">
              <span className="ping-dot absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-secondary" />
            </span>
            <span className="font-mono text-xs text-secondary">SYSTEM ONLINE</span>
          </div>
        </nav>

        {/* Mobile toggle */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="text-foreground md:hidden"
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 top-16 z-40 bg-background/95 backdrop-blur-lg md:hidden">
          <nav className="flex flex-col gap-6 p-8">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setMobileOpen(false)}
                className={`font-display text-2xl font-bold tracking-wider transition-colors ${
                  location.pathname === link.path
                    ? "text-primary"
                    : "text-muted-foreground"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
};

export default Header;
