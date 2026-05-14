import { Link } from "react-router-dom";
import AshokaChakra from "./AshokaChakra";

const Footer = () => (
  <footer className="border-t border-primary/10 bg-[#030812]">
    <div className="container mx-auto px-4 py-10">
      <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
        {/* Left - Logo */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <AshokaChakra size={20} className="text-primary" />
            <span className="font-display text-lg font-bold tracking-[0.3em] text-primary">VIDYA</span>
          </div>
          <p className="font-body text-sm text-muted-foreground max-w-xs">
            AI-powered access to government schemes for every Indian
          </p>
        </div>

        {/* Center - Links */}
        <div className="flex items-center justify-center gap-6">
          {[
            { label: "Discover", path: "/discover" },
            { label: "Voice Hub", path: "/hub" },
            { label: "Dashboard", path: "/dashboard" },
          ].map(link => (
            <Link
              key={link.path}
              to={link.path}
              className="font-display text-sm text-muted-foreground transition-colors hover:text-primary"
            >
              {link.label}
            </Link>
          ))}
        </div>

        {/* Right - Tech */}
        <div className="flex items-center justify-end">
          <p className="font-mono text-xs text-muted-foreground/60">
            Built with Groq LPU • FAISS • React
          </p>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="mt-8 border-t border-primary/10 pt-6 text-center">
        <p className="font-body text-[13px] text-muted-foreground">
          Made with ❤️ for Bharat
        </p>
      </div>
    </div>
  </footer>
);

export default Footer;
