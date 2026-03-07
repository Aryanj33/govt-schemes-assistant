import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { MessageSquare, Sparkles, Compass, Info } from "lucide-react";

const navLinks = [
    { label: "Discover Schemes", to: "/discover", icon: <Compass className="h-4 w-4" /> },
    { label: "How it Works", to: "/about", icon: <Info className="h-4 w-4" /> },
];

export default function Navbar() {
    return (
        <header
            className="sticky top-0 z-50 w-full border-b border-white/10"
            style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)" }}
        >
            <div className="container px-8 mx-auto flex h-16 items-center justify-between">
                <Link to="/" className="flex items-center space-x-2">
                    <div className="bg-primary/10 p-2 rounded-xl">
                        <Sparkles className="h-5 w-5 text-primary" />
                    </div>
                    <span className="inline-block font-bold text-2xl tracking-tight">Sarkari Mitra</span>
                </Link>
                <nav className="hidden md:flex items-center gap-2 text-sm font-medium">
                    {navLinks.map(({ label, to, icon }) => (
                        <Link
                            key={to}
                            to={to}
                            className="flex items-center gap-1.5 px-4 py-2 rounded-lg transition-colors text-foreground/60 hover:text-foreground hover:bg-accent"
                        >
                            {icon}
                            {label}
                        </Link>
                    ))}
                </nav>
                <div className="flex items-center space-x-4">
                    <Link to="/vidya-hub">
                        <Button size="sm" className="rounded-full px-5 gap-2 shadow-md text-sm font-semibold">
                            <MessageSquare className="h-4 w-4" />
                            Talk to Vidya
                        </Button>
                    </Link>
                </div>
            </div>
        </header>
    );
}
