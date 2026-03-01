import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { MessageSquare, Sparkles } from "lucide-react";

export default function Navbar() {
    return (
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container px-4 mx-auto flex h-16 items-center justify-between">
                <Link to="/" className="flex items-center space-x-2">
                    <div className="bg-primary/10 p-2 rounded-xl">
                        <Sparkles className="h-5 w-5 text-primary" />
                    </div>
                    <span className="inline-block font-bold text-xl tracking-tight">Sarkari Mitra</span>
                </Link>
                <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
                    <Link to="/discover" className="transition-colors hover:text-foreground/80 text-foreground/60">Discover Schemes</Link>
                    <Link to="/about" className="transition-colors hover:text-foreground/80 text-foreground/60">How it Works</Link>
                </nav>
                <div className="flex items-center space-x-4">
                    <Link to="/vidya-hub">
                        <Button className="rounded-full shadow-sm">
                            <MessageSquare className="mr-2 h-4 w-4" />
                            Talk to Vidya
                        </Button>
                    </Link>
                </div>
            </div>
        </header>
    );
}
