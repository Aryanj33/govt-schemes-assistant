export default function Footer() {
    return (
        <footer className="border-t py-6 md:py-0">
            <div className="container px-4 mx-auto flex flex-col items-center justify-between gap-4 md:h-16 md:flex-row">
                <p className="text-sm leading-loose text-center text-muted-foreground md:text-left">
                    Built to empower citizens. Not an official government website.
                </p>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <a href="#" className="hover:underline underline-offset-4">Terms</a>
                    <a href="#" className="hover:underline underline-offset-4">Privacy</a>
                </div>
            </div>
        </footer>
    );
}
