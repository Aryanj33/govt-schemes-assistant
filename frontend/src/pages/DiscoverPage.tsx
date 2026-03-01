import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Search, MapPin, Tag, ArrowRight, Loader2 } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";

// Scheme Type based on the backend data structure
interface Scheme {
    id: string;
    name: string;
    details: string;
    benefits: string;
    eligibility: string;
    application_process: string;
    state: string;
    category: string;
    source: string;
}

export default function DiscoverPage() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<Scheme[]>([]);
    const [loading, setLoading] = useState(false);
    const [selectedScheme, setSelectedScheme] = useState<Scheme | null>(null);

    const handleSearch = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        try {
            const response = await fetch("http://localhost:8080/search", { // Adjust URL based on actual backend route
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, limit: 10 }),
            });
            if (response.ok) {
                const data = await response.json();
                // The backend returns an array of tuples: [[scheme, score], ...]
                if (data.results && Array.isArray(data.results)) {
                    setResults(data.results.map((r: any) => r[0]));
                } else {
                    console.error("Unexpected search response format", data);
                    setResults([]);
                }
            } else {
                console.error("Search failed");
                setResults([]);
            }
        } catch (error) {
            console.error("Search error:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container px-4 py-8 mx-auto max-w-6xl">
            <div className="flex flex-col md:flex-row gap-8 mb-8">
                <div className="flex-1 space-y-4">
                    <h1 className="text-3xl font-bold tracking-tight">Discover Schemes</h1>
                    <p className="text-muted-foreground text-lg">
                        Search through thousands of government schemes using natural language.
                    </p>
                </div>
                <div className="flex-1">
                    <form id="search-form" onSubmit={handleSearch} className="flex gap-2 relative">
                        <div className="relative flex-1">
                            <Search className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
                            <Input
                                placeholder="E.g. Engineering scholarships in Maharashtra for SC category"
                                className="pl-10 h-12 text-base rounded-full shadow-sm"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                            />
                        </div>
                        <Button type="submit" className="h-12 px-6 rounded-full" disabled={loading}>
                            {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : "Search"}
                        </Button>
                    </form>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {loading ? (
                    Array.from({ length: 6 }).map((_, i) => (
                        <Card key={i} className="animate-pulse border-none shadow-sm h-64 bg-muted/50" />
                    ))
                ) : results.length > 0 ? (
                    results.map((scheme, idx) => (
                        <Card key={idx} className="flex flex-col h-full hover:shadow-md transition-all duration-300 border-border/50 bg-background/60 backdrop-blur">
                            <CardHeader>
                                <div className="flex justify-between items-start mb-2 gap-2">
                                    <Badge variant="outline" className="bg-primary/5 text-primary border-primary/20">
                                        {scheme.state !== "NaN" ? scheme.state : "Central"}
                                    </Badge>
                                    {scheme.category && scheme.category !== "NaN" && (
                                        <Badge variant="secondary" className="whitespace-nowrap">
                                            {scheme.category.substring(0, 15)}{scheme.category.length > 15 ? '...' : ''}
                                        </Badge>
                                    )}
                                </div>
                                <CardTitle className="text-xl leading-tight line-clamp-2">{scheme.name}</CardTitle>
                            </CardHeader>
                            <CardContent className="flex-1">
                                <p className="text-sm text-muted-foreground line-clamp-3">
                                    {scheme.details}
                                </p>
                            </CardContent>
                            <CardFooter className="pt-4 border-t border-border/50">
                                <Button variant="ghost" className="w-full justify-between" onClick={() => setSelectedScheme(scheme)}>
                                    Read Full Details
                                    <ArrowRight className="h-4 w-4" />
                                </Button>
                            </CardFooter>
                        </Card>
                    ))
                ) : query && !loading ? (
                    <div className="col-span-full py-20 text-center text-muted-foreground">
                        <Search className="h-12 w-12 mx-auto mb-4 opacity-20" />
                        <p className="text-xl">No schemes found matching your query.</p>
                        <p className="mt-2">Try rephrasing or using fewer keywords.</p>
                    </div>
                ) : (
                    <div className="col-span-full py-16 text-center">
                        <div className="bg-primary/5 rounded-full p-6 inline-flex items-center justify-center mb-6">
                            <Search className="h-10 w-10 text-primary/40" />
                        </div>
                        <h3 className="text-xl font-medium text-foreground mb-2">Discover Government Schemes</h3>
                        <p className="text-muted-foreground mb-8 max-w-lg mx-auto">
                            Search naturally for scholarships, business loans, agricultural subsidies, or health benefits.
                        </p>

                        <div className="max-w-2xl mx-auto">
                            <p className="text-sm font-medium text-muted-foreground mb-4 uppercase tracking-wider">Try asking about:</p>
                            <div className="flex flex-wrap justify-center gap-3">
                                {[
                                    "Scholarships for SC students in Maharashtra",
                                    "Mudra loan for new business",
                                    "Financial help for pregnant women",
                                    "Subsidies for farmers growing wheat",
                                    "Pension schemes for senior citizens"
                                ].map((example, i) => (
                                    <Badge
                                        key={i}
                                        variant="secondary"
                                        className="px-4 py-2 cursor-pointer hover:bg-primary/10 transition-colors text-sm font-normal"
                                        onClick={() => {
                                            setQuery(example);
                                            // We use a slight timeout to ensure state updates before form submission
                                            setTimeout(() => {
                                                const form = document.getElementById('search-form');
                                                if (form) form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
                                            }, 50);
                                        }}
                                    >
                                        "{example}"
                                    </Badge>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Detail Dialog */}
            <Dialog open={!!selectedScheme} onOpenChange={(open) => !open && setSelectedScheme(null)}>
                <DialogContent className="max-w-3xl max-h-[90vh] flex flex-col p-0 overflow-hidden">
                    {selectedScheme && (
                        <>
                            <DialogHeader className="p-6 border-b bg-muted/30">
                                <div className="flex gap-2 mb-3">
                                    <Badge variant="outline" className="bg-primary/5 text-primary">
                                        <MapPin className="mr-1 h-3 w-3" />
                                        {selectedScheme.state !== "NaN" ? selectedScheme.state : "Central Scheme"}
                                    </Badge>
                                </div>
                                <DialogTitle className="text-2xl">{selectedScheme.name}</DialogTitle>
                                <DialogDescription>
                                    Source: <a href={selectedScheme.source} target="_blank" rel="noreferrer" className="text-primary hover:underline">{selectedScheme.source.substring(0, 50)}...</a>
                                </DialogDescription>
                            </DialogHeader>
                            <ScrollArea className="flex-1 p-6">
                                <div className="space-y-6">
                                    <div>
                                        <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                                            <Tag className="h-5 w-5 text-blue-500" /> Details
                                        </h3>
                                        <p className="text-muted-foreground leading-relaxed">{selectedScheme.details}</p>
                                    </div>
                                    {selectedScheme.benefits && selectedScheme.benefits !== "NaN" && (
                                        <div className="p-4 bg-green-500/5 rounded-xl border border-green-500/10">
                                            <h3 className="text-lg font-semibold text-green-700 dark:text-green-400 mb-2">Benefits</h3>
                                            <p className="leading-relaxed whitespace-pre-line">{selectedScheme.benefits}</p>
                                        </div>
                                    )}
                                    {selectedScheme.eligibility && selectedScheme.eligibility !== "NaN" && (
                                        <div>
                                            <h3 className="text-lg font-semibold mb-2">Eligibility Criteria</h3>
                                            <p className="text-muted-foreground leading-relaxed whitespace-pre-line">{selectedScheme.eligibility}</p>
                                        </div>
                                    )}
                                    {selectedScheme.application_process && selectedScheme.application_process !== "NaN" && (
                                        <div className="p-4 bg-blue-500/5 rounded-xl border border-blue-500/10">
                                            <h3 className="text-lg font-semibold text-blue-700 dark:text-blue-400 mb-2">Application Process</h3>
                                            <p className="leading-relaxed whitespace-pre-line">{selectedScheme.application_process}</p>
                                        </div>
                                    )}
                                </div>
                            </ScrollArea>
                            <div className="p-4 border-t bg-muted/30 flex justify-end">
                                <Button onClick={() => setSelectedScheme(null)}>Close</Button>
                            </div>
                        </>
                    )}
                </DialogContent>
            </Dialog>
        </div>
    );
}
