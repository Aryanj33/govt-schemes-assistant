import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Mic, X, Filter, ChevronDown, ChevronUp } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import SchemeCard from "@/components/SchemeCard";
import SchemeDetailPanel from "@/components/SchemeDetailModal";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { api } from "@/lib/api";
import { type SchemeData } from "@/data/useSchemes";

const categoryOptions = ["SC", "ST", "OBC", "General", "Women", "Farmers", "MSME", "Students", "Divyang"];
const levels = ["Central", "State"];
const states = [
  "All India", "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa",
  "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
  "Kerala", "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab",
  "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal",
];

const Discover = () => {
  const [query, setQuery] = useState("");
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [selectedLevels, setSelectedLevels] = useState<string[]>([]);
  const [selectedState, setSelectedState] = useState("");
  const [selectedScheme, setSelectedScheme] = useState<SchemeData | null>(null);
  const [schemes, setSchemes] = useState<SchemeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchMessage, setSearchMessage] = useState("Searching schemes...");
  const [searchSessionId, setSearchSessionId] = useState<string>(() => {
    const existing = localStorage.getItem("vidya-discover-session");
    if (existing) return existing;
    const sessionId = `discover-${crypto.randomUUID()}`;
    localStorage.setItem("vidya-discover-session", sessionId);
    return sessionId;
  });
  const [showFilters, setShowFilters] = useState(false);
  const [stateSearch, setStateSearch] = useState("");
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    category: true, level: true, state: true,
  });
  const { isRecording, toggleRecording } = useAudioRecorder();

  const toggleCategory = (cat: string) => {
    setSelectedCategories(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat]);
  };
  const toggleLevel = (lvl: string) => {
    setSelectedLevels(prev => prev.includes(lvl) ? prev.filter(l => l !== lvl) : [...prev, lvl]);
  };
  const clearAll = () => { setSelectedCategories([]); setSelectedLevels([]); setSelectedState(""); setQuery(""); };

  const activeFilterCount = selectedCategories.length + selectedLevels.length + (selectedState ? 1 : 0);
  const filteredStates = states.filter(s => s.toLowerCase().includes(stateSearch.toLowerCase()));

  useEffect(() => {
    const timeoutId = window.setTimeout(async () => {
      setLoading(true);
      try {
        const level = selectedLevels.length === 1 ? selectedLevels[0] : undefined;
        const state = selectedState && selectedState !== "All India" ? selectedState : undefined;
        const response = await api.search(query, {
          categories: selectedCategories.length ? selectedCategories : undefined,
          level,
          state,
          session_id: searchSessionId,
        });
        if (response.session_id && response.session_id !== searchSessionId) {
          setSearchSessionId(response.session_id);
          localStorage.setItem("vidya-discover-session", response.session_id);
        }
        setSchemes(Array.isArray(response.schemes) ? response.schemes : []);
        setSearchMessage(response.message || "Search complete");
      } catch {
        setSchemes([]);
        setSearchMessage("Search failed. Check if backend is running on port 8080.");
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => window.clearTimeout(timeoutId);
  }, [query, selectedCategories, selectedLevels, selectedState, searchSessionId]);

  const toggleSection = (key: string) => {
    setExpandedSections(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const SectionHeader = ({ label, sectionKey }: { label: string; sectionKey: string }) => (
    <button onClick={() => toggleSection(sectionKey)} className="flex w-full items-center justify-between mb-2">
      <h3 className="font-display text-xs font-bold uppercase tracking-wider text-muted-foreground/70">{label}</h3>
      {expandedSections[sectionKey] ? <ChevronUp size={14} className="text-muted-foreground/50" /> : <ChevronDown size={14} className="text-muted-foreground/50" />}
    </button>
  );

  const FilterPanel = () => (
    <div className="space-y-6">
      <h2 className="font-display text-sm font-bold uppercase tracking-[0.15em] text-muted-foreground flex items-center gap-2">
        Filters
        {activeFilterCount > 0 && (
          <span className="rounded-full bg-primary/20 px-2 py-0.5 font-mono text-[10px] text-primary">{activeFilterCount}</span>
        )}
      </h2>

      {/* Category */}
      <div>
        <SectionHeader label="Category" sectionKey="category" />
        {expandedSections.category && (
          <div className="flex flex-wrap gap-2">
            {categoryOptions.map((cat) => (
              <button
                key={cat}
                onClick={() => toggleCategory(cat)}
                className={`rounded-full px-3 py-1 font-mono text-xs transition-all ${
                  selectedCategories.includes(cat)
                    ? "bg-primary text-primary-foreground"
                    : "border border-border text-muted-foreground hover:border-primary/40"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Level */}
      <div>
        <SectionHeader label="Level" sectionKey="level" />
        {expandedSections.level && (
          <div className="flex flex-col gap-2">
            {levels.map((lvl) => (
              <label key={lvl} className="flex items-center gap-2.5 cursor-pointer group">
                <div
                  className={`h-4 w-4 rounded border-2 flex items-center justify-center transition-all ${
                    selectedLevels.includes(lvl)
                      ? "border-primary bg-primary"
                      : "border-muted-foreground/30 group-hover:border-primary/50"
                  }`}
                  onClick={() => toggleLevel(lvl)}
                >
                  {selectedLevels.includes(lvl) && (
                    <svg width="10" height="8" viewBox="0 0 10 8" fill="none"><path d="M1 4L3.5 6.5L9 1" stroke="hsl(var(--primary-foreground))" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  )}
                </div>
                <span className="font-body text-sm text-muted-foreground">{lvl} Government</span>
              </label>
            ))}
          </div>
        )}
      </div>

      {/* State */}
      <div>
        <SectionHeader label="State" sectionKey="state" />
        {expandedSections.state && (
          <div>
            <input
              type="text"
              value={stateSearch}
              onChange={(e) => setStateSearch(e.target.value)}
              placeholder="Search states..."
              className="mb-2 w-full rounded border border-border bg-card px-3 py-2 font-body text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-primary/40"
            />
            <div className="max-h-40 overflow-y-auto space-y-1">
              {filteredStates.map(s => (
                <button
                  key={s}
                  onClick={() => setSelectedState(selectedState === s ? "" : s)}
                  className={`block w-full text-left rounded px-3 py-1.5 font-body text-sm transition-colors ${
                    selectedState === s
                      ? "bg-primary/15 text-primary"
                      : "text-muted-foreground hover:bg-muted/30"
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <button
        onClick={clearAll}
        className="w-full rounded-lg bg-primary py-2.5 font-display text-sm font-bold text-primary-foreground transition-all hover:brightness-110"
      >
        Apply Filters
      </button>
      {activeFilterCount > 0 && (
        <button onClick={clearAll} className="block w-full text-center font-body text-xs text-primary hover:underline">
          Clear All
        </button>
      )}
    </div>
  );

  // Active filter chips
  const activeChips = [
    ...selectedCategories.map(c => ({ label: c, remove: () => toggleCategory(c) })),
    ...selectedLevels.map(l => ({ label: l, remove: () => toggleLevel(l) })),
    ...(selectedState ? [{ label: selectedState, remove: () => setSelectedState("") }] : []),
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <div className="container mx-auto px-4 pt-24 pb-12">
        {/* Search bar */}
        <div className="relative mb-4">
          <div
            className="flex items-center rounded-lg bg-card px-4 transition-all border"
            style={{ height: 64, borderColor: "rgba(255,153,51,0.2)" }}
          >
            <button
              onClick={async () => {
                const blob = await toggleRecording();
                if (blob) console.log("Audio recorded", blob.size);
              }}
              className={`mr-3 flex h-10 w-10 items-center justify-center rounded-full transition-all ${
                isRecording
                  ? "bg-primary text-primary-foreground animate-pulse"
                  : "bg-muted text-muted-foreground hover:text-primary"
              }`}
            >
              <Mic size={18} />
            </button>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search in Hindi or English... e.g. 'engineering scholarship SC category'"
              className="flex-1 bg-transparent font-body text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none"
            />
            {query && (
              <button onClick={() => setQuery("")} className="ml-2 text-muted-foreground hover:text-foreground">
                <X size={16} />
              </button>
            )}
            <button className="ml-2 flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Search size={18} />
            </button>
          </div>
        </div>

        {/* Active filter chips */}
        {activeChips.length > 0 && (
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs text-muted-foreground">Query interpreted as →</span>
            {activeChips.map(chip => (
              <span key={chip.label} className="flex items-center gap-1 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-xs text-primary">
                {chip.label}
                <button onClick={chip.remove} className="ml-1 hover:text-foreground"><X size={12} /></button>
              </span>
            ))}
          </div>
        )}

        <div className="flex gap-8">
          {/* Sidebar - desktop */}
          <aside className="hidden w-64 shrink-0 lg:block">
            <FilterPanel />
          </aside>

          {/* Main */}
          <main className="flex-1">
            {/* Mobile filter toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="mb-4 flex items-center gap-2 rounded border border-border px-3 py-2 font-display text-sm text-muted-foreground lg:hidden"
            >
              <Filter size={14} />
              Filters {activeFilterCount > 0 && `(${activeFilterCount})`}
            </button>

            {showFilters && (
              <div className="mb-6 rounded-lg border border-border bg-card p-4 lg:hidden">
                <FilterPanel />
              </div>
            )}

            {/* Result count */}
            <p className="mb-6 font-mono text-xs text-muted-foreground">
              Showing {schemes.length} results
              {query && <> for &quot;{query}&quot;</>}
            </p>

            {/* Loading skeleton */}
            {loading && (
              <div className="grid gap-4 sm:grid-cols-2">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="rounded-lg border border-border/50 bg-card p-5" style={{ borderLeftWidth: 3, borderLeftColor: "hsl(var(--primary) / 0.3)" }}>
                    <div className="skeleton-shimmer h-4 w-20 rounded mb-3" />
                    <div className="skeleton-shimmer h-5 w-3/4 rounded mb-2" />
                    <div className="skeleton-shimmer h-4 w-full rounded mb-1" />
                    <div className="skeleton-shimmer h-4 w-2/3 rounded mb-4" />
                    <div className="skeleton-shimmer h-3 w-1/2 rounded" />
                  </div>
                ))}
              </div>
            )}

            {/* Results grid */}
            {!loading && (
              <div className="grid gap-4 sm:grid-cols-2">
                {schemes.map((scheme, i) => (
                  <SchemeCard
                    key={scheme.id}
                    scheme={scheme}
                    index={i}
                    onClick={() => setSelectedScheme(scheme)}
                  />
                ))}
              </div>
            )}

            {!loading && schemes.length === 0 && (
              <div className="py-20 text-center">
                <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full border border-primary/20">
                  <Search size={32} className="text-primary/40" />
                </div>
                <p className="font-display text-xl text-muted-foreground">No schemes found</p>
                <p className="mt-2 font-body text-sm text-muted-foreground/70">{searchMessage}</p>
                <div className="mt-4 flex justify-center gap-2">
                  {["engineering SC Maharashtra", "farming loan", "women scholarship"].map(sug => (
                    <button
                      key={sug}
                      onClick={() => setQuery(sug)}
                      className="rounded-full border border-primary/20 px-3 py-1 font-mono text-xs text-primary hover:bg-primary/10 transition-colors"
                    >
                      {sug}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </main>
        </div>
      </div>

      {/* Scheme Detail Panel */}
      <AnimatePresence>
        {selectedScheme && (
          <SchemeDetailPanel
            scheme={selectedScheme}
            onClose={() => setSelectedScheme(null)}
          />
        )}
      </AnimatePresence>

      <Footer />
    </div>
  );
};

export default Discover;
