import { motion } from "framer-motion";
import { X } from "lucide-react";
import { type SchemeData } from "@/data/useSchemes";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

const tabs = ["Overview", "Eligibility", "Documents", "Apply Now"];

const SchemeDetailPanel = ({
  scheme,
  onClose,
}: {
  scheme: SchemeData;
  onClose: () => void;
}) => {
  const [activeTab, setActiveTab] = useState("Overview");
  const navigate = useNavigate();

  const handleAskVidya = () => {
    // Navigate to hub with scheme context
    navigate("/hub", {
      state: {
        askAboutScheme: {
          name: scheme.name,
          description: scheme.description,
          amount: scheme.amount,
          level: scheme.level,
          categories: scheme.categories,
          eligibility: scheme.eligibility,
          deadline: scheme.deadline
        }
      }
    });
    onClose(); // Close the modal
  };

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-40 bg-background/60 backdrop-blur-sm"
        onClick={onClose}
      />
      {/* Slide-in panel */}
      <motion.div
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", damping: 30, stiffness: 300 }}
        className="fixed right-0 top-0 bottom-0 z-50 w-full max-w-2xl border-l border-border bg-background overflow-y-auto"
      >
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-background/90 backdrop-blur px-6 py-4">
          <div className="flex items-center gap-3 min-w-0">
            <h2 className="font-display text-xl font-bold text-foreground truncate">{scheme.name}</h2>
            <span
              className={`shrink-0 rounded px-2 py-0.5 font-mono text-[11px] uppercase border ${
                scheme.level === "Central"
                  ? "border-primary/30 bg-primary/10 text-primary"
                  : "border-secondary/30 bg-secondary/10 text-secondary"
              }`}
            >
              {scheme.level}
            </span>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground shrink-0 ml-4">
            <X size={20} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 border-b border-border px-6">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-3 font-display text-sm font-bold transition-colors ${
                activeTab === tab
                  ? "border-b-2 border-b-primary text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
          className="p-6 space-y-4"
        >
          {activeTab === "Overview" && (
            <div>
              <p className="font-body text-sm text-muted-foreground leading-relaxed">{scheme.description}</p>
              <div className="mt-4 flex flex-wrap gap-3">
                {scheme.amount && (
                  <div className="rounded border border-border bg-muted/30 px-4 py-3">
                    <div className="font-mono text-[11px] text-muted-foreground uppercase">Amount</div>
                    <div className="font-display text-lg font-bold text-foreground">{scheme.amount}</div>
                  </div>
                )}
                <div className="rounded border border-border bg-muted/30 px-4 py-3">
                  <div className="font-mono text-[11px] text-muted-foreground uppercase">Level</div>
                  <div className="font-display text-lg font-bold text-foreground">{scheme.level}</div>
                </div>
                <div className="rounded border border-border bg-muted/30 px-4 py-3">
                  <div className="font-mono text-[11px] text-muted-foreground uppercase">Match</div>
                  <div className="font-display text-lg font-bold text-primary">{scheme.relevance}%</div>
                </div>
                {scheme.deadline && (
                  <div className="rounded border border-border bg-muted/30 px-4 py-3">
                    <div className="font-mono text-[11px] text-muted-foreground uppercase">Deadline</div>
                    <div className="font-display text-lg font-bold text-foreground">{scheme.deadline}</div>
                  </div>
                )}
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {scheme.categories.map(c => (
                  <span key={c} className="rounded border border-border bg-muted/30 px-2.5 py-1 font-mono text-xs text-muted-foreground">{c}</span>
                ))}
              </div>
            </div>
          )}
          {activeTab === "Eligibility" && (
            <div className="space-y-3">
              {scheme.eligibility ? (
                <p className="font-body text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">{scheme.eligibility}</p>
              ) : (
                <div>
                  <p className="font-body text-sm text-muted-foreground">Eligibility criteria:</p>
                  <ul className="list-inside list-disc space-y-2 font-body text-sm text-muted-foreground mt-2">
                    <li>Must be an Indian citizen</li>
                    <li>Category: {scheme.categories.join(", ")}</li>
                    <li>Region: {scheme.state || "All India"}</li>
                  </ul>
                </div>
              )}
            </div>
          )}
          {activeTab === "Documents" && (
            <ul className="list-inside list-disc space-y-2 font-body text-sm text-muted-foreground">
              {scheme.documents?.length ? (
                scheme.documents.map((doc, i) => <li key={i}>{doc}</li>)
              ) : (
                <>
                  <li>Aadhaar Card</li>
                  <li>Income Certificate</li>
                  <li>Caste Certificate (if applicable)</li>
                  <li>Bank Account Passbook</li>
                </>
              )}
            </ul>
          )}
          {activeTab === "Apply Now" && (
            <div className="text-center py-8">
              <p className="font-body text-sm text-muted-foreground mb-4">
                Application portal will redirect to the official government website.
              </p>
              {scheme.applicationLink ? (
                <a
                  href={scheme.applicationLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-block rounded-lg bg-primary px-6 py-3 font-display font-bold text-primary-foreground transition-all hover:brightness-110"
                >
                  Apply on Official Portal →
                </a>
              ) : (
                <button className="rounded-lg bg-primary px-6 py-3 font-display font-bold text-primary-foreground transition-all hover:brightness-110">
                  Apply on Official Portal →
                </button>
              )}
            </div>
          )}
        </motion.div>

        {/* Sticky bottom bar */}
        <div className="sticky bottom-0 flex items-center justify-between border-t border-border bg-background px-6 py-4">
          <button className="rounded-lg bg-primary px-6 py-2.5 font-display text-sm font-bold text-primary-foreground transition-all hover:brightness-110">
            Apply Now →
          </button>
          <button
            onClick={handleAskVidya}
            className="rounded-lg border border-primary/30 px-4 py-2.5 font-display text-sm font-bold text-primary transition-colors hover:bg-primary/10"
          >
            Ask Vidya about this
          </button>
        </div>
      </motion.div>
    </>
  );
};

export default SchemeDetailPanel;
