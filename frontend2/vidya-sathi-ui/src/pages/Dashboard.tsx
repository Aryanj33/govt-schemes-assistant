import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Database, Phone, Server, Timer, UserRound } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { api } from "@/lib/api";

type ActiveSession = {
  session_id: string;
  updated_at?: number;
  turn_count: number;
  message_count: number;
  last_scheme_count: number;
  profile?: Record<string, unknown>;
};

type EndpointMetric = {
  count: number;
  errors: number;
  avg_ms: number;
  p95_ms: number;
};

type MetricsSummary = {
  uptime_seconds: number;
  active_sessions: number;
  session_store_backend: string;
  twilio_configured: boolean;
  cache?: {
    hit_rate?: string;
    cache_size?: number;
    total_queries?: number;
  };
  endpoints?: Record<string, EndpointMetric>;
};

const StatCard = ({
  label,
  value,
  sublabel,
  icon: Icon,
  accent = "text-primary",
}: {
  label: string;
  value: string | number;
  sublabel: string;
  icon: typeof Activity;
  accent?: string;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 12 }}
    animate={{ opacity: 1, y: 0 }}
    className="rounded-xl border border-border bg-card p-5"
  >
    <div className="mb-3 flex items-center justify-between">
      <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-muted-foreground">{label}</span>
      <Icon size={18} className={accent} />
    </div>
    <div className="font-display text-3xl font-bold text-foreground">{value}</div>
    <p className="mt-1 font-body text-sm text-muted-foreground">{sublabel}</p>
  </motion.div>
);

const formatUpdatedAt = (updatedAt?: number) => {
  if (!updatedAt) return "unknown";
  return new Date(updatedAt * 1000).toLocaleTimeString("en-IN");
};

const Dashboard = () => {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [sessions, setSessions] = useState<ActiveSession[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [metricsResponse, sessionsResponse] = await Promise.all([
          api.metricsSummary(),
          api.activeSessions(),
        ]);
        if (cancelled) return;
        setMetrics(metricsResponse);
        setSessions(Array.isArray(sessionsResponse.sessions) ? sessionsResponse.sessions : []);
        setError("");
      } catch {
        if (!cancelled) {
          setError("Could not load live operator data from the backend.");
        }
      }
    };

    load();
    const intervalId = window.setInterval(load, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  const endpointEntries = Object.entries(metrics?.endpoints ?? {});

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <div className="container mx-auto px-4 pt-24 pb-12">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-display text-3xl font-bold text-foreground">Operator Dashboard</h1>
            <p className="mt-2 font-body text-sm text-muted-foreground">
              Live prototype health, session continuity, and request performance.
            </p>
          </div>
          <div className="rounded-full border border-border bg-card px-4 py-2 font-mono text-xs text-muted-foreground">
            Auto-refresh every 5 seconds
          </div>
        </div>

        {error && (
          <div className="mb-6 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 font-body text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Active Sessions"
            value={metrics?.active_sessions ?? 0}
            sublabel="Anonymous live browser and phone sessions"
            icon={UserRound}
          />
          <StatCard
            label="Session Store"
            value={(metrics?.session_store_backend ?? "memory").toUpperCase()}
            sublabel="Durable state backend"
            icon={Database}
            accent="text-secondary"
          />
          <StatCard
            label="Cache Hit Rate"
            value={metrics?.cache?.hit_rate ?? "0.0%"}
            sublabel={`${metrics?.cache?.cache_size ?? 0} cached query shapes`}
            icon={Activity}
          />
          <StatCard
            label="Telephony"
            value={metrics?.twilio_configured ? "READY" : "OFF"}
            sublabel="Outbound and inbound phone support"
            icon={Phone}
            accent={metrics?.twilio_configured ? "text-secondary" : "text-primary"}
          />
        </div>

        <div className="mb-8 grid grid-cols-1 gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-xl border border-border bg-card p-5">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="font-display text-lg font-bold text-foreground">Endpoint Latency</h2>
              <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-muted-foreground">
                avg / p95
              </span>
            </div>

            {endpointEntries.length === 0 ? (
              <p className="font-body text-sm text-muted-foreground">No endpoint timings recorded yet.</p>
            ) : (
              <div className="space-y-3">
                {endpointEntries.map(([name, metric]) => (
                  <div key={name} className="rounded-lg border border-border/70 bg-background/50 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="font-display text-base font-bold text-foreground">{name}</p>
                        <p className="font-mono text-[11px] text-muted-foreground">
                          {metric.count} calls · {metric.errors} errors
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="font-mono text-sm text-foreground">{metric.avg_ms}ms</p>
                        <p className="font-mono text-[11px] text-muted-foreground">p95 {metric.p95_ms}ms</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-xl border border-border bg-card p-5">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="font-display text-lg font-bold text-foreground">Runtime Health</h2>
              <Server size={18} className="text-primary" />
            </div>

            <div className="space-y-4">
              <div className="rounded-lg border border-border/70 bg-background/50 p-4">
                <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.15em] text-muted-foreground">Uptime</div>
                <div className="font-display text-2xl font-bold text-foreground">
                  {metrics ? `${Math.round(metrics.uptime_seconds / 60)} min` : "--"}
                </div>
              </div>
              <div className="rounded-lg border border-border/70 bg-background/50 p-4">
                <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.15em] text-muted-foreground">Cached Queries</div>
                <div className="font-display text-2xl font-bold text-foreground">
                  {metrics?.cache?.total_queries ?? 0}
                </div>
              </div>
              <div className="rounded-lg border border-border/70 bg-background/50 p-4">
                <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.15em] text-muted-foreground">Prototype Status</div>
                <div className="flex items-center gap-2 font-body text-sm text-foreground">
                  <span className="h-2.5 w-2.5 rounded-full bg-secondary" />
                  Operator surface is live and reading backend state
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-display text-lg font-bold text-foreground">Live Sessions</h2>
            <Timer size={18} className="text-primary" />
          </div>

          {sessions.length === 0 ? (
            <p className="font-body text-sm text-muted-foreground">No active sessions yet.</p>
          ) : (
            <div className="space-y-3">
              {sessions.map((session) => (
                <div key={session.session_id} className="rounded-lg border border-border/70 bg-background/50 p-4">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <p className="font-display text-base font-bold text-foreground">{session.session_id}</p>
                      <p className="mt-1 font-mono text-[11px] text-muted-foreground">
                        Updated {formatUpdatedAt(session.updated_at)} · {session.turn_count} turns · {session.message_count} messages
                      </p>
                    </div>
                    <div className="rounded-full border border-primary/20 bg-primary/10 px-3 py-1 font-mono text-[11px] text-primary">
                      {session.last_scheme_count} schemes in context
                    </div>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2">
                    {Object.entries(session.profile ?? {}).slice(0, 4).map(([key, value]) => (
                      <span
                        key={key}
                        className="rounded-full border border-border bg-card px-3 py-1 font-mono text-[11px] text-muted-foreground"
                      >
                        {key}: {String(value)}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default Dashboard;
