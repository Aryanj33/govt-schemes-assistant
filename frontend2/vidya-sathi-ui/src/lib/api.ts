const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

export const api = {
  search: async (query: string, filters?: Record<string, unknown>) => {
    const res = await fetch(`${API_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, ...filters }),
    });
    if (!res.ok) throw new Error("Search failed");
    return res.json();
  },

  textChat: async (message: string, sessionId?: string) => {
    const res = await fetch(`${API_URL}/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    if (!res.ok) throw new Error("Chat failed");
    return res.json();
  },

  sendAudio: async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    const res = await fetch(`${API_URL}/audio`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Audio processing failed");
    return res.json();
  },

  streamAudio: async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    const res = await fetch(`${API_URL}/audio/stream`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Audio stream failed");
    return res;
  },

  health: async () => {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) throw new Error("Health check failed");
    return res.json();
  },

  reset: async (sessionId?: string) => {
    const res = await fetch(`${API_URL}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) throw new Error("Reset failed");
    return res.json();
  },

  initiateCall: async (phoneNumber: string, language?: string) => {
    const res = await fetch(`${API_URL}/call`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phone_number: phoneNumber, language }),
    });
    if (!res.ok) throw new Error("Call initiation failed");
    return res.json();
  },

  callStatus: async (callId: string) => {
    const res = await fetch(`${API_URL}/call/status/${callId}`);
    if (!res.ok) throw new Error("Call status check failed");
    return res.json();
  },

  activeSessions: async () => {
    const res = await fetch(`${API_URL}/sessions/active`);
    if (!res.ok) throw new Error("Active sessions fetch failed");
    return res.json();
  },

  sessionDetails: async (sessionId: string) => {
    const res = await fetch(`${API_URL}/sessions/${sessionId}`);
    if (!res.ok) throw new Error("Session details fetch failed");
    return res.json();
  },

  metricsSummary: async () => {
    const res = await fetch(`${API_URL}/metrics/summary`);
    if (!res.ok) throw new Error("Metrics summary fetch failed");
    return res.json();
  },
};
