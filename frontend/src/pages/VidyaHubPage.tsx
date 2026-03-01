import { useState } from "react";
import { LiveKitRoom, RoomAudioRenderer, VoiceAssistantControlBar, AudioVisualizer, useVoiceAssistant } from "@livekit/components-react";
import { Volume2, Loader2, Disc3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { motion, AnimatePresence } from "framer-motion";

export default function VidyaHubPage() {
    const [token, setToken] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const [inRoom, setInRoom] = useState<boolean>(false);

    // Hardcoded for now, real app should get from /livekit/url endpoint if dynamic
    const serverUrl = import.meta.env.VITE_LIVEKIT_URL || "wss://your-livekit-url";

    const connectToVoice = async () => {
        setLoading(true);
        try {
            // In a real app, this would hit your Python backend to get a specific token
            // Currently, the python backend directly serves the html with the token injected, 
            // but here we simulate a token fetch endpoint that we would need to add to the python backend.
            const response = await fetch("http://localhost:8080/token", { method: "POST" });
            if (response.ok) {
                const data = await response.json();
                setToken(data.token);
                setInRoom(true);
            } else {
                alert("Failed to get connection token. Ensure the Python backend is running and GROQ/LiveKit keys are present.");
            }
        } catch (e) {
            console.error(e);
            alert("Error connecting to backend server.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container px-4 py-12 mx-auto max-w-4xl flex flex-col items-center">
            <div className="text-center mb-10">
                <Badge variant="outline" className="mb-4 bg-primary/5 text-primary">Live Voice Assistant</Badge>
                <h1 className="text-4xl font-bold tracking-tight mb-4">Talk to Vidya</h1>
                <p className="text-muted-foreground text-lg max-w-xl mx-auto">
                    Vidya is an AI government counselor connected to a database of 3,400+ schemes. Ask her about scholarships, loans, or pensions.
                </p>
            </div>

            <Card className="w-full max-w-2xl bg-gradient-to-br from-card to-muted/20 border-border/50 shadow-xl overflow-hidden relative min-h-[400px] flex items-center justify-center">
                {!inRoom ? (
                    <div className="flex flex-col items-center justify-center p-10 text-center z-10">
                        <div className="w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center mb-6">
                            <Volume2 className="h-10 w-10 text-primary" />
                        </div>
                        <h2 className="text-2xl font-semibold mb-2">Ready to assist you</h2>
                        <p className="text-muted-foreground mb-8">Click connect to establish a secure voice connection.</p>
                        <Button size="lg" className="rounded-full h-12 px-8 text-base shadow-lg" onClick={connectToVoice} disabled={loading}>
                            {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Volume2 className="mr-2 h-5 w-5" />}
                            {loading ? "Connecting..." : "Connect to Voice"}
                        </Button>
                    </div>
                ) : (
                    <div className="w-full h-full flex flex-col pt-8">
                        <LiveKitRoom
                            token={token}
                            serverUrl={serverUrl}
                            connect={true}
                            audio={true}
                            video={false}
                            className="flex flex-col w-full h-full"
                        >
                            <ActiveVoiceSession onDisconnect={() => setInRoom(false)} />
                            <RoomAudioRenderer />
                        </LiveKitRoom>
                    </div>
                )}
            </Card>

            <div className="mt-12 w-full max-w-2xl">
                <h3 className="font-semibold text-lg mb-4 text-center">Try asking:</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Card className="p-4 bg-muted/30 border-none cursor-help hover:bg-muted/50 transition-colors">
                        <p className="text-sm">"मुझे इंजीनियरिंग की पढ़ाई के लिए छात्रवृत्ति चाहिए, मैं उत्तर प्रदेश से हूँ।"</p>
                    </Card>
                    <Card className="p-4 bg-muted/30 border-none cursor-help hover:bg-muted/50 transition-colors">
                        <p className="text-sm">"How can I apply for a Mudra loan for my new business?"</p>
                    </Card>
                </div>
            </div>
        </div>
    );
}

// Inner component that actually uses the LiveKit hooks
function ActiveVoiceSession({ onDisconnect }: { onDisconnect: () => void }) {
    const { state, audioTrack } = useVoiceAssistant();

    return (
        <div className="flex flex-col h-full bg-transparent w-full pb-8">
            <div className="flex-1 flex flex-col items-center justify-center py-10 relative">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={state}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="flex flex-col items-center justify-center z-10"
                    >
                        {/* The "Vidya" Avatar / Visualizer */}
                        <div className={`relative flex items-center justify-center w-48 h-48 rounded-full mb-8 transition-all duration-500
              ${state === "listening" ? "shadow-[0_0_40px_rgba(59,130,246,0.5)] border-4 border-blue-500/50" : ""}
              ${state === "speaking" ? "shadow-[0_0_60px_rgba(34,197,94,0.6)] border-4 border-green-500/50" : ""}
              ${state === "thinking" ? "shadow-[0_0_30px_rgba(168,85,247,0.5)] border-4 border-purple-500/50" : ""}
              ${state === "connecting" ? "scale-90 opacity-50 border-4 border-gray-500/50" : ""}
              ${state === "disconnected" ? "opacity-30 border-4 border-red-500/50" : ""}
              bg-background
            `}>
                            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-primary/20 to-transparent m-2"></div>

                            {state === "speaking" && audioTrack && (
                                <div className="absolute inset-0 opacity-80 z-0">
                                    <AudioVisualizer trackRef={audioTrack} className="w-full h-full text-green-500" />
                                </div>
                            )}

                            {state === "thinking" && (
                                <Disc3 className="h-16 w-16 text-purple-500 animate-spin z-10 opacity-80" />
                            )}

                            {state === "listening" && (
                                <Volume2 className="h-16 w-16 text-blue-500 animate-pulse z-10" />
                            )}

                            {state === "idle" || state === "connecting" || state === "disconnected" ? (
                                <div className="h-16 w-16 rounded-full bg-primary/20 flex items-center justify-center z-10">
                                    <span className="text-3xl">🤖</span>
                                </div>
                            ) : null}
                        </div>

                        <Badge variant="secondary" className="px-4 py-1.5 text-sm uppercase tracking-wider bg-background/80 backdrop-blur-md">
                            {state.toUpperCase()}
                        </Badge>
                    </motion.div>
                </AnimatePresence>
            </div>

            <div className="w-full flex justify-center mt-auto pb-6">
                <div className="bg-background/80 backdrop-blur-md p-2 rounded-full shadow-lg border flex gap-2">
                    <VoiceAssistantControlBar controls={{ leave: false }} />
                    <Button variant="destructive" size="icon" className="rounded-full" onClick={onDisconnect}>
                        <Volume2 className="h-4 w-4" />
                    </Button>
                </div>
            </div>
        </div>
    );
}
