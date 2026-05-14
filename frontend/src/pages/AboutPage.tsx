import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";
import { Shield, Zap, HeartHandshake, Database } from "lucide-react";

export default function AboutPage() {
    const features = [
        {
            icon: <Zap className="h-6 w-6 text-yellow-500" />,
            title: "Lightning Fast Search",
            description: "Powered by an advanced AI hybrid-search engine (BM25 + FAISS) processing over 3,400 schemes instantly."
        },
        {
            icon: <Shield className="h-6 w-6 text-blue-500" />,
            title: "Accurate Information",
            description: "Data directly sourced and cross-encoded from official government databases to ensure high reliability."
        },
        {
            icon: <HeartHandshake className="h-6 w-6 text-green-500" />,
            title: "Accessible to All",
            description: "Interact using natural conversational voice in multiple Indian languages, breaking the digital literacy barrier."
        },
        {
            icon: <Database className="h-6 w-6 text-purple-500" />,
            title: "Comprehensive Coverage",
            description: "Includes central and state schemes covering scholarships, health, agriculture, and business loans."
        }
    ];

    return (
        <div className="container px-4 py-16 mx-auto max-w-4xl">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-center mb-16"
            >
                <Badge variant="outline" className="mb-4 bg-primary/5 text-primary border-primary/20">The Mission</Badge>
                <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-6">
                    Democratizing Government <br className="hidden sm:block" /> Scheme Discovery
                </h1>
                <p className="text-xl text-muted-foreground leading-relaxed max-w-2xl mx-auto">
                    Sarkari Mitra (Vidya) is an AI-powered counselor designed to bridge the gap between citizens and the benefits they are entitled to. We believe finding help should be as easy as having a conversation.
                </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
                {features.map((feature, idx) => (
                    <motion.div
                        key={idx}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.4, delay: idx * 0.1 }}
                    >
                        <Card className="h-full border-border/50 bg-background/50 backdrop-blur hover:shadow-md transition-shadow">
                            <CardContent className="p-6">
                                <div className="p-3 bg-muted rounded-xl inline-block mb-4">
                                    {feature.icon}
                                </div>
                                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                                <p className="text-muted-foreground leading-relaxed">
                                    {feature.description}
                                </p>
                            </CardContent>
                        </Card>
                    </motion.div>
                ))}
            </div>

            <Card className="bg-primary/5 border-primary/10 overflow-hidden relative">
                <div className="absolute top-0 right-0 w-64 h-64 bg-primary/10 rounded-full blur-3xl -mr-32 -mt-32"></div>
                <div className="absolute bottom-0 left-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl -ml-32 -mb-32"></div>

                <CardContent className="p-8 md:p-12 text-center relative z-10">
                    <h2 className="text-2xl md:text-3xl font-bold mb-4">Open Source Frontend</h2>
                    <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
                        This beautiful new frontend was entirely rebuilt using React, Vite, Tailwind CSS, and Shadcn/ui to provide a premium, accessible citizen experience.
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}
