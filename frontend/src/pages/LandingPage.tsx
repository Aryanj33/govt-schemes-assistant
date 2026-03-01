import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Mic, Search, Shield, BookOpen, Stethoscope, Briefcase } from "lucide-react";
import { motion } from "framer-motion";

export default function LandingPage() {
    const containerVariants = {
        hidden: { opacity: 0 },
        show: {
            opacity: 1,
            transition: { staggerChildren: 0.1 }
        }
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0 }
    };

    return (
        <div className="flex flex-col min-h-screen">
            {/* Hero Section */}
            <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-background to-background" />
                <div className="container px-4 mx-auto relative z-10 text-center">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5 }}
                    >
                        <Badge variant="secondary" className="mb-6 px-4 py-1.5 text-sm font-medium rounded-full border bg-background/50 backdrop-blur-sm">
                            ✨ Introducing Vidya AI Counselor
                        </Badge>
                    </motion.div>
                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6"
                    >
                        Find government <br className="hidden md:block" />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-600">
                            schemes with your voice
                        </span>
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className="text-xl text-muted-foreground mb-10 max-w-2xl mx-auto"
                    >
                        Talk to Vidya in Hindi, English, or Hinglish to discover the benefits and scholarships you are eligible for in seconds.
                    </motion.p>
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.3 }}
                        className="flex flex-col sm:flex-row items-center justify-center gap-4"
                    >
                        <Link to="/vidya-hub">
                            <Button size="lg" className="h-14 px-8 rounded-full text-lg shadow-lg shadow-primary/20 transition-transform hover:scale-105">
                                <Mic className="mr-2 h-5 w-5" />
                                Talk to Vidya Now
                            </Button>
                        </Link>
                        <Link to="/discover">
                            <Button size="lg" variant="outline" className="h-14 px-8 rounded-full text-lg bg-background/50 backdrop-blur-sm">
                                <Search className="mr-2 h-5 w-5" />
                                Browse 3,000+ Schemes
                            </Button>
                        </Link>
                    </motion.div>
                </div>
            </section>

            {/* Categories Grid */}
            <section className="py-20 bg-muted/30">
                <div className="container px-4 mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold tracking-tight mb-4">Who is it for?</h2>
                        <p className="text-muted-foreground">Schemes categorized to help every citizen of India.</p>
                    </div>
                    <motion.div
                        variants={containerVariants}
                        initial="hidden"
                        whileInView="show"
                        viewport={{ once: true, margin: "-100px" }}
                        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto"
                    >
                        {[
                            { title: "Students", icon: <BookOpen className="h-8 w-8 text-blue-500" />, desc: "Scholarships & Hostels" },
                            { title: "Farmers", icon: <Shield className="h-8 w-8 text-green-500" />, desc: "PM-KISAN & Insurance" },
                            { title: "MSME & Business", icon: <Briefcase className="h-8 w-8 text-purple-500" />, desc: "Mudra Loans & Registration" },
                            { title: "Health", icon: <Stethoscope className="h-8 w-8 text-red-500" />, desc: "Ayushman Bharat & Camps" },
                        ].map((cat, i) => (
                            <motion.div key={i} variants={itemVariants}>
                                <Card className="hover:shadow-md transition-shadow border-none bg-background/60 backdrop-blur supports-[backdrop-filter]:bg-background/40">
                                    <CardContent className="p-6 flex flex-col items-center text-center">
                                        <div className="p-4 bg-muted rounded-full mb-4">
                                            {cat.icon}
                                        </div>
                                        <h3 className="font-semibold text-lg mb-2">{cat.title}</h3>
                                        <p className="text-sm text-muted-foreground">{cat.desc}</p>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>
        </div>
    );
}
