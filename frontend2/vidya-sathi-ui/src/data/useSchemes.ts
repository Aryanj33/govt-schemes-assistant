import { useState, useEffect, useMemo } from "react";

export interface SchemeData {
  id: string;
  name: string;
  description: string;
  level: "Central" | "State";
  relevance: number;
  amount?: string;
  categories: string[];
  state?: string;
  eligibility?: string;
  documents?: string[];
  applicationLink?: string;
  tags?: string[];
  deadline?: string;
}

interface RawScholarship {
  id: string;
  name: string;
  description: string;
  eligibility?: { category?: string; education_level?: string; gender?: string; income_limit?: number; marks_criteria?: number; disability?: boolean };
  award_amount?: string;
  deadline?: string;
  documents?: string[];
  application_link?: string;
  category?: string[];
  applicable_regions?: string;
  course_types?: string[];
}

interface RawScheme {
  id: string;
  name: string;
  slug?: string;
  details?: string;
  benefits?: string;
  eligibility?: string;
  application_process?: string;
  documents?: string;
  level?: string;
  tags?: string[];
  category?: string[];
}

function normalizeScholarship(s: RawScholarship, idx: number): SchemeData {
  const cats: string[] = [];
  if (s.eligibility?.category && s.eligibility.category !== "All") cats.push(s.eligibility.category);
  if (s.eligibility?.gender === "Female") cats.push("Women");
  if (s.eligibility?.disability) cats.push("Divyang");
  if (s.category) cats.push(...s.category.filter(c => !cats.includes(c)));
  if (cats.length === 0) cats.push("General");

  return {
    id: s.id || `scholarship-${idx}`,
    name: s.name,
    description: s.description,
    level: "Central",
    relevance: Math.floor(70 + Math.random() * 28),
    amount: s.award_amount,
    categories: cats,
    state: s.applicable_regions || "All India",
    eligibility: s.eligibility ? `Education: ${s.eligibility.education_level || "N/A"}${s.eligibility.income_limit ? `, Income < ₹${(s.eligibility.income_limit / 100000).toFixed(1)}L` : ""}${s.eligibility.marks_criteria ? `, Min ${s.eligibility.marks_criteria}%` : ""}` : undefined,
    documents: s.documents,
    applicationLink: s.application_link,
    tags: s.course_types,
    deadline: s.deadline,
  };
}

function normalizeScheme(s: RawScheme, idx: number): SchemeData {
  const desc = s.details ? s.details.slice(0, 200) + (s.details.length > 200 ? "..." : "") : s.benefits?.slice(0, 200) || "";
  const amount = s.benefits?.match(/₹[\s\d,]+(?:\/[-\w]+)?/)?.[0]?.trim();

  return {
    id: s.id || `scheme-${idx}`,
    name: s.name.replace(/^"|"$/g, ""),
    description: desc,
    level: (s.level === "State" ? "State" : "Central") as "Central" | "State",
    relevance: Math.floor(65 + Math.random() * 30),
    amount: amount,
    categories: s.category || s.tags || ["General"],
    state: "All India",
    eligibility: typeof s.eligibility === "string" ? s.eligibility.slice(0, 300) : undefined,
    documents: typeof s.documents === "string" ? s.documents.split(/[.]\s+/).filter(Boolean).slice(0, 8) : undefined,
    tags: s.tags,
  };
}

export function useSchemes() {
  const [scholarships, setScholarships] = useState<SchemeData[]>([]);
  const [schemes, setSchemes] = useState<SchemeData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/data/scholarships.json").then(r => r.json()).catch(() => []),
      fetch("/data/schemes.json").then(r => r.json()).catch(() => []),
    ]).then(([schData, scmData]: [RawScholarship[], RawScheme[]]) => {
      setScholarships(schData.map(normalizeScholarship));
      // Load 500 schemes for comprehensive coverage
      setSchemes(scmData.slice(0, 500).map(normalizeScheme));
      setLoading(false);
    });
  }, []);

  const allSchemes = useMemo(() => [...scholarships, ...schemes], [scholarships, schemes]);

  return { allSchemes, loading };
}

export function filterSchemes(
  schemes: SchemeData[],
  query: string,
  categories: string[],
  levels: string[],
  state: string,
): SchemeData[] {
  return schemes.filter(s => {
    if (query) {
      const q = query.toLowerCase();
      if (!s.name.toLowerCase().includes(q) && !s.description.toLowerCase().includes(q) && !s.categories.some(c => c.toLowerCase().includes(q))) return false;
    }
    if (levels.length) {
      if (!levels.some(l => l.startsWith(s.level))) return false;
    }
    if (categories.length) {
      if (!s.categories.some(c => categories.some(sel => c.toLowerCase().includes(sel.toLowerCase())))) return false;
    }
    if (state && s.state !== state && s.state !== "All India") return false;
    return true;
  });
}
