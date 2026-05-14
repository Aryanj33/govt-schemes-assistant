const schemeNames = [
  "PM-KISAN", "MUDRA Loan", "Post Matric Scholarship", "Ayushman Bharat",
  "PM Fasal Bima", "INSPIRE Fellowship", "Startup India", "Digital India",
  "Ujjwala Yojana", "Swachh Bharat", "Jan Dhan Yojana", "Sukanya Samriddhi",
  "PM Awas Yojana", "MGNREGA", "Skill India", "प्रधानमंत्री किसान", "मुद्रा ऋण",
];

const Marquee = () => {
  const items = [...schemeNames, ...schemeNames];
  return (
    <div className="overflow-hidden border-y border-primary/15 bg-card py-3.5 fade-mask-x">
      <div className="marquee-track flex whitespace-nowrap">
        {items.map((name, i) => (
          <span key={i} className="mx-6 font-mono text-[13px] text-foreground">
            <span className="mr-3 text-primary text-[10px]">◆</span>
            {name}
          </span>
        ))}
      </div>
    </div>
  );
};

export default Marquee;
