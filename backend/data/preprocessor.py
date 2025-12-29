"""
Scholarship Preprocessor - Data Processing
==========================================
Cleans and structures scraped scholarship data.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger
from rag.embeddings import get_embedding_generator, create_scholarship_text
from rag.vectorstore import VectorStore

logger = get_logger()


class ScholarshipPreprocessor:
    """
    Preprocesses and structures scholarship data for RAG.
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.embeddings_dir = self.data_dir / "embeddings"
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\xa0', ' ')   # Non-breaking space
        
        return text.strip()
    
    def normalize_amount(self, amount: str) -> str:
        """
        Normalize award amount formatting.
        
        Args:
            amount: Raw amount string
            
        Returns:
            Normalized amount with proper formatting
        """
        if not amount:
            return "Varies"
        
        # Clean the text
        amount = self.clean_text(amount)
        
        # Convert numbers to Indian format (e.g., 100000 -> ₹1,00,000)
        def format_indian(match):
            num = int(match.group(0))
            if num >= 100000:
                return f"₹{num:,.0f}".replace(',', '_').replace('_', ',', 1).replace('_', ',')
            return f"₹{num:,}"
        
        # Add rupee symbol if missing
        if '₹' not in amount and 'INR' not in amount and 'Rs' not in amount:
            amount = re.sub(r'\b(\d{4,})\b', format_indian, amount)
        
        return amount
    
    def extract_categories(self, scholarship: Dict[str, Any]) -> List[str]:
        """
        Extract categories from scholarship data.
        
        Args:
            scholarship: Scholarship dictionary
            
        Returns:
            List of category tags
        """
        categories = set()
        
        # From explicit category field
        if 'category' in scholarship:
            cat = scholarship['category']
            if isinstance(cat, list):
                categories.update(cat)
            else:
                categories.add(str(cat))
        
        # From name and description
        text = f"{scholarship.get('name', '')} {scholarship.get('description', '')}".lower()
        
        # Category keywords
        if any(word in text for word in ['sc ', 'scheduled caste']):
            categories.add('SC')
        if any(word in text for word in ['st ', 'scheduled tribe']):
            categories.add('ST')
        if 'obc' in text or 'other backward' in text:
            categories.add('OBC')
        if 'minority' in text or 'muslim' in text or 'christian' in text:
            categories.add('Minority')
        if 'girl' in text or 'women' in text or 'female' in text:
            categories.add('Female')
        if 'merit' in text or 'topper' in text:
            categories.add('Merit')
        if 'income' in text or 'need' in text or 'ews' in text:
            categories.add('Need-based')
        
        return list(categories)
    
    def process_scholarship(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single scholarship entry.
        
        Args:
            raw: Raw scholarship data
            
        Returns:
            Processed scholarship dictionary
        """
        processed = {
            "id": raw.get("id", "").lower().replace(' ', '-'),
            "name": self.clean_text(raw.get("name", "")),
            "description": self.clean_text(raw.get("description", "")),
            "eligibility": raw.get("eligibility", {}),
            "award_amount": self.normalize_amount(raw.get("award_amount", "")),
            "deadline": self.clean_text(raw.get("deadline", "Check website")),
            "documents": raw.get("documents", []),
            "application_link": raw.get("application_link", ""),
            "category": self.extract_categories(raw),
            "applicable_regions": raw.get("applicable_regions", "All India"),
            "course_types": raw.get("course_types", [])
        }
        
        # Clean eligibility subfields
        if processed["eligibility"]:
            for key, value in processed["eligibility"].items():
                if isinstance(value, str):
                    processed["eligibility"][key] = self.clean_text(value)
        
        return processed
    
    def process_all(self, input_path: Path, output_path: Path) -> int:
        """
        Process all scholarships from input file.
        
        Args:
            input_path: Path to raw scholarships JSON
            output_path: Path to save processed JSON
            
        Returns:
            Number of scholarships processed
        """
        if not input_path.exists():
            logger.error(f"❌ Input file not found: {input_path}")
            return 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = []
        for raw in raw_data:
            try:
                processed = self.process_scholarship(raw)
                processed_data.append(processed)
            except Exception as e:
                logger.warning(f"⚠️ Failed to process: {raw.get('name', 'unknown')}: {e}")
        
        # Save processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Processed {len(processed_data)} scholarships → {output_path}")
        return len(processed_data)
    
    def build_embeddings(self, scholarships_path: Path, index_path: Path) -> bool:
        """
        Build FAISS embeddings for scholarships.
        
        Args:
            scholarships_path: Path to processed scholarships JSON
            index_path: Path to save FAISS index
            
        Returns:
            True if successful
        """
        if not scholarships_path.exists():
            logger.error(f"❌ Scholarships file not found: {scholarships_path}")
            return False
        
        with open(scholarships_path, 'r', encoding='utf-8') as f:
            scholarships = json.load(f)
        
        logger.info(f"📊 Building embeddings for {len(scholarships)} scholarships...")
        
        # Create text representations
        texts = [create_scholarship_text(s) for s in scholarships]
        
        # Generate embeddings
        generator = get_embedding_generator()
        embeddings = generator.encode_documents(texts)
        
        # Create and save index
        vectorstore = VectorStore(dimension=generator.dimension)
        vectorstore.create_index(embeddings, scholarships)
        vectorstore.save(index_path)
        
        logger.info(f"✅ Built FAISS index at {index_path}")
        return True


def main():
    """Main preprocessing function."""
    preprocessor = ScholarshipPreprocessor()
    
    processed_path = preprocessor.processed_dir / "scholarships.json"
    index_path = preprocessor.embeddings_dir / "faiss_index"
    
    # If processed data exists, just build embeddings
    if processed_path.exists():
        logger.info("📚 Using existing processed data")
        preprocessor.build_embeddings(processed_path, index_path)
    else:
        # Look for raw data to process
        raw_path = preprocessor.raw_dir / "scholarships_raw.json"
        
        if raw_path.exists():
            count = preprocessor.process_all(raw_path, processed_path)
            if count > 0:
                preprocessor.build_embeddings(processed_path, index_path)
        else:
            logger.warning("⚠️ No raw data found. Please run scraper first or use sample data.")


if __name__ == "__main__":
    main()
