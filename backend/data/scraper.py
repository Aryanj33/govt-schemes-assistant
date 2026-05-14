"""
Scholarship Scraper - Data Collection
=====================================
Scrapes scholarship data from Buddy4Study and other sources.
Note: For hackathon, we use pre-collected sample data.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger()


@dataclass
class Scholarship:
    """Scholarship data structure."""
    id: str
    name: str
    description: str
    eligibility: Dict[str, Any]
    award_amount: str
    deadline: str
    documents: List[str]
    application_link: str
    category: List[str]
    applicable_regions: str
    course_types: List[str]


class ScholarshipScraper:
    """
    Scrapes scholarship data from various Indian scholarship portals.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.scraped_data: List[Scholarship] = []
    
    def scrape_buddy4study(self, max_pages: int = 3) -> List[Dict[str, Any]]:
        """
        Scrape scholarships from Buddy4Study.
        
        Note: This is a template. Buddy4Study may have anti-scraping measures.
        For production, use their official API if available.
        
        Args:
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of scholarship dictionaries
        """
        base_url = "https://www.buddy4study.com/scholarships"
        scholarships = []
        
        logger.info(f"🔍 Starting Buddy4Study scrape (max {max_pages} pages)")
        
        for page in range(1, max_pages + 1):
            try:
                url = f"{base_url}?page={page}"
                response = self.session.get(url, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"⚠️ Page {page} returned {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Find scholarship cards (selector may change)
                cards = soup.select('.scholarship-card, .card-scholarship')
                
                for card in cards:
                    try:
                        scholarship = self._parse_buddy4study_card(card)
                        if scholarship:
                            scholarships.append(scholarship)
                    except Exception as e:
                        logger.debug(f"Failed to parse card: {e}")
                
                logger.info(f"📄 Page {page}: Found {len(cards)} cards")
                
                # Respectful delay
                time.sleep(2)
                
            except requests.RequestException as e:
                logger.error(f"❌ Request failed for page {page}: {e}")
                break
        
        logger.info(f"✅ Scraped {len(scholarships)} scholarships from Buddy4Study")
        return scholarships
    
    def _parse_buddy4study_card(self, card) -> Optional[Dict[str, Any]]:
        """Parse a single scholarship card from Buddy4Study."""
        try:
            # These selectors are examples - actual site structure may differ
            name = card.select_one('.scholarship-name, h3, .title')
            name_text = name.get_text(strip=True) if name else None
            
            if not name_text:
                return None
            
            description = card.select_one('.description, .excerpt')
            amount = card.select_one('.amount, .award')
            deadline = card.select_one('.deadline, .last-date')
            link = card.select_one('a[href*="scholarship"]')
            
            return {
                "id": name_text.lower().replace(' ', '-')[:50],
                "name": name_text,
                "description": description.get_text(strip=True) if description else "",
                "eligibility": {},  # Would need detail page scrape
                "award_amount": amount.get_text(strip=True) if amount else "Varies",
                "deadline": deadline.get_text(strip=True) if deadline else "Check website",
                "documents": [],
                "application_link": link['href'] if link else "",
                "category": [],
                "applicable_regions": "All India",
                "course_types": []
            }
            
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None
    
    def scrape_nsp(self) -> List[Dict[str, Any]]:
        """
        Scrape from National Scholarship Portal.
        
        Note: NSP uses dynamic content loading, requires Selenium for full scrape.
        This is a placeholder that would need browser automation.
        """
        logger.info("📚 NSP scraping requires browser automation - using cached data")
        return []
    
    def save_to_json(self, scholarships: List[Dict[str, Any]], output_path: Path):
        """
        Save scraped scholarships to JSON file.
        
        Args:
            scholarships: List of scholarship dictionaries
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scholarships, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(scholarships)} scholarships to {output_path}")
    
    def load_existing(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load existing scholarships from JSON."""
        if not json_path.exists():
            return []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """Main scraping function."""
    scraper = ScholarshipScraper()
    
    # Check existing data
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "scholarships.json"
    
    if data_path.exists():
        existing = scraper.load_existing(data_path)
        logger.info(f"📚 Found {len(existing)} existing scholarships")
        
        # For hackathon, we use the curated sample data
        logger.info("✅ Using curated scholarship data for demo")
        return
    
    # If no data exists, attempt to scrape
    logger.warning("⚠️ No existing data found. Attempting scrape...")
    
    # Note: Live scraping may fail due to anti-bot measures
    # The project includes pre-curated data in scholarships.json
    
    try:
        scholarships = scraper.scrape_buddy4study(max_pages=2)
        
        if scholarships:
            scraper.save_to_json(scholarships, data_path)
        else:
            logger.warning("⚠️ No scholarships scraped. Please use the included sample data.")
            
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        logger.info("💡 Tip: The project includes curated data in data/processed/scholarships.json")


if __name__ == "__main__":
    main()
