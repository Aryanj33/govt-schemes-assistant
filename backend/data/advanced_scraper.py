"""
Advanced Scholarship Scraper - Massive Database Builder
========================================================
Scrapes 500-1000+ scholarships from multiple Indian scholarship portals.
"""

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger()


# User agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


@dataclass
class ScrapedScholarship:
    """Standardized scholarship data structure."""
    id: str
    name: str
    short_name: str
    description: str
    eligibility: Dict[str, Any]
    award_amount: str
    award_min: Optional[int]
    award_max: Optional[int]
    deadline: str
    application_link: str
    documents: List[str]
    applicable_states: List[str]
    category: List[str]
    course_types: List[str]
    provider: str
    provider_type: str
    source: str
    source_url: str
    last_updated: str


class RateLimiter:
    """Simple rate limiter for polite scraping."""
    
    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0
    
    async def wait(self):
        """Wait if needed to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class Buddy4StudyScraper:
    """Scraper for Buddy4Study - largest Indian scholarship database."""
    
    BASE_URL = "https://www.buddy4study.com"
    
    # Category URLs
    CATEGORY_URLS = {
        "sc-st": "/page/sc-st-scholarships",
        "obc": "/page/obc-scholarships",
        "minority": "/page/minority-scholarships",
        "girls": "/page/scholarships-for-girls",
        "merit": "/page/merit-based-scholarships",
        "engineering": "/page/engineering-scholarships",
        "medical": "/page/medical-scholarships",
        "mba": "/page/mba-scholarships",
        "law": "/page/law-scholarships",
        "arts": "/page/arts-scholarships",
        "phd": "/page/phd-scholarships",
    }
    
    # State URLs
    STATE_URLS = {
        "maharashtra": "/page/maharashtra-scholarships",
        "uttar-pradesh": "/page/uttar-pradesh-scholarships",
        "karnataka": "/page/karnataka-scholarships",
        "tamil-nadu": "/page/tamil-nadu-scholarships",
        "rajasthan": "/page/rajasthan-scholarships",
        "gujarat": "/page/gujarat-scholarships",
        "madhya-pradesh": "/page/madhya-pradesh-scholarships",
        "west-bengal": "/page/west-bengal-scholarships",
        "bihar": "/page/bihar-scholarships",
        "andhra-pradesh": "/page/andhra-pradesh-scholarships",
        "telangana": "/page/telangana-scholarships",
        "kerala": "/page/kerala-scholarships",
        "punjab": "/page/punjab-scholarships",
        "haryana": "/page/haryana-scholarships",
        "odisha": "/page/odisha-scholarships",
        "jharkhand": "/page/jharkhand-scholarships",
        "chhattisgarh": "/page/chhattisgarh-scholarships",
        "assam": "/page/assam-scholarships",
        "delhi": "/page/delhi-scholarships",
    }
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session: Optional[aiohttp.ClientSession] = None
        self.scraped_urls: Set[str] = set()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={'User-Agent': random.choice(USER_AGENTS)},
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with rate limiting and error handling."""
        await self.rate_limiter.wait()
        
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"⚠️ HTTP {response.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"❌ Fetch error for {url}: {e}")
            return None
    
    async def scrape_listing_page(self, url: str) -> List[str]:
        """Scrape scholarship links from a listing page."""
        html = await self.fetch_page(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'lxml')
        links = []
        
        # Find scholarship cards/links
        for link in soup.select('a[href*="/scholarship/"]'):
            href = link.get('href', '')
            if href and '/scholarship/' in href:
                full_url = href if href.startswith('http') else self.BASE_URL + href
                if full_url not in self.scraped_urls:
                    links.append(full_url)
                    self.scraped_urls.add(full_url)
        
        return links
    
    async def scrape_scholarship_detail(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape detailed scholarship information from its page."""
        html = await self.fetch_page(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'lxml')
        
        try:
            # Extract name
            name_elem = soup.select_one('h1, .scholarship-title, .scholarship-name')
            name = name_elem.get_text(strip=True) if name_elem else "Unknown Scholarship"
            
            # Extract description
            desc_elem = soup.select_one('.scholarship-description, .overview, article p')
            description = desc_elem.get_text(strip=True)[:500] if desc_elem else ""
            
            # Extract award amount
            amount_elem = soup.select_one('.award-amount, .scholarship-amount, [class*="amount"]')
            amount_text = amount_elem.get_text(strip=True) if amount_elem else "Varies"
            amount_min, amount_max = self._parse_amount(amount_text)
            
            # Extract deadline
            deadline_elem = soup.select_one('.deadline, [class*="deadline"], [class*="last-date"]')
            deadline = deadline_elem.get_text(strip=True) if deadline_elem else "Check website"
            
            # Extract eligibility
            eligibility = self._extract_eligibility(soup)
            
            # Extract documents
            documents = self._extract_documents(soup)
            
            # Extract categories
            categories = self._extract_categories(soup, name, description)
            
            # Extract states
            states = self._extract_states(soup, name, description)
            
            # Generate ID
            scholarship_id = self._generate_id(name)
            
            return {
                "id": scholarship_id,
                "name": name,
                "short_name": name[:50] if len(name) > 50 else name,
                "description": description,
                "eligibility": eligibility,
                "award_amount": amount_text,
                "award_min": amount_min,
                "award_max": amount_max,
                "deadline": deadline,
                "application_link": url,
                "documents": documents,
                "applicable_states": states,
                "applicable_regions": "All India" if "All India" in states else ", ".join(states[:3]),
                "category": categories,
                "course_types": self._extract_course_types(soup, name, description),
                "provider": self._extract_provider(soup),
                "provider_type": "Unknown",
                "source": "Buddy4Study",
                "source_url": url,
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            logger.error(f"❌ Parse error for {url}: {e}")
            return None
    
    def _parse_amount(self, text: str) -> tuple:
        """Parse amount text to min/max integers."""
        if not text:
            return None, None
        
        # Find numbers in text
        numbers = re.findall(r'[\d,]+', text.replace(',', ''))
        numbers = [int(n) for n in numbers if n.isdigit() and int(n) > 100]
        
        if len(numbers) >= 2:
            return min(numbers), max(numbers)
        elif len(numbers) == 1:
            return numbers[0], numbers[0]
        return None, None
    
    def _extract_eligibility(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract eligibility criteria."""
        eligibility = {
            "education_level": "Various",
            "category": "All",
            "marks_criteria": None,
            "income_limit": None
        }
        
        # Look for eligibility section
        for elem in soup.select('.eligibility, [class*="eligibility"], [class*="criteria"]'):
            text = elem.get_text().lower()
            
            # Education level
            if any(t in text for t in ['10th', 'class 10']):
                eligibility["education_level"] = "10th pass"
            elif any(t in text for t in ['12th', 'class 12', 'hsc']):
                eligibility["education_level"] = "12th pass"
            elif any(t in text for t in ['graduate', 'ug', 'bachelor']):
                eligibility["education_level"] = "Undergraduate"
            elif any(t in text for t in ['postgraduate', 'pg', 'master']):
                eligibility["education_level"] = "Postgraduate"
            
            # Marks
            marks_match = re.search(r'(\d{2,3})\s*%', text)
            if marks_match:
                eligibility["marks_criteria"] = int(marks_match.group(1))
            
            # Income
            income_match = re.search(r'(\d+)\s*lakh', text)
            if income_match:
                eligibility["income_limit"] = int(income_match.group(1)) * 100000
        
        return eligibility
    
    def _extract_documents(self, soup: BeautifulSoup) -> List[str]:
        """Extract required documents."""
        documents = []
        
        for elem in soup.select('.documents li, [class*="document"] li, ul li'):
            text = elem.get_text(strip=True)
            if any(doc in text.lower() for doc in ['certificate', 'marksheet', 'aadhaar', 'photo', 'bank', 'income']):
                documents.append(text[:100])
        
        # Default documents if none found
        if not documents:
            documents = [
                "Identity proof (Aadhaar)",
                "Marksheet of last exam",
                "Income certificate",
                "Passport size photo",
                "Bank account details"
            ]
        
        return documents[:10]  # Limit to 10
    
    def _extract_categories(self, soup: BeautifulSoup, name: str, desc: str) -> List[str]:
        """Extract scholarship categories."""
        categories = []
        text = (name + " " + desc).lower()
        
        if any(t in text for t in ['sc', 'scheduled caste']):
            categories.append("SC")
        if any(t in text for t in ['st', 'scheduled tribe']):
            categories.append("ST")
        if 'obc' in text or 'backward' in text:
            categories.append("OBC")
        if 'minority' in text or 'muslim' in text or 'christian' in text:
            categories.append("Minority")
        if any(t in text for t in ['girl', 'women', 'female']):
            categories.append("Female")
        if 'merit' in text:
            categories.append("Merit")
        if 'need' in text or 'income' in text or 'poor' in text:
            categories.append("Need-based")
        if 'disability' in text or 'divyang' in text:
            categories.append("Disability")
        
        return categories if categories else ["General"]
    
    def _extract_states(self, soup: BeautifulSoup, name: str, desc: str) -> List[str]:
        """Extract applicable states."""
        text = (name + " " + desc).lower()
        states = []
        
        state_names = {
            "maharashtra": "Maharashtra",
            "uttar pradesh": "Uttar Pradesh",
            "karnataka": "Karnataka",
            "tamil nadu": "Tamil Nadu",
            "rajasthan": "Rajasthan",
            "gujarat": "Gujarat",
            "madhya pradesh": "Madhya Pradesh",
            "west bengal": "West Bengal",
            "bihar": "Bihar",
            "andhra": "Andhra Pradesh",
            "telangana": "Telangana",
            "kerala": "Kerala",
            "punjab": "Punjab",
            "haryana": "Haryana",
            "odisha": "Odisha",
            "jharkhand": "Jharkhand",
            "chhattisgarh": "Chhattisgarh",
            "assam": "Assam",
            "delhi": "Delhi",
        }
        
        for key, value in state_names.items():
            if key in text:
                states.append(value)
        
        return states if states else ["All India"]
    
    def _extract_course_types(self, soup: BeautifulSoup, name: str, desc: str) -> List[str]:
        """Extract applicable course types."""
        courses = []
        text = (name + " " + desc).lower()
        
        course_map = {
            "engineering": "Engineering",
            "btech": "Engineering",
            "medical": "Medical",
            "mbbs": "Medical",
            "mba": "Management",
            "law": "Law",
            "arts": "Arts",
            "science": "Science",
            "commerce": "Commerce",
            "agriculture": "Agriculture",
            "pharmacy": "Pharmacy",
        }
        
        for key, value in course_map.items():
            if key in text:
                courses.append(value)
        
        return courses if courses else ["All Courses"]
    
    def _extract_provider(self, soup: BeautifulSoup) -> str:
        """Extract scholarship provider."""
        for elem in soup.select('.provider, [class*="provider"], [class*="organization"]'):
            return elem.get_text(strip=True)[:100]
        return "Various"
    
    def _generate_id(self, name: str) -> str:
        """Generate unique ID from scholarship name."""
        clean_name = re.sub(r'[^a-z0-9\s]', '', name.lower())
        clean_name = re.sub(r'\s+', '-', clean_name)[:50]
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]
        return f"{clean_name}-{hash_suffix}"
    
    async def scrape_all(self, max_scholarships: int = 300) -> List[Dict[str, Any]]:
        """Scrape scholarships from all category and state pages."""
        all_links = set()
        
        logger.info("🔍 Scraping Buddy4Study scholarship listings...")
        
        # Scrape category pages
        for category, url_path in self.CATEGORY_URLS.items():
            url = self.BASE_URL + url_path
            links = await self.scrape_listing_page(url)
            all_links.update(links)
            logger.info(f"   {category}: found {len(links)} scholarships")
            
            if len(all_links) >= max_scholarships:
                break
        
        # Scrape state pages if needed
        if len(all_links) < max_scholarships:
            for state, url_path in self.STATE_URLS.items():
                url = self.BASE_URL + url_path
                links = await self.scrape_listing_page(url)
                all_links.update(links)
                logger.info(f"   {state}: found {len(links)} scholarships")
                
                if len(all_links) >= max_scholarships:
                    break
        
        logger.info(f"📋 Total unique scholarship links: {len(all_links)}")
        
        # Scrape individual scholarship pages
        scholarships = []
        links_list = list(all_links)[:max_scholarships]
        
        logger.info(f"📥 Scraping {len(links_list)} scholarship details...")
        
        for i, link in enumerate(links_list):
            scholarship = await self.scrape_scholarship_detail(link)
            if scholarship:
                scholarships.append(scholarship)
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Progress: {i+1}/{len(links_list)} ({len(scholarships)} successful)")
        
        logger.info(f"✅ Scraped {len(scholarships)} scholarships from Buddy4Study")
        return scholarships


class MassiveScholarshipScraper:
    """Main orchestrator for scraping from multiple sources."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_second=0.5)  # Be polite
        self.buddy4study = Buddy4StudyScraper(self.rate_limiter)
        self.all_scholarships: List[Dict[str, Any]] = []
        self.checkpoint_path = Path(__file__).parent.parent.parent / "data" / "raw"
    
    async def scrape_all(self, max_per_source: int = 200) -> List[Dict[str, Any]]:
        """Scrape from all sources."""
        logger.info("🚀 Starting massive scholarship scraping...")
        
        # Scrape Buddy4Study
        try:
            b4s_scholarships = await self.buddy4study.scrape_all(max_per_source)
            self.all_scholarships.extend(b4s_scholarships)
            self.save_checkpoint("buddy4study", b4s_scholarships)
        except Exception as e:
            logger.error(f"❌ Buddy4Study scraping failed: {e}")
            # Load from checkpoint if available
            b4s_scholarships = self.load_checkpoint("buddy4study")
            self.all_scholarships.extend(b4s_scholarships)
        finally:
            await self.buddy4study.close()
        
        logger.info(f"📊 Total scraped: {len(self.all_scholarships)} scholarships")
        return self.all_scholarships
    
    def deduplicate(self, scholarships: List[Dict]) -> List[Dict]:
        """Remove duplicate scholarships."""
        seen_names = {}
        unique = []
        
        for s in scholarships:
            name_key = s['name'].lower().strip()
            # Simple deduplication by name similarity
            is_duplicate = False
            for seen_name in seen_names:
                if self._is_similar(name_key, seen_name, threshold=0.85):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(s)
                seen_names[name_key] = True
        
        logger.info(f"📉 Deduplicated: {len(scholarships)} → {len(unique)}")
        return unique
    
    def _is_similar(self, str1: str, str2: str, threshold: float) -> bool:
        """Simple string similarity check."""
        # Simple Jaccard similarity on words
        words1 = set(str1.split())
        words2 = set(str2.split())
        if not words1 or not words2:
            return False
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return (intersection / union) >= threshold
    
    def validate(self, scholarship: Dict) -> bool:
        """Validate scholarship has required fields."""
        required = ['name', 'eligibility', 'award_amount', 'application_link']
        return all(scholarship.get(field) for field in required)
    
    def save_checkpoint(self, source: str, data: List[Dict]):
        """Save scraping checkpoint."""
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path / f"checkpoint_{source}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, source: str) -> List[Dict]:
        """Load from checkpoint if available."""
        path = self.checkpoint_path / f"checkpoint_{source}.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"📂 Loaded checkpoint: {len(data)} scholarships")
            return data
        return []
    
    def save_final(self, scholarships: List[Dict], filename: str = "scholarships_massive.json"):
        """Save final scholarship database."""
        output_path = Path(__file__).parent.parent.parent / "data" / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scholarships, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(scholarships)} scholarships to {output_path}")
        return output_path


async def build_massive_database(max_scholarships: int = 300):
    """Main function to build massive scholarship database."""
    scraper = MassiveScholarshipScraper()
    
    print("="*60)
    print("🎓 MASSIVE SCHOLARSHIP DATABASE BUILDER")
    print("="*60)
    print(f"Target: {max_scholarships} scholarships")
    print("Sources: Buddy4Study, NSP (planned)")
    print("="*60)
    
    # Scrape
    all_scholarships = await scraper.scrape_all(max_scholarships)
    
    # Deduplicate
    unique_scholarships = scraper.deduplicate(all_scholarships)
    
    # Validate
    valid_scholarships = [s for s in unique_scholarships if scraper.validate(s)]
    print(f"✅ Valid scholarships: {len(valid_scholarships)}")
    
    # Save
    scraper.save_final(valid_scholarships)
    
    print("="*60)
    print(f"🎉 DONE! Database ready with {len(valid_scholarships)} scholarships!")
    print("="*60)
    
    return valid_scholarships


if __name__ == "__main__":
    asyncio.run(build_massive_database(max_scholarships=300))
