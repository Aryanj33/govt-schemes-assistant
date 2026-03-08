"""
Government Schemes CSV to JSON Converter
Converts updated_data.csv to schemes.json for the RAG system
"""
import pandas as pd
import json
import hashlib
import re
from pathlib import Path

# Paths
csv_path = Path(__file__).parent.parent.parent / "data" / "updated_data.csv"
json_path = Path(__file__).parent.parent.parent / "data" / "processed" / "schemes.json"

print(f"📖 Reading {csv_path}...")
df = pd.read_csv(csv_path, encoding='utf-8')
print(f"📊 Found {len(df)} schemes")

schemes = []

for idx, row in df.iterrows():
    try:
        name = str(row.get('scheme_name', '')).strip()
        if not name or name == 'nan':
            continue
            
        # Generate unique ID
        slug = str(row.get('slug', '')).strip()
        if not slug or slug == 'nan':
            slug = re.sub(r'[^a-z0-9]+', '-', name.lower())[:50]
        
        # Get tags as list
        tags_str = str(row.get('tags', ''))
        tags = [t.strip() for t in tags_str.split(',') if t.strip() and t.strip() != 'nan']
        
        # Get level (state/central)
        level = str(row.get('level', 'Central')).strip()
        if level == 'nan':
            level = 'Central'
        
        scheme = {
            'id': f"{slug}-{hashlib.md5(name.encode()).hexdigest()[:6]}",
            'name': name,
            'slug': slug,
            'details': str(row.get('details', '')).strip() if str(row.get('details', '')) != 'nan' else '',
            'benefits': str(row.get('benefits', '')).strip() if str(row.get('benefits', '')) != 'nan' else '',
            'eligibility': str(row.get('eligibility', '')).strip() if str(row.get('eligibility', '')) != 'nan' else '',
            'application_process': str(row.get('application', '')).strip() if str(row.get('application', '')) != 'nan' else '',
            'documents': str(row.get('documents', '')).strip() if str(row.get('documents', '')) != 'nan' else '',
            'level': level,
            'tags': tags,
            'category': tags[:3] if tags else ['Government Scheme']
        }
        
        schemes.append(scheme)
        
        if (idx + 1) % 500 == 0:
            print(f"  ✅ Processed {idx + 1} schemes...")
            
    except Exception as e:
        print(f"  ⚠️ Error processing row {idx}: {e}")
        continue

print(f"\n📝 Total schemes processed: {len(schemes)}")

# Save to JSON
print(f"💾 Saving to {json_path}...")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(schemes, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Saved {len(schemes)} government schemes to schemes.json")

# Show sample
print("\n📋 Sample scheme:")
if schemes:
    sample = schemes[0]
    print(f"  Name: {sample['name'][:60]}")
    print(f"  Level: {sample['level']}")
    print(f"  Tags: {sample['tags'][:3]}")
