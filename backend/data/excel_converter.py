"""
Excel to JSON Converter for Scholarship Dataset
"""
import pandas as pd
import json
import hashlib
import re
from pathlib import Path

# Read Excel
excel_path = Path(__file__).parent.parent.parent / "data" / "dataset_combined.xlsx"
json_path = Path(__file__).parent.parent.parent / "data" / "processed" / "scholarships.json"

print(f"Reading {excel_path}...")
df = pd.read_excel(excel_path)
print(f"Processing {len(df)} rows...")

# Group by scholarship name and aggregate eligibility
scholarships = []

for name in df['Name'].unique():
    subset = df[df['Name'] == name]
    eligible_rows = subset[subset['Outcome'] == 1]
    
    # Extract unique values for each criterion
    education = eligible_rows['Education Qualification'].unique().tolist()
    genders = eligible_rows['Gender'].unique().tolist()
    communities = eligible_rows['Community'].unique().tolist()
    religions = eligible_rows['Religion'].unique().tolist()
    income_levels = eligible_rows['Income'].unique().tolist()
    percentages = eligible_rows['Annual-Percentage'].unique().tolist()
    
    # Clean name
    clean_name = str(name).replace('?', '-').replace('  ', ' ').strip()
    
    # Generate ID
    name_slug = re.sub(r'[^a-z0-9]+', '-', clean_name.lower())[:40]
    hash_suffix = hashlib.md5(clean_name.encode()).hexdigest()[:6]
    
    # Determine categories
    cats = [str(c) for c in communities if str(c) not in ['General', 'nan']]
    
    scholarship = {
        'id': f'{name_slug}-{hash_suffix}',
        'name': clean_name,
        'description': f'Scholarship for eligible students. Eligibility based on education, community, and income criteria.',
        'eligibility': {
            'education_level': ', '.join([str(e) for e in education[:3]]),
            'category': ', '.join(cats[:3]) if cats else 'All',
            'gender': 'All' if len(genders) > 1 else str(genders[0]) if genders else 'All',
            'income_limit': ', '.join([str(i) for i in income_levels[:3]]),
            'marks_criteria': ', '.join([str(p) for p in percentages[:3]])
        },
        'award_amount': 'Varies - Check official website',
        'deadline': 'Check official website',
        'documents': ['Marksheet', 'Income certificate', 'Category certificate', 'Aadhaar'],
        'application_link': 'https://scholarships.gov.in/',
        'category': cats if cats else ['Merit'],
        'applicable_regions': 'All India',
        'course_types': [str(e) for e in education[:3]]
    }
    
    scholarships.append(scholarship)
    print(f'  + {clean_name[:60]}')

# Load existing scholarships
print(f"\nLoading existing scholarships from {json_path}...")
with open(json_path, 'r', encoding='utf-8') as f:
    existing = json.load(f)

print(f'Existing scholarships: {len(existing)}')
print(f'New from Excel: {len(scholarships)}')

# Merge (avoid duplicates by name)
existing_names = [s['name'].lower() for s in existing]
added = 0
for new_s in scholarships:
    if new_s['name'].lower() not in existing_names:
        existing.append(new_s)
        print(f'  + Added: {new_s["name"][:50]}')
        added += 1

print(f'\nAdded {added} new scholarships')
print(f'Total combined: {len(existing)}')

# Save merged
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

print(f'\n✅ Saved to {json_path}')
