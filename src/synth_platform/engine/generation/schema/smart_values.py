"""
LLM-powered smart value generation for context-aware data.

This module generates realistic domain-specific values by:
1. Detecting semantic domain from column/table names
2. Using LLM to generate domain-appropriate data pools
3. Caching pools for fast repeated generation
"""

import json
import os
import hashlib
import string
from typing import Dict, List, Optional, Any
from pathlib import Path


def _rng_int(rng: Any, low: int, high: int) -> int:
    """Return an integer in [low, high) from numpy/random-compatible RNGs."""
    if rng is not None and hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    if rng is not None and hasattr(rng, "randint"):
        return int(rng.randint(low, high - 1))
    import random
    return random.randrange(low, high)


def _rng_choice(rng: Any, values: List[Any]) -> Any:
    if rng is not None and hasattr(rng, "choice"):
        return rng.choice(values)
    import random
    return random.choice(values)


def luhn_check_digit(number_without_check: str) -> str:
    """Compute the Luhn check digit for a numeric string."""
    digits = [int(ch) for ch in str(number_without_check) if ch.isdigit()]
    total = 0
    parity = (len(digits) + 1) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return str((10 - (total % 10)) % 10)


def is_luhn_valid(value: str) -> bool:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return bool(digits) and luhn_check_digit(digits[:-1]) == digits[-1]


def generate_luhn_number(length: int = 16, rng: Any = None, prefix: str = "4") -> str:
    """Generate a synthetic Luhn-valid number, useful for card-like IDs."""
    length = max(int(length), len(prefix) + 1)
    body_len = length - len(prefix) - 1
    body = "".join(str(_rng_int(rng, 0, 10)) for _ in range(body_len))
    base = f"{prefix}{body}"
    return base + luhn_check_digit(base)


def aba_routing_check_digit(first_eight_digits: str) -> str:
    """Compute the ABA routing check digit from the first eight digits."""
    digits = [int(ch) for ch in first_eight_digits]
    if len(digits) != 8:
        raise ValueError("ABA routing number check digit requires exactly 8 digits")
    weights = [3, 7, 1, 3, 7, 1, 3, 7]
    total = sum(d * w for d, w in zip(digits, weights))
    return str((10 - (total % 10)) % 10)


def is_aba_routing_number(value: str) -> bool:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) != 9:
        return False
    prefix = int(digits[:2])
    valid_prefix = 1 <= prefix <= 12 or 21 <= prefix <= 32 or 61 <= prefix <= 72 or prefix == 80
    return valid_prefix and aba_routing_check_digit(digits[:8]) == digits[-1]


def generate_routing_number(rng: Any = None) -> str:
    """Generate a synthetic ABA routing number with a valid checksum."""
    prefixes = [f"{i:02d}" for i in range(1, 13)] + [f"{i:02d}" for i in range(21, 33)] + [f"{i:02d}" for i in range(61, 73)] + ["80"]
    first_eight = str(_rng_choice(rng, prefixes)) + "".join(str(_rng_int(rng, 0, 10)) for _ in range(6))
    return first_eight + aba_routing_check_digit(first_eight)


def generate_account_number(digits: int = 12, rng: Any = None, prefix: Optional[str] = None) -> str:
    """Generate a synthetic bank account number."""
    digits = max(6, min(int(digits), 18))
    prefix = "" if prefix is None else str(prefix)
    body_len = max(0, digits - len(prefix))
    first = str(_rng_int(rng, 1, 10)) if not prefix and body_len else ""
    if first:
        body_len -= 1
    body = "".join(str(_rng_int(rng, 0, 10)) for _ in range(body_len))
    return (prefix + first + body)[:digits]


def generate_ssn(rng: Any = None, hyphenated: bool = True) -> str:
    """Generate a synthetic US SSN-shaped value avoiding invalid area/group/serial ranges."""
    area = _rng_int(rng, 1, 900)
    while area == 666:
        area = _rng_int(rng, 1, 900)
    group = _rng_int(rng, 1, 100)
    serial = _rng_int(rng, 1, 10000)
    if hyphenated:
        return f"{area:03d}-{group:02d}-{serial:04d}"
    return f"{area:03d}{group:02d}{serial:04d}"


_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6], [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4], [9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2], [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8],
]
_VERHOEFF_INV = [0,4,3,2,1,5,6,7,8,9]


def verhoeff_check_digit(number_without_check: str) -> str:
    c = 0
    for i, item in enumerate(reversed(str(number_without_check))):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(item)]]
    return str(_VERHOEFF_INV[c])


def is_verhoeff_valid(value: str) -> bool:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return False
    c = 0
    for i, item in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(item)]]
    return c == 0


def generate_aadhaar(rng: Any = None, spaced: bool = True) -> str:
    """Generate a synthetic 12-digit Aadhaar-shaped value with Verhoeff checksum."""
    base = str(_rng_int(rng, 2, 10)) + "".join(str(_rng_int(rng, 0, 10)) for _ in range(10))
    value = base + verhoeff_check_digit(base)
    if spaced:
        return f"{value[:4]} {value[4:8]} {value[8:]}"
    return value


def generate_ifsc_code(rng: Any = None) -> str:
    bank = "".join(str(_rng_choice(rng, list(string.ascii_uppercase))) for _ in range(4))
    branch = "".join(str(_rng_choice(rng, list(string.ascii_uppercase + string.digits))) for _ in range(6))
    return f"{bank}0{branch}"


def generate_swift_bic(rng: Any = None, country: str = "US", length: int = 11) -> str:
    letters = list(string.ascii_uppercase)
    alnum = list(string.ascii_uppercase + string.digits)
    bank = "".join(str(_rng_choice(rng, letters)) for _ in range(4))
    country = (country or "US").upper()[:2].ljust(2, "X")
    location = "".join(str(_rng_choice(rng, alnum)) for _ in range(2))
    branch = "".join(str(_rng_choice(rng, alnum)) for _ in range(3))
    return (bank + country + location + branch)[: 8 if int(length) == 8 else 11]


def generate_iban(rng: Any = None, country: str = "GB") -> str:
    """Generate a synthetic IBAN-shaped value. The check digits are placeholders."""
    country = (country or "GB").upper()[:2].ljust(2, "X")
    bank = "".join(str(_rng_choice(rng, list(string.ascii_uppercase))) for _ in range(4))
    acct = generate_account_number(14, rng=rng)
    return f"{country}00{bank}{acct}"


def banking_pool(domain: str, size: int = 50, rng: Any = None) -> List[str]:
    """Generate deterministic synthetic banking primitive pools."""
    generators = {
        "routing_number": lambda: generate_routing_number(rng),
        "account_number": lambda: generate_account_number(12, rng),
        "bank_account": lambda: generate_account_number(12, rng),
        "ssn": lambda: generate_ssn(rng),
        "aadhaar": lambda: generate_aadhaar(rng),
        "ifsc_code": lambda: generate_ifsc_code(rng),
        "swift_bic": lambda: generate_swift_bic(rng),
        "iban": lambda: generate_iban(rng),
        "credit_card": lambda: generate_luhn_number(16, rng),
    }
    fn = generators.get(domain)
    if not fn:
        return []
    seen = set()
    out: List[str] = []
    attempts = 0
    while len(out) < size and attempts < size * 10:
        value = fn()
        if value not in seen:
            seen.add(value)
            out.append(value)
        attempts += 1
    return out


class SmartValueGenerator:
    """
    Generate context-aware realistic values using LLM.
    
    Detects domains from column/table names and generates
    appropriate data pools using LLM or curated fallbacks.
    """
    
    # Domain detection patterns
    DOMAIN_PATTERNS = {
        # Medical
        "disease": ["disease", "diagnosis", "condition", "illness", "ailment", "disorder", "pathology"],
        "prescription": ["prescription", "medication", "drug", "medicine", "rx", "pharma"],
        "procedure": ["procedure", "surgery", "treatment", "operation", "therapy", "intervention"],
        "symptom": ["symptom", "complaint", "sign", "manifestation"],
        "medical_specialty": ["specialty", "department", "ward"],
        
        # Legal
        "case_type": ["case_type", "legal_matter", "litigation_type"],
        "law_firm": ["law_firm", "legal_firm", "attorney_firm"],
        "legal_status": ["legal_status", "case_status", "court_status"],
        
        # Retail/E-commerce  
        "product": ["product", "item", "merchandise", "goods"],
        "category": ["category", "department", "section"],
        "brand": ["brand", "manufacturer", "vendor"],
        
        # Finance
        "transaction_type": ["transaction_type", "payment_type", "transfer_type"],
        "account_type": ["account_type", "bank_account", "financial_account"],
        "routing_number": ["routing_number", "routing", "aba", "aba_routing"],
        "account_number": ["account_number", "acct_number", "bank_account_number"],
        "bank_account": ["bank_account", "account_id", "deposit_account"],
        "ssn": ["ssn", "social_security_number", "social security"],
        "aadhaar": ["aadhaar", "aadhar", "aadhaar_number"],
        "ifsc_code": ["ifsc", "ifsc_code"],
        "swift_bic": ["swift", "bic", "swift_bic"],
        "iban": ["iban"],
        "credit_card": ["credit_card", "card_number", "pan"],
        
        # HR/Employment
        "job_title": ["job_title", "position", "role", "designation"],
        "department": ["department", "division", "unit", "team"],
        "skill": ["skill", "competency", "expertise", "qualification"],
        
        # NEW: Food & Restaurant
        "restaurant_name": ["restaurant", "diner", "cafe", "eatery", "bistro", "tavern"],
        "cuisine_type": ["cuisine", "food_type", "culinary"],
        "menu_item": ["menu_item", "dish", "meal", "entree", "appetizer"],
        
        # NEW: Education
        "course_name": ["course", "class", "lecture", "module", "subject"],
        "university": ["university", "college", "institution", "school"],
        "degree": ["degree", "certification", "diploma", "qualification"],
        
        # NEW: Events & Meetings
        "event_name": ["event", "conference", "meeting", "workshop", "seminar", "webinar"],
        "venue": ["venue", "location", "hall", "auditorium", "center"],
        
        # NEW: Projects & Work
        "project_name": ["project", "initiative", "campaign", "program"],
        "task_name": ["task", "todo", "action_item", "work_item"],
        "milestone": ["milestone", "deliverable", "goal", "objective"],
        
        # NEW: Reviews & Feedback
        "review_title": ["review_title", "feedback_title", "comment_title"],
        "review_text": ["review", "feedback", "comment", "testimonial", "opinion"],
        
        # NEW: Location
        "city": ["city", "town", "municipality", "metro"],
        "country": ["country", "nation", "region"],
        "address": ["address", "location", "street", "postal"],
        
        # NEW: Business
        "company_name": ["company", "organization", "business", "corporation", "enterprise", "firm"],
        "industry": ["industry", "sector", "vertical", "market"],
        
        # NEW: Tech/Software
        "feature_name": ["feature", "capability", "functionality"],
        "bug_type": ["bug", "issue", "defect", "error"],
        "api_endpoint": ["endpoint", "api", "route", "path"],
        
        # NEW v0.5.0: Additional domain patterns
        "payment_method": ["payment_method", "pay_type", "payment_option"],
        "order_status": ["order_status", "status", "state"],
        "customer_segment": ["segment", "customer_type", "tier", "classification"],
        "license_type": ["license", "licence"],
        "file_type": ["file_type", "document_type", "mime_type"],
        "priority_level": ["priority", "urgency", "importance"],
        "subscription_plan": ["plan", "subscription", "tier", "package"],
        
        # Generic patterns - lowest priority but always match on exact column names
        "name": ["name"],
        "description": ["description", "desc", "about", "summary", "details"],
        "title": ["title", "heading"],
        "status": ["status", "state"],
        "type": ["type", "kind", "category"],
    }
    
    # Curated fallback pools (no LLM needed)
    FALLBACK_POOLS = {
        "disease": [
            "Type 2 Diabetes Mellitus", "Essential Hypertension", "Chronic Obstructive Pulmonary Disease",
            "Major Depressive Disorder", "Generalized Anxiety Disorder", "Acute Myocardial Infarction",
            "Atrial Fibrillation", "Chronic Kidney Disease Stage 3", "Rheumatoid Arthritis",
            "Osteoarthritis", "Migraine without Aura", "Asthma", "Hypothyroidism", "Hyperlipidemia",
            "Gastroesophageal Reflux Disease", "Irritable Bowel Syndrome", "Obesity", "Sleep Apnea",
            "Chronic Lower Back Pain", "Urinary Tract Infection", "Pneumonia", "Bronchitis",
            "Anemia", "Osteoporosis", "Fibromyalgia", "Seizure Disorder", "Glaucoma",
            "Allergic Rhinitis", "Eczema", "Psoriasis", "Hepatitis C", "Cirrhosis",
            "Congestive Heart Failure", "Coronary Artery Disease", "Peripheral Artery Disease",
            "Deep Vein Thrombosis", "Pulmonary Embolism", "Stroke", "Transient Ischemic Attack",
            "Multiple Sclerosis", "Parkinson's Disease", "Alzheimer's Disease", "Epilepsy",
            "Lupus", "Crohn's Disease", "Ulcerative Colitis", "Celiac Disease",
            "Polycystic Ovary Syndrome", "Endometriosis", "Benign Prostatic Hyperplasia",
        ],
        "prescription": [
            "Metformin 500mg - Take twice daily with meals",
            "Lisinopril 10mg - Take once daily",
            "Atorvastatin 20mg - Take at bedtime",
            "Levothyroxine 50mcg - Take on empty stomach",
            "Amlodipine 5mg - Take once daily",
            "Metoprolol 25mg - Take twice daily",
            "Omeprazole 20mg - Take before breakfast",
            "Sertraline 50mg - Take once daily",
            "Gabapentin 300mg - Take three times daily",
            "Tramadol 50mg - Take as needed for pain",
            "Prednisone 10mg - Take with food",
            "Albuterol Inhaler - Use as needed for breathing",
            "Fluticasone Nasal Spray - Use twice daily",
            "Insulin Glargine 20 units - Inject at bedtime",
            "Warfarin 5mg - Take as directed with INR monitoring",
            "Clopidogrel 75mg - Take once daily",
            "Furosemide 40mg - Take in the morning",
            "Losartan 50mg - Take once daily",
            "Hydrochlorothiazide 25mg - Take in the morning",
            "Duloxetine 60mg - Take once daily",
            "Escitalopram 10mg - Take once daily",
            "Alprazolam 0.5mg - Take as needed for anxiety",
            "Zolpidem 10mg - Take at bedtime as needed",
            "Simvastatin 40mg - Take at bedtime",
            "Pantoprazole 40mg - Take before breakfast",
        ],
        "procedure": [
            "Complete Blood Count", "Comprehensive Metabolic Panel", "Lipid Panel",
            "Chest X-Ray", "CT Scan - Abdomen", "MRI - Brain", "Echocardiogram",
            "Colonoscopy", "Upper Endoscopy", "Cardiac Catheterization",
            "Knee Arthroscopy", "Laparoscopic Cholecystectomy", "Appendectomy",
            "Total Hip Replacement", "Total Knee Replacement", "Coronary Artery Bypass",
            "Angioplasty with Stent Placement", "Pacemaker Implantation",
            "Lumbar Puncture", "Bone Marrow Biopsy", "Bronchoscopy",
            "Thyroidectomy", "Mastectomy", "Prostatectomy", "Hysterectomy",
            "Cataract Surgery", "LASIK Eye Surgery", "Tonsillectomy",
            "Cesarean Section", "Spinal Fusion", "Hernia Repair",
        ],
        "symptom": [
            "Chest pain", "Shortness of breath", "Fatigue", "Headache",
            "Dizziness", "Nausea", "Vomiting", "Abdominal pain", "Back pain",
            "Joint pain", "Muscle weakness", "Numbness", "Tingling sensation",
            "Blurred vision", "Hearing loss", "Cough", "Fever", "Chills",
            "Night sweats", "Weight loss", "Weight gain", "Loss of appetite",
            "Insomnia", "Excessive thirst", "Frequent urination", "Swelling",
            "Rash", "Itching", "Bruising", "Bleeding", "Difficulty swallowing",
            "Heartburn", "Constipation", "Diarrhea", "Blood in stool",
            "Difficulty concentrating", "Memory problems", "Anxiety", "Depression",
            "Palpitations", "Leg cramps", "Cold intolerance", "Heat intolerance",
        ],
        "job_title": [
            "Software Engineer", "Senior Software Engineer", "Staff Engineer",
            "Product Manager", "Senior Product Manager", "Director of Product",
            "Data Scientist", "Machine Learning Engineer", "Data Analyst",
            "UX Designer", "UI Designer", "Product Designer",
            "DevOps Engineer", "Site Reliability Engineer", "Platform Engineer",
            "Engineering Manager", "VP of Engineering", "CTO",
            "Sales Representative", "Account Executive", "Sales Manager",
            "Marketing Manager", "Content Strategist", "Growth Manager",
            "HR Manager", "Recruiter", "People Operations",
            "Financial Analyst", "Controller", "CFO",
            "Customer Success Manager", "Support Engineer", "Technical Writer",
        ],
        "department": [
            "Engineering", "Product", "Design", "Marketing", "Sales",
            "Human Resources", "Finance", "Operations", "Customer Success",
            "Research & Development", "Legal", "IT", "Security",
            "Quality Assurance", "Business Development", "Analytics",
            "Supply Chain", "Manufacturing", "Procurement",
        ],
        "product": [
            "Wireless Bluetooth Headphones", "Mechanical Gaming Keyboard",
            "Ultra HD 4K Monitor", "Ergonomic Office Chair",
            "Portable Power Bank 20000mAh", "Smart Home Speaker",
            "Fitness Tracking Smartwatch", "Noise Cancelling Earbuds",
            "USB-C Docking Station", "Laptop Cooling Pad",
            "Wireless Mouse", "Gaming Mouse Pad XL",
            "Webcam 1080p HD", "Ring Light with Tripod",
            "Desk Organizer Set", "Cable Management Kit",
        ],
        "category": [
            "Electronics", "Computers", "Office Supplies", "Home & Garden",
            "Sports & Outdoors", "Clothing", "Beauty & Personal Care",
            "Toys & Games", "Books", "Food & Grocery",
            "Automotive", "Health & Wellness", "Pet Supplies",
            "Baby & Kids", "Jewelry & Watches", "Arts & Crafts",
        ],
        # NEW DOMAIN POOLS
        "restaurant_name": [
            "The Golden Fork", "Bella Italia", "Tokyo Garden", "Blue Ocean Grill",
            "Mountain View Cafe", "The Rustic Table", "Sakura Sushi", "Le Petit Bistro",
            "Spice Route", "The Green Leaf", "Urban Kitchen", "Fire & Ice",
            "The Hungry Bear", "Sunset Terrace", "Casa del Sol", "The Laughing Lobster",
            "Emerald Thai", "Brooklyn Deli", "The Olive Branch", "Maple Street Diner",
        ],
        "cuisine_type": [
            "Italian", "Japanese", "Mexican", "Chinese", "Indian", "Thai",
            "French", "Mediterranean", "American", "Korean", "Vietnamese",
            "Greek", "Middle Eastern", "Spanish", "Brazilian", "Ethiopian",
        ],
        "menu_item": [
            "Grilled Salmon with Lemon Butter", "Margherita Pizza", "Chicken Tikka Masala",
            "Pad Thai with Shrimp", "Caesar Salad", "Beef Bourguignon",
            "Sushi Platter Deluxe", "Fish and Chips", "Vegetable Stir Fry",
            "Lobster Bisque", "BBQ Ribs", "Mushroom Risotto", "Tacos al Pastor",
            "Greek Moussaka", "Tom Yum Soup", "Eggs Benedict", "Avocado Toast",
            "Butter Chicken", "Pho Bo", "Beef Wellington", "Crème Brûlée",
        ],
        "course_name": [
            "Introduction to Machine Learning", "Advanced Data Structures",
            "Web Development Fundamentals", "Cloud Computing Essentials",
            "Digital Marketing Strategy", "Financial Accounting 101",
            "Project Management Professional", "UX Design Principles",
            "Python for Data Science", "Business Analytics", "Agile Methodology",
            "Cybersecurity Fundamentals", "Leadership and Management",
            "Public Speaking Mastery", "Creative Writing Workshop",
        ],
        "university": [
            "MIT", "Stanford University", "Harvard University", "UC Berkeley",
            "Cambridge University", "Oxford University", "ETH Zurich",
            "Carnegie Mellon University", "Georgia Tech", "University of Michigan",
            "UCLA", "Columbia University", "Yale University", "Princeton University",
            "University of Toronto", "National University of Singapore",
        ],
        "degree": [
            "Bachelor of Science in Computer Science", "Master of Business Administration",
            "Doctor of Philosophy in Physics", "Bachelor of Arts in Economics",
            "Master of Science in Data Science", "Bachelor of Engineering",
            "Master of Public Health", "Doctor of Medicine",
            "Master of Fine Arts", "Bachelor of Commerce",
            "Professional Certificate in Project Management",
        ],
        "event_name": [
            "Annual Tech Summit 2024", "Global Innovation Conference",
            "Product Launch Webinar", "Quarterly Business Review",
            "Team Building Workshop", "Customer Success Meetup",
            "Developer Conference", "Marketing Strategy Session",
            "Leadership Retreat", "Industry Networking Event",
            "Hackathon: Build the Future", "AI & ML Symposium",
        ],
        "venue": [
            "Grand Convention Center", "The Ritz-Carlton Ballroom",
            "Silicon Valley Conference Hall", "Downtown Marriott",
            "Tech Hub Auditorium", "Innovation Campus", "The Summit Club",
            "Waterfront Event Space", "Metropolitan Convention Center",
            "Hilton Garden Terrace", "The Forum", "Sunrise Pavilion",
        ],
        "project_name": [
            "Project Phoenix", "Operation Streamline", "Initiative Alpha",
            "Digital Transformation 2024", "Customer Experience Revamp",
            "Platform Migration", "Security Enhancement Program",
            "Market Expansion Initiative", "Product Innovation Sprint",
            "Process Automation Project", "Brand Refresh Campaign",
            "Infrastructure Modernization", "Data Lake Implementation",
        ],
        "task_name": [
            "Review pull request", "Update documentation", "Fix login bug",
            "Design landing page mockup", "Set up CI/CD pipeline",
            "Conduct user interviews", "Write unit tests", "Optimize database queries",
            "Create marketing copy", "Schedule team meeting", "Prepare quarterly report",
            "Refactor authentication module", "Deploy to production",
        ],
        "milestone": [
            "MVP Launch", "Beta Release", "General Availability",
            "100K Users Milestone", "Series A Funding", "Product Market Fit",
            "First Enterprise Customer", "International Expansion",
            "SOC 2 Certification", "Mobile App Launch", "API v2 Release",
        ],
        "review_title": [
            "Great product, highly recommend!", "Exceeded my expectations",
            "Solid quality for the price", "Good but room for improvement",
            "Not what I expected", "Amazing customer service",
            "Would buy again", "Mixed feelings about this",
            "Perfect for my needs", "Disappointing experience",
        ],
        "review_text": [
            "This product exceeded all my expectations. The quality is outstanding and it arrived faster than expected. Highly recommend!",
            "Solid purchase. Works exactly as described. Customer service was helpful when I had questions.",
            "Good value for the money, but the instructions could be clearer. Otherwise satisfied with my purchase.",
            "The quality is excellent and it's clear a lot of thought went into the design. Will definitely buy from this brand again.",
            "Decent product but took longer to arrive than expected. The product itself works fine.",
            "Amazing! This is exactly what I was looking for. The attention to detail is impressive.",
            "Not bad, but I've seen better at this price point. It does the job adequately.",
            "Fantastic experience from start to finish. Easy to set up and works perfectly.",
            "The product is good but packaging was damaged on arrival. Fortunately, the item was intact.",
            "Exceptional quality and great customer support. They went above and beyond to help me.",
        ],
        "city": [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "San Francisco", "Seattle", "Boston", "Austin", "Denver",
            "Miami", "Atlanta", "Portland", "San Diego", "Dallas",
            "London", "Paris", "Tokyo", "Singapore", "Sydney",
            "Toronto", "Berlin", "Amsterdam", "Dubai", "Mumbai",
        ],
        "country": [
            "United States", "United Kingdom", "Canada", "Germany", "France",
            "Japan", "Australia", "India", "Brazil", "Mexico",
            "Italy", "Spain", "Netherlands", "Sweden", "Singapore",
            "South Korea", "United Arab Emirates", "Switzerland", "Ireland", "Israel",
        ],
        "address": [
            "123 Main Street, Suite 100", "456 Oak Avenue, Floor 3",
            "789 Innovation Drive", "1001 Tech Boulevard, Building A",
            "2500 Market Street", "350 Fifth Avenue, 21st Floor",
            "1600 Amphitheatre Parkway", "One Microsoft Way",
            "410 Terry Avenue North", "1 Infinite Loop",
        ],
        "company_name": [
            "Acme Corporation", "TechVision Inc.", "Global Dynamics",
            "Innovate Solutions", "Summit Technologies", "Blue Horizon Labs",
            "Apex Industries", "Quantum Systems", "Pioneer Analytics",
            "Stellar Ventures", "Nexus Consulting", "Atlas Enterprises",
            "Synergy Partners", "Velocity Software", "Horizon Digital",
        ],
        "industry": [
            "Technology", "Healthcare", "Finance", "E-commerce", "Manufacturing",
            "Education", "Real Estate", "Consulting", "Telecommunications",
            "Energy", "Retail", "Media & Entertainment", "Transportation",
            "Hospitality", "Insurance", "Pharmaceuticals", "Aerospace",
        ],
        "feature_name": [
            "Single Sign-On (SSO)", "Two-Factor Authentication", "Real-time Analytics",
            "Custom Dashboards", "API Integration", "Advanced Reporting",
            "Role-Based Access Control", "Automated Workflows", "Data Export",
            "Mobile App Support", "Bulk Import", "Audit Logging",
            "Collaborative Editing", "Version History", "Custom Branding",
        ],
        "bug_type": [
            "UI rendering issue", "Authentication failure", "Data sync error",
            "Performance degradation", "Memory leak", "API timeout",
            "Incorrect calculation", "Missing validation", "Broken link",
            "Cross-browser compatibility", "Mobile responsiveness issue",
            "Localization error", "Permission denied unexpectedly",
        ],
        "api_endpoint": [
            "/api/v1/users", "/api/v1/auth/login", "/api/v1/products",
            "/api/v1/orders", "/api/v1/payments", "/api/v1/analytics",
            "/api/v1/notifications", "/api/v1/settings", "/api/v1/search",
            "/api/v1/reports", "/api/v1/webhooks", "/api/v1/integrations",
        ],
        # NEW v0.5.0: Additional high-quality domain pools
        "medical_specialty": [
            "Cardiology", "Dermatology", "Emergency Medicine", "Endocrinology",
            "Family Medicine", "Gastroenterology", "General Surgery", "Geriatrics",
            "Hematology", "Infectious Disease", "Internal Medicine", "Nephrology",
            "Neurology", "Obstetrics & Gynecology", "Oncology", "Ophthalmology",
            "Orthopedic Surgery", "Otolaryngology", "Pediatrics", "Psychiatry",
            "Pulmonology", "Radiology", "Rheumatology", "Urology", "Anesthesiology",
        ],
        "transaction_type": [
            "Purchase", "Refund", "Transfer", "Deposit", "Withdrawal",
            "Payment", "Credit", "Debit", "Fee", "Interest",
            "Dividend", "Commission", "Bonus", "Adjustment", "Reversal",
            "Wire Transfer", "ACH Transfer", "Direct Deposit", "Check Payment",
            "Cash Advance", "Balance Transfer", "Loan Disbursement", "Bill Payment",
        ],
        "account_type": [
            "Checking Account", "Savings Account", "Money Market Account",
            "Certificate of Deposit", "Individual Retirement Account (IRA)",
            "401(k) Account", "Brokerage Account", "Business Checking",
            "Business Savings", "Health Savings Account (HSA)", "Joint Account",
            "Trust Account", "Custodial Account", "Student Account", "Premium Account",
        ],
        "brand": [
            "Apple", "Samsung", "Sony", "LG", "Nike", "Adidas", "Puma", "Under Armour",
            "Toyota", "Honda", "Ford", "Tesla", "Microsoft", "Google", "Amazon",
            "Dell", "HP", "Lenovo", "ASUS", "Acer", "Canon", "Nikon", "Bose",
            "JBL", "Philips", "Panasonic", "Whirlpool", "GE", "Bosch", "Dyson",
            "IKEA", "Williams-Sonoma", "Crate & Barrel", "West Elm", "Pottery Barn",
        ],
        "payment_method": [
            "Credit Card (Visa)", "Credit Card (Mastercard)", "Credit Card (Amex)",
            "Debit Card", "PayPal", "Apple Pay", "Google Pay", "Bank Transfer",
            "Wire Transfer", "Check", "Cash", "Cryptocurrency", "Venmo",
            "Klarna", "Afterpay", "Shop Pay", "Amazon Pay", "ACH Direct Debit",
        ],
        "order_status": [
            "Pending", "Confirmed", "Processing", "Shipped", "In Transit",
            "Out for Delivery", "Delivered", "Completed", "Cancelled", "Refunded",
            "On Hold", "Backordered", "Returned", "Partially Shipped", "Failed",
        ],
        "customer_segment": [
            "Enterprise", "Mid-Market", "Small Business", "Startup", "Individual",
            "Premium", "Standard", "Basic", "Trial", "Churned", "At-Risk",
            "Champion", "Loyal", "New Customer", "VIP", "Wholesale", "Retail",
        ],
        "license_type": [
            "MIT License", "Apache License 2.0", "GNU GPL v3", "BSD 3-Clause",
            "Creative Commons BY 4.0", "Proprietary", "Commercial", "Educational",
            "Open Source", "Freeware", "Shareware", "Enterprise License",
            "Single User", "Multi-User", "Site License", "Perpetual License",
        ],
        "file_type": [
            "PDF Document", "Word Document", "Excel Spreadsheet", "PowerPoint Presentation",
            "JPEG Image", "PNG Image", "MP4 Video", "MP3 Audio", "ZIP Archive",
            "CSV File", "JSON File", "XML File", "HTML Page", "Python Script",
            "JavaScript File", "SQL Database", "Markdown Document", "Text File",
        ],
        "priority_level": [
            "Critical", "High", "Medium", "Low", "Trivial",
            "Urgent", "Normal", "Deferred", "Blocked", "In Review",
        ],
        "subscription_plan": [
            "Free Tier", "Basic Plan", "Professional Plan", "Business Plan",
            "Enterprise Plan", "Starter Plan", "Growth Plan", "Scale Plan",
            "Team Plan", "Individual Plan", "Student Plan", "Nonprofit Plan",
            "Annual Pro", "Monthly Basic", "Lifetime Access", "Pay-As-You-Go",
        ],
        # Generic fallbacks for common column patterns
        "name": [
            "Alpha Project", "Beta Initiative", "Gamma Solution", "Delta System",
            "Epsilon Framework", "Zeta Platform", "Eta Service", "Theta Module",
            "Iota Component", "Kappa Engine", "Lambda Protocol", "Mu Architecture",
            "Strategic Modernization", "Digital Transformation", "Innovation Hub",
            "Next Generation Platform", "Cloud Migration", "Data Integration Suite",
        ],
        "description": [
            "High-performance solution designed for enterprise-scale deployments with robust security features.",
            "User-friendly platform offering seamless integration with existing workflows and systems.",
            "Cutting-edge technology stack built for reliability, scalability, and maintainability.",
            "Comprehensive toolkit featuring advanced analytics and real-time monitoring capabilities.",
            "Industry-leading service with proven track record of customer satisfaction and uptime.",
            "Streamlined workflow automation reducing manual effort and improving efficiency.",
            "Innovative approach combining best practices with modern architectural patterns.",
            "Full-featured solution supporting multiple deployment options and configuration flexibility.",
        ],
        "title": [
            "Senior Software Engineer", "Product Manager", "Data Analyst",
            "Marketing Director", "Sales Representative", "Customer Success Manager",
            "Technical Lead", "UX Designer", "DevOps Engineer", "Quality Analyst",
            "Project Coordinator", "Business Analyst", "Account Executive",
        ],
        "status": [
            "Active", "Inactive", "Pending", "Approved", "Rejected",
            "Under Review", "Completed", "In Progress", "On Hold", "Archived",
            "Draft", "Published", "Expired", "Suspended", "Verified",
        ],
        "type": [
            "Standard", "Premium", "Custom", "Default", "Advanced",
            "Basic", "Professional", "Enterprise", "Starter", "Legacy",
            "Internal", "External", "Public", "Private", "Hybrid",
        ],
        "skill": [
            "Python", "JavaScript", "SQL", "Machine Learning", "Data Analysis",
            "Project Management", "Communication", "Leadership", "Problem Solving",
            "AWS", "Docker", "Kubernetes", "React", "Node.js", "TensorFlow",
            "Agile/Scrum", "Public Speaking", "Negotiation", "Strategic Planning",
        ],
    }
    
    def __init__(
        self,
        provider: str = "groq",
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the smart value generator.
        
        Args:
            provider: LLM provider ("groq", "openai", "ollama")
            api_key: API key for the provider
            cache_dir: Directory to cache generated pools
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".mvp" / "value_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._pool_cache: Dict[str, List[str]] = {}
        self._client = None
    
    def _get_client(self):
        """Lazily initialize LLM client."""
        if self._client is None:
            if self.provider == "groq":
                try:
                    from groq import Groq
                    self._client = Groq(api_key=self.api_key)
                except ImportError:
                    return None
            elif self.provider == "openai":
                try:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=self.api_key)
                except ImportError:
                    return None
        return self._client
    
    def detect_domain(self, column_name: str, table_name: str = "") -> Optional[str]:
        """
        Detect semantic domain from column and table names.
        
        Args:
            column_name: Name of the column
            table_name: Name of the table (optional context)
            
        Returns:
            Detected domain name or None
        """
        # Normalize names
        col_lower = column_name.lower().replace("_", " ")
        table_lower = table_name.lower().replace("_", " ")
        combined = f"{table_lower} {col_lower}"
        
        # Check each domain pattern
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if pattern in col_lower or pattern in combined:
                    return domain
        
        return None
    
    def _get_cache_key(self, domain: str, context: str, size: int) -> str:
        """Generate a cache key for a pool request."""
        content = f"{domain}:{context}:{size}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _load_cached_pool(self, cache_key: str) -> Optional[List[str]]:
        """Load a pool from disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def _save_pool_to_cache(self, cache_key: str, pool: List[str]) -> None:
        """Save a pool to disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(pool, f)
        except Exception:
            pass
    
    def generate_pool_with_llm(
        self,
        domain: str,
        context: str = "",
        size: int = 50,
    ) -> List[str]:
        """
        Generate a pool of realistic values using LLM.
        
        Args:
            domain: Semantic domain (e.g., "disease", "prescription")
            context: Additional context (e.g., "hospital emergency room")
            size: Number of values to generate
            
        Returns:
            List of generated values
        """
        client = self._get_client()
        if client is None:
            # Fall back to curated pools
            return self.FALLBACK_POOLS.get(domain, [])[:size]
        
        # Build prompt
        context_str = f" for a {context}" if context else ""
        prompt = f"""Generate exactly {size} realistic {domain.replace('_', ' ')} values{context_str}.

Requirements:
- Be specific and realistic (not generic placeholders)
- Include variety (different types, severities, categories)
- Use proper terminology for the domain
- Each value should be unique

Return ONLY a JSON array of strings, no explanation. Example:
["Value 1", "Value 2", "Value 3"]"""

        try:
            if self.provider == "groq":
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a domain expert generating realistic test data. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content.strip()
            elif self.provider == "openai":
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a domain expert generating realistic test data. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
                content = response.choices[0].message.content.strip()
            else:
                return self.FALLBACK_POOLS.get(domain, [])[:size]
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            pool = json.loads(content)
            
            if isinstance(pool, list) and len(pool) > 0:
                return pool[:size]
            else:
                return self.FALLBACK_POOLS.get(domain, [])[:size]
                
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self.FALLBACK_POOLS.get(domain, [])[:size]
    
    def get_pool(
        self,
        column_name: str,
        table_name: str = "",
        domain_hint: Optional[str] = None,
        context: str = "",
        size: int = 50,
        use_llm: bool = True,
    ) -> List[str]:
        """
        Get or create a value pool for a column.
        
        Args:
            column_name: Name of the column
            table_name: Name of the table
            domain_hint: Explicit domain override
            context: Additional context for LLM
            size: Pool size
            use_llm: Whether to use LLM for generation
            
        Returns:
            List of domain-appropriate values (NEVER empty - falls back to generic pools)
        """
        # Determine domain
        domain = domain_hint or self.detect_domain(column_name, table_name)
        
        # If no domain detected, infer from column name patterns
        if domain is None:
            col_lower = column_name.lower()
            # Try to match generic patterns
            if "name" in col_lower:
                domain = "name"
            elif "desc" in col_lower or "about" in col_lower:
                domain = "description"
            elif "title" in col_lower:
                domain = "title"
            elif "status" in col_lower or "state" in col_lower:
                domain = "status"
            elif "type" in col_lower or "kind" in col_lower:
                domain = "type"
            else:
                # Ultimate fallback - use "name" pool for any unknown TEXT column
                domain = "name"
        
        # Banking primitives are generated locally so they stay deterministic,
        # fast, and never depend on an LLM.
        banking = banking_pool(domain, size=size)
        if banking:
            return banking

        # Build context string
        full_context = context or f"{table_name} {column_name}".strip()
        
        # Check in-memory cache first
        cache_key = self._get_cache_key(domain, full_context, size)
        if cache_key in self._pool_cache:
            return self._pool_cache[cache_key]
        
        # Check disk cache
        cached = self._load_cached_pool(cache_key)
        if cached:
            self._pool_cache[cache_key] = cached
            return cached
        
        # Generate new pool
        if use_llm:
            pool = self.generate_pool_with_llm(domain, full_context, size)
        else:
            pool = self.FALLBACK_POOLS.get(domain, [])[:size]
        
        # Ensure we never return empty - cascade through fallbacks
        if not pool:
            pool = self.FALLBACK_POOLS.get(domain, [])[:size]
        if not pool:
            # Absolute fallback - use generic name pool
            pool = self.FALLBACK_POOLS.get("name", ["Item A", "Item B", "Item C"])[:size]
        
        # Cache the pool
        self._pool_cache[cache_key] = pool
        self._save_pool_to_cache(cache_key, pool)
        
        return pool
    
    def get_fallback_pool(self, domain: str) -> List[str]:
        """Get curated fallback pool for a domain."""
        return self.FALLBACK_POOLS.get(domain, [])
    
    def generate_with_template(
        self,
        template: str,
        size: int,
        components: Dict[str, List[str]],
    ) -> List[str]:
        """Generate text by substituting template components.
        
        This creates more variety by combining parts rather than
        picking from a fixed pool.
        
        Args:
            template: String template with {component_name} placeholders
            size: Number of values to generate
            components: Dict mapping component names to value lists
            
        Returns:
            List of generated strings
            
        Example:
            template = "{first_name} {last_name}"
            components = {
                "first_name": ["John", "Jane", "Alex"],
                "last_name": ["Smith", "Johnson", "Williams"],
            }
            values = gen.generate_with_template(template, 100, components)
            # Returns: ["John Smith", "Jane Williams", "Alex Johnson", ...]
        """
        import random
        
        results = []
        for _ in range(size):
            text = template
            for key, values in components.items():
                if f"{{{key}}}" in text:
                    text = text.replace(f"{{{key}}}", random.choice(values), 1)
            results.append(text)
        
        return results
    
    def generate_composite_pool(
        self,
        domain: str,
        size: int = 200,
    ) -> List[str]:
        """Generate larger pools using template composition.
        
        Instead of calling LLM for 200 values, we compose
        templates with varied components.
        
        Args:
            domain: Semantic domain
            size: Target pool size
            
        Returns:
            List of composed values
        """
        import random
        
        # Domain-specific templates
        templates = {
            "address": {
                "template": "{number} {street_name} {street_type}, {city}, {state}",
                "components": {
                    "number": [str(i) for i in range(100, 10000)],
                    "street_name": ["Oak", "Maple", "Cedar", "Pine", "Elm", "Birch", "Walnut", "Cherry", "Willow", "Aspen",
                                   "Main", "First", "Second", "Third", "Park", "Lake", "River", "Hill", "Valley", "Spring"],
                    "street_type": ["Street", "Avenue", "Boulevard", "Lane", "Drive", "Court", "Place", "Road", "Way", "Circle"],
                    "city": ["Springfield", "Riverside", "Franklin", "Georgetown", "Clinton", "Salem", "Madison", "Bristol", "Fairview", "Newport"],
                    "state": ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "MI", "NC", "WA", "CO", "AZ", "MA", "VA"],
                },
            },
            "email": {
                "template": "{name_part}{separator}{domain_part}@{provider}.{tld}",
                "components": {
                    "name_part": ["john", "jane", "alex", "sam", "chris", "pat", "taylor", "jordan", "casey", "morgan",
                                 "mike", "lisa", "david", "emma", "ryan", "kate", "nick", "amy", "steve", "jen"],
                    "separator": ["", ".", "_", ""],
                    "domain_part": ["smith", "jones", "work", "mail", "pro", "dev", "biz", "123", "2024", "online"],
                    "provider": ["gmail", "yahoo", "outlook", "hotmail", "icloud", "proton", "fastmail", "zoho"],
                    "tld": ["com", "com", "com", "org", "net", "io", "co"],
                },
            },
            "product": {
                "template": "{adjective} {material} {item_type} - {size_color}",
                "components": {
                    "adjective": ["Premium", "Ultra", "Pro", "Classic", "Modern", "Sleek", "Essential", "Deluxe", "Elite", "Smart"],
                    "material": ["Stainless Steel", "Bamboo", "Ceramic", "Leather", "Cotton", "Titanium", "Wood", "Glass", "Silicone", "Carbon Fiber"],
                    "item_type": ["Water Bottle", "Phone Case", "Backpack", "Wallet", "Watch Band", "Desk Lamp", "Speaker", "Charging Dock", "Notebook", "Organizer"],
                    "size_color": ["Black/Large", "White/Medium", "Navy/Standard", "Gray/Compact", "Red/XL", "Brown/Regular", "Silver/Slim", "Green/Mini"],
                },
            },
            "company_name": {
                "template": "{prefix} {industry_word} {suffix}",
                "components": {
                    "prefix": ["Nova", "Apex", "Prime", "Vertex", "Quantum", "Fusion", "Nexus", "Stellar", "Vector", "Atlas",
                              "Blue", "Red", "Green", "Global", "United", "First", "New", "Smart", "Tech", "Digital"],
                    "industry_word": ["Solutions", "Systems", "Tech", "Labs", "Works", "Group", "Partners", "Dynamics", "Innovations", "Ventures",
                                     "Digital", "Logic", "Flow", "Wave", "Net", "Cloud", "Data", "Edge", "Core", "Sync"],
                    "suffix": ["Inc", "Corp", "LLC", "Co", "Ltd", "GmbH", "Technologies", "International", "Enterprises", "Holdings"],
                },
            },
        }
        
        if domain in templates:
            config = templates[domain]
            return self.generate_with_template(
                config["template"],
                size,
                config["components"]
            )
        
        # Fall back to curated pool with random sampling
        base_pool = self.FALLBACK_POOLS.get(domain, [])
        if len(base_pool) >= size:
            return random.sample(base_pool, size)
        elif len(base_pool) > 0:
            # Repeat with slight variations
            result = []
            for i in range(size):
                base = random.choice(base_pool)
                if random.random() < 0.3:  # 30% chance to add suffix
                    suffix = random.choice([" (v2)", " Pro", " Plus", " - Updated", " 2.0", ""])
                    base = base + suffix
                result.append(base)
            return result
        
        return []


# ============ Template Registry ============

COMPOSITION_TEMPLATES = {
    "order_id": "{prefix}-{year}-{number}",
    "invoice_number": "INV-{year}{month}-{number}",
    "tracking_number": "{carrier}{number}{check}",
    "sku": "{category}-{brand}-{variant}-{size}",
    "username": "{adjective}{noun}{number}",
}

TEMPLATE_COMPONENTS = {
    "order_id": {
        "prefix": ["ORD", "SO", "PO", "WO", "REQ"],
        "year": ["2023", "2024", "2025"],
        "number": [str(i).zfill(6) for i in range(1, 1000)],
    },
    "invoice_number": {
        "year": ["23", "24", "25"],
        "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
        "number": [str(i).zfill(4) for i in range(1, 10000)],
    },
    "tracking_number": {
        "carrier": ["1Z", "9400", "92", "420"],
        "number": [str(i).zfill(12) for i in range(100000000000, 100001000000)],
        "check": [str(i) for i in range(10)],
    },
    "sku": {
        "category": ["ELC", "CLO", "HOM", "SPT", "TOY", "BOK"],
        "brand": ["APP", "SAM", "NIK", "ADI", "SON", "LG"],
        "variant": ["BLK", "WHT", "RED", "BLU", "GRN", "GRY"],
        "size": ["S", "M", "L", "XL", "XXL", "OS"],
    },
    "username": {
        "adjective": ["cool", "super", "mega", "ultra", "epic", "pro", "fast", "swift", "bold", "smart"],
        "noun": ["ninja", "tiger", "dragon", "wolf", "hawk", "bear", "lion", "eagle", "shark", "fox"],
        "number": [str(i) for i in range(1, 1000)],
    },
}


# Convenience function for quick testing
def smart_generate(column_name: str, table_name: str = "", size: int = 10) -> List[str]:
    """Quick smart value generation for testing."""
    gen = SmartValueGenerator()
    pool = gen.get_pool(column_name, table_name, size=max(size * 2, 50))
    if pool:
        import random
        return random.choices(pool, k=size)
    return []
