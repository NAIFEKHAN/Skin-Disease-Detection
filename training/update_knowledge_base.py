"""
Update Knowledge Base (Disease Library & Treatment Costs) for 9 Common Skin Diseases.
Prepares two-mode application architecture without altering existing dermoscopic records.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DISEASE_INFO_FILE = ROOT_DIR / "data" / "disease_info.json"
TREATMENT_COSTS_FILE = ROOT_DIR / "data" / "treatment_costs.json"


def update_disease_library():
    print(f"Updating {DISEASE_INFO_FILE}...")
    with open(DISEASE_INFO_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for d in data["diseases"]:
        if "mode" not in d:
            d["mode"] = "dermoscopy"

    existing_codes = set(d["code"].upper() for d in data["diseases"])

    new_diseases = [
        {
            "code": "ACNE",
            "name": "Acne",
            "category_type": "Inflammatory Pilosebaceous Disorder",
            "mode": "common_skin",
            "short_description": "A multifactorial inflammatory disease of the pilosebaceous unit characterized by comedones, erythematous papules, pustules, and occasional nodules or cysts.",
            "representative_image": "common_acne.jpg",
            "risk_level": "Low Concern",
            "common_characteristics": [
                "Closed comedones (whiteheads) and open comedones (blackheads)",
                "Inflammatory erythematous papules and superficial pustules",
                "Occasional deeper tender nodules or pseudocysts in severe grades",
                "Commonly localized to sebum-rich areas (face, chest, upper back)",
                "Risk of post-inflammatory hyperpigmentation or scarring"
            ],
            "diagnostic_approaches": [
                "Clinical grading of lesion count and severity (PGA / GAGS scale)",
                "Evaluation of comedonal vs. inflammatory predominant components",
                "Assessment of contributing endocrinological factors (e.g., PCOS in females)",
                "Screening for medication-induced acneiform eruptions"
            ],
            "treatment_approaches": [
                "Topical retinoids (adapalene, tretinoin) for comedolytic regulation",
                "Topical benzoyl peroxide and topical/oral antibiotics for inflammatory control",
                "Oral isotretinoin for severe, recalcitrant, or scarring nodulocystic acne",
                "Hormonal therapies (combined oral contraceptives, spironolactone)",
                "Gentle non-comedogenic skincare and daily SPF sunscreen"
            ],
            "when_to_seek_care": "Consult a dermatologist if over-the-counter washes fail, if painful deep cysts or permanent scarring appear, or if acne significantly impacts emotional well-being.",
            "sources": [
                "American Academy of Dermatology (AAD) Acne Clinical Guidelines",
                "British Association of Dermatologists (BAD) Acne Management Guidelines",
                "DermNet NZ (Acne Vulgaris Reference)"
            ]
        },
        {
            "code": "ACD",
            "name": "Allergic Contact Dermatitis",
            "category_type": "Type IV Delayed Hypersensitivity Reaction",
            "mode": "common_skin",
            "short_description": "An immunologically mediated delayed hypersensitivity eruption occurring after cutaneous exposure to an exogenous environmental allergen.",
            "representative_image": "common_allergic_contact_dermatitis.jpg",
            "risk_level": "Moderate Concern",
            "common_characteristics": [
                "Erythematous, intensely pruritic macules, plaques, and tiny vesicles",
                "Well-demarcated geometric or artificial distribution matching contact zone",
                "Oozing, crusting, and scaling in acute stages; lichenification in chronic phases",
                "Latency period of 24–72 hours following exposure to sensitized contactant"
            ],
            "diagnostic_approaches": [
                "Detailed exposure history (occupational, cosmetics, jewelry, plants)",
                "Standardized cutaneous patch testing with baseline allergen series",
                "Repeated Open Application Testing (ROAT) for leave-on products",
                "Differential exclusion of irritant contact dermatitis and atopic dermatitis"
            ],
            "treatment_approaches": [
                "Strict identification and permanent elimination of culprit allergens",
                "Short-course mid-to-high potency topical corticosteroids for acute flares",
                "Cool compresses and barrier restoration creams (ceramide-based emollients)",
                "Short course of oral corticosteroids for widespread or facial involvement",
                "Oral antihistamines for symptomatic control of intense pruritus"
            ],
            "when_to_seek_care": "Seek medical care if rash is widespread, involves the eyes/face/genitalia, shows yellow crusting indicating secondary infection, or fails to resolve with trigger elimination.",
            "sources": [
                "American Contact Dermatitis Society (ACDS) Core Guidelines",
                "European Society of Contact Dermatitis (ESCD)",
                "DermNet NZ (Allergic Contact Dermatitis)"
            ]
        },
        {
            "code": "ECZEMA",
            "name": "Eczema (Atopic Dermatitis)",
            "category_type": "Chronic Pruritic Inflammatory Dermatosis",
            "mode": "common_skin",
            "short_description": "A chronic, relapsing, pruritic inflammatory skin disorder associated with epidermal barrier dysfunction (filaggrin mutations) and immune dysregulation.",
            "representative_image": "common_eczema.jpg",
            "risk_level": "Moderate Concern",
            "common_characteristics": [
                "Intense itch (pruritus) leading to persistent scratch-itch cycle",
                "Erythema, scaling, excoriations, and xerosis (severe dryness)",
                "Classic flexural distribution (antecubital and popliteal fossae) in adults",
                "Facial and extensor involvement common in infants and young children",
                "Lichenification (thickened leather-like skin) from chronic rubbing"
            ],
            "diagnostic_approaches": [
                "Hanifin and Rajka diagnostic criteria or UK Working Party criteria",
                "Evaluation of personal or family history of atopic triad (asthma, allergic rhinitis)",
                "Assessment of disease severity via EASI (Eczema Area and Severity Index) or SCORAD",
                "Bacterial swab if honey-colored crusts suggest secondary Staphylococcal infection"
            ],
            "treatment_approaches": [
                "Liberal daily application of lipid-rich barrier repair emollients",
                "Intermittent proactive topical corticosteroids or calcineurin inhibitors",
                "Topical PDE4 inhibitors (crisaborole) or JAK inhibitors for mild-moderate cases",
                "Systemic biologics (dupilumab, tralokinumab) or JAK inhibitors for severe disease",
                "Bleach baths or antiseptic cleansers to reduce Staphylococcus aureus colonization"
            ],
            "when_to_seek_care": "Consult a dermatologist if itching disturbs sleep, skin shows oozing/pus indicating bacterial infection, or sudden punched-out sores appear (possible eczema herpeticum).",
            "sources": [
                "American Academy of Dermatology (AAD) Atopic Dermatitis Guidelines",
                "National Eczema Association (NEA) Clinical Guidelines",
                "DermNet NZ (Atopic Dermatitis)"
            ]
        },
        {
            "code": "IMPETIGO",
            "name": "Impetigo",
            "category_type": "Contagious Superficial Bacterial Infection",
            "mode": "common_skin",
            "short_description": "A highly contagious superficial bacterial skin infection, predominantly caused by Staphylococcus aureus and Streptococcus pyogenes, presenting with honey-colored crusted erosions.",
            "representative_image": "common_impetigo.jpg",
            "risk_level": "Moderate Concern",
            "common_characteristics": [
                "Classic non-bullous form with golden honey-colored adherence crusts",
                "Bullous variant with flaccid, clear-to-turbid fluid-filled blisters",
                "Erythematous base exposed upon detachment of superficial crusts",
                "Commonly clustered around the perioral and perinasal areas of the face",
                "Rapid spread via autoinoculation or direct physical contact"
            ],
            "diagnostic_approaches": [
                "Clinical recognition of pathognomonic honey-colored crusted erosions",
                "Bacterial Gram stain and wound culture with antibiotic sensitivity testing",
                "Differential diagnosis against herpes simplex, contact dermatitis, and scabies"
            ],
            "treatment_approaches": [
                "Topical prescription antibiotic ointment (mupirocin, retapamulin) for localized lesions",
                "Oral beta-lactamase resistant antibiotics (cephalexin, amoxicillin-clavulanate) for widespread lesions",
                "Gentle debridement of crusts using warm soapy soaks before ointment application",
                "Strict hygiene measures: daily hot laundry of towels and clothes, frequent handwashing"
            ],
            "when_to_seek_care": "Seek prompt healthcare consultation for definitive prescription antimicrobial therapy to halt contagion and prevent post-streptococcal complications.",
            "sources": [
                "Infectious Diseases Society of America (IDSA) Skin Infection Guidelines",
                "Centers for Disease Control and Prevention (CDC) Impetigo Resource",
                "DermNet NZ (Impetigo)"
            ]
        },
        {
            "code": "PSORIASIS",
            "name": "Psoriasis",
            "category_type": "Chronic Autoimmune Erythematosquamous Disease",
            "mode": "common_skin",
            "short_description": "A chronic, systemic immune-mediated inflammatory dermatosis characterized by keratinocyte hyperproliferation, manifesting as well-demarcated erythematous plaques with silvery-white scales.",
            "representative_image": "common_psoriasis.jpg",
            "risk_level": "Moderate Concern",
            "common_characteristics": [
                "Sharply circumscribed erythematous plaques covered by silvery micaceous scale",
                "Predilection for extensor surfaces (elbows, knees), scalp, and lumbosacral region",
                "Auspitz sign (punctate bleeding upon gentle detachment of scales)",
                "Koebner phenomenon (appearance of new psoriatic lesions along lines of trauma)",
                "Nail changes (pitting, oil drop sign, onycholysis) in up to 50% of patients"
            ],
            "diagnostic_approaches": [
                "Clinical morphologic assessment and PASI (Psoriasis Area & Severity Index) scoring",
                "Screening for psoriatic arthritis (CASPAR criteria) in all psoriatic patients",
                "Punch biopsy in atypical or erythrodermic variants to confirm epidermal hyperplasia",
                "Cardiovascular and metabolic risk factor comorbidity assessment"
            ],
            "treatment_approaches": [
                "Topical fixed-combination vitamin D analogues (calcipotriol) and potent corticosteroids",
                "Narrowband UVB (NB-UVB) phototherapy or targeted excimer laser therapy",
                "Conventional systemic agents (methotrexate, cyclosporine, acitretin)",
                "Targeted biologic therapeutics (anti-TNF, anti-IL-17, anti-IL-23 monoclonal antibodies)",
                "Keratolytic agents (salicylic acid) to facilitate topical medication penetration"
            ],
            "when_to_seek_care": "Seek specialist dermatological care for personalized treatment planning, if joint pain or stiffness occurs, or if psoriasis covers greater than 5% of body surface area.",
            "sources": [
                "AAD-NPF Joint Guidelines of Care for the Management of Psoriasis",
                "National Psoriasis Foundation (NPF) Clinical Resource",
                "DermNet NZ (Psoriasis Vulgaris)"
            ]
        },
        {
            "code": "RINGWORM",
            "name": "Ringworm (Tinea Corporis)",
            "category_type": "Superficial Dermatophyte Fungal Infection",
            "mode": "common_skin",
            "short_description": "A superficial dermatophytosis of glabrous skin caused by keratinophilic fungi (Trichophyton, Microsporum, or Epidermophyton), presenting as annular erythematous scaly plaques with active expanding borders.",
            "representative_image": "common_ringworm.jpg",
            "risk_level": "Moderate Concern",
            "common_characteristics": [
                "Annular (ring-shaped) erythematous plaque with central clearing",
                "Active, raised, scaly or vesicular advancing outer perimeter",
                "Mild to intense pruritus localized to the active lesion edge",
                "Transmission from infected pets (zoophilic), humans (anthropophilic), or soil"
            ],
            "diagnostic_approaches": [
                "Potassium hydroxide (KOH) direct microscopic mount showing septate hyphae",
                "Fungal culture on Sabouraud dextrose agar for definitive species identification",
                "Wood lamp fluorescence examination for select microsporum infections",
                "Clinical differentiation from annular psoriasis, pityriasis rosea, and erythema annulare"
            ],
            "treatment_approaches": [
                "Topical allylamine antifungals (terbinafine) or imidazoles (clotrimazole, ketoconazole)",
                "Application continued for 1–2 weeks past complete visible clinical clearance",
                "Oral antifungal therapy (terbinafine, itraconazole) for extensive or refractory cases",
                "STRICT WARNING: Avoid topical steroid creams alone (prevents tinea incognito)",
                "Veterinary screening and treatment of household pets to prevent reinfection"
            ],
            "when_to_seek_care": "Consult a doctor if lesions fail to respond to over-the-counter antifungals within 2 weeks, spread rapidly, or if systemic symptoms emerge.",
            "sources": [
                "Centers for Disease Control and Prevention (CDC) Fungal Diseases Resource",
                "British Association of Dermatologists (BAD) Tinea Guidelines",
                "DermNet NZ (Tinea Corporis)"
            ]
        },
        {
            "code": "SCABIES",
            "name": "Scabies",
            "category_type": "Parasitic Mite Infestation (Sarcoptes scabiei)",
            "mode": "common_skin",
            "short_description": "A contagious, intensely pruritic ectoparasitic dermatosis caused by the microscopic burrowing mite Sarcoptes scabiei var. hominis, characterized by nocturnal itch and interdigital burrows.",
            "representative_image": "common_scabies.jpg",
            "risk_level": "Moderate Concern",
            "common_characteristics": [
                "Severe, intractable pruritus, classically exacerbated at nighttime",
                "Serpiginous, thread-like greyish-white cutaneous burrows (2–10 mm in length)",
                "Erythematous papules, excoriations, and secondary bacterial impetiginization",
                "Characteristic anatomic predilection: web spaces of fingers, flexor wrists, axillae, umbilicus, genitalia",
                "Frequent history of itching in close family members or household cohabitants"
            ],
            "diagnostic_approaches": [
                "Dermoscopy revealing the characteristic delta-wing jet sign of the female mite",
                "Skin scraping of unexcoriated burrows with mineral oil microscopy for mites, ova, or scybala",
                "Burrow ink test (pigment concentrates inside mite tunnel)",
                "Clinical diagnosis based on characteristic distribution and nocturnal itching"
            ],
            "treatment_approaches": [
                "Topical 5% permethrin cream applied from neck to toes, left on for 8–14 hours, repeated after 7 days",
                "Oral ivermectin (200 mcg/kg) as two doses spaced 7–14 days apart",
                "Simultaneous treatment of ALL household members and sexual partners, regardless of symptoms",
                "Decontamination: laundering bedding, towels, and worn clothing in hot water (>60°C)",
                "Oral antihistamines and bland moisturizers for post-scabetic itch (can persist 2–4 weeks)"
            ],
            "when_to_seek_care": "Seek immediate medical attention for prescription scabicide therapy. Contact tracing and simultaneous household treatment are mandatory to stop cyclical transmission.",
            "sources": [
                "World Health Organization (WHO) Scabies Factsheet and Guidelines",
                "Centers for Disease Control and Prevention (CDC) Scabies Protocol",
                "DermNet NZ (Scabies)"
            ]
        },
        {
            "code": "SEB_DERM",
            "name": "Seborrheic Dermatitis",
            "category_type": "Chronic Erythematosquamous Scalp & Facial Dermatosis",
            "mode": "common_skin",
            "short_description": "A common chronic, relapsing inflammatory disorder affecting sebum-rich cutaneous areas, linked to Malassezia yeast colonization, abnormal lipid composition, and individual susceptibility.",
            "representative_image": "common_seborrheic_dermatitis.jpg",
            "risk_level": "Low Concern",
            "common_characteristics": [
                "Greasy, yellowish, adherent or bran-like scales over erythematous patches",
                "Symmetrical involvement of scalp (dandruff), eyebrows, nasolabial folds, and chest",
                "Mild to moderate pruritus, frequently fluctuating with seasons and stress",
                "Infantile variant (cradle cap) typically self-limiting within first year of life",
                "Adult form follows a chronic waxing-and-waning lifelong clinical course"
            ],
            "diagnostic_approaches": [
                "Clinical morphological evaluation of characteristic scale texture and anatomical zones",
                "Exclusion of psoriasis (sebopsoriasis overlap), contact dermatitis, and rosacea",
                "Screening for underlying conditions in severe or sudden refractory cases (HIV, Parkinsonism)"
            ],
            "treatment_approaches": [
                "Medicated antifungal shampoos (ketoconazole 2%, zinc pyrithione, selenium sulfide, ciclopirox)",
                "Topical low-potency corticosteroids or calcineurin inhibitors for acute facial erythema",
                "Keratolytic agents (salicylic acid, coal tar) to loosen thick adherent scalp crusts",
                "Maintenance regimen with antifungal washes once weekly to prevent relapses",
                "Avoidance of heavy oil-based hair pomades and harsh alcohol-containing toners"
            ],
            "when_to_seek_care": "Consult a dermatologist if over-the-counter medicated shampoos fail, facial redness becomes bothersome, or scalp scaling becomes painful or secondarily infected.",
            "sources": [
                "American Academy of Dermatology (AAD) Seborrheic Dermatitis Guide",
                "National Eczema Association (NEA) Seborrheic Dermatitis Resource",
                "DermNet NZ (Seborrhoeic Dermatitis)"
            ]
        },
        {
            "code": "URTICARIA",
            "name": "Urticaria (Hives)",
            "category_type": "Mast-Cell Mediated Vascular Reaction",
            "mode": "common_skin",
            "short_description": "A vascular reaction of the dermis characterized by transient, intensely pruritic, erythematous edematous wheals that typically resolve within 24 hours without leaving residual marks.",
            "representative_image": "common_urticaria.jpg",
            "risk_level": "Moderate Concern",
            "common_characteristics": [
                "Circumscribed, raised erythematous wheals with central pallor and surrounding flare",
                "Evanescent nature: individual wheals appear, migrate, and resolve within 24 hours",
                "Intense pruritus (itching) or stinging/burning sensation",
                "May be accompanied by deeper subcutaneous swelling (angioedema) in up to 40% of cases",
                "Classified as acute (<6 weeks duration) or chronic spontaneous/inducible (>6 weeks)"
            ],
            "diagnostic_approaches": [
                "Comprehensive clinical history focusing on triggers (infections, foods, medications, NSAIDs)",
                "Evaluation of inducible physical stimuli (dermographism, cold, heat, pressure, exercise)",
                "Complete blood count and inflammatory markers for chronic persistent urticaria",
                "Emergency assessment to immediately rule out systemic anaphylactic features"
            ],
            "treatment_approaches": [
                "First-line: Second-generation non-sedating oral H1 antihistamines (cetirizine, fexofenadine, bilastine)",
                "Up-titration of non-sedating antihistamines up to 4-fold standard dose under medical supervision",
                "Third-line biologic therapy (omalizumab, anti-IgE) for chronic spontaneous urticaria",
                "Short rescue course of oral corticosteroids (3–5 days) only for severe acute exacerbations",
                "EMERGENCY WARNING: Intramuscular epinephrine (auto-injector) for any signs of anaphylaxis"
            ],
            "when_to_seek_care": "EMERGENCY: Seek immediate emergency room care if hives are accompanied by throat tightness, difficulty breathing, tongue/lip swelling, or dizziness. Consult a doctor for hives lasting over 6 weeks.",
            "sources": [
                "EAACI/GA2LEN/EuroGuiDerm/APAAACI Urticaria Clinical Guideline",
                "American Academy of Allergy, Asthma & Immunology (AAAAI) Urticaria Resource",
                "DermNet NZ (Urticaria)"
            ]
        }
    ]

    for nd in new_diseases:
        if nd["code"].upper() not in existing_codes:
            data["diseases"].append(nd)

    with open(DISEASE_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Disease library updated: {len(data['diseases'])} total entries.")


def update_treatment_costs():
    print(f"Updating {TREATMENT_COSTS_FILE}...")
    with open(TREATMENT_COSTS_FILE, "r", encoding="utf-8") as f:
        costs = json.load(f)

    cats = costs["categories"]

    new_costs = {
        "acne": {
            "disease_name": "Acne",
            "nature": "Inflammatory Pilosebaceous Condition",
            "consultation": {
                "description": "Dermatologist clinical evaluation & acne grading",
                "min_cost": 500,
                "max_cost": 1000
            },
            "diagnostics": {
                "description": "Comedone assessment & optional hormonal panel",
                "min_cost": 500,
                "max_cost": 1500
            },
            "medication": {
                "description": "Topical retinoids, benzoyl peroxide / topical antibiotics",
                "min_cost": 800,
                "max_cost": 2500
            },
            "procedure": {
                "description": "Optional comedone extraction or superficial chemical peel",
                "min_cost": 1500,
                "max_cost": 4000
            },
            "follow_up": {
                "description": "Treatment review & tolerance assessment (6–8 weeks)",
                "min_cost": 400,
                "max_cost": 800
            },
            "total_min": 3700,
            "total_max": 9800
        },
        "allergic_contact_dermatitis": {
            "disease_name": "Allergic Contact Dermatitis",
            "nature": "Type IV Hypersensitivity Eruption",
            "consultation": {
                "description": "Allergy & dermatology clinical consultation",
                "min_cost": 600,
                "max_cost": 1200
            },
            "diagnostics": {
                "description": "Cutaneous patch testing (standard allergen series)",
                "min_cost": 1500,
                "max_cost": 4000
            },
            "medication": {
                "description": "Potent topical corticosteroids & barrier restoration creams",
                "min_cost": 500,
                "max_cost": 1500
            },
            "procedure": {
                "description": "Acute flare skin barrier soothing & allergen counseling",
                "min_cost": 500,
                "max_cost": 1500
            },
            "follow_up": {
                "description": "Patch test reading (48h & 96h) and allergen review",
                "min_cost": 400,
                "max_cost": 800
            },
            "total_min": 3500,
            "total_max": 9000
        },
        "eczema": {
            "disease_name": "Eczema (Atopic Dermatitis)",
            "nature": "Chronic Inflammatory Dermatosis",
            "consultation": {
                "description": "Dermatologist clinical assessment & EASI scoring",
                "min_cost": 600,
                "max_cost": 1200
            },
            "diagnostics": {
                "description": "Barrier integrity assessment & optional secondary swab",
                "min_cost": 800,
                "max_cost": 2000
            },
            "medication": {
                "description": "Lipid barrier emollients, calcineurin inhibitors, topical steroids",
                "min_cost": 1200,
                "max_cost": 3500
            },
            "procedure": {
                "description": "Wet wrap therapy instruction or initial phototherapy session",
                "min_cost": 1000,
                "max_cost": 3000
            },
            "follow_up": {
                "description": "Chronic disease monitoring & flare mitigation review",
                "min_cost": 400,
                "max_cost": 800
            },
            "total_min": 4000,
            "total_max": 10500
        },
        "impetigo": {
            "disease_name": "Impetigo",
            "nature": "Superficial Bacterial Infection",
            "consultation": {
                "description": "Clinical examination & contagion assessment",
                "min_cost": 500,
                "max_cost": 1000
            },
            "diagnostics": {
                "description": "Bacterial Gram stain & culture with antibiotic sensitivity",
                "min_cost": 500,
                "max_cost": 1500
            },
            "medication": {
                "description": "Prescription mupirocin ointment / oral beta-lactam antibiotics",
                "min_cost": 400,
                "max_cost": 1200
            },
            "procedure": {
                "description": "Antiseptic soak & gentle crust removal instruction",
                "min_cost": 400,
                "max_cost": 800
            },
            "follow_up": {
                "description": "Infection clearance confirmation (7–10 days)",
                "min_cost": 300,
                "max_cost": 600
            },
            "total_min": 2100,
            "total_max": 5100
        },
        "psoriasis": {
            "disease_name": "Psoriasis",
            "nature": "Autoimmune Erythematosquamous Disease",
            "consultation": {
                "description": "Specialist dermatological consultation & PASI assessment",
                "min_cost": 700,
                "max_cost": 1500
            },
            "diagnostics": {
                "description": "Joint screening, baseline metabolic panel & optional punch biopsy",
                "min_cost": 1500,
                "max_cost": 3500
            },
            "medication": {
                "description": "Topical vitamin D3 analogues, potent corticosteroids, keratolytics",
                "min_cost": 1500,
                "max_cost": 4500
            },
            "procedure": {
                "description": "Narrowband UVB phototherapy session / plaque descaling",
                "min_cost": 2000,
                "max_cost": 5000
            },
            "follow_up": {
                "description": "Periodic response evaluation & systemic toxicity monitoring",
                "min_cost": 500,
                "max_cost": 1000
            },
            "total_min": 6200,
            "total_max": 15500
        },
        "ringworm": {
            "disease_name": "Ringworm (Tinea Corporis)",
            "nature": "Superficial Dermatophyte Infection",
            "consultation": {
                "description": "Dermatologist clinical evaluation",
                "min_cost": 500,
                "max_cost": 1000
            },
            "diagnostics": {
                "description": "Potassium hydroxide (KOH) direct fungal microscopy",
                "min_cost": 500,
                "max_cost": 1200
            },
            "medication": {
                "description": "Topical allylamine/imidazole creams (terbinafine/clotrimazole)",
                "min_cost": 600,
                "max_cost": 1800
            },
            "procedure": {
                "description": "Antifungal hygiene & recurrence prevention protocol",
                "min_cost": 300,
                "max_cost": 700
            },
            "follow_up": {
                "description": "Mycological cure confirmation after 2–3 weeks",
                "min_cost": 300,
                "max_cost": 600
            },
            "total_min": 2200,
            "total_max": 5300
        },
        "scabies": {
            "disease_name": "Scabies",
            "nature": "Parasitic Mite Infestation",
            "consultation": {
                "description": "Clinical examination & household contact tracing",
                "min_cost": 500,
                "max_cost": 1000
            },
            "diagnostics": {
                "description": "Dermoscopic burrow inspection & skin scraping microscopy",
                "min_cost": 600,
                "max_cost": 1500
            },
            "medication": {
                "description": "Topical 5% permethrin lotion & oral ivermectin for family",
                "min_cost": 800,
                "max_cost": 2200
            },
            "procedure": {
                "description": "Household decontamination instruction & antipruritic soak",
                "min_cost": 400,
                "max_cost": 1000
            },
            "follow_up": {
                "description": "Re-evaluation at day 14 to verify complete eradication",
                "min_cost": 400,
                "max_cost": 800
            },
            "total_min": 2700,
            "total_max": 6500
        },
        "seborrheic_dermatitis": {
            "disease_name": "Seborrheic Dermatitis",
            "nature": "Chronic Scalp & Facial Dermatosis",
            "consultation": {
                "description": "Clinical examination & trigger assessment",
                "min_cost": 500,
                "max_cost": 1000
            },
            "diagnostics": {
                "description": "Sebum & scalp clinical evaluation",
                "min_cost": 400,
                "max_cost": 1000
            },
            "medication": {
                "description": "Ketoconazole / zinc pyrithione shampoo & mild anti-inflammatory lotion",
                "min_cost": 600,
                "max_cost": 1600
            },
            "procedure": {
                "description": "Keratolytic scalp de-scaling treatment",
                "min_cost": 500,
                "max_cost": 1200
            },
            "follow_up": {
                "description": "Maintenance frequency adjustment (4 weeks)",
                "min_cost": 300,
                "max_cost": 600
            },
            "total_min": 2300,
            "total_max": 5400
        },
        "urticaria": {
            "disease_name": "Urticaria (Hives)",
            "nature": "Mast-Cell Mediated Vascular Reaction",
            "consultation": {
                "description": "Emergency / clinical dermatology consultation",
                "min_cost": 600,
                "max_cost": 1200
            },
            "diagnostics": {
                "description": "CBC, inflammatory markers & allergen trigger screening",
                "min_cost": 1200,
                "max_cost": 3500
            },
            "medication": {
                "description": "Second-generation H1 antihistamines & short rescue corticosteroids",
                "min_cost": 600,
                "max_cost": 2000
            },
            "procedure": {
                "description": "Acute flare observation & trigger avoidance counseling",
                "min_cost": 800,
                "max_cost": 2500
            },
            "follow_up": {
                "description": "Chronic wheal monitoring & antihistamine titration",
                "min_cost": 400,
                "max_cost": 800
            },
            "total_min": 3600,
            "total_max": 10000
        }
    }

    for k, v in new_costs.items():
        cats[k] = v

    with open(TREATMENT_COSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(costs, f, indent=2, ensure_ascii=False)
    print(f"Treatment costs updated: {len(cats)} categories supported.")


if __name__ == "__main__":
    update_disease_library()
    update_treatment_costs()
