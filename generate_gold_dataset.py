# File: generate_gold_dataset.py
"""
Gold Test Dataset Generator for Policy Documents
Extracts meaningful Q&A pairs from policy documents for evaluation
"""

import json
from typing import List, Dict
from pathlib import Path

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')    

class GoldDatasetGenerator:
    """Generate gold standard Q&A pairs from policy documents"""
    
    def __init__(self):
        self.dataset = []
        
    def generate_policy1_questions(self) -> List[Dict]:
        """
        Generate gold standard Q&A pairs for Bajaj Allianz Global Health Care
        Based on policy1.pdf content
        """
        
        questions = [
            {
                "id": "p1_q1",
                "category": "coverage_scope",
                "question": "What is the pre-existing disease waiting period for the Global Health Care policy?",
                "ground_truth": "Pre-existing diseases are excluded until 36 months of continuous coverage after the date of inception of the first Global Health Care Policy.",
                "source_section": "Exclusions - Standard Exclusions (Code -Excl01)",
                "difficulty": "easy",
                "expected_chunks": ["Pre-Existing Diseases", "waiting period", "36 months"]
            },
            {
                "id": "p1_q2",
                "category": "benefits",
                "question": "Does the Imperial Plus Plan cover air ambulance services?",
                "ground_truth": "Yes, the Imperial Plus Plan covers Air Ambulance + Medical Evacuation up to the In-patient Sum Insured. This service is on cashless basis only and must be organized by the insurer.",
                "source_section": "Part B-I, Section 8",
                "difficulty": "medium",
                "expected_chunks": ["Air Ambulance", "Medical Evacuation", "Imperial Plus"]
            },
            {
                "id": "p1_q3",
                "category": "claims",
                "question": "What is the timeline for submitting medical claims under the international cover?",
                "ground_truth": "All claims must be submitted no later than 30 days after the date of discharge from the Hospital.",
                "source_section": "Claims Procedure for International Cover",
                "difficulty": "easy",
                "expected_chunks": ["claims", "30 days", "discharge", "timeline"]
            },
            {
                "id": "p1_q4",
                "category": "exclusions",
                "question": "Are dental treatments covered under the base policy?",
                "ground_truth": "No, dental treatments are excluded except for Emergency Inpatient Dental Treatment arising from an Accident. A separate optional Dental Plan is available with 20% co-payment.",
                "source_section": "Specific Exclusions, Dental Plan Benefits",
                "difficulty": "medium",
                "expected_chunks": ["dental", "accident", "emergency", "optional"]
            },
            {
                "id": "p1_q5",
                "category": "waiting_period",
                "question": "What conditions require a 24-month waiting period?",
                "ground_truth": "Specified diseases/procedures require 24 months waiting period including cataracts, hernia, stones in urinary and biliary systems, joint replacement surgery, varicose veins, and many others as listed in the policy.",
                "source_section": "Exclusions (Code - Excl02)",
                "difficulty": "hard",
                "expected_chunks": ["24 months", "specified disease", "waiting period", "cataract"]
            },
            {
                "id": "p1_q6",
                "category": "benefits",
                "question": "What is the pre-hospitalization coverage period for domestic cover?",
                "ground_truth": "The policy covers medical expenses for 60 days immediately before hospitalization, provided such expenses were incurred for the same illness/injury requiring hospitalization.",
                "source_section": "Part A-I, Section 2",
                "difficulty": "easy",
                "expected_chunks": ["pre-hospitalization", "60 days", "domestic"]
            },
            {
                "id": "p1_q7",
                "category": "mental_health",
                "question": "Is mental illness treatment covered under this policy?",
                "ground_truth": "Yes, mental illness treatment is covered for in-patient treatment in a recognized psychiatric unit, diagnosed and treated by a psychiatrist, clinical psychologist or licensed psychotherapist. Out-patient mental illness treatment is excluded.",
                "source_section": "Mental Illness Treatment",
                "difficulty": "medium",
                "expected_chunks": ["mental illness", "psychiatric", "in-patient", "covered"]
            },
            {
                "id": "p1_q8",
                "category": "maternity",
                "question": "Does the policy cover maternity expenses?",
                "ground_truth": "No, maternity expenses are excluded including childbirth, complicated deliveries, caesarean sections (except ectopic pregnancy), and lawful medical termination of pregnancy.",
                "source_section": "Exclusions (Code -Excl18)",
                "difficulty": "easy",
                "expected_chunks": ["maternity", "excluded", "childbirth"]
            },
            {
                "id": "p1_q9",
                "category": "day_care",
                "question": "What types of procedures are covered under day care treatment?",
                "ground_truth": "Day care procedures include treatments undertaken under anesthesia in less than 24 hours that would otherwise require hospitalization, such as cataract surgery, ERCP, colonoscopy, arthroscopy, and many others listed in Annexure I.",
                "source_section": "Day Care Procedures, Annexure I",
                "difficulty": "hard",
                "expected_chunks": ["day care", "less than 24 hours", "procedures", "Annexure"]
            },
            {
                "id": "p1_q10",
                "category": "policy_terms",
                "question": "Can the policy be cancelled and what is the refund policy?",
                "ground_truth": "The insured can cancel with 15 days written notice. Refund varies by time: 65% if within 3 months, 45% for 3-6 months, 20% for 6-9 months, 0% after 9 months. No refund if any claim has been made.",
                "source_section": "Cancellation",
                "difficulty": "medium",
                "expected_chunks": ["cancellation", "refund", "15 days", "notice"]
            }
        ]
        
        return questions
    
    def generate_policy2_questions(self) -> List[Dict]:
        """
        Generate gold standard Q&A pairs for ICICI Lombard Golden Shield
        Based on policy2.pdf content
        """
        
        questions = [
            {
                "id": "p2_q1",
                "category": "copayment",
                "question": "What is the base co-payment percentage for Golden Shield policy?",
                "ground_truth": "The Golden Shield policy has a mandatory 50% base co-payment, meaning the insured must pay 50% of admissible claim amount for each and every claim.",
                "source_section": "Base Co-payment, Section 12",
                "difficulty": "easy",
                "expected_chunks": ["base co-payment", "50%", "Golden Shield"]
            },
            {
                "id": "p2_q2",
                "category": "benefits",
                "question": "What is the cumulative bonus structure in this policy?",
                "ground_truth": "The policy provides 10% cumulative bonus of the Annual Sum Insured at the end of each claim-free policy year, with maximum accumulation up to 100% of the Annual Sum Insured.",
                "source_section": "Cumulative Bonus/Additional Sum Insured, Section 13",
                "difficulty": "medium",
                "expected_chunks": ["cumulative bonus", "10%", "100%", "claim-free"]
            },
            {
                "id": "p2_q3",
                "category": "coverage",
                "question": "Does the policy cover organ donor expenses?",
                "ground_truth": "Yes, living donor medical costs are covered up to INR 500,000 for harvesting of the donated organ, provided the organ donation complies with the Transplantation of Human Organs Act and the in-patient claim is accepted.",
                "source_section": "Donor Expenses, Section 6",
                "difficulty": "medium",
                "expected_chunks": ["donor", "INR 500,000", "organ", "harvesting"]
            },
            {
                "id": "p2_q4",
                "category": "ayush",
                "question": "Are AYUSH treatments covered under this policy?",
                "ground_truth": "Yes, in-patient AYUSH hospitalization is covered up to the Annual Sum Insured for treatment at AYUSH hospitals or day-care centres. Pre and post hospitalization expenses are excluded for AYUSH treatment.",
                "source_section": "In Patient AYUSH Hospitalization, Section 9",
                "difficulty": "medium",
                "expected_chunks": ["AYUSH", "covered", "in-patient", "excluded pre-post"]
            },
            {
                "id": "p2_q5",
                "category": "sub_limits",
                "question": "What are the sub-limits for treatment of cancer for a sum insured of 10L?",
                "ground_truth": "For sum insured of 10L/15L/20L, the sub-limit for cancer treatment (including chemo/radio/oral) is INR 3,50,000 per policy period.",
                "source_section": "Sub-limits applicable, Section 15",
                "difficulty": "hard",
                "expected_chunks": ["cancer", "sub-limit", "3,50,000", "chemo"]
            },
            {
                "id": "p2_q6",
                "category": "ambulance",
                "question": "What is the air ambulance coverage limit?",
                "ground_truth": "Air ambulance coverage varies by sum insured: INR 500,000 for 3.75L SI, INR 675,000 for 5.6L SI, and INR 750,000 for 7.5L and above sum insured options.",
                "source_section": "Air Ambulance, Section 9",
                "difficulty": "medium",
                "expected_chunks": ["air ambulance", "INR", "limit", "sum insured"]
            },
            {
                "id": "p2_q7",
                "category": "reset_benefit",
                "question": "How does the reset benefit work in this policy?",
                "ground_truth": "The reset benefit restores 100% of Annual Sum Insured unlimited times for future claims related to different illness/injury within the same policy year. It triggers only after the first claim and doesn't apply to same/related illness within 45 days.",
                "source_section": "Reset Benefit, Section 14",
                "difficulty": "hard",
                "expected_chunks": ["reset", "100%", "unlimited", "different illness"]
            },
            {
                "id": "p2_q8",
                "category": "exclusions",
                "question": "Is treatment for obesity covered under the policy?",
                "ground_truth": "Obesity treatment is excluded unless it meets specific conditions: BMI ≥40 or BMI ≥35 with severe co-morbidities (obesity-related cardiomyopathy, coronary heart disease, severe sleep apnea, or uncontrolled Type 2 diabetes) after failure of less invasive methods.",
                "source_section": "Obesity/Weight Control (Code- Excl06)",
                "difficulty": "hard",
                "expected_chunks": ["obesity", "BMI", "excluded", "conditions"]
            },
            {
                "id": "p2_q9",
                "category": "road_accident",
                "question": "What special benefit is available for road traffic accidents?",
                "ground_truth": "For road traffic accidents, the Annual Sum Insured is doubled if safety precautions were taken (seat-belt/helmet) as evidenced by police and hospital records. This enhanced sum is available only once per policy period and only after exhaustion of regular sum insured.",
                "source_section": "Enhanced Annual Sum insured for Road Traffic Accidents, Section 16",
                "difficulty": "medium",
                "expected_chunks": ["road traffic accident", "doubled", "safety", "helmet"]
            },
            {
                "id": "p2_q10",
                "category": "health_checkup",
                "question": "What preventive health check-up benefits are provided?",
                "ground_truth": "Insured persons can avail a preventive health check-up once a year per person at network providers on cashless basis. The benefit is provided via mobile application or health check-up coupons and doesn't impact the sum insured or cumulative bonus.",
                "source_section": "Preventive health check-up, Section 17",
                "difficulty": "easy",
                "expected_chunks": ["preventive", "health check-up", "once a year", "cashless"]
            }
        ]
        
        return questions
    
    def generate_cross_policy_questions(self) -> List[Dict]:
        """
        Generate comparative questions across both policies
        """
        
        questions = [
            {
                "id": "cross_q1",
                "category": "comparison",
                "question": "Which policy has a shorter pre-existing disease waiting period?",
                "ground_truth": "The ICICI Lombard Golden Shield has a shorter waiting period of 24 months for pre-existing diseases compared to Bajaj Allianz Global Health Care's 36 months waiting period.",
                "source_section": "Exclusions sections of both policies",
                "difficulty": "hard",
                "expected_chunks": ["pre-existing", "24 months", "36 months", "waiting period"]
            },
            {
                "id": "cross_q2",
                "category": "comparison",
                "question": "Do both policies cover mental illness treatment?",
                "ground_truth": "Yes, both policies cover mental illness treatment. Bajaj policy covers it under base plan while ICICI Golden Shield also provides coverage. Both require treatment in recognized psychiatric units and exclude out-patient mental illness treatment.",
                "source_section": "Mental Illness sections",
                "difficulty": "medium",
                "expected_chunks": ["mental illness", "both", "covered", "psychiatric"]
            }
        ]
        
        return questions
    
    def generate_complete_dataset(self, output_path: str = "gold_test_dataset.json"):
        """Generate complete gold dataset and save to JSON"""
        
        print("🔨 Generating Gold Test Dataset...")
        print("="*60)
        
        # Collect all questions
        policy1_qa = self.generate_policy1_questions()
        policy2_qa = self.generate_policy2_questions()
        cross_policy_qa = self.generate_cross_policy_questions()
        
        dataset = {
            "metadata": {
                "name": "Policy RAG Gold Test Dataset",
                "version": "1.0",
                "created_date": "2024-01-10",
                "total_questions": len(policy1_qa) + len(policy2_qa) + len(cross_policy_qa),
                "documents": [
                    "Bajaj Allianz Global Health Care (policy1.pdf)",
                    "ICICI Lombard Golden Shield (policy2.pdf)"
                ]
            },
            "policy1_bajaj": policy1_qa,
            "policy2_icici": policy2_qa,
            "cross_policy": cross_policy_qa
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n✅ Dataset Generated Successfully!")
        print(f"   Total Questions: {dataset['metadata']['total_questions']}")
        print(f"   - Policy 1 (Bajaj): {len(policy1_qa)}")
        print(f"   - Policy 2 (ICICI): {len(policy2_qa)}")
        print(f"   - Cross-Policy: {len(cross_policy_qa)}")
        print(f"\n📄 Saved to: {output_path}")
        print("\n📊 Question Distribution:")
        
        # Category distribution
        categories = {}
        all_questions = policy1_qa + policy2_qa + cross_policy_qa
        for q in all_questions:
            cat = q['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            print(f"   - {cat}: {count}")
        
        print("\n📈 Difficulty Distribution:")
        difficulties = {}
        for q in all_questions:
            diff = q.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        for diff, count in sorted(difficulties.items()):
            print(f"   - {diff}: {count}")
        
        return dataset


if __name__ == "__main__":
    generator = GoldDatasetGenerator()
    dataset = generator.generate_complete_dataset()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("1. Review and verify the ground truth answers")
    print("2. Run: python rag_evaluator.py")
    print("="*60)