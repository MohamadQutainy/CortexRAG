"""
============================================================================
Sham Real Estate Dataset Generator
============================================================================
Creates a rich, interconnected synthetic dataset for demonstrating RAG capabilities.
Includes smart noise injection to test RAG robustness.

Usage:
    python generate_dataset.py --mode fast
    python generate_dataset.py --mode hybrid --seed 42
    python generate_dataset.py --mode llm / python generate_dataset.py --mode llm --seed 99 if i would a different dataset - just change the seed

Modes:
- fast: Procedural templating only (instant, free, good for structural testing).
- hybrid: Procedural metadata + LLM generated rich descriptions (Recommended).
- llm: Full LLM generation for every file (Slow, costs API credits).
============================================================================
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from litellm import completion  # pyright: ignore[reportMissingImports]

load_dotenv(override=True)

# --- Configuration ---
DATASET_PATH = Path("real_estate_dataset")
MODEL_NAME = "gpt-4o-mini"

# Limits to keep total files between 70-120
COUNTS = {
    "properties": 25,
    "clients": 20,
    "contracts": 20,
    "transactions": 20,
    "reports": 10,
    "emails": 15,
}

# --- Shared Dictionaries for Interconnection ---
DB = {
    "properties": [],
    "clients": [],
    "contracts": [],
    "transactions": []
}

LOCATIONS = [
    ("Damascus", "Al-Malki"), ("Damascus", "Kfar Souseh"), ("Damascus", "Abu Rummaneh"),
    ("Aleppo", "Al-Aziziyeh"), ("Aleppo", "Al-Mogambo"),
    ("Homs", "Al-Inshaat"),
    ("Latakia", "Project 10"), ("Tartus", "Corniche")
]

TYPES = ["Apartment", "Villa", "Office", "Commercial Land"]

# --- Smart Noise Generator ---
def inject_noise(text: str, file_type: str) -> str:
    """Injects subtle inconsistencies 10% of the time."""
    if random.random() > 0.15:
        return text
    
    noise_type = random.choice(["missing_field", "format_variation", "contradiction"])
    # Simple manipulation
    lines = text.split("\n")
    if noise_type == "missing_field":
        lines = [l for l in lines if not l.startswith("- **Size**:")]
    elif noise_type == "format_variation":
        lines = [l.replace("- **", "* ") for l in lines]
    elif noise_type == "contradiction":
        # Introduce a subtle contradiction replacing a number
        lines = [l.replace("USD", "SYP") if random.random() > 0.8 else l for l in lines]

    return "\n".join(lines)


# --- LLM Helper ---
def ask_llm(prompt: str, sys_msg: str = "You are a professional real estate dataset generator.") -> str:
    """Send requests to the LLM."""
    try:
        response = completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Content generation failed due to API error."


# --- Category Generators ---

def generate_properties(mode: str):
    print("Generating Properties...")
    path = DATASET_PATH / "properties"
    for i in range(1, COUNTS["properties"] + 1):
        prop_id = f"PROP-{i:03d}"
        city, district = random.choice(LOCATIONS)
        prop_type = random.choice(TYPES)
        price = random.randint(50, 500) * 1000
        size = random.randint(80, 600)
        
        DB["properties"].append({"id": prop_id, "city": city, "price": price, "type": prop_type})

        base_md = f"""# Property Information: {prop_id}

## Core Details
- **ID**: {prop_id}
- **Location**: {city}, {district}
- **Type**: {prop_type}
- **Price**: ${price:,} USD
- **Size**: {size} sqm

## Detailed Description
"""
        description = f"A beautiful {size} sqm {prop_type.lower()} located in the prestigious {district} district of {city}. Ideal for investment or residential use. The legal status is fully clear, ready for immediate title deed transfer. Close to local amenities."

        if mode in ["hybrid", "llm"]:
            prompt = f"Write a professional, detailed 2-paragraph real estate listing description for a {prop_type} in {district}, {city} costing ${price}. RAG friendly formatting."
            description = ask_llm(prompt)

        content = base_md + description + "\n\n## Features\n- 24/7 Security\n- Renovated in 2022\n- Clear Legal Title"
        
        if mode == "llm":
            content = ask_llm(f"Rewrite this property markdown file into a more varied and professional RAG-friendly document, but keep the headers:\n{content}")

        content = inject_noise(content, "property")
        with open(path / f"{prop_id}.md", "w", encoding="utf-8") as f:
            f.write(content)


def generate_clients(mode: str):
    print("Generating Clients...")
    path = DATASET_PATH / "clients"
    names = ["Ahmad Al-Saleh", "Youssef Ibrahim", "Sarah Khalil", "Rami Nader", "Lina Haddad", "Omar Zidan", "Tarek Fawaz", "Maya Kanaan", "Hassan Jaber", "Nour Al-Din", "Fadi Abboud", "Zeina Masri", "Sami Khoury"]
    
    for i in range(1, COUNTS["clients"] + 1):
        client_id = f"CLI-{i:03d}"
        name = random.choice(names) + f" (ID: {i})"
        budget = random.randint(100, 800) * 1000
        
        DB["clients"].append({"id": client_id, "name": name, "budget": budget})

        base_md = f"""# Client Profile: {name}

## Profile Summary
- **Client ID**: {client_id}
- **Name**: {name}
- **Budget**: Up to ${budget:,} USD
- **Preferences**: Looking for properties in Damascus or coastal areas.

## Interaction History & Behavioral Notes
"""
        history = f"- **2025-01-10**: Client called inquiring about villas.\n- **2025-02-15**: Sent list of properties via email.\n- **Behavior**: Client is a hard negotiator, requires fast legal processing."

        if mode in ["hybrid", "llm"]:
            prompt = f"Write a detailed interaction history and behavioral profile (in bullet points) for a high-net-worth real estate client named {name} looking in Syria. RAG friendly."
            history = ask_llm(prompt)

        content = base_md + history
        content = inject_noise(content, "client")
        with open(path / f"{client_id}.md", "w", encoding="utf-8") as f:
            f.write(content)


def generate_contracts_and_transactions(mode: str):
    print("Generating Contracts & Transactions...")
    c_path = DATASET_PATH / "contracts"
    t_path = DATASET_PATH / "transactions"
    
    for i in range(1, COUNTS["contracts"] + 1):
        cont_id = f"CONT-{i:03d}"
        trans_id = f"TRANS-{i:03d}"
        
        # Link existing entities
        prop = random.choice(DB["properties"])
        client = random.choice(DB["clients"])
        
        DB["contracts"].append({"id": cont_id, "prop_id": prop["id"], "client_id": client["id"]})

        # --- Contract Generation ---
        cont_md = f"""# Legal Contract: {cont_id}

## Contract Details
- **Contract Type**: Sale Agreement
- **Property ID**: {prop['id']} ({prop['city']})
- **Client ID**: {client['id']} ({client['name']})
- **Agreed Value**: ${prop['price'] * 0.95:,.0f} USD (Post-negotiation)
- **Status**: Executed

## Legal Clauses & Obligations
1. **Payment Terms**: 50% upfront, 50% upon deed transfer.
2. **Liabilities**: Sham Real Estate Group is not liable for delayed municipal approvals.
3. **Condition**: Property sold "as is".
"""
        if mode == "llm":
            cont_md = ask_llm(f"Rewrite this real estate contract summary to be more formal and expand on the legal clauses. RAG friendly formatting:\n{cont_md}")
        
        cont_md = inject_noise(cont_md, "contract")
        with open(c_path / f"{cont_id}.md", "w", encoding="utf-8") as f:
            f.write(cont_md)

        # --- Transaction Generation ---
        trans_md = f"""# Transaction Log: {trans_id}

## Transaction Overview
- **Linked Contract**: {cont_id}
- **Property**: {prop['id']}
- **Buyer**: {client['id']}

## Negotiation & Closing Process
The client {client['name']} initially offered ${client['budget']:,} USD. After 3 rounds of negotiation regarding the property in {prop['city']}, we settled at a 5% discount from the listed price of ${prop['price']:,}. M. Nader managed the legal paperwork. Deal closed successfully.
"""
        if mode in ["hybrid", "llm"]:
            trans_md = ask_llm(f"Write a detailed internal transaction log explaining the negotiation for property {prop['id']} sold to {client['name']}. Mention the contract {cont_id}. Use headers. RAG friendly.")
            
        with open(t_path / f"{trans_id}.md", "w", encoding="utf-8") as f:
            f.write(trans_md)


def generate_reports_and_emails(mode: str):
    print("Generating Reports & Emails...")
    r_path = DATASET_PATH / "reports"
    e_path = DATASET_PATH / "emails"

    # Reports
    for i in range(1, COUNTS["reports"] + 1):
        rep_id = f"REP-{i:03d}"
        topic = random.choice(["Damascus Market Trends", "Coastal Investment ROI", "Legal Tax Changes 2025", "Aleppo Reconstruction Analysis"])
        
        rep_md = f"""# Sham Real Estate Report: {rep_id}
## Topic: {topic}

## Executive Summary
This report outlines the latest insights regarding {topic}. 
- Market volume increased by 12%.
- Main drivers: Expatriate investments and new zoning laws.

## Detailed Analysis
The sector is experiencing shifts. Properties in central areas retain high value. Legal processes have slowed down municipal deed transfers.
"""
        if mode in ["hybrid", "llm"]:
            rep_md = ask_llm(f"Write a highly professional market analysis report for '{topic}' in the Syrian real estate sector. Include headers for Executive Summary, Key Drivers, and Conclusion. RAG friendly.")

        with open(r_path / f"{rep_id}.md", "w", encoding="utf-8") as f:
            f.write(rep_md)

    # Emails
    for i in range(1, COUNTS["emails"] + 1):
        email_id = f"EMAIL-{i:03d}"
        prop_id = f"PROP-{random.randint(1, COUNTS['properties']):03d}"
        
        email_md = f"""# Internal Email Communication: {email_id}

- **From**: Sales Department
- **To**: Legal Team
- **Subject**: Urgent Deed Transfer for {prop_id}

## Message Thread
Hello Legal,
Can we expedite the municipal approval for {prop_id}? The buyer is threatening to pull out due to the delay. Let me know the exact fee needed to fast-track this.
Thanks,
Sales Team.
"""
        if mode == "llm":
            email_md = ask_llm(f"Write a realistic corporate email thread regarding a delay in selling property {prop_id} in Syria. Format with From, To, Subject, and the thread content. RAG friendly.")

        email_md = inject_noise(email_md, "email")
        with open(e_path / f"{email_id}.md", "w", encoding="utf-8") as f:
            f.write(email_md)


def generate_qa_evaluation():
    print("Generating Evaluation Dataset...")
    qa_path = DATASET_PATH / "evaluation" / "qa_eval.jsonl"
    
    # We will pick 10 random valid contracts to ensure truth
    valid_contracts = random.sample(DB["contracts"], min(10, len(DB["contracts"])))
    
    questions = []
    category = "Contract Knowledge"
    
    for c in valid_contracts:
        client = next((cl for cl in DB["clients"] if cl["id"] == c["client_id"]), None)
        prop = next((p for p in DB["properties"] if p["id"] == c["prop_id"]), None)
        
        if client and prop:
            q1 = {
                "question": f"Which client purchased property {prop['id']}?",
                "reference_answer": f"Client {client['name']} ({client['id']}) purchased property {prop['id']} under contract {c['id']}.",
                "category": category,
                "keywords": [client['id'], client['name'].split()[0], prop['id']]
            }
            questions.append(q1)

            q2 = {
                "question": f"In which city is the property linked to contract {c['id']} located?",
                "reference_answer": f"The property {prop['id']} linked to contract {c['id']} is located in {prop['city']}.",
                "category": "Location Knowledge",
                "keywords": [prop['city'], prop['id'], c['id']]
            }
            questions.append(q2)

    with open(qa_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Generated {len(questions)} evaluation QA pairs.")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Generate Sham Real Estate Dataset")
    parser.add_argument("--mode", choices=["fast", "hybrid", "llm"], default="hybrid",
                        help="fast: procedural only | hybrid: procedural + LLM descriptions | llm: full LLM text")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"--- Starting Generation (Mode: {args.mode}, Seed: {args.seed}) ---")

    # Build Dirs
    folders = ["properties", "clients", "contracts", "transactions", "reports", "emails", "evaluation"]
    for folder in folders:
        os.makedirs(DATASET_PATH / folder, exist_ok=True)

    generate_properties(args.mode)
    generate_clients(args.mode)
    generate_contracts_and_transactions(args.mode)
    generate_reports_and_emails(args.mode)
    generate_qa_evaluation()

    print("--- Dataset Generation Complete! ---")
    print(f"Total files stored in: {DATASET_PATH.resolve()}")

if __name__ == "__main__":
    main()
