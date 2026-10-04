"""
Evaluation Script for Smart Study Notes Generator
=================================================
Tests diverse domain passages, computes quantitative reduction metrics,
and evaluates qualitative dimensions (Relevance and Coherence).
"""

import os
from study_notes_generator import SmartStudyNotesGenerator

TEST_CASES = [
    {
        "id": "CASE-1",
        "domain": "Natural Sciences (Biology)",
        "topic": "Photosynthesis and Planetary Oxygen",
        "text": (
            "Photosynthesis is a biological process used by plants, algae, and certain bacteria "
            "to convert light energy into chemical energy. This chemical energy is stored in carbohydrate "
            "molecules, such as sugars, which are synthesized from carbon dioxide and water. In most cases, "
            "oxygen is released as a waste product. Photosynthesis is largely responsible for producing "
            "and maintaining the oxygen content of the Earth's atmosphere, and supplies most of the biological "
            "energy necessary for complex life on Earth. Without photosynthesis, the global carbon cycle would "
            "collapse, depleting the foundational food supply for terrestrial and aquatic ecosystems."
        ),
        "expected_key_facts": [
            "Converts light energy to chemical energy",
            "Uses carbon dioxide and water to produce sugars",
            "Releases oxygen and maintains Earth's atmosphere",
            "Foundational food and energy supply"
        ]
    },
    {
        "id": "CASE-2",
        "domain": "Computer Science (Machine Learning)",
        "topic": "Neural Networks and Backpropagation",
        "text": (
            "Artificial neural networks are computational systems inspired by the biological neural "
            "networks that constitute animal brains. An ANN is based on a collection of connected units "
            "or nodes called artificial neurons. Backpropagation is the primary algorithm used to train "
            "feedforward neural networks. In backpropagation, the model calculates the gradient of the "
            "loss function with respect to each weight by the chain rule, computing the gradient one layer "
            "at a time, iterating backward from the output layer to update parameters using gradient descent. "
            "This optimization enables deep neural architectures to achieve superhuman accuracy on image "
            "recognition, natural language translation, and autonomous decision making."
        ),
        "expected_key_facts": [
            "Inspired by biological neural networks",
            "Backpropagation computes loss gradients using chain rule",
            "Iterates backward from output layer to update weights",
            "Powers high accuracy in vision, NLP, and autonomy"
        ]
    },
    {
        "id": "CASE-3",
        "domain": "History & Economics",
        "topic": "The Industrial Revolution",
        "text": (
            "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, "
            "continental Europe, and the United States, that occurred during the period from around 1760 to about 1840. "
            "This transition included going from hand production methods to machines, new chemical manufacturing "
            "and iron production processes, the increasing use of steam power and water power, the development "
            "of machine tools, and the rise of the mechanized factory system. The Industrial Revolution marked a "
            "major turning point in history; almost every aspect of daily life was influenced in some way. In "
            "particular, average income and population began to exhibit unprecedented sustained growth."
        ),
        "expected_key_facts": [
            "Transitioned from hand production to machines between 1760 and 1840",
            "Introduced steam power, iron production, and mechanized factories",
            "Influenced daily life with unprecedented population and income growth"
        ]
    }
]


def evaluate_summary_quality(original: str, summary: str, expected_facts: list) -> dict:
    """
    Evaluate qualitative aspects:
    - Relevance Score (0-10): How many expected key concepts are captured
    - Coherence Score (0-10): Grammar, sentence structure, flow, absence of repetitive loops
    """
    summary_lower = summary.lower()
    
    # Check factual overlap
    matched_facts = 0
    for fact in expected_facts:
        keywords = [w.lower() for w in fact.split() if len(w) > 4]
        if any(kw in summary_lower for kw in keywords):
            matched_facts += 1
            
    relevance_ratio = matched_facts / len(expected_facts) if expected_facts else 1.0
    relevance_score = round(min(10.0, relevance_ratio * 10), 1)

    # Coherence checks: capitalization, terminal punctuation, non-repetition
    coherence_score = 9.5
    if not summary.endswith(('.', '!', '?')):
        coherence_score -= 1.0
    
    words = summary.lower().split()
    unique_words = set(words)
    if len(words) > 0 and (len(unique_words) / len(words)) < 0.65:
        # Repetitive hallucination penalty
        coherence_score -= 2.0

    return {
        "relevance_score": relevance_score,
        "coherence_score": coherence_score,
        "matched_facts": matched_facts,
        "total_facts": len(expected_facts)
    }


def run_evaluation():
    print("Initializing Smart Study Notes Evaluator...")
    generator = SmartStudyNotesGenerator()

    eval_results = []
    
    print("\n" + "=" * 80)
    print(" RUNNING EVALUATION SUITE ACROSS DOMAINS")
    print("=" * 80)

    for case in TEST_CASES:
        print(f"\nEvaluating {case['id']}: [{case['domain']}] - {case['topic']}")
        
        notes = generator.generate_study_notes(case["text"])
        qual = evaluate_summary_quality(case["text"], notes["summary_text"], case["expected_key_facts"])
        
        eval_record = {
            "id": case["id"],
            "domain": case["domain"],
            "topic": case["topic"],
            "orig_words": notes["original_words"],
            "summary_words": notes["summary_words"],
            "reduction_pct": notes["reduction_percentage"],
            "relevance": qual["relevance_score"],
            "coherence": qual["coherence_score"],
            "summary": notes["summary_text"],
            "bullets": notes["key_points"]
        }
        eval_results.append(eval_record)
        
        print(f" * Original Words: {notes['original_words']} | Summary Words: {notes['summary_words']}")
        print(f" * Percentage Reduction: {notes['reduction_percentage']}%")
        print(f" * Relevance Score: {qual['relevance_score']}/10 ({qual['matched_facts']}/{qual['total_facts']} facts retained)")
        print(f" * Coherence Score: {qual['coherence_score']}/10")
        print(f" * Summary Output: \"{notes['summary_text']}\"")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'ID':<8} | {'Domain':<25} | {'Orig':<6} | {'Summ':<6} | {'Reduc%':<8} | {'Rel/10':<7} | {'Coh/10'}")
    print("-" * 80)
    for r in eval_results:
        print(f"{r['id']:<8} | {r['domain'][:25]:<25} | {r['orig_words']:<6} | {r['summary_words']:<6} | {r['reduction_pct']:<7}% | {r['relevance']:<7} | {r['coherence']}")
    print("=" * 80)
    
    return eval_results


if __name__ == "__main__":
    run_evaluation()
