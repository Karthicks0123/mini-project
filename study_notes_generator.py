"""
Smart Study Notes Generator using Hugging Face Transformers
===========================================================
This module provides a complete pipeline to summarize academic and technical text,
compute word count statistics, calculate percentage compression, and format summaries
into clean, high-yield study notes.
"""

import sys
import re
import warnings
from typing import Dict, Any, List, Optional

# Suppress unnecessary hub warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class SmartStudyNotesGenerator:
    """
    A smart study notes assistant powered by Hugging Face Transformers.
    Utilizes an encoder-decoder architecture (e.g., DistilBART / BART / T5)
    to perform abstractive summarization and analytical reporting.
    """

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the tokenizer and model.
        
        Args:
            model_name: Hugging Face model repository or local path.
                        Defaults to 'sshleifer/distilbart-cnn-12-6', an efficient 
                        distilled BART model optimized for text summarization.
        """
        print(f"Loading summarization model '{model_name}'...")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("Model and tokenizer loaded successfully!\n")

    @staticmethod
    def count_words(text: str) -> int:
        """
        Calculate word count by splitting text on whitespace.
        
        Args:
            text: Input string.
            
        Returns:
            Number of words.
        """
        if not text or not text.strip():
            return 0
        return len(text.strip().split())

    @staticmethod
    def calculate_percentage_reduction(original_count: int, summary_count: int) -> float:
        r"""
        Calculate the percentage reduction in text length:
        $$\text{Percentage Reduction} = \left( \frac{\text{Original Word Count} - \text{Summary Word Count}}{\text{Original Word Count}} \right) \times 100$$
        
        Args:
            original_count: Word count of the original passage.
            summary_count: Word count of the generated summary.
            
        Returns:
            Reduction percentage as a float rounded to 2 decimal places.
        """
        if original_count <= 0:
            return 0.0
        reduction = ((original_count - summary_count) / original_count) * 100.0
        return max(0.0, round(reduction, 2))

    @staticmethod
    def extract_key_points(summary_text: str) -> List[str]:
        """
        Split summary text into clean, readable bullet points.
        """
        # Split by sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', summary_text.strip())
        points = [s.strip() for s in sentences if len(s.strip()) > 8]
        return points

    def generate_study_notes(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Summarize the given text and compute quantitative and qualitative analytics.
        
        Args:
            text: Input text to summarize.
            max_length: Maximum token length of generated summary.
            min_length: Minimum token length of generated summary.
            
        Returns:
            A dictionary containing:
              - original_text
              - summary_text
              - key_points (list of bullet sentences)
              - original_words (int)
              - summary_words (int)
              - reduction_percentage (float)
        """
        cleaned_text = text.strip()
        orig_words = self.count_words(cleaned_text)

        if orig_words == 0:
            raise ValueError("Input text cannot be empty.")

        # Dynamically adapt max/min lengths for shorter passages
        if max_length is None:
            max_length = min(140, max(30, int(orig_words * 0.7)))
        if min_length is None:
            min_length = min(25, max(10, int(orig_words * 0.25)))

        if min_length >= max_length:
            min_length = max(5, max_length - 10)

        # Tokenize input text
        inputs = self.tokenizer(
            cleaned_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate summary tokens using beam search
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        # Decode tokens to human-readable string
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        # Clean extra spaces before punctuation (e.g. "word ." -> "word.")
        summary_text = re.sub(r'\s+([,.:;!?])', r'\1', summary_text)

        summary_words = self.count_words(summary_text)
        reduction_pct = self.calculate_percentage_reduction(orig_words, summary_words)
        key_points = self.extract_key_points(summary_text)

        return {
            "original_text": cleaned_text,
            "summary_text": summary_text,
            "key_points": key_points,
            "original_words": orig_words,
            "summary_words": summary_words,
            "reduction_percentage": reduction_pct
        }

    def print_study_notes(self, results: Dict[str, Any], title: str = "Smart Study Notes"):
        """
        Nicely format and display the study notes and statistics.
        """
        border = "=" * 70
        divider = "-" * 70
        print(f"\n{border}")
        print(f" 📚 {title.upper()}")
        print(border)
        
        print("\n📝 [EXECUTIVE SUMMARY]:")
        print(f"   {results['summary_text']}")

        if results.get("key_points"):
            print("\n📌 [KEY TAKEAWAYS / FLASHCARD BULLETS]:")
            for idx, pt in enumerate(results["key_points"], start=1):
                print(f"   {idx}. {pt}")

        print(f"\n{divider}")
        print(" 📊 [QUANTITATIVE REDUCTION METRICS]:")
        print(f"   • Original Word Count:  {results['original_words']} words")
        print(f"   • Summary Word Count:   {results['summary_words']} words")
        print(f"   • Words Saved:          {results['original_words'] - results['summary_words']} words")
        print(f"   • Percentage Reduction: {results['reduction_percentage']}%")
        print(f"{border}\n")


def interactive_cli():
    """
    Interactive command-line interface for generating study notes.
    """
    generator = SmartStudyNotesGenerator()

    sample_passage = (
        "Photosynthesis is a biological process used by plants, algae, and certain bacteria "
        "to convert light energy into chemical energy. This chemical energy is stored in carbohydrate "
        "molecules, such as sugars, which are synthesized from carbon dioxide and water. In most cases, "
        "oxygen is released as a waste product. Photosynthesis is largely responsible for producing "
        "and maintaining the oxygen content of the Earth's atmosphere, and supplies most of the biological "
        "energy necessary for complex life on Earth. Without photosynthesis, the global carbon cycle would "
        "collapse, depleting the foundational food supply for terrestrial and aquatic ecosystems."
    )

    print("Options:")
    print("1. Run with built-in sample passage (Biology: Photosynthesis)")
    print("2. Enter custom study text")
    
    choice = input("\nEnter choice (1 or 2, default is 1): ").strip()
    
    if choice == "2":
        print("\nEnter or paste your passage below (Type 'END' on a new line when finished):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            except EOFError:
                break
        user_text = "\n".join(lines).strip()
        if not user_text:
            print("No text entered. Using sample passage instead.")
            user_text = sample_passage
    else:
        user_text = sample_passage

    print("\nGenerating study notes...")
    results = generator.generate_study_notes(user_text)
    generator.print_study_notes(results, title="Generated Study Notes")


if __name__ == "__main__":
    interactive_cli()
