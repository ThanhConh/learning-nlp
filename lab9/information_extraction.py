"""
Information Extraction Pipeline
================================

Pipeline: Raw text → NER → Entity extraction → Structured representation

Example:
    Input:  "Barack Obama visited Paris."
    Output: {
        "PERSON": ["Barack Obama"],
        "LOCATION": ["Paris"]
    }
"""

from transformers import pipeline
import pandas as pd
from typing import Dict, List


# ============================================================================
# STEP 1: Initialize NER Pipeline
# ============================================================================
# NOTE: Requires trained model and tokenizer from baseline_ner.py
# Uncomment and modify based on your trained model:
#
# ner_pipeline = pipeline(
#     "ner",
#     model=trainer.model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple"
# )


# ============================================================================
# STEP 2: Entity Extraction Function
# ============================================================================

def extract_entities(
    text: str,
    ner_pipeline,
    entity_types: List[str] = None
) -> Dict[str, List[str]]:
    """
    Extract named entities from text using NER pipeline.
    
    Args:
        text: Input text to process
        ner_pipeline: HuggingFace NER pipeline
        entity_types: List of entity types to extract (optional)
        
    Returns:
        Dictionary mapping entity types to lists of extracted entities
    """
    # Run NER
    ner_output = ner_pipeline(text)
    
    # Define entity mapping (model labels → output labels)
    entity_mapping = {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "MISC": "MISCELLANEOUS"
    }
    
    # Initialize result structure
    if entity_types:
        result = {entity_type: [] for entity_type in entity_types}
    else:
        result = {
            "PERSON": [],
            "ORGANIZATION": [],
            "LOCATION": [],
            "MISCELLANEOUS": []
        }
    
    # Extract and map entities
    for entity in ner_output:
        entity_type = entity["entity_group"]
        mapped_type = entity_mapping.get(entity_type)
        
        if mapped_type and mapped_type in result:
            entity_text = entity["word"].strip()
            # Avoid duplicates
            if entity_text not in result[mapped_type]:
                result[mapped_type].append(entity_text)
    
    return result


# ============================================================================
# STEP 3: Information Extraction Pipeline
# ============================================================================

class InformationExtractionPipeline:
    """
    Complete IE Pipeline: Raw text → NER → Structured data
    """
    
    def __init__(self, ner_pipeline, option="news"):
        """
        Initialize IE Pipeline.
        
        Args:
            ner_pipeline: HuggingFace NER pipeline
            option: "news", "academic", or "domain"
        """
        self.ner_pipeline = ner_pipeline
        self.option = option
        
        # Define entity types for each option
        self.option_configs = {
            "news": ["PERSON", "ORGANIZATION", "LOCATION"],
            "academic": ["PERSON", "ORGANIZATION", "LOCATION"],
            "domain": ["PERSON", "ORGANIZATION", "DISEASE", "DRUG"]
        }
        
        self.entity_types = self.option_configs.get(
            option, 
            ["PERSON", "ORGANIZATION", "LOCATION"]
        )
    
    def process_text(self, text: str) -> Dict:
        """
        Process single text through IE pipeline.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text and extracted entities
        """
        entities = extract_entities(
            text, 
            self.ner_pipeline, 
            self.entity_types
        )
        
        return {
            "text": text,
            "entities": entities
        }
    
    def process_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Process multiple texts through IE pipeline.
        
        Args:
            texts: List of input texts
            
        Returns:
            DataFrame with texts and extracted entities
        """
        results = []
        
        for text in texts:
            result = self.process_text(text)
            results.append({
                "Text": result["text"],
                "Entities": result["entities"]
            })
        
        return pd.DataFrame(results)


# ============================================================================
# STEP 4: Test Examples
# ============================================================================

# Option A — News Articles
news_articles = [
    "Barack Obama visited Paris.",
    "Microsoft announced a new product.",
    "Apple is headquartered in California.",
    "The United Nations held a meeting in New York.",
    "Elon Musk leads Tesla and SpaceX in California."
]

# Option B — Academic Texts
academic_texts = [
    "Professor John Smith from Stanford University published a paper.",
    "The research was conducted at MIT in Cambridge.",
    "Dr. Maria Garcia presented at Harvard University.",
    "Oxford University researchers collaborated with Cambridge."
]

# Option C — Domain-specific (Medical - example)
medical_texts = [
    "Patient presented with diabetes and was prescribed metformin.",
    "Dr. Johnson treated the patient at Mayo Clinic.",
    "The study examined COVID-19 treatment with remdesivir."
]


# ============================================================================
# STEP 5: Run Information Extraction
# ============================================================================

def run_information_extraction_demo(ner_pipeline):
    """
    Demonstrate information extraction on different options.
    
    Args:
        ner_pipeline: Trained NER pipeline
    """
    print("=" * 80)
    print("INFORMATION EXTRACTION PIPELINE DEMO")
    print("=" * 80)
    
    # Option A: News
    print("\n[OPTION A — NEWS ARTICLES]")
    print("-" * 80)
    news_ie = InformationExtractionPipeline(ner_pipeline, option="news")
    news_results = news_ie.process_batch(news_articles)
    print(news_results.to_string())
    
    # Option B: Academic
    print("\n[OPTION B — ACADEMIC TEXTS]")
    print("-" * 80)
    academic_ie = InformationExtractionPipeline(ner_pipeline, option="academic")
    academic_results = academic_ie.process_batch(academic_texts)
    print(academic_results.to_string())
    
    # Option C: Domain (if available)
    print("\n[OPTION C — DOMAIN-SPECIFIC (MEDICAL)]")
    print("-" * 80)
    print("Note: Requires specialized NER model trained on medical entities")
    print("Showing structure only - results may not be accurate without proper model")
    domain_ie = InformationExtractionPipeline(ner_pipeline, option="domain")
    # Uncomment if you have a domain-specific model:
    # medical_results = domain_ie.process_batch(medical_texts)
    # print(medical_results.to_string())
    
    return news_results, academic_results


# ============================================================================
# STEP 6: Visualization Helper
# ============================================================================

def display_structured_output(text: str, entities: Dict[str, List[str]]):
    """
    Pretty print the IE result showing the transformation:
    Raw text → Structured representation
    """
    print("\n" + "=" * 80)
    print("UNSTRUCTURED TEXT → STRUCTURED DATA")
    print("=" * 80)
    print(f"\nInput (Raw Text):")
    print(f'  "{text}"')
    print(f"\nOutput (Structured):")
    print("  {")
    for entity_type, entities_list in entities.items():
        if entities_list:  # Only show non-empty
            print(f'    "{entity_type}": {entities_list},')
    print("  }")
    print("=" * 80)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # NOTE: This requires a trained NER model from baseline_ner.py
    # Uncomment and run after training your model:
    
    from baseline_ner import trainer, tokenizer
    
    # Initialize NER pipeline
    ner_pipeline = pipeline(
        "ner",
        model=trainer.model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    
    # Run demo
    news_results, academic_results = run_information_extraction_demo(ner_pipeline)
    
    # Show example transformation
    example_text = "Barack Obama visited Paris."
    example_entities = extract_entities(example_text, ner_pipeline)
    display_structured_output(example_text, example_entities)
    
    print("\nInformation Extraction module loaded.")
    print("Import and use after training your NER model in baseline_ner.py")
