import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import random
from loguru import logger
from datetime import datetime

class DataGenerator:
    """
    Generates synthetic data for sentiment analysis using a specified causal language model
    with a few-shot prompting strategy.
    """
    def __init__(self, model_name: str = 'google/gemma-3-1b-it'):
        logger.info(f"Initializing generator with model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            token=True
        )
        logger.info("Model and tokenizer loaded successfully.")

        self.few_shot_examples = [
            ("The GeoSolutions technology will leverage Benefon 's GPS solutions by providing Location Based Search Technology , a Communities Platform , location relevant multimedia content and a new and powerful commercial model .", "positive"),
            ("$ESI on lows, down .50 to $2.50 BK a real possibility", "negative"),
            ("For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m .", "positive"),
            ("According to the Finnish-Russian Chamber of Commerce , all the major construction companies of Finland are operating in Russia .", "neutral"),
            ("The Swedish buyout firm has sold its remaining 22.4 percent stake , almost eighteen months after taking the company public in Finland .", "neutral"),
            ("$SPY wouldn't be surprised to see a green close", "positive"),
            ("Shell's $70 Billion BG Deal Meets Shareholder Skepticism", "negative"),
            ("SSH COMMUNICATIONS SECURITY CORP STOCK EXCHANGE RELEASE OCTOBER 14 , 2008 AT 2:45 PM The Company updates its full year outlook and estimates its results to remain at loss for the full year .", "negative"),
            ("Kone 's net sales rose by some 14 % year-on-year in the first nine months of 2008 .", "positive"),
            ("The Stockmann department store will have a total floor space of over 8,000 square metres and Stockmann 's investment in the project will have a price tag of about EUR 12 million .", "neutral"),
            ("Circulation revenue has increased by 5 % in Finland and 4 % in Sweden in 2008 .", "positive"),
            ("$SAP Q1 disappoints as #software licenses down. Real problem? #Cloud growth trails $MSFT $ORCL $GOOG $CRM $ADBE https://t.co/jNDphllzq5", "negative"),
            ("The subdivision made sales revenues last year of EUR 480.7 million EUR 414.9 million in 2008 , and operating profits of EUR 44.5 million EUR 7.4 million .", "positive"),
            ("Viking Line has canceled some services .", "neutral")
        ]

    def _determine_generation_counts(self, examples_per_sentiment, distribution_from_file, total_examples):
        if examples_per_sentiment:
            sentiments = ['positive', 'neutral', 'negative']
            return {s: examples_per_sentiment for s in sentiments}
        
        if distribution_from_file and total_examples:
            logger.info(f"Reading sentiment distribution from {distribution_from_file}")
            dist_df = pd.read_csv(distribution_from_file)
            distribution = dist_df['Sentiment'].value_counts(normalize=True)
            counts = {s: int(total_examples * dist) for s, dist in distribution.items()}
            current_total = sum(counts.values())
            if current_total < total_examples:
                most_frequent_sentiment = distribution.idxmax()
                counts[most_frequent_sentiment] += total_examples - current_total
            logger.info(f"Target generation counts: {counts}")
            return counts
        
        raise ValueError("Invalid arguments.")

    def _generate_one_example(self, sentiment: str, lang: str) -> str:
        # Build the few-shot prompt
        prompt = f"Generate a short, single sentence in {lang} expressing a given sentiment related to finance or stock markets. Follow the examples below.\n\n"
        
        # Select 2 random examples for the prompt
        shot_examples = random.sample(self.few_shot_examples, 2)
        for ex_sentence, ex_sentiment in shot_examples:
            prompt += f"Sentence: {ex_sentence}\nSentiment: {ex_sentiment}\n\n"

        # Add the final instruction for the model to complete
        prompt += f"Sentence: " # The model should generate the sentence here
        final_prompt_part = f"\nSentiment: {sentiment}"
        prompt += final_prompt_part

        input_ids = self.tokenizer(prompt.replace(final_prompt_part, ''), return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=60,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the output to get only the newly generated sentence
        base_prompt = prompt.replace(final_prompt_part, '').replace('<bos>', '')
        generated_sentence = full_text.replace(base_prompt, '').strip()
        # Stop at the first newline to avoid capturing extra generated content
        generated_sentence = generated_sentence.split('\n')[0]

        return generated_sentence

    def generate(self, name: str, examples_per_sentiment: int = None, distribution_from_file: str = None, total_examples: int = None, lang: str = "en"):
        counts = self._determine_generation_counts(examples_per_sentiment, distribution_from_file, total_examples)
        
        results = []
        for sentiment, count in counts.items():
            logger.info(f"Generating {count} examples for sentiment: '{sentiment}' in language: {lang}")
            for i in range(count):
                sentence = self._generate_one_example(sentiment, lang)
                results.append({"Sentence": sentence, "Sentiment": sentiment})
                logger.info(f"({i+1}/{count}) Generated for '{sentiment}': {sentence}")

        df = pd.DataFrame(results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = os.path.join("artifacts", "new_data", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved {len(df)} examples to {file_path}")