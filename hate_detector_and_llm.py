import argparse
from typing import List, Dict, Any

import pandas as pd

from generate_prediction import predict
import torch
import logging
from hugging_face_configuration import hugging_face_login

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

hugging_face_login()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class HateSpeechDetector:
    def __init__(self, model_name: str, checkpoint_path: str = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(device)
        self.model.eval()

    def make_prediction(self, text: str) -> Dict[str, Any]:
        return predict(text, model=self.model, threshold=0.5, tokenizer=self.tokenizer, device=device)


class LLMJudge:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            do_sample=True
        )

    def judge(self, classification_result: Dict[str, Any]) -> str:
        prompt = (
            f"### Role: Hate speech detection expert\n\n"
            f"### Task: Review the classification below:\n"
            f"- Text: {classification_result['text']}\n"
            f"- Prediction: {classification_result['prediction']}\n"
            f"- Hate Probability: {classification_result['hate_probability']:.4f}\n\n"
            f"### Instructions:\n"
            f"1. Assess accuracy of prediction\n"
            f"2. Consider context, tone, and subtle hate\n"
            f"3. Explain your reasoning\n"
            f"4. End with: **Final Verdict: HATE** or **Final Verdict: NOT HATE**\n\n"
            f"### Analysis:"
        )

        try:
            llm_response = self.pipeline(
                prompt,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9
            )[0]["generated_text"]

            analysis_start = llm_response.find("### Analysis:")
            response_from_analysis = llm_response[
                                     analysis_start:].strip() if analysis_start != -1 else llm_response.strip()

            if "**Final Verdict:" in response_from_analysis:
                verdict = response_from_analysis.split("**Final Verdict:")[1].split("**")[0].strip()
                response_from_analysis = response_from_analysis.replace(f"**Final Verdict: {verdict}**", "")
                response_from_analysis += f"\n\nFinal Verdict: {verdict}"

            return response_from_analysis.strip()
        except Exception as e:
            logger.error(f"LLM judgment failed: {str(e)}")
            return "LLM evaluation unavailable at this time."


class HateSpeechDetectionSystem:
    def __init__(self, detector_model_name: str, checkpoint_path: str = None):
        self.detector = HateSpeechDetector(detector_model_name, checkpoint_path)
        self.llm_judge = LLMJudge()

    def evaluate_test_cases(self, test_texts: List[str] = None):
        logger.info(f"\nEvaluating {len(test_texts)} test cases...")

        test_texts = pd.read_csv("additional_text_cases.csv")

        for i, text in enumerate(test_texts, 1):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Test Case {i}/{len(test_texts)}")
            logger.info(f"Text: {text}")

            try:
                result = self.detector.make_prediction(text)
                logger.info(f"Classification: {result['prediction']}")
                logger.info(f"Confidence: {result['confidence']:.2%}")
                logger.info(f"Hate Probability: {result['hate_probability']:.4f}")

                # logger.info("\nLLM Evaluation:")
                # explanation = self.llm_judge.judge(result)
                # logger.info(explanation)

            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="GroNLP/hateBERT")
    parser.add_argument("--checkpoint", default="model_epoch_20.pt")
    args = parser.parse_args()

    try:
        system = HateSpeechDetectionSystem(args.model_name, args.checkpoint)
        system.evaluate_test_cases()
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")