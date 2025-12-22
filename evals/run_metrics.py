import json
import os
from pathlib import Path

from langchain_openai import ChatOpenAI


from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
llm = ChatOpenAI(
            model="fireworks/gpt-oss-20b",
            api_key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
            temperature=0
        )


REPORT_PATH = Path("evals/results/latest_report.json")


def run_metrics():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    rows = []
    for item in report["results"]:
        if not item["answerable"]:
            continue  # RAGAS is for answerable questions

        rows.append({
            "question": item["question"],
            "answer": item["generated_answer"],
            "contexts": item["contexts"],
            "ground_truth": item["expected_answer"]
        })

    rows = rows[:3]

    dataset = Dataset.from_list(rows)

    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm
    )

    print("\nRAG Evaluation Metrics:\n")
    print(results)


if __name__ == "__main__":
    run_metrics()
