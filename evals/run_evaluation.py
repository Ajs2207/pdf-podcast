import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import json
from datetime import datetime, timezone

from src.agents.rag_agent import RAGAgent


DATASET_PATH = Path("evals/datasets/rag_eval.json")
OUTPUT_PATH = Path("evals/results/latest_report.json")


def run_evaluation():
    agent = RAGAgent(use_memory=False)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    results = []

    for sample in eval_data:
        question = sample["question"]
        answerable = sample["answerable"]

        try:
            answer = agent.answer(
                question=question,
                session_id="eval_session"  # fixed session, memory not used
            )
        except Exception as e:
            answer = f"ERROR: {str(e)}"

        hallucinated = False
        if not answerable:
            hallucinated = not (
                "i don't know" in answer.lower()
                or "not mentioned" in answer.lower()
            )

        results.append({
            "question": question,
            "answerable": answerable,
            "generated_answer": answer,
            "hallucinated": hallucinated
        })

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(results),
        "results": results
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation completed. Report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()
