import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from embed_and_store import load_vectorstore, load_bm25_data
from pipeline import answer

from langchain_anthropic import ChatAnthropic
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall


def load_eval_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def fill_nan_with_retry(df, samples, metric_by_col, max_retries=2):
    for col, metric in metric_by_col.items():
        for idx in df.index:
            if not math.isnan(df.at[idx, col]):
                continue
            print(f"sample {idx+1} {col} returned NaN, retrying")
            score = float("nan")
            for attempt in range(max_retries):
                try:
                    score = metric.single_turn_score(samples[idx])
                    if not math.isnan(score):
                        break
                except Exception as e:
                    print(f"  try {attempt+1} failed: {e}")
            if math.isnan(score):
                print(f"  still NaN, falling back to 0.0")
                score = 0.0
            else:
                print(f"  ok: {score:.3f}")
            df.at[idx, col] = score


def build_sample(item: dict, chunks, raw_texts, vectorstore, *, multiquery, rerank, compress) -> SingleTurnSample:
    result = answer(
        query=item["question"],
        chunks=chunks,
        raw_texts=raw_texts,
        vectorstore=vectorstore,
        multiquery=multiquery,
        rerank=rerank,
        compress=compress,
    )
    return SingleTurnSample(
        user_input=result["question"],
        response=result["answer"],
        retrieved_contexts=[doc.page_content for doc in result["retrieved_docs"]],
        reference=item["ground_truth"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiquery", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    print(f"config: multiquery={args.multiquery} rerank={args.rerank} compress={args.compress}")
    print("loading vectorstore and bm25 data")
    vectorstore = load_vectorstore()
    chunks, raw_texts = load_bm25_data()
    print(f"loaded {len(chunks)} chunks\n")

    dataset_path = Path(__file__).parent / "eval_dataset.json"
    eval_items = load_eval_dataset(dataset_path)
    print(f"running {len(eval_items)} questions through the pipeline")

    samples = []
    for i, item in enumerate(eval_items, 1):
        print(f"  {i}/{len(eval_items)}  {item['question']}")
        samples.append(build_sample(item, chunks, raw_texts, vectorstore,
                                    multiquery=args.multiquery,
                                    rerank=args.rerank,
                                    compress=args.compress))

    ragas_dataset = EvaluationDataset(samples=samples)

    evaluator_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    )

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    context_recall_metric = LLMContextRecall(llm=evaluator_llm)
    metric_by_col = {
        "faithfulness": faithfulness_metric,
        "context_recall": context_recall_metric,
    }

    print("\nscoring with ragas")
    results = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness_metric, context_recall_metric],
        llm=evaluator_llm,
        raise_exceptions=False,
    )

    df = results.to_pandas()
    fill_nan_with_retry(df, samples, metric_by_col, max_retries=2)

    print(f"\nfaithfulness   {df['faithfulness'].mean():.3f}")
    print(f"context recall {df['context_recall'].mean():.3f}")

    print()
    print(f"{'#':<4}{'faith':>7}{'recall':>8}   question")
    for idx, row in df.iterrows():
        q = eval_items[idx]["question"]
        print(f"{idx+1:<4}{row['faithfulness']:>7.3f}{row['context_recall']:>8.3f}   {q}")


if __name__ == "__main__":
    main()
