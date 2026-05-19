import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_PATH = "../data/fever_1000.csv"
OUTPUT_PATH = "../data/llm_answers_Qwen2.5-0.5B-Instruct.csv"


def load_data(path, limit=None):
    df = pd.read_csv(path)
    df = df.dropna(subset=["claim"]).copy()

    if limit is not None:
        df = df.head(limit).copy()

    return df


def slugify_model_name(model_name):
    return model_name.replace("/", "__").replace("\\", "__").replace(":", "_")


def resolve_output_path(output_path, model_name):
    if output_path:
        return output_path

    model_slug = slugify_model_name(model_name)
    return f"../data/llm_answers_{model_slug}.csv"


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_type = "seq2seq"

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model_type = "causal"

    return tokenizer, model, model_type


def build_prompt(claim):
    return f"""
    First write True or False for the following claim:

Claim: "{claim}"

and only THEN give 1 or 2 sentence explanation.
"""


def generate_answer(claim, tokenizer, model, model_type, max_new_tokens=64):
    prompt = build_prompt(claim)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_length = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    if model_type == "causal":
        generated_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    else:
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return answer


def main():
    parser = argparse.ArgumentParser(description="Generate LLM answers for FEVER claims.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model id.")
    parser.add_argument("--input", default=INPUT_PATH, help="Input CSV path.")
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Output CSV path. Default: ../data/llm_answers_<model>.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional row limit. Default: full file. Use a positive integer to test a subset.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per answer.",
    )
    args = parser.parse_args()

    limit = None if args.limit is not None and args.limit < 0 else args.limit
    output_path = resolve_output_path(args.output, args.model_name)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input, limit=limit)

    print(f"Loaded {len(df)} claims")
    print(f"Loading model: {args.model_name}")
    print("Tip: if Hugging Face downloads time out, try a smaller model or set HF_TOKEN in your shell.")

    tokenizer, model, model_type = load_model(args.model_name)
    print(f"Detected model type: {model_type}")

    answers = []

    print("Generating explanatory answers...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        claim = str(row["claim"]).strip()
        answer = generate_answer(
            claim,
            tokenizer,
            model,
            model_type,
            max_new_tokens=args.max_new_tokens,
        )

        answers.append({
            "id": row["id"],
            "verifiable": row["verifiable"],
            "label": row["label"],
            "claim": claim,
            "llm_answer": answer,
            "llm_model": args.model_name,
        })

    output_df = pd.DataFrame(answers)
    output_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nDone. Saved answers to: {output_path}")
    print("\nSample outputs:")
    print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
