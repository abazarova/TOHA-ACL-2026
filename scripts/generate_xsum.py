import argparse
import json
import os
from pathlib import Path

import nltk
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def prompt_to_template(prompt: str, tokenizer: AutoTokenizer):
    messages = [
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_model_and_tokenizer(model_id: str, device: str) -> tuple:
    """Load the model and the corresponding tokenizer and place the model at the selected device."""
    login(os.environ["HUGGING_FACE_API_KEY"])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).half().to(device)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_dataset(path: Path):
    """Load dataset from the given path."""
    with open(path) as f:
        data = json.load(f)
    return data


def generate_prompt(context: str, model_name: str, tokenizer: AutoTokenizer) -> str:
    """Generate the prompt for the model given the dataset."""
    text = f"Article: {context}\nSummarize the article in one sentence. Summary:"
    if model_name.split("/")[-1] in [
        "Mistral-7B-Instruct-v0.1",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
    ]:
        text = f"<s>[INST] {text} [/INST]"
    elif model_name.split("/")[-1] == "Llama-3.1-8B-Instruct":
        text = f"<|begin_of_text|> {text}"
    elif model_name.split("/")[-1] == "Qwen2.5-7B-Instruct":
        text = prompt_to_template(text, tokenizer)
    return text


def save_generations(generations: list, save_path: Path):
    """Save generations of the model."""
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "generations.json", "w") as f:
        json.dump(generations, f)

    df = pd.DataFrame(generations)
    df.to_csv(save_path / "generations.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/app/data/raw/")
    parser.add_argument("model_id", type=str)
    parser.add_argument("--dataset", type=str, choices=["XSum"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--stop_sequences", type=str, default="default")

    args = parser.parse_args()

    nltk.download("wordnet")
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.PorterStemmer()

    root_dir = Path(args.data_dir) / args.dataset
    load_path = root_dir / "preprocessed.json"
    save_path = root_dir / args.model_id.split("/")[1]

    data = load_dataset(load_path)

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id, device=args.device
    )

    generations = []
    for sample in tqdm(data):
        context = sample["document"].strip()
        input_text = generate_prompt(context, args.model_id, tokenizer)
        input_ids = tokenizer(
            text=input_text, add_special_tokens=False, return_tensors="pt"
        ).to(args.device)
        generated_ids = model.generate(
            **input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0][len(input_ids[0]) :].cpu()
        summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        sample["generated_summary"] = summary
        generations.append(sample)

    save_generations(generations, save_path)
