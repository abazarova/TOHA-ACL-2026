import argparse
import json
import logging
import os
import re
from pathlib import Path

# import evaluate
import nltk
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from num2words import num2words
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from utils import STOP_SEQUENCES, StoppingCriteriaSub

load_dotenv()


def load_model_and_tokenizer(model_id: str, device: str) -> tuple:
    """Load the model and the corresponding tokenizer and place the model at the selected device."""
    login(os.environ["HUGGING_FACE_API_KEY"])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float16").to(
        device
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_coqa_dataset(path: Path):
    """Load CoQA dataset from the given path."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_squad_dataset(path: Path):
    """Load SQuAD dataset from the given path."""
    with open(path) as f:
        data = json.load(f)
    return data


def prompt_to_template(prompt: str, tokenizer: AutoTokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_prompt(
    question: str, context: str, dataset: str, model_name: str, tokenizer: AutoTokenizer
) -> str:
    """Generate the prompt for the model given the dataset."""
    if dataset == "CoQA":
        text = context + " Q: " + question + " A:"
    elif dataset == "HotpotQA":
        text = f"Using the following knowledge: [{context}], answer the question in a brief but complete sentence: {question} "
    else:
        text = (
            "Given the context, answer the question in a single brief but complete sentence."
            + "Note that your answer should be strictly based on the given context."
            + "In case the context does not contain the necessary information to answer the question, "
            + 'please reply with: "Unable to answer based on given context. "'
            + f"Context: {context}\n"
            + f"Question: {question}\n"
            + "Answer: "
        )
    if model_name.split("/")[-1] in [
        "Mistral-7B-Instruct-v0.1",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
    ]:
        if dataset != "CoQA":
            text = f"<s>[INST] {text} [/INST]"
        else:
            text = f"<s>{text}"
    elif model_name.split("/")[-1] == "Llama-3.1-8B-Instruct":
        text = f"<|begin_of_text|>{text}"
    elif model_name.split("/")[-1] == "Qwen2.5-7B-Instruct":
        if dataset != "CoQA":
            text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            text = f"<|im_start|>user\n Given the context and the subsequent series of question-answer pairs, answer the last question. \n{text}<|im_end|>\n<|im_start|>assistant\n"
    elif model_name.split("/")[-1] == "Mistral-Small-24B-Instruct-2501":
        text = f"<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST]{text}[/INST]"
    return text


def generate_answers(
    question: str,
    context: str,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    model_name: str,
    device: str = "cuda:0",
):
    """Generate answers to the given query."""
    input_text = generate_prompt(
        question,
        context,
        dataset=dataset_name,
        model_name=model_name,
        tokenizer=tokenizer,
    )
    input_ids = tokenizer(
        text=input_text, add_special_tokens=False, return_tensors="pt"
    ).to(args.device)

    stopping_criteria = None
    if args.stop_sequences == "default" and dataset_name == "CoQA":
        stopping_criteria = StoppingCriteriaList(
            [
                StoppingCriteriaSub(
                    stops=STOP_SEQUENCES,
                    initial_length=len(input_ids[0]),
                    tokenizer=tokenizer,
                )
            ]
        )

    generated_ids = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
    )[0][len(input_ids[0]) :].cpu()

    generation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # generated_answers = postprocess_generation(generated_ids, tokenizer=tokenizer)

    return generation


def check_if_substring(
    generated_answer: str,
    correct_answers: str,
    lemmatizer: nltk.WordNetLemmatizer,
    stemmer: nltk.PorterStemmer,
):
    """Check if any of the correct answers are a substring of the generated answer."""
    generated_answer = generated_answer.lower()
    is_substring = any(
        all(
            (lemmatizer.lemmatize(w) in generated_answer)
            or (stemmer.stem(w) in generated_answer)
            for w in answ.split(" ")
        )
        for answ in correct_answers
    )
    return is_substring


def generate_alias(answer: str):
    """Create an alias for the answer with the numbers written as words."""
    return re.sub(r"[0-9]+", repl=lambda x: num2words(x.group()), string=answer)


def preprocess_answers(answers: list[str]) -> list[str]:
    """Remove anything but Latin letters and digits."""
    clean_answers = set()
    for answ in answers:
        answ = re.sub(r"[^a-zA-Z0-9\s]", " ", answ)
        clean_answers.add(answ)
        clean_answers.add(generate_alias(answ))

    return list(clean_answers)


def postprocess_generation(
    generated_ids: torch.tensor, tokenizer: AutoTokenizer
) -> tuple[str, str]:
    """Decode the generation and posprocess it."""
    # remove stop-words
    generation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    stop_at = len(generation)
    sliced_answer = generation
    for stop in STOP_SEQUENCES:
        if generation.endswith(stop):
            stop_at = len(generation) - len(stop)
            sliced_answer = generation[:stop_at]
            break
    if not all([stop not in sliced_answer for stop in STOP_SEQUENCES]):
        error_msg = "Error: Stop words not removed successfully!"
        logging.error(error_msg)

    if sliced_answer.startswith("A: "):
        sliced_answer = sliced_answer.lstrip("A: ")
    sliced_answer = sliced_answer.rstrip()

    try:
        answer_alias = generate_alias(sliced_answer)
    except Exception as e:
        print(sliced_answer)
        print(e)
        return sliced_answer, sliced_answer
    return sliced_answer, answer_alias


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
    parser.add_argument(
        "dataset",
        type=str,
        choices=["SQuAD", "CoQA", "SQuAD2", "HotpotQA", "NQ_Swap"],
    )
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

    if args.dataset == "CoQA":
        data = load_coqa_dataset(load_path)
    else:
        data = load_squad_dataset(load_path)

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id, device=args.device
    )

    # rouge = evaluate.load("rouge")
    generations = []
    for qa in tqdm(data):
        if not qa["answers"]:
            continue
        # gt_answers = preprocess_answers(qa["answers"])
        question = qa["question"].strip()
        context = qa["context"].strip()

        answer = generate_answers(
            question,
            context,
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
            model_name=args.model_id,
        )

        qa["generated_answer"] = answer
        generations.append(qa)

    save_generations(generations, save_path)
