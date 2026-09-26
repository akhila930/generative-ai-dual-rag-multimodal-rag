import time
import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama

from .config import SMOLLM_MODEL, LLAMA_MODEL, MAX_NEW_TOKENS


# Exact local GGUF model path
LLAMA_GGUF_PATH = (
    "/home/akhila/.var/app/com.visualstudio.code/cache/huggingface/hub/"
    "models--hugging-quants--Llama-3.2-1B-Instruct-Q4_K_M-GGUF/"
    "snapshots/"
)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_smollm():
    """Load SmolLM2 using Transformers."""
    tokenizer = AutoTokenizer.from_pretrained(SMOLLM_MODEL)

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        SMOLLM_MODEL,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model.to("cpu")

    model.eval()

    return tokenizer, model


@lru_cache(maxsize=1)
def load_llama():
    """Load quantized Llama 3.2 1B using llama.cpp."""

    import glob

    matches = glob.glob(
        LLAMA_GGUF_PATH +
        "*/llama-3.2-1b-instruct-q4_k_m.gguf"
    )

    if not matches:
        raise FileNotFoundError(
            "Llama GGUF model not found. "
            "Expected file inside Hugging Face snapshot cache."
        )

    model_path = matches[0]

    print(f"\nLoading quantized Llama model:")
    print(model_path)

    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=6,
        verbose=False,
    )

    return llm


def generate_smollm(question, contexts):
    """Generate answer using SmolLM2."""

    tokenizer, model = load_smollm()

    context_text = "\n\n".join(
        f"[Handbook page {c['page']}]\n{c['text']}"
        for c in contexts
    )

    system_message = (
        "You are a university handbook assistant. "
        "Answer the user's question using ONLY the supplied handbook context. "
        "Do not invent information. "
        "If the answer is not in the context, say that the handbook "
        "context does not contain enough information."
    )

    user_message = (
        f"HANDBOOK CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = (
            f"{system_message}\n\n"
            f"{user_message}\n\n"
            "ANSWER:"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    device = next(model.parameters()).device

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.perf_counter() - start

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return {
        "answer": answer,
        "latency_seconds": round(latency, 3),
        "model": SMOLLM_MODEL,
    }


def generate_llama(question, contexts):
    """Generate answer using quantized Llama 3.2 1B."""

    llm = load_llama()

    context_text = "\n\n".join(
        f"[Handbook page {c['page']}]\n{c['text']}"
        for c in contexts
    )

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a university handbook assistant.
Answer the user's question using ONLY the supplied handbook context.
Do not invent information.
If the answer is not in the context, say that the handbook context does not contain enough information.

<|eot_id|><|start_header_id|>user<|end_header_id|>

HANDBOOK CONTEXT:

{context_text}

QUESTION:

{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    start = time.perf_counter()

    output = llm(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
        stop=[
            "<|eot_id|>",
            "<|end_of_text|>"
        ],
    )

    latency = time.perf_counter() - start

    answer = output["choices"][0]["text"].strip()

    return {
        "answer": answer,
        "latency_seconds": round(latency, 3),
        "model": "Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
    }


def generate_answer(question, contexts, model_key="smollm"):

    if model_key == "smollm":
        return generate_smollm(
            question,
            contexts
        )

    elif model_key == "llama":
        return generate_llama(
            question,
            contexts
        )

    else:
        raise ValueError(
            f"Unknown model: {model_key}"
        )