import os
import time
import subprocess

from .vector_store import retrieve


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VLM_MODEL = "ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0"
LLAMA_CLI = os.path.expanduser("~/.llama-app/llama")


# ---------------------------------------------------------
# Multimodal Answer Generation
# ---------------------------------------------------------

def generate_multimodal_answer(question, contexts):
    """
    Generate an answer using:
    1. Retrieved handbook text
    2. Retrieved handbook page images
    3. SmolVLM-256M-Instruct through llama.cpp
    """

    # -----------------------------------------------------
    # No context found
    # -----------------------------------------------------

    if not contexts:
        return {
            "answer": "No relevant context was retrieved.",
            "latency_seconds": 0,
            "image_paths": [],
            "pages": []
        }

    # -----------------------------------------------------
    # Use top 3 retrieved contexts
    # -----------------------------------------------------

    selected_contexts = contexts[:3]

    image_paths = []
    text_context = []

    for context in selected_contexts:

        # Get image path associated with this page
        image_path = context.get("image_path")

        if image_path and os.path.exists(image_path):
            image_paths.append(image_path)

        # Get text associated with this page
        text_context.append(
            f"Page {context['page']}:\n{context['text']}"
        )

    combined_text = "\n\n".join(text_context)

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are answering a question using a university handbook.

Use BOTH:
1. The retrieved handbook text
2. The provided handbook page images

Important instructions:
- Answer ONLY from the provided handbook.
- Carefully check numerical values, tables, grades, percentages, and labels.
- Do not guess.
- If the information is not clearly available, say that it is not clearly available.
- Give a concise and accurate answer.
- Do not explain your reasoning.
- Do not mention the model, retrieval process, or images.

Retrieved handbook text:

{combined_text}

Question:
{question}

Answer:
"""

    # -----------------------------------------------------
    # Build llama.cpp command
    # -----------------------------------------------------

    command = [
        LLAMA_CLI,
        "cli",

        # Hugging Face SmolVLM model
        "-hf",
        VLM_MODEL,

        # Single question / single response
        "--single-turn",

        # Cleaner command-line I/O
        "--simple-io",
        "--no-display-prompt",
        "--no-show-timings",

        # Maximum generated tokens
        "-n",
        "128"
    ]

    # -----------------------------------------------------
    # Add retrieved images
    # -----------------------------------------------------

    for image_path in image_paths:
        command.extend([
            "--image",
            image_path
        ])

    # -----------------------------------------------------
    # Add prompt
    # -----------------------------------------------------

    command.extend([
        "-p",
        prompt
    ])

    # -----------------------------------------------------
    # Run SmolVLM
    # -----------------------------------------------------

    start = time.time()

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )

    except subprocess.TimeoutExpired:

        return {
            "answer": "The multimodal model timed out while generating the answer.",
            "latency_seconds": round(time.time() - start, 3),
            "image_paths": image_paths,
            "pages": [
                c["page"]
                for c in selected_contexts
            ]
        }

    latency = time.time() - start

    # -----------------------------------------------------
    # Check for CLI error
    # -----------------------------------------------------

    if result.returncode != 0:

        error_message = result.stderr.strip()

        if not error_message:
            error_message = result.stdout.strip()

        return {
            "answer": f"Multimodal model error: {error_message}",
            "latency_seconds": round(latency, 3),
            "image_paths": image_paths,
            "pages": [
                c["page"]
                for c in selected_contexts
            ]
        }

    # -----------------------------------------------------
    # Get raw llama.cpp output
    # -----------------------------------------------------

    raw_output = result.stdout.strip()

    # -----------------------------------------------------
    # Extract ONLY the generated answer
    #
    # llama.cpp produces output similar to:
    #
    # Loading model...
    # build...
    # model...
    # Loaded media...
    #
    # > Question
    # Actual generated answer
    #
    # Exiting...
    #
    # We only want:
    #
    # Actual generated answer
    # -----------------------------------------------------

    answer = raw_output

    # Find the CLI prompt marker.
    # We intentionally use the FIRST prompt marker because
    # everything before it is llama.cpp startup information.
    prompt_marker = "\n>"

    if prompt_marker in answer:

        answer = answer.split(
            prompt_marker,
            1
        )[1]

        # Remove the question itself if llama.cpp displayed it.
        answer_lines = answer.splitlines()

        if answer_lines:

            # The first line normally contains the question.
            first_line = answer_lines[0].strip()

            if first_line == question.strip():
                answer_lines = answer_lines[1:]

            elif first_line.startswith(question.strip()):
                answer_lines = answer_lines[1:]

        answer = "\n".join(answer_lines)

    # -----------------------------------------------------
    # Remove llama.cpp termination text
    # -----------------------------------------------------

    if "Exiting..." in answer:

        answer = answer.split(
            "Exiting...",
            1
        )[0]

    # -----------------------------------------------------
    # Remove common CLI artifacts
    # -----------------------------------------------------

    cleaned_lines = []

    for line in answer.splitlines():

        stripped = line.strip()

        # Empty lines are okay, but we don't need excessive ones.
        if not stripped:
            cleaned_lines.append("")
            continue

        # Remove accidental prompt markers
        if stripped == ">":
            continue

        if stripped.startswith("> "):
            # If this is still a prompt marker, remove it.
            stripped = stripped[2:].strip()

            if stripped:
                cleaned_lines.append(stripped)

            continue

        # Ignore common llama.cpp messages
        if stripped in {
            "Loading model...",
            "available commands:"
        }:
            continue

        # Ignore media-loading messages
        if stripped.startswith("Loaded media from"):
            continue

        cleaned_lines.append(line)

    # -----------------------------------------------------
    # Final cleanup
    # -----------------------------------------------------

    answer = "\n".join(cleaned_lines).strip()

    # Remove excessive blank lines
    while "\n\n\n" in answer:
        answer = answer.replace(
            "\n\n\n",
            "\n\n"
        )

    # Fallback if parsing somehow removed everything
    if not answer:

        answer = (
            "The multimodal model did not return a readable answer. "
            "Please try the question again."
        )

    # -----------------------------------------------------
    # Return structured result
    # -----------------------------------------------------

    return {
        "answer": answer,
        "latency_seconds": round(latency, 3),
        "image_paths": image_paths,
        "pages": [
            c["page"]
            for c in selected_contexts
        ]
    }


# ---------------------------------------------------------
# Multimodal RAG
# ---------------------------------------------------------

def ask_multimodal(question, top_k=3):
    """
    Complete multimodal RAG pipeline:

    Question
       ↓
    FAISS retrieval
       ↓
    Retrieved text + page images
       ↓
    SmolVLM
       ↓
    Answer
    """

    # Retrieve relevant handbook pages
    contexts = retrieve(
        question,
        top_k=top_k
    )

    # Generate multimodal answer
    result = generate_multimodal_answer(
        question,
        contexts
    )

    # Add question and contexts to result
    result["question"] = question
    result["contexts"] = contexts

    return result


# ---------------------------------------------------------
# Command-line test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "What grades and grade points are specified "
        "for the Research Incubation course?"
    )

    result = ask_multimodal(
        question,
        top_k=3
    )

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED PAGES:")
    print(result["pages"])

    print("\nIMAGES:")

    for image in result["image_paths"]:
        print(image)

    print("\nLATENCY:")
    print(result["latency_seconds"], "seconds")