import argparse
from .vector_store import retrieve
from .llm import generate_answer

def ask(question, model_key="smollm", top_k=3):
    contexts = retrieve(question, top_k=top_k)
    result = generate_answer(question, contexts, model_key=model_key)
    result["question"] = question
    result["contexts"] = contexts
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["smollm", "llama"], default="smollm")
    parser.add_argument("--question", default="What is the purpose of the Research Incubation program?")
    args = parser.parse_args()

    result = ask(args.question, args.model)
    print("\nANSWER:\n", result["answer"])
    print("\nRETRIEVED PAGES:", [x["page"] for x in result["contexts"]])
    print("LATENCY:", result["latency_seconds"], "seconds")
