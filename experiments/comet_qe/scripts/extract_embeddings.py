import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

def get_instruction_prompt(model_name: str) -> str:
    """Returns the MTEB task instruction if required by the chosen model."""
    if "e5" in model_name.lower():
        # E5 models typically expect 'query: ' or 'passage: ' or a specific instruction
        return "Instruct: Given a context sentence, retrieve a representation that clarifies the difficulty of the target word.\nQuery: "
    elif "bge" in model_name.lower() or "gte" in model_name.lower() or "qwen" in model_name.lower():
        return "Represent this sentence to determine the lexical difficulty of its target word: "
    return ""

def main():
    parser = argparse.ArgumentParser(description="Extract dense embeddings using sub-1B multilingual models")
    parser.add_argument("--data_file", default="data/all.csv")
    parser.add_argument("--output_file", default="data/all_embeddings.pt")
    # Using multilingual-e5-large-instruct by default (~560M params). 
    # Can securely swap to Qwen/BGE equivalent here:
    parser.add_argument("--model", default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    # Initialize SentenceTransformer on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    
    # Enable Half-precision if compatible
    if device == "cuda":
        model.half()

    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} rows. Starting extraction...")

    instruction = get_instruction_prompt(args.model)
    
    # We embed a composite text: The L1 context + the target word + the source word
    # This captures the semantic alignment problem directly.
    texts_to_embed = []
    for _, row in df.iterrows():
        context = str(row.get("L1_context", ""))
        target = str(row.get("en_target_word", ""))
        source = str(row.get("L1_source_word", ""))
        composite_string = f"{instruction}Context: {context} | Target Word: {target} | Native Word: {source}"
        texts_to_embed.append(composite_string)

    # Encode efficiently in batches
    print("Encoding sequences...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True # Normalizing to unit length for cosine similarities
    )
    
    # Move to CPU for saving
    embeddings_cpu = embeddings.cpu()
    
    # Save a dictionary mapping item_id to the embedding tensor
    item_ids = df["item_id"].tolist()
    save_dict = {item_id: emb for item_id, emb in zip(item_ids, embeddings_cpu)}
    
    torch.save(save_dict, args.output_file)
    print(f"Saved dense embeddings (Dim: {embeddings_cpu.shape[-1]}) for {len(save_dict)} rows to {args.output_file}")

if __name__ == "__main__":
    main()
