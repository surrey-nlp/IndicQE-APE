"""
extract_llm_embeddings.py

Extracts contextual embeddings from large language models (e.g., Qwen3, F2LLM)
to be used as dense features in the BEA2026 Lexical Difficulty Prediction framework.

Usage:
  python src/extract_llm_embeddings.py --model Qwen/Qwen3-Embedding-8B --data data/final_production_split.csv --output data/llm_embeddings_qwen3.pt
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_qwen3_instruct(query: str) -> str:
    """Format input for Qwen3-Embedding with task instruction."""
    task = 'Given a short context and a target word in a second language, encode the semantic representation of the word in context for lexical difficulty prediction.'
    return f'Instruct: {task}\nQuery:{query}'


def format_input_text(row: pd.Series, use_qwen_instruct: bool = False) -> str:
    """Format the DataFrame row into the same text string the encoder sees."""
    segments = []
    
    # Optional LID prepend (matching prod models)
    l1 = str(row.get('L1', '')).strip()
    if l1:
        segments.append(f"[L1: {l1}]")
        
    for col in ["L1_context", "L1_source_word", "en_target_clue", "en_target_pos", "en_target_word"]:
        val = str(row.get(col, "")).strip()
        if val and not pd.isna(row.get(col)):
            segments.append(val)
            
    text = " [SEP] ".join(segments)
    
    if use_qwen_instruct:
        return get_qwen3_instruct(text)
    return text


def extract_with_sentence_transformers(model_name: str, df: pd.DataFrame, device: str, batch_size: int, is_qwen: bool = False):
    """Extraction using sentence-transformers (works well for F2LLM)."""
    print(f"Loading SentenceTransformer: {model_name}")
    from sentence_transformers import SentenceTransformer
    
    # Load model (SentenceTransformers handles dtype internally or we can cast)
    model = SentenceTransformer(
        model_name, 
        device=device,
        trust_remote_code=True
    )
    # Cast to bfloat16 to save memory
    model.to(torch.bfloat16)
    
    texts = [format_input_text(row, use_qwen_instruct=is_qwen) for _, row in df.iterrows()]
    
    print(f"Encoding {len(texts)} items...")
    
    # F2LLM-v2 uses encode_query for prompts, but standard encode is fine if not using custom prompts
    # Since we formatted the text ourselves, we use standard encode
    encode_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_tensor": True,
        "normalize_embeddings": True
    }
    
    with torch.inference_mode():
        embeddings = model.encode(texts, **encode_kwargs)
        
    return embeddings.cpu()


def extract_with_transformers(model_name: str, df: pd.DataFrame, device: str, batch_size: int, is_qwen: bool = False):
    """Extraction using pure transformers (best for Qwen3-Embedding-8B)."""
    print(f"Loading Transformers model: {model_name}")
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    
    # Try to load with flash attention if available
    try:
        model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        print("✓ Loaded with Flash Attention 2")
    except (ImportError, ValueError):
        print("⚠️ Flash Attention 2 not available/supported, falling back to SDPA")
        model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
    model = model.to(device)
    model.eval()
    
    texts = [format_input_text(row, use_qwen_instruct=is_qwen) for _, row in df.iterrows()]
    all_embeddings = []
    
    print(f"Encoding {len(texts)} items...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        
        # Qwen3 uses 8192 context; we cap at 1024 for these short lexical items
        batch_dict = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=1024, 
            return_tensors="pt"
        ).to(device)
        
        with torch.inference_mode():
            outputs = model(**batch_dict)
            
            # Last token pooling (standard for causal LM embeddings like Qwen/F2LLM)
            attention_mask = batch_dict['attention_mask']
            sequence_lengths = attention_mask.sum(dim=1) - 1
            bsz = outputs.last_hidden_state.shape[0]
            
            embeddings = outputs.last_hidden_state[torch.arange(bsz, device=device), sequence_lengths]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            
    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Extract LLM embeddings")
    parser.add_argument("--model", type=str, required=True, help="HF model name (e.g., Qwen/Qwen3-Embedding-8B)")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--output", type=str, required=True, help="Output .pt path")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--use_st", action="store_true", help="Force use sentence-transformers")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if "item_id" not in df.columns:
        raise ValueError("Data must have an 'item_id' column")

    is_qwen = "qwen3" in args.model.lower()
    
    # Qwen3 is officially supported via pure transformers; F2LLM via ST or transformers
    # We default appropriately
    try:
        if args.use_st or "f2llm" in args.model.lower():
            embeddings = extract_with_sentence_transformers(
                args.model, df, device, args.batch_size, is_qwen
            )
        else:
            embeddings = extract_with_transformers(
                args.model, df, device, args.batch_size, is_qwen
            )
            
        # Create dict mapping item_id -> embedding tensor
        emb_dict = {}
        for idx, row in df.iterrows():
            item_id = row['item_id']
            # Store as float32 for training compatibility (MLP uses fp32/fp16 mixed precision)
            emb_dict[item_id] = embeddings[idx].float()
            
        print(f"Saving {len(emb_dict)} embeddings of size {next(iter(emb_dict.values())).shape} to {args.output}")
        torch.save(emb_dict, args.output)
        print("Done!")
        
    except Exception as e:
        print(f"Extraction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
