"""
Text generation module for custom transformer language model.

Loads trained custom transformer models and generates text from prompts.
"""

import argparse
import os
from typing import Optional, List

import torch
from transformers import GPT2TokenizerFast

from .utils.config import load_config
from .models.transformer_lm import TransformerLM


def load_model_for_generation(
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda"
):
    """
    Load a trained custom transformer model for text generation.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to configuration file.
        device: Device to load model on.
        
    Returns:
        Tuple of (model, tokenizer, config).
    """
    # Load tokenizer (try to load from preprocessed data, fallback to downloading)
    print("[INFO] Loading tokenizer...")
    tokenizer_path = "data/preprocessed/tokenizer"
    if os.path.exists(tokenizer_path):
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        print(f"[INFO] Loaded tokenizer from: {tokenizer_path}")
    else:
        cache_dir = os.environ.get('HF_HOME', None)
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir=cache_dir)
        print("[INFO] Loaded tokenizer from HuggingFace cache")
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load configuration
    print(f"[INFO] Loading model configuration from: {config_path}")
    config = load_config(config_path)
    
    # Create model
    print("[INFO] Creating custom transformer model...")
    model = TransformerLM(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint and state_dict only
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded checkpoint (epoch: {checkpoint.get('epoch', 'N/A')}, "
              f"step: {checkpoint.get('step', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("[INFO] Loaded checkpoint (state_dict only)")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Model loaded successfully on {device}")
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, config


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    num_samples: int = 1,
    device: str = "cuda"
) -> List[str]:
    """
    Generate text from a prompt.
    
    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        top_p: Nucleus sampling.
        num_samples: Number of samples to generate.
        device: Device to run generation on.
        
    Returns:
        List of generated texts.
    """
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    
    print(f"\n[INFO] Prompt: {prompt}")
    print(f"[INFO] Generating {num_samples} sample(s)...")
    print("-" * 80)
    
    generated_texts = []
    
    for i in range(num_samples):
        with torch.no_grad():
            # Generate (with AMP if on GPU for faster inference)
            if device.startswith('cuda') and torch.cuda.is_available():
                from torch.cuda.amp import autocast
                with autocast():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        eos_token_id=tokenizer.eos_token_id
                    )
            else:
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            
            print(f"\n[Sample {i+1}]")
            print(generated_text)
            print("-" * 80)
    
    return generated_texts


def interactive_generation(
    model,
    tokenizer,
    config,
    device: str = "cuda"
):
    """
    Interactive text generation loop.
    
    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        config: Configuration object.
        device: Device to run generation on.
    """
    print("\n" + "="*80)
    print("Interactive Text Generation")
    print("="*80)
    print("Enter a prompt to generate text. Type 'quit' or 'exit' to stop.")
    print("="*80 + "\n")
    
    # Get default generation parameters from config
    if config is not None and hasattr(config, 'generation'):
        default_max_tokens = config.generation.max_new_tokens
        default_temp = config.generation.temperature
        default_top_k = config.generation.top_k
        default_top_p = config.generation.top_p
        default_num_samples = 1
    else:
        default_max_tokens = 100
        default_temp = 0.8
        default_top_k = 50
        default_top_p = 0.9
        default_num_samples = 1
    
    while True:
        # Get prompt
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not prompt:
            print("Please enter a non-empty prompt.")
            continue
        
        # Generate with default parameters
        try:
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=default_max_tokens,
                temperature=default_temp,
                top_k=default_top_k,
                top_p=default_top_p,
                num_samples=default_num_samples,
                device=device
            )
            
        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            continue
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            continue


def main():
    """
    Main entry point for text generation.
    """
    parser = argparse.ArgumentParser(description="Generate text using trained custom Transformer LM")
    
    # Model loading arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus (top-p) sampling"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save generated text"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run generation on"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("[INFO] Loading model for generation...")
    model, tokenizer, config = load_model_for_generation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Interactive or single generation
    if args.interactive:
        interactive_generation(model, tokenizer, config, args.device)
    else:
        if args.prompt is None:
            print("[ERROR] Please provide --prompt or use --interactive mode")
            return
        
        # Generate text
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples=args.num_samples,
            device=args.device
        )
        
        # Save to file if specified
        if args.output is not None:
            with open(args.output, 'w') as f:
                f.write(f"Prompt: {args.prompt}\n")
                f.write("="*80 + "\n\n")
                for i, text in enumerate(generated_texts):
                    f.write(f"Sample {i+1}:\n")
                    f.write(text + "\n\n")
                    f.write("-"*80 + "\n\n")
            print(f"\n[INFO] Generated text saved to: {args.output}")


if __name__ == "__main__":
    main()
