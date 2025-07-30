import torch
from tokenizer import tokenizer
from model import Llama2Model
from rich.console import Console

console=Console()
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_text_sample(model, idx, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    
    
    
    # Generate
    with torch.no_grad():
        generated = model.generate(idx, max_tokens, temperature, top_k)
    print(generated)
    # Decode and return
    result = token_ids_to_text(generated,tokenizer)
    return result
def generate_text(model_path, config, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = Llama2Model(config)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    
    context = text_to_token_ids(prompt,tokenizer)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)
    
    # Decode and return
    result = token_ids_to_text(generated,tokenizer)
    return result

def run_inference_examples():
    """Run inference examples with different prompts."""
    try:
        from model import LLAMA2_CONFIG_7B
        
        config = LLAMA2_CONFIG_7B
        
        test_prompts = [
            "Once upon a time",
            "The little girl",
            "In a magical forest",
            "The brave knight"
        ]
        
        console.print("=" * 50,style="bold green underline")
        print("LLAMA-2 INFERENCE EXAMPLES")
        console.print("=" * 50,style="bold green underline")
        
        for prompt in test_prompts:
            result = generate_text(
                "training/best_llama_v2.pt", 
                config, 
                prompt, 
                max_tokens=80, 
                temperature=0.7, 
                top_k=40
            )
            
            print(f"\nPrompt: '{prompt}'")
            print("Generated:", result)
            print("-" * 30)
            
    except FileNotFoundError:
        console.print("Model file 'training/best_llama_v2.pt' not found. Please train the model first.",style="red")
    except Exception as e:
        console.print(f"red during inference: {e}",style="red",)