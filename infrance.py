import torch
from tokenizer import LlamaTokenizer
from model import Llama2Model
from huggingface_hub import hf_hub_download

tokenizer_file = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="tokenizer.model",
    local_dir="Llama-2-7b"
)
tokenizer = LlamaTokenizer(tokenizer_file)
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_text(model_path, config, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = Llama2Model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    
    context = text_to_token_ids(prompt,tokenizer)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)
    
    # Decode and return
    result = token_ids_to_text.decode(generated)
    return result

def run_inference_examples():
    """Run inference examples with different prompts."""
    try:
        from model import LLAMA2_CONFIG_7B
        
        config = LLAMA2_CONFIG_7B()
        
        test_prompts = [
            "Once upon a time",
            "The little girl",
            "In a magical forest",
            "The brave knight"
        ]
        
        print("=" * 50)
        print("LLAMA-2 INFERENCE EXAMPLES")
        print("=" * 50)
        
        for prompt in test_prompts:
            result = generate_text(
                "best_llama_v2.pt", 
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
        print("Model file 'best_llama_v2.pt' not found. Please train the model first.")
    except Exception as e:
        print(f"Error during inference: {e}")