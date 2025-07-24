import torch
from model import Llama2Model
from config import LLAMA2_CONFIG_7B
from huggingface_hub import hf_hub_download
from infrance import generate_text,run_inference_examples
from utils import total_parms,model_memory_size
from weight_loader import load_weights_into_llama

device= "cuda" if torch.cuda.is_available() else "cpu"

weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b",
   filename="consolidated.00.pth",
   local_dir="Llama-2-7b"
)
weights = torch.load(weights_file, weights_only=True)

models= Llama2Model(LLAMA2_CONFIG_7B)

load_weights_into_llama(models, LLAMA2_CONFIG_7B, weights)
models.to(device)
torch.save(models.parameters(),"best_llama_v2.pt")
config=LLAMA2_CONFIG_7B
print(total_parms(models))
print(model_memory_size(models))

batch_size, seq_len = 2, 32
test_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    
with torch.no_grad():
    logits = models(test_input)
print(logits.shape)