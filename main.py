import torch
from rich import Console,print
from rich.panel import Panel
from model import Llama2Model
from rich.markdown import Markdown
from rich.console import Console
from rich import box
from rich.panel import Panel
from config import LLAMA2_CONFIG_7B
from huggingface_hub import hf_hub_download
from infrance import generate_text,run_inference_examples
from utils import total_parms,model_memory_size
from weight_loader import load_weights_into_llama
from training.training import trainer


console=Console()
device= "cuda" if torch.cuda.is_available() else "cpu"
com=Console()
h="[bold bright_red]LLAMA2[/bold bright_red] "
com.print(Panel(h,border_style="bright_magenta",padding=(1,20),width=100,box=box.DOUBLE,highlight=True))
weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b",
   filename="consolidated.00.pth",
   local_dir="Llama-2-7b"
)
weights = torch.load(weights_file, weights_only=True)

models= Llama2Model(LLAMA2_CONFIG_7B)

load_weights_into_llama(models, LLAMA2_CONFIG_7B, weights)
models.to(device)
torch.save(models.parameters(),"training/best_llama_v2.pt")
config=LLAMA2_CONFIG_7B
print(total_parms(models))
print(model_memory_size(models))
def demo():
    console.print("=" * 50,style="bold green underline")
    print("Llama2 DEMO")
    print("=" * 50,style="bold green underline")
    
    batch_size, seq_len = 2, 32
    test_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

        
    with torch.no_grad():
        logits = models(test_input)
    print(logits.shape)
    run_inference_examples()


def chat():
    console.print("if you want to quit press 'quit'\n\n\n",style="bold red")
    while True:
        
        inp= input(">>>>>>   ")
        if inp!='quit':
            res=generate_text("training/best_llama_v2.pt",LLAMA2_CONFIG_7B,inp)
            console.print(f"\n\n[bold magenta]output->>> [/bold magenta]  {res}")


def main():

    import sys
    console.print("Usage: python main.py [demo|train|chat]",style="error")
    if len(sys.argv) < 2:
        demo()
        return
    
    mode = sys.argv[1]
    
    if mode == "demo":
        demo()
    elif mode == "train":
        print("Starting training...")
        trainer()
    elif mode=="chat":
        chat()
    else:
        print("Usage: python main.py [demo|train|chat]")



# if __name__ == "__main__":
#     main()