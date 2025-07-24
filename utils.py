import torch


def total_parms(model):
    total_params = sum(p.numel() for p in model.parameters())
    return f"Total number of parameters: {total_params:,}"

def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
       
        param_size = param.numel()
        total_params += param_size
        
        if param.requires_grad:
            total_grads += param_size

    
    total_buffers = sum(buf.numel() for buf in model.buffers())

    
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    
    total_memory_gb = total_memory_bytes / (1024**3)

    return f"{total_memory_gb:.2f} GB"