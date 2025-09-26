import torch
from scipy.linalg import orth
import numpy as np

def generate_on_text(model, tokenizer, input_text, **kwargs):
        
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    # Generate output
    outputs = model.generate(
        **inputs,
        **kwargs,
    )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0])
    return generated_text
    
def hook_model(model, directions, layers_to_control, control_coef, component_idx=0, anti="no", last=True):
    hooks = {}
    for layer_idx in layers_to_control:
        if anti == "yes":
            if isinstance(component_idx, int):
                if component_idx == 0:
                    control_vec = directions[layer_idx] # [300, 4096]
                else:
                    control_vec = directions[layer_idx][:component_idx] # [300, 4096]
            elif isinstance(component_idx, dict):
                control_vec = directions[layer_idx][:component_idx[layer_idx]]

            C = control_vec.to(dtype=torch.float64)
            C_np = C.detach().cpu().numpy() # [300, 4096]

            O_np = orth(C_np.T) # (4096, 300)

            # print(O_np.shape)
            # print(np.allclose(O_np.T @ O_np, np.eye(O_np.shape[1]))) # True
            # print(np.allclose(C_np @ O_np @ O_np.T, C_np)) # True

            O = torch.from_numpy(O_np).to(device=control_vec.device, dtype=control_vec.dtype).contiguous()

            block = model.model.layers[layer_idx]

            def block_hook_anti(module, input, output, O=O, control_coef=control_coef, last=last):
                """
                Anti-steer: remove the component along control_vec via orthogonal projection.
                """
                new_output = output[0] # [batch, seq, d]
                
                if new_output.shape[1] == 1:
                    # generation phase: only one position
                    h = new_output # 1,1,4096

                    proj = (h @ O) @ O.T
                    h_projected = h - control_coef * proj

                    new_output = h_projected

                else:
                    # prefill phase
                    if last:
                        # only last token position
                        h = new_output[:, -1, :].unsqueeze(1) # [B, 1, d]

                        proj = (h @ O) @ O.T
                        h_projected = h - control_coef * proj
                        new_output[:, -1, :] = h_projected.squeeze(1)

                        # dot_prod = (h * c_norm).sum(dim=-1, keepdim=True)
                        # h_projected = h - extent * dot_prod * c_norm
                        # new_output[:, -1, :] = control_coef * h_projected
                    else:
                        # all positions
                        h = new_output # 1,38,4096

                        proj = (h @ O) @ O.T  # [B, T, d]
                        h_projected = h - control_coef * proj

                        new_output = h_projected
                
                if isinstance(output, tuple):
                    new_output = (new_output,) + output[1:]
                
                return new_output

            hook_handle = block.register_forward_hook(block_hook_anti)

        
        elif anti == "mixed":
            print("todo mixed")
            # delete the direction
            # add the other direction
            pass
        
        
        elif anti == "no":
            # original
            control_vec = directions[layer_idx][component_idx]
            if len(control_vec.shape)==1:
                control_vec = control_vec.reshape(1,1,-1)
                
                
            block = model.model.layers[layer_idx]

            def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef):
                """
                note that module, input are unused, but are
                required by torch.
                """ 
                
                new_output = output[0]

                new_output = new_output + control_coef*control_vec.to(dtype=new_output.dtype, device=new_output.device)
                
                if isinstance(output, tuple):
                    new_output = (new_output,) + output[1:] 
                
                return new_output
            
            hook_handle = block.register_forward_hook(block_hook)

        else:
            print("Check the anti variable!!")
        
        hooks[layer_idx] = hook_handle
    
    return hooks

def clear_hooks(hooks) -> None:
    for hook_handle in hooks.values():
        hook_handle.remove()