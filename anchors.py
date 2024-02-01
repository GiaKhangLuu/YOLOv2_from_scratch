import torch
import numpy as np

def generate_anchors(scales, aspect_ratios, dtype=torch.float32):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #scales = torch.as_tensor(scales, dtype=dtype, device=device)
    #aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)

    #sqrt_aspect_ratio = torch.sqrt(aspect_ratios)
    #scales, sqrt_aspect_ratio = torch.meshgrid(scales, sqrt_aspect_ratio, indexing='ij')
    #scales, sqrt_aspect_ratio = scales.flatten(), sqrt_aspect_ratio.flatten()
    
    #return torch.stack([scales * sqrt_aspect_ratio, scales / sqrt_aspect_ratio], dim=-1)
    ANCHOR_BOXS = [[1.08,1.19],
                   [3.42,4.41],
                   [6.63,11.38],
                   [9.42,5.11],
                   [16.62,10.52]]
    
    return torch.tensor(ANCHOR_BOXS, device=device)