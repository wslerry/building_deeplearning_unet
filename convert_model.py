# import argparse
# import logging

import torch
import torch.onnx
from unet import UNet


def export2ONNX(model_dir,device='cpu'):

    net = UNet(n_channels=3, n_classes=2)
    net.to(device='cpu') if device == 'cpu' else net.to(device='cuda:0')
    net.load_state_dict(torch.load(model_dir, map_location='cuda:0'))

    # Create the right input shape (e.g. for an image)
    dummy_input = torch.randn(4, 3, 512, 512)

    torch.onnx.export(net, dummy_input, "test_out.onnx")
    
    

export2ONNX('./checkpoints/checkpoint_epoch1.pth',device='cpu')
