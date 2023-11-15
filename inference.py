import torch
import numpy as np
import argparse
from ruamel.yaml import YAML
from pathlib import Path
from dataset import MLDataSet
from torch.utils.data import DataLoader 

from evaluate import evaluate
from model import get_model
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--config', type=str, default='./config.yml', help='config yaml path')
    parser.add_argument('--data', type=str, default='./data', help='data path')
    parser.add_argument('--model', type=str, default='./checkpoint/checkpoint_epoch100.pth', help='inference checkpoint')
    return parser.parse_args()


def inference(args, config):
    
    batch_size = config['train']['batch_size']
    amp = config['train']['amp']
    num_work = config['train']['num_work']
    
    dataset = MLDataSet(args, config, 'infer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_args = dict(batch_size=batch_size, num_workers=num_work, pin_memory=True)
    
    dataloader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
    
    model = get_model(config).to(device)
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    
    result = evaluate(model, dataloader, device, amp)
    
    for k, v in result.items():
        print(k, ":", v)
    

if __name__ == "__main__":
    args = get_args()
    yaml = YAML(typ='safe')
    with Path(args.config).open('r') as f:
        config = yaml.load(f)
        
    inference(args=args, config=config)
