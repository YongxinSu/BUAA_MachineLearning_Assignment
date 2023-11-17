from dataset import MLDataSet
import numpy as np
import argparse
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, random_split
import torch
import os
from pathlib import Path
from model import get_model
from tqdm import tqdm
import wandb
import logging

from utils.metric import KLD_gpu, calc_cc_score_GPU
from evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--config', type=str, default='./config.yml', help='config yaml path')
    parser.add_argument('--data', type=str, default='./data', help='data path')

    return parser.parse_args()



def run(
        args, config,
):
    dataset = MLDataSet(args, config, mode='train')
    
    epochs = config['epochs']
    val_percent = config['val_percent']
    batch_size = config['batch_size']
    learning_rate = config['lr']
    weight_decay = config['weight_decay']
    momentum = config['momentum']
    amp = config['amp']
    gradient_clipping = config['gradient_clipping']
    save_checkpoint = config['save_checkpoint']
    img_scale = config['img_scale']
    dir_checkpoint = config['dir_checkpoint']
    num_work = config['num_work']
    writer = SummaryWriter()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_work, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # model = UNet(config).to(device)
    # model = SalPreNet().to(device)
    model = get_model(config).to(device)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: maximize Dice score
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = lambda pred, target: 0.5 * KLD_gpu(pred, target) + 0.5 * torch.nn.MSELoss()(pred, target) # + 0.5 * KLD_gpu(1.0 - pred, 1.0 - target) # torch.nn.CrossEntropyLoss() 
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['label']
                # assert images.shape[1] == model.n_channels, \
                #     f'Network has been defined with {model.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                # print(images.max())
                # print(images.min())
                # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                # print('#', masks_pred.sum(), " ### ", true_masks.sum())
                loss = criterion(masks_pred, true_masks)
                
                # print(masks_pred.sum())
                optimizer.zero_grad()
                # grad_scaler.scale(loss).backward()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                writer.add_scalar('learningrate', optimizer.state_dict()['param_groups'][0]['lr'], global_step)
                writer.add_scalar('train loss', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 1:
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        score_dict = evaluate(model, val_loader, device, amp)
                        scheduler.step(score_dict['kld'])

                        try:
                            writer.add_scalar('KLD', score_dict['kld'], global_step)
                            writer.add_scalar('cc', score_dict['cc'], global_step)
                            writer.add_scalar('valDice', score_dict['dice'], global_step)
                            
                            writer.add_images('image_batch', images, global_step)
                            writer.add_images('mask_batch', masks_pred, global_step)
                            writer.add_images('gt_batch', true_masks, global_step)
                        except:
                            pass

        if save_checkpoint and epoch  % 10 == 1:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(Path(dir_checkpoint) / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

if __name__ == "__main__":
    
    args = get_args()
    
    yaml = YAML(typ='safe')
    with Path(args.config).open('r') as f:
        config = yaml.load(f)
    
    run(args=args, config=config)
    
    