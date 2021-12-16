import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb TODO
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from data_loader import BasicDataset, RemoteSensingDataset
from dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from torch_utils import select_device

dir_img = Path('./data_for_training2/images/')
dir_mask = Path('./data_for_training2/labels/')
dir_checkpoint = Path('./checkpoints/')


def train_net(net, 
              dir_img, 
              dir_mask, 
              dir_checkpoint,
              device,
              epochs: int = 10,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5
              ):
    # 1. Create dataset
    try:
        dataset = RemoteSensingDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    # train_set, val_set = random_split(dataset, [n_train, n_val],
    #                                   generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    writer = SummaryWriter(comment=f'LR_{learning_rate}_BS_{batch_size}_SCALE_{img_scale}')
    
    # # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # with torch.cuda.amp.autocast(enabled=amp):
                #     masks_pred = net(images)
                #     loss = criterion(masks_pred, true_masks) \
                #            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                #                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                #                        multiclass=True)
                
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks) \
                    + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                # TODO for PyTorch >= 1.9
                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        for tag, value in net.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        
                        # histograms = {}
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                        
                        if net.n_classes > 1:
                            logging.info('Validation cross entropy: {}'.format(val_score))
                            writer.add_scalar('Loss/test', val_score, global_step)
                        else:
                            logging.info('Validation Dice Coeff: {}'.format(val_score))
                            writer.add_scalar('Dice/test', val_score, global_step)

                        writer.add_images('images', images, global_step)
                        if net.n_classes == 1:
                            writer.add_images('masks/true', true_masks, global_step)
                            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                        
                        # experiment.log({
                        #     'learning rate': optimizer.param_groups[0]['lr'],
                        #     'validation Dice': val_score,
                        #     'images': wandb.Image(images[0].cpu()),
                        #     'masks': {
                        #         'true': wandb.Image(true_masks[0].float().cpu()),
                        #         'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                        #     },
                        #     'step': global_step,
                        #     'epoch': epoch,
                        #     **histograms
                        # })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
    
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='[INFO] Train the UNet on images and target masks')
    parser.add_argument('-i','--images',type=str,default=None,required=True,
                        help='Directory to images')
    parser.add_argument('-m','--masks',type=str,default=None,required=True,
                        help='Directory to masks')
    parser.add_argument('-s','--save',type=str,default='./checkpoints/',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, 
                        help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, 
                        help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, 
                        help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, 
                        help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=30.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = select_device(device='cpu' if torch.cuda.is_available() is False else 'cuda:0')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net, 
                  dir_img=args.image,
                  dir_mask =args.masks, 
                  dir_checkpoint=args.save,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
