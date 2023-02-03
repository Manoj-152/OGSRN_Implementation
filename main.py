from Models.SORTN import SORTN
from Models.PatchGAN import PatchGAN
from SORTN_dataset import SAR_optic_dataset
import yaml
import argparse

from Runner.train import train
from Runner.inference import inference

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
torch.manual_seed(10)


parser = argparse.ArgumentParser(description = 'calling for training or performing inference on the model')
parser.add_argument("to_do", help = "Valid Arguments: train, inference")
args = parser.parse_args()
if args.to_do != "train" and args.to_do != "inference":
    print('Please enter a valid argument. (train, inference)')
to_do = args.to_do

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


generator = SORTN()
discriminator = PatchGAN(1)
generator = generator.to(cfg['DEVICE'])
discriminator = discriminator.to(cfg['DEVICE'])

# loading pretrained models
if to_do == 'train':
    
    print('Loading dataset for training')
    dataset = SAR_optic_dataset(cfg, tensor_transform=True)
    a = int(cfg['TRAIN']['TRAIN_TEST_SPLIT'] * len(dataset))
    b = len(dataset) - a
    train_ds, val_ds = torch.utils.data.random_split(dataset, (a, b))
    trainloader = DataLoader(train_ds, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=2, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=2, shuffle=False)
    
    if cfg['TRAIN']['START_FROM_PRETRAINED_WEIGHTS'] == True:
        
        print('Loading pretrained weights for training')
        weights = torch.load(cfg['TRAIN']['PRETRAINED_WEIGHTS'])
        generator.load_state_dict(weights['generator'])
        discriminator.load_state_dict(weights['discriminator'])
        start_epoch = weights['epoch'] + 1

        print('Starting training for epoch ', start_epoch)
    else:
        print('Training the model from scratch')
        start_epoch = 0
      
    # initializing optimizers
    initial_lr = cfg['TRAIN']['INITIAL_LR']
    beta_1 = cfg['TRAIN']['BETA_1']
    beta_2 = cfg['TRAIN']['BETA_2']
    optimizer_gen = optim.Adam(generator.parameters(),lr=initial_lr,betas=(beta_1,beta_2))
    optimizer_disc = optim.Adam(discriminator.parameters(),lr=initial_lr,betas=(beta_1,beta_2)) 
    train(cfg, trainloader, valloader, generator, discriminator, optimizer_gen, optimizer_disc, start_epoch)
        
else:
    print('Loading the model for inference')
    weights = torch.load(cfg['BEST_CKPT'])
    generator.load_state_dict(weights['generator'])
    discriminator.load_state_dict(weights['discriminator'])
    inference(cfg, generator)