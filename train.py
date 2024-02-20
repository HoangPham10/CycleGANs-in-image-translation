import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from dataset import ImageDataset
from config import TRAIN_DIR
from model.Generator import Generator
from model.Discriminator import Discriminator
from utils import weights_init, visualize_images
from loss import get_disc_loss, get_gen_loss
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    #TODO: Create ArgumentParser object 
    parser = argparse.ArgumentParser(description="Training script for Face Identification")
    # Add arguments
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.001 , help='learning rate')
    parser.add_argument('--input_size', type=int, default=640, help='input size image')
    parser.add_argument('--logdir', type=str, default='./exp', help='tensorboard')
    parser.add_argument('--ckpt_path', type=str, default = './weights/cycleGAN_last.pth')
    # Parse the command-line arguments
    args = parser.parse_args()

    #TODO: Hyper-parameters initialization
    print('-------------------- Hyper-parameters initialization -----------------------')
    LR, EPOCHS, BATCH_SIZE, INPUT_SIZE  = args.lr, args.epochs, args.batch_size, args.input_size
    ckpt_path = args.ckpt_path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter(log_dir=args.logdir)
    print('-------------------- ------------------------------- -----------------------')

    #TODO: dataset initialization
    print('-------------------- Dataset initialization -----------------------')
    train_tfms = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trainset = ImageDataset(TRAIN_DIR, train_tfms)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Number of training samples: {trainset.__len__()}")
    print('-------------------- ------------------------------- -----------------------')

 

    #TODO: Model initialization
    print('-------------------- Model initialization -----------------------')
    gen_AB = Generator(3,3).to(DEVICE)
    gen_BA = Generator(3,3).to(DEVICE)
    disc_A = Discriminator(3).to(DEVICE)
    disc_B = Discriminator(3).to(DEVICE)
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=LR, betas=(0.5, 0.999))
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=LR, betas=(0.5, 0.999))
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=LR, betas=(0.5, 0.999))
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location = torch.device(DEVICE))
        gen_AB.load_state_dict(checkpoint['gen_AB'])
        gen_BA.load_state_dict(checkpoint['gen_BA'])
        disc_A.load_state_dict(checkpoint['disc_A'])
        disc_B.load_state_dict(checkpoint['disc_B'])
        gen_opt.load_state_dict(checkpoint['gen_opt'])
        gen_opt.param_groups[0]['capturable'] = True
        #disc_A_opt.load_state_dict(checkpoint['disc_A_opt'])
        #disc_A_opt.param_groups[0]['capturable'] = True
        #disc_B_opt.load_state_dict(checkpoint['disc_B_opt'])
        #disc_B_opt.param_groups[0]['capturable'] = True
        if 'epoch' in checkpoint.keys():
            init_epoch = checkpoint['epoch']
        else:
            init_epoch = 100
    else:
        gen_AB = gen_AB.apply(weights_init)
        gen_BA = gen_BA.apply(weights_init)
        disc_A = disc_A.apply(weights_init)
        disc_B = disc_B.apply(weights_init)
        init_epoch = 0
    print('-------------------- ------------------------------- -----------------------')

    

    #TODO: Optimizer and loss function initialization
    print('--------------------  Optimizer and loss function initialization -----------------------')
    
    
    adv_criterion = nn.MSELoss() 
    recon_criterion = nn.L1Loss()
    
    print('-------------------- ------------------------------- -----------------------')

    
    #TODO: Train epochs
    print('-------------------- Train -----------------------')
    best_loss, best_epoch = 10000, 0
    for epoch in range(init_epoch, init_epoch + EPOCHS):
        discrimination_loss, generator_loss = 0,0
        pbar = tqdm(trainloader)
        for idx, (real_A, real_B) in enumerate(pbar):
            # real_A = nn.functional.interpolate(real_A, size=target_shape)
            # real_B = nn.functional.interpolate(real_B, size=target_shape)
            batch_size = real_A.shape[0]
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            ### Update discriminator A ###
            disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True) # Update gradients
            disc_A_opt.step() # Update optimizer

            ### Update discriminator B ###
            disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True) # Update gradients
            disc_B_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            discrimination_loss += disc_A_loss.item() 
            # Keep track of the average generator loss
            generator_loss += gen_loss.item() 
            
            pbar.set_description(f"Epoch {epoch+1}: Iteration {idx+1}/{len(trainloader)}: Generator (U-Net) loss: {generator_loss/(idx+1)}, Discriminator loss: {discrimination_loss/(idx+1)}")
        
        writer.add_scalar('Loss/discrimination', discrimination_loss/len(trainloader),global_step=epoch+1)
        writer.add_scalar('Loss/generator', generator_loss/len(trainloader), global_step=epoch+1)
        
        
            
        real_img = visualize_images(torch.cat([real_A, real_B]), size=(3, INPUT_SIZE, INPUT_SIZE)).numpy()
        fake_img = visualize_images(torch.cat([fake_B, fake_A]), size=(3, INPUT_SIZE, INPUT_SIZE)).numpy()
        #print(real_img.shape, real_img.max(), real_img.min())
        real_img = (real_img*255).astype(np.uint8)
        fake_img = (fake_img*255).astype(np.uint8)
        img = cv2.vconcat([cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR), cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)])
        cv2.imwrite(os.path.join('./visualization', f"{epoch+1}.jpg"), img)
        # writer.add_image('images', img, global_step=epoch)
                # You can change save_model to True if you'd like to save the model
        
        torch.save({
            'epoch': epoch,
            'gen_AB': gen_AB.state_dict(),
            'gen_BA': gen_BA.state_dict(),
            'gen_opt': gen_opt.state_dict(),
            'disc_A': disc_A.state_dict(),
            'disc_A_opt': disc_A_opt.state_dict(),
            'disc_B': disc_B.state_dict(),
            'disc_B_opt': disc_B_opt.state_dict()
        }, f"./weights/cycleGAN_last.pth")
        
        if best_loss > (discrimination_loss + generator_loss) / len(trainloader):
            best_loss = (discrimination_loss + generator_loss) / len(trainloader)
            best_epoch = epoch
            torch.save({
                    'epoch': epoch,
                    'gen_AB': gen_AB.state_dict(),
                    'gen_BA': gen_BA.state_dict(),
                    'gen_opt': gen_opt.state_dict(),
                    'disc_A': disc_A.state_dict(),
                    'disc_A_opt': disc_A_opt.state_dict(),
                    'disc_B': disc_B.state_dict(),
                    'disc_B_opt': disc_B_opt.state_dict()
                }, f"./weights/cycleGAN_best.pth")
                
        if best_epoch - epoch > 10:
            break
