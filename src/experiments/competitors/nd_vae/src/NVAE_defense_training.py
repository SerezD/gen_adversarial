
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

from src.experiments.competitors.nd_vae.src.models.NVAE_utils import DiscMixLogistic

# Find and use GPU's if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''The generalized defense vae training function adapted for NVAE'''

def NVAE_Defense_training(test_name,vae_model,train_dataset,batch_size,n_epochs,lr,pre_trained=False):


    if not os.path.exists(f'./{test_name}/model_parameters'):
        os.makedirs(f'./{test_name}/model_parameters')

    if not os.path.exists(f'./{test_name}/figures'):
        os.makedirs(f'./{test_name}/figures')

    print('Training...')


    if(pre_trained == True):
        model_path = '../model_parameters/' + test_name + '_vaeModel.pth'
        vae_model.load_state_dict(torch.load(model_path))



    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size, shuffle=True)



    '''
    #Initialize vae model weights to be between 0.08 and -0.08 to increase network stability
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight,-0.08,0.08)
            if(m.bias is not None):
                m.bias.data.fill_(0.01)
    #vae_model.logvar_layer.weight.data.uniform_(-.08,0.08)
    #vae_model = VAE()
    
    vae_model.apply(init_weights)
    '''
    
    vae_model.to(device)
    optimizer =  torch.optim.Adamax(vae_model.parameters(), lr, weight_decay=1e-2, eps=1e-3)
    print("###### Beginning VAE Training... ######")

    #Count steps for annealing purposes
    global_step = 0
    num_total_iter = n_epochs * len(train_loader)
    KL_losses = []
    Recon_losses = []
    for epoch in range(n_epochs):
        avg_batch_KL = []
        avg_batch_recon = []
        for idx,sample in enumerate(tqdm(train_loader), 0):
            x_adv = sample['x_adv']
            x_adv = x_adv.to(device).clamp(0, 1)
            x_orig = sample['x_orig']
            x_orig = x_orig.to(device).clamp(0, 1)

            out, log_q, log_p, kl_all, kl_diag = vae_model(x_adv)

            loss, Recon_loss, KL_loss= vae_model.loss(out,x_orig,log_q, log_p, kl_all, kl_diag, global_step,num_total_iter)

            #Record batch loss  
            avg_batch_recon.append(torch.mean(Recon_loss).item())
            avg_batch_KL.append(torch.mean(KL_loss).item())
            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            # if idx == 8:
            #     break

        # Reconstruct and save images
        recons = DiscMixLogistic(out).mean()
        display = make_grid(
            torch.cat([x_orig, x_adv, recons], dim=0), nrow=batch_size
        ).permute(1, 2, 0).cpu().numpy()

        plt.imshow(display)
        plt.savefig(f'./{test_name}/figures/epoch={epoch}.png')
        plt.close()

        #Record average losses for epoch
        KL_losses.append(sum(avg_batch_KL)/len(avg_batch_KL))
        Recon_losses.append(sum(avg_batch_recon)/len(avg_batch_recon))

        print('Epoch {}: Loss {}'.format(epoch, loss))
    
    print("###### VAE Training Complete! ######")

    print("############ Saving Models... ##########")

    torch.save(vae_model.state_dict(), f'./{test_name}/model_parameters/nd_vae.pth')

    print("################   Done.   ##############")

    return vae_model,KL_losses,Recon_losses



def display_NVAE_output(vae_model,test_data,location='defense_nvae_samples'):

    Path('../results/'+location).mkdir(parents=True, exist_ok=True)

    test_loader = torch.utils.data.DataLoader(test_data,batch_size=1)

    sample_count = 10
    image_count = 1
    vae_model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            benign_imgs = sample['x_orig']
            benign_imgs = benign_imgs.to(device)
            benign_imgs = np.transpose(benign_imgs[0].cpu().numpy(), [1,0,2])
            plt.suptitle(sample['label'].item())
            plt.subplot(131)
            plt.imshow(np.squeeze(benign_imgs))
            plt.title("Benign")
            imgs = sample['x_adv']
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1,0,2])
            plt.subplot(132)
            plt.imshow(np.squeeze(img))
            plt.title("Adversarial")
            out = vae_model.generate_sample(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1,0,2])
            plt.subplot(133)
            plt.imshow(np.squeeze(outimg))
            plt.title("Defence VAE")
            plt.savefig('../results/'+location+'/defense_vae_example'+str(image_count)+'.png')
            plt.clf()
            image_count += 1
            if (image_count > sample_count):
                break

    vae_model.train()