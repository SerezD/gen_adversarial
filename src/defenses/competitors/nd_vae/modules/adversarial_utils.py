'''
Based on the following tutorial by Trusted-AI: 
https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py

'''

#from test_adversarial import BATCH_SIZE
#from types import NoneType
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import transforms

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier


from sklearn.model_selection import train_test_split

import os
from pathlib import Path
import matplotlib.pyplot as plt

from data_utils import Generate_attack_data, ImgDataset, ImgDataset_Basic, generate_adv_datasets
from models.basic_vae_mnist import VAE
from NVAE_defense_training import display_NVAE_output


# Find and use GPU's if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def adversarial_test(train, model, lr, n_classes, n_epochs, batch_size, x_train, y_train, x_test, y_test, load_data=False):

    # Use first data item to determine input size

    input_size = x_train[0].shape

    # Create the model

    model = model

    # Define the loss function and the optimizer


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_size,
        nb_classes=n_classes,
        device_type='gpu'
    )

    # Train the ART classifier
    if(train == True):
        classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=n_epochs)

    # Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test)
    accuracy_benign = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)

    # Manually check accuracy to validate
    '''
    correct_count = 0
    for i in range(predictions.shape[0]):
        pred = np.argmax(predictions[i])
        if pred == y_test[i]:
            correct_count += 1
    acc = correct_count / len(y_test)
    print("VERIFICATION OF ACCURACY: ", acc)
    '''


    print("Accuracy on benign test examples: {}%".format(accuracy_benign * 100))

    # Generate adversarial test examples
    attack = FastGradientMethod(estimator=classifier, eps=0.35)
    x_test_adv = attack.generate(x=x_test)

    # Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy_adv = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=0)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy_adv * 100))

    return accuracy_benign, accuracy_adv



'''
As per the Defense VAE paper, the training process for whitebox attacks is as follows:

There are 3 attacks: FGSM, RAND-FGSM, and CW

1. Train the CNN on the original images for 10 epochs.
2. For each clean training image:
    a. using FGSM and RAND-FGSM generate 4 images using
       epsilon values e=.25, .3, .35, and .4
    b. using CW attack, generate 4 images using learning
       rates: lr=6,8,10,12
4. Label all generated images with their corresponding 
   Clean image

'''
def whitebox_vae_training(test_name, train_end_to_end, vae, network, attack_type,
lr, n_classes,initial_epochs, n_epochs, tuning_epochs, batch_size, x_train, y_train, x_test, y_test,
load_data=False,classifier_pretrained=False,classifier_pre_tuned=False,vae_pretrained=False,
vae_training_function=None,nvae_params=None,save_to=None, guided=False, noisy=False):

    if(save_to is None):
        save_to = test_name

    torch.autograd.set_detect_anomaly(True)
    ####################################
    # STEP 1: Train the base model
    ####################################

    print('Training Base Model')

    net_model = network()

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net_model.parameters(),lr=lr)
    print(x_train.dtype,y_train.dtype,x_test.dtype,y_test.dtype)
    train_dataset_basic = ImgDataset_Basic(x_train,y_train,transform=transforms.ToTensor())
    test_dataset_basic = ImgDataset_Basic(x_test,y_test,transform=transforms.ToTensor())

    if(classifier_pretrained == True):
        print("###########   Loading initial classifier parameters   #############")
        net_model.load_state_dict(torch.load("../model_parameters/"+test_name+"/initial_classifier.pth"))
    else:
        net_model, accuracy_benign = standard_training(save_to,
                                net_model,train_dataset_basic,test_dataset_basic,
                                batch_size,initial_epochs,lr)


    # Switch the model parameters back to trainable now that evaluation is finished
    #net_model.train()

    # Use first data item to determine input size
    input_size = x_train[0].shape


    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=net_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_size,
        nb_classes=n_classes,
        device_type='gpu'
    )

    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    # Evaluate the ART classifier on adversarial test examples
    #TODO try this with a normal evaluation loop
    predictions = classifier.predict(x_test_adv)

    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

    
    correct_count = 0
    for i in range(predictions.shape[0]):
        pred = np.argmax(predictions[i])
        if pred == y_test[i]:
            correct_count += 1
    acc = correct_count / len(y_test)
    print("VERIFICATION OF ACCURACY: ", acc)
    

    #########################################
    # STEP 2: Generate adversarial Dataset
    #########################################
    using_NVAE = False
    if(nvae_params is not None):
        using_NVAE = True
    train_dataset, test_dataset = generate_adv_datasets('../data/',classifier,attack_type,
            x_train,y_train,x_test,y_test,
            load_data=load_data,using_NVAE=using_NVAE,benign=False,noisy=noisy)
    
    if(guided == True):
        guided_train_dataset, _ = generate_adv_datasets('../data/',classifier,attack_type,
            x_train,y_train,x_test,y_test,
            load_data=load_data,using_NVAE=using_NVAE,benign=False,guided=True,noisy=noisy)
 
    #########################################
    # STEP 3: Train the VAE
    #########################################

    

    if(nvae_params is not None):
        x_channels = nvae_params['x_channels']
        pre_proc_groups = nvae_params['pre_proc_groups']
        encoding_channels = nvae_params['encoding_channels']
        scales = nvae_params['scales']
        groups = nvae_params['groups']
        cells = nvae_params['cells']
        vae_model = vae(x_channels,encoding_channels,pre_proc_groups,scales,groups,cells)
        
        if(vae_pretrained == True):
            vae_model.load_state_dict(torch.load("../model_parameters/"+test_name+"/vae.pth"))
        elif(guided == True):
            raise NotImplementedError("Guided NVAE training not yet implemented!")
        else:
            vae_model,KL_losses,Recon_losses = vae_training_function(save_to,vae_model,
                                train_dataset,batch_size,n_epochs,lr)
    else:
        vae_model = vae()
        if(vae_pretrained == True):
            vae_model.load_state_dict(torch.load("../model_parameters/"+test_name+"/vae.pth"))
        elif(guided == True):
            pass
        else:
            vae_model,KL_losses,Recon_losses = vae_training(save_to,vae_model,
                                train_dataset,batch_size,n_epochs,lr)

    #########################################
    # STEP 3b: Display example VAE output
    #########################################

    if(nvae_params is not None):
        display_NVAE_output(vae_model,test_dataset,location=save_to)
    else:
        display_vae_output(vae_model,test_dataset,location=save_to)

    #########################################
    # STEP 4: Tune the CNN to the VAE outputs
    #########################################
    
    train_dataset, test_dataset = generate_adv_datasets('../data/',classifier,attack_type,
            x_train,y_train,x_test,y_test,
            load_data=load_data,using_NVAE=using_NVAE,benign=True,noisy=noisy)
    # Create fresh optimization objects (to reset loss/optimizer parameters)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net_model.parameters(), lr=.01,
                        #momentum=.001)
    optimizer = optim.Adam(net_model.parameters(),lr=lr)
    

    if(classifier_pre_tuned == False):
        net_model.train()
        print("########## Tuning Model to trained VAE... ##########")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)

        for epoch in range(tuning_epochs):#range(n_epochs):
            for idx, sample in enumerate(train_loader):

                x = sample['x'].to(device)#.permute(0,2,1,3).to(device)
                label = sample['label'].to(device)

                if (nvae_params is not None):
                    #input = vae_model.generate_sample(x_adv)
                    input = vae_model.generate_sample(x)
                    #undo the padding of 2
                    input = input[:,:,2:-2,2:-2]
                else:
                    #input, _, _ = vae_model(x_adv)
                    input, _, _ = vae_model(x)
                input = input.to(device)
                output = net_model(input)
                
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch {}: Loss {}'.format(epoch, loss))

        print("############ Saving Models... ##########")

        torch.save(net_model.state_dict(), '../model_parameters/'+ save_to +'/tuned_model.pth')
        #torch.save(vae_model.state_dict(), '../model_parameters/'+ test_name +'_vaeModel.pth')
    else: 
        net_model.load_state_dict(torch.load("../model_parameters/"+test_name+"/tuned_model.pth"))

    print("############ Testing Performance... ##########")

    test_labels = []
    predictions = []
    net_model.eval()
    for idx, sample in enumerate(tqdm(test_loader)):

        if (nvae_params is not None):
            x_adv = sample['x_adv'].to(device)
        else:
            #x_adv = sample['x_adv'].permute(0, 2, 1, 3).to(device)
            x_adv = sample['x_adv'].to(device)
        label = sample['label']
        test_labels.append(label.numpy())
        label.to(device)
        if(nvae_params is not None):
            output = vae_model.generate_sample(x_adv)
            # undo the padding of 2
            output = output[:,:,2:-2,2:-2]
        else:
            output, _, _ = vae_model(x_adv)
        output = output.to(device)
        output = net_model(output)
        predictions.append(np.argmax(output.detach().cpu().numpy(),axis=1))


    accuracy_adv = 0

    for batch_num in range(len(predictions)):
        pred_batch = predictions[batch_num]
        label_batch = test_labels[batch_num]
        # Sum the number of correct (matching) elements per batch

        accuracy_adv += np.count_nonzero(pred_batch==label_batch)


    accuracy_adv = accuracy_adv / len(test_dataset)

    print("Accuracy on adversarial test examples: {}%".format(accuracy_adv * 100))
    


    ################################################################
    # STEP 4: Create new adversarial attacks and verify performance
    ################################################################

    if(using_NVAE):
        # Apply the necesary preproccessing for NVAE
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(padding=2),
            transforms.ToTensor(),
            # Binarize(),
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(padding=2),
            transforms.ToTensor(),
            # Binarize(),
        ])
    else:
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()

    classifier = PyTorchClassifier(
            model=net_model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_size,
            nb_classes=n_classes,
            device_type='gpu'
    )

    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    # x values are the adversarial images
    # y values are the original (benign) images
    
    x_fgsm, y_fgsm, x_fgsm_test, y_fgsm_test, y_train_labels, y_test_labels = Generate_attack_data(attack_type,classifier,x_train,y_train,x_test,y_test)

    #TODO test on CW data as well
    #CW_attack = CarliniL2Method(classifier=classifier, max_iter=10, verbose=False)

    train_dataset = ImgDataset(x_fgsm,y_fgsm,y_train_labels,transform=train_transform,noisy_input=noisy)
    test_dataset = ImgDataset(x_fgsm_test,y_fgsm_test,y_test_labels,transform=test_transform,noisy_input=noisy)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)



    print("############ Validating Performance... ##########")

    torch.cuda.empty_cache() # Clear cache to avoid memory problems

    test_labels = []
    predictions = []
    net_model.eval()
    for idx, sample in enumerate(tqdm(test_loader)):

        if (nvae_params is not None):
            x_adv = sample['x_adv'].to(device)
        else:
            x_adv = sample['x_adv'].to(device)
        label = sample['label']
        test_labels.append(label.numpy())
        label.to(device)
        if(nvae_params is not None):
            output = vae_model.generate_sample(x_adv)
            # undo the padding of 2
            output = output[:,:,2:-2,2:-2]
        else:
            output, _, _ = vae_model(x_adv)
        output = output.to(device)
        output = net_model(output)
        predictions.append(np.argmax(output.detach().cpu().numpy(),axis=1))


    accuracy_adv = 0

    for batch_num in range(len(predictions)):
        pred_batch = predictions[batch_num]
        label_batch = test_labels[batch_num]
        # Sum the number of correct (matching) elements per batch
        accuracy_adv += np.count_nonzero(pred_batch==label_batch)


    accuracy_adv = accuracy_adv / len(test_dataset)

    print("Validation Accuracy on adversarial test examples: {}%".format(accuracy_adv * 100))

    return accuracy_benign, accuracy_adv, KL_losses, Recon_losses








def standard_training(test_name,model,train_set,test_set,batch_size,n_epochs,lr=0.001,pre_trained=False):


        
    Path("../model_parameters/"+test_name).mkdir(parents=True, exist_ok=True)
        
    model_path = '../model_parameters/'+ test_name +'/initial_classifier.pth'

    print('Training...')


    if(pre_trained == True):
        model.load_state_dict(torch.load(model_path))

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)


    train_loader_basic = torch.utils.data.DataLoader(
        train_set,batch_size=batch_size, shuffle=True)
    test_loader_basic = torch.utils.data.DataLoader(test_set,batch_size=batch_size)



    model.to(device)

    for epoch in range(n_epochs):
        for idx, sample in enumerate(train_loader_basic, 0):
            x = sample['x'].permute(0, 2, 1, 3).to(device)
            label = sample['label'].to(device)

            output = model(x)
            output = output.to(device)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}: Loss {}'.format(epoch, loss))

    # Evaluate the classifier on benign test examples

    print('Evaluating base model on Benign samples:')

    model.eval()

    test_labels = []
    predictions = []

    for idx, sample in enumerate(tqdm(test_loader_basic)):
        x = sample['x'].permute(0, 2, 1, 3).to(device)
        label = sample['label']
        test_labels.append(label.numpy())
        label = label.to(device)
        output = model(x)

        predictions.append(np.argmax(output.detach().cpu().numpy(),axis=1))



    accuracy = 0

    for batch_num in range(len(predictions)):
        pred_batch = predictions[batch_num]
        label_batch = test_labels[batch_num]
        # Sum the number of correct (matching) elements per batch
        accuracy += np.count_nonzero(pred_batch==label_batch)


    accuracy = accuracy / len(test_set)


    print("Accuracy on benign test examples: {}%".format(accuracy * 100))


    print("############ Saving Models... ##########")

    torch.save(model.state_dict(), model_path)

    print("################   Done.   ##############")

    # Switch the model parameters back to trainable now that evaluation is finished
    model.train()


    return model, accuracy



# TODO Determine if the classifier object needs the optimizer for the gradient
def generate_adv_data(model):
    pass




def vae_training(test_name,vae_model,train_dataset,batch_size,n_epochs,lr,pre_trained=False):

    model_path = '../model_parameters/'+ test_name +'/vae.pth'

    print('Training...')


    if(pre_trained == True):
        vae_model.load_state_dict(torch.load(model_path))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size, shuffle=True)

        # Define custom Loss function
    def defence_vae_loss(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD


    #Initialize vae model weights to be between 0.08 and -0.08 to increase network stability
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight,-0.08,0.08)
            if(m.bias is not None):
                m.bias.data.fill_(0.01)



    
    vae_model.apply(init_weights)

    
    vae_model.to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)
    print("###### Beginning VAE Training... ######")
    KL_losses = []
    Recon_losses = []
    for epoch in range(n_epochs):
        avg_batch_KL = []
        avg_batch_recon = []
        for idx,sample in enumerate(train_loader, 0):
            x_adv = sample['x_adv']
            x_adv = x_adv.to(device)
            x_orig = sample['x_orig']
            x_orig = x_orig.to(device)
           
            # Feeding a batch of images into the network to obtain the output image, mu, and logVar

            out, mu, logVar = vae_model(x_adv)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            #kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            #loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

            cur_loss, Recon_loss, KL_loss = defence_vae_loss(out,x_orig,mu,logVar)

            #Record batch loss
            avg_batch_recon.append(torch.mean(Recon_loss).item())
            avg_batch_KL.append(torch.mean(KL_loss).item())
            ''' Un comment to view inputs and labels
            plt.subplot(121)
            plt.imshow(np.transpose(input[0].cpu().detach().numpy(), [1,2,0]))
            plt.subplot(122)
            plt.title(y_train_labels[i])
            plt.imshow(np.transpose(y_fgsm[i][0].cpu().detach().numpy(), [1,2,0]))
            plt.show()
            plt.clf()
            '''

            # Backpropagation based on the loss
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
        
        #Record average losses for epoch
        KL_losses.append(sum(avg_batch_KL)/len(avg_batch_KL))
        Recon_losses.append(sum(avg_batch_recon)/len(avg_batch_recon))



        print('Epoch {}: Loss {}'.format(epoch, cur_loss))
    
    print("###### VAE Training Complete! ######")

    print("############ Saving Models... ##########")

    torch.save(vae_model.state_dict(), '../model_parameters/'+ test_name +'/vae.pth')

    print("################   Done.   ##############")

    return vae_model,KL_losses,Recon_losses



def display_vae_output(vae_model,test_data,location="defense_vae_samples"):

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
            #out, mu, logVAR = vae_model(imgs.permute(0,2,1,3))
            out, mu, logVAR = vae_model(imgs)
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













