
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

from models.basic_vae_mnist import VAE
from models.mnist_classifier import Net
from models.NVAE import Defence_NVAE
from data_utils import Generate_attack_data, ImgDataset, ImgDataset_Basic, AddGaussianNoise, prep_pytorch_dataset
from config import EPOCHS, TUNING_EPOCHS, DEF_VAE_EPOCHS, BATCH_SIZE, LOAD_DATA, CLASSIFIER_PRETRAINED, CLASSIFIER_PRETUNED, VAE_PRETRAINED, REDUCED_DATA, NVAE_PARAMS, NOISE_MAX
# Find and use GPU's if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = 0.1307
std = .3081
transform_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    AddGaussianNoise(0.0,NOISE_MAX)
    ])





def run_noise_test(test_name,parameter_source,dataset,n_classes,net_model,vae_model,using_NVAE,attack_method):

    

    if(using_NVAE):
        # Apply the necesary preproccessing for NVAE
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(padding=2),
            transforms.ToTensor(),
            # Binarize(),
        ])
    else:
        test_transform = transforms.ToTensor()



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net_model.parameters(), lr=.01,
                        momentum=.001)
    
    print("###########   Loading classifier parameters   #############")
    net_model.load_state_dict(torch.load("../model_parameters/"+parameter_source+"/tuned_model.pth"))
    print("###########   Loading VAE parameters   #############")
    vae_model.load_state_dict(torch.load("../model_parameters/"+parameter_source+"/vae.pth"))

    input_size = x_train[0].shape

    classifier = PyTorchClassifier(
            model=net_model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_size,
            nb_classes=n_classes,
    )

    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    # x values are the adversarial images
    # y values are the original (benign) images
    x_fgsm, y_fgsm, x_fgsm_test, y_fgsm_test, y_train_labels, y_test_labels = attack_method(classifier,x_train,y_train,x_test,y_test)

    test_dataset = ImgDataset(x_fgsm_test,y_fgsm_test,y_test_labels,transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE)



    print("############ Validating Performance... ##########")

    torch.cuda.empty_cache() # Clear cache to avoid memory problems

    test_labels = []
    predictions = []
    net_model.eval()
    for idx, sample in enumerate(tqdm(test_loader)):

        if (using_NVAE is True):
            x_adv = sample['x_adv'].to(device)
        else:
            #x_adv = sample['x_adv'].permute(0, 2, 1, 3).to(device)
            x_adv = sample['x_adv'].to(device)
        label = sample['label']
        test_labels.append(label.numpy())
        label.to(device)
        if(using_NVAE is True):
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

    return accuracy_benign, accuracy_adv


test_name = "NVAE_defence_vae_MNIST"
save_to= "NVAE_defence_vae_MNIST_NOISY"
def_nvae = Defence_NVAE
net_model3 = Net

run_noise_test("MNIST_Noisy","MNIST/NVAE_defence_vae",datasets.MNIST,using_NVAE=True,attack_method=Generate_attack_data)

