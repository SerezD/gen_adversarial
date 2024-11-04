import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from art.estimators.classification import PyTorchClassifier

from models.basic_vae_mnist import VAE
from models.deeper_vae import deeper_VAE
from models.mnist_classifier import Net
from models.vae_ncell import VAE_Ncell
from models.NVAE import Defence_NVAE
from data_utils import prep_pytorch_dataset,ImgDataset, Generate_attack_data

from config import EPOCHS, TUNING_EPOCHS, DEF_VAE_EPOCHS, BATCH_SIZE, LOAD_DATA, CLASSIFIER_PRETRAINED, CLASSIFIER_PRETUNED, VAE_PRETRAINED, REDUCED_DATA, NVAE_PARAMS, USE_NOISE

# Find and use GPU's if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_eval(net_model, vae_model, dataset, attack_type, n_classes=10, using_NVAE=False):

    x_train, y_train, x_test, y_test = prep_pytorch_dataset(dataset)

    if(REDUCED_DATA):
        x_train = x_train[:30]
        y_train = y_train[:30]
        x_test = x_test[:30]
        y_test = y_test[:30]
    else:
        x_train = x_train[:8000]
        y_train = y_train[:8000]
        x_test = x_test[:8000]
        y_test = y_test[:8000]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net_model.parameters(),lr=0.01)
    input_size = x_train[0].shape

    ################################################################
    #      Create adversarial attacks and verify performance       #
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
            device_type='gpu',
            #clip_values=(0,1)
    )


    # x values are the adversarial images
    # y values are the original (benign) images
    
    x_cw, y_cw, x_cw_test, y_cw_test, y_train_labels, y_test_labels = Generate_attack_data(attack_type,classifier,x_train,y_train,x_test,y_test,eval_only=True)


    test_dataset = ImgDataset(x_cw_test,y_cw_test,y_test_labels,transform=test_transform,noisy_input=USE_NOISE)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE)

    print("############ Validating Performance... ##########")

    torch.cuda.empty_cache() # Clear cache to avoid memory problems

    test_labels = []
    predictions = []
    net_model = net_model.to(device)
    net_model.eval()
    vae_model.to(device)
    vae_model.eval()
    for idx, sample in enumerate(tqdm(test_loader)):

        if (using_NVAE):
            x_adv = sample['x_adv'].to(device)
        else:
            #x_adv = sample['x_adv'].permute(0, 2, 1, 3).to(device)
            x_adv = sample['x_adv'].to(device)
        label = sample['label']
        test_labels.append(label.numpy())
        label.to(device)
        if(using_NVAE):
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

    return accuracy_adv


def zero_shot_eval(test_name,dataset,attack_list):

    results = []

    path = '../model_parameters/'+test_name+'/'

    def_vae_classifier = Net()
    def_vae_classifier.load_state_dict(torch.load(path+'defence_vae/tuned_model.pth'))
    def_vae_classifier.eval()
    def_vae = VAE()
    def_vae.load_state_dict(torch.load(path+'defence_vae/vae.pth'))
    def_vae.eval()

    for attack in attack_list:
        acc = model_eval(def_vae_classifier,def_vae,dataset,attack)
        results.append(['Defense VAE',attack,acc])

    torch.cuda.empty_cache()

    deep_def_vae_classifier = Net()
    deep_def_vae_classifier.load_state_dict(torch.load(path+'deeper_defence_vae/tuned_model.pth'))
    deep_def_vae_classifier.eval()
    deep_def_vae = deeper_VAE()
    deep_def_vae.load_state_dict(torch.load(path+'deeper_defence_vae/vae.pth'))
    deep_def_vae.eval()

    for attack in attack_list:
        acc = model_eval(deep_def_vae_classifier,deep_def_vae,dataset,attack)
        results.append(['Deeper Defense VAE',attack,acc]) 

    torch.cuda.empty_cache()

    ncell_def_vae_classifier = Net()
    ncell_def_vae_classifier.load_state_dict(torch.load(path+'Ncell_defence_vae/tuned_model.pth'))
    ncell_def_vae_classifier.eval()
    ncell_def_vae = VAE_Ncell()
    ncell_def_vae.load_state_dict(torch.load(path+'Ncell_defence_vae/vae.pth'))
    ncell_def_vae.eval()

    for attack in attack_list:
        acc = model_eval(ncell_def_vae_classifier,ncell_def_vae,dataset,attack)
        results.append(['Ncell Defense VAE',attack,acc]) 

    torch.cuda.empty_cache()
    
    nvae_def_vae_classifier = Net()
    nvae_def_vae_classifier.load_state_dict(torch.load(path+'NVAE_defence_vae/tuned_model.pth'))
    nvae_def_vae_classifier.eval()
    x_channels = NVAE_PARAMS['x_channels']
    pre_proc_groups = NVAE_PARAMS['pre_proc_groups']
    encoding_channels = NVAE_PARAMS['encoding_channels']
    scales = NVAE_PARAMS['scales']
    groups = NVAE_PARAMS['groups']
    cells = NVAE_PARAMS['cells']
    nvae_def_vae = Defence_NVAE(x_channels,encoding_channels,pre_proc_groups,scales,groups,cells)
    nvae_def_vae.load_state_dict(torch.load(path+'NVAE_defence_vae/vae.pth'))
    nvae_def_vae.eval()

    for attack in attack_list:
        acc = model_eval(nvae_def_vae_classifier,nvae_def_vae,dataset,attack,using_NVAE=True)
        results.append(['NVAE Defense VAE',attack,acc]) 

        ######################################################

    # Create data table with results:

    #results = np.array([[NcellVAE_acc,NcellVAE_acc_adv]])
    result_table = pd.DataFrame(results,columns=['Model Name','Attack Name', 'Accuracy'])
    result_table.to_csv('../results/'+test_name+'/DefenseVAE_ZeroShot_table.csv')

    #dfi.export(result_table,"../DefenseVAE_Results_Model_A.png")
    fig = plt.figure()#(figsize = (8, 2))
    ax = fig.add_subplot(111)

    ax.table(cellText = result_table.values,
            rowLabels = result_table.index,
            colLabels = result_table.columns,
            loc = "center"
            )
    ax.set_title("Defense VAE Zero-Shot Evaluation")

    ax.axis("off")
    plt.savefig('../results/'+test_name+'/DefenseVAE_ZeroShot.png')
    plt.clf()


#zero_shot_eval('MNIST_Noisy_FGSM','MNIST')
#attack_list = ['HopSkipJump','CW']
#zero_shot_eval('FashionMNIST_Noisy_FGSM','FashionMNIST',attack_list)
#zero_shot_eval('FashionMNIST_FGSM','FashionMNIST',attack_list)