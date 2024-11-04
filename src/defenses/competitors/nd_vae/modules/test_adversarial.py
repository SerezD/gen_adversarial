


import torch





import os
import pandas as pd
import matplotlib.pyplot as plt

from models.basic_vae_mnist import VAE
from models.deeper_vae import deeper_VAE
from models.mnist_classifier import Net
from models.vae_ncell import VAE_Ncell
from models.NVAE import Defence_NVAE
from adversarial_utils import  whitebox_vae_training
from data_utils import prep_pytorch_dataset
from NVAE_defense_training import NVAE_Defense_training
from generate_plots import generate_plots
from zero_shot_test import zero_shot_eval
from config import EPOCHS, TUNING_EPOCHS, DEF_VAE_EPOCHS, BATCH_SIZE, LOAD_DATA, CLASSIFIER_PRETRAINED, CLASSIFIER_PRETUNED, VAE_PRETRAINED, REDUCED_DATA, NVAE_PARAMS, USE_NOISE


'''Below are all the tests, feel free to comment out all the ones you don't want'''


def run_defVAE_tests(dataset,dataset_Name,attack_type,include_noise=False):
    data_path = '../data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        #LOAD_DATA = False
    result_path = '../results/'+dataset_Name+'/'
    print(result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(result_path)


    results = []
    plot_values = []


    x_train, y_train, x_test, y_test = prep_pytorch_dataset(dataset)

    if(REDUCED_DATA):
        x_train = x_train[:30]
        y_train = y_train[:30]
        x_test = x_test[:30]
        y_test = y_test[:30]



    
    print('###################################################')
    print('#########   DEFENSE-VAE MNIST Model   #############')
    print('###################################################')

    vae_model = VAE
    net_model = Net

    #These will be the training parameters/utilities for the non-vae model


    current_test_name = dataset_Name+'/defence_vae'
    basic_defVAE_acc, basic_defVAE_acc_adv, basic_KL_losses, basic_recon_losses= whitebox_vae_training(test_name=current_test_name,train_end_to_end=False, 
                                    vae=vae_model, network=net_model, attack_type=attack_type, lr=0.001, 
                                    n_classes=10, batch_size=BATCH_SIZE,initial_epochs=EPOCHS, n_epochs=DEF_VAE_EPOCHS, tuning_epochs=TUNING_EPOCHS,
                                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,load_data=False,noisy=include_noise)
    results.append(['defence_vae',basic_defVAE_acc, basic_defVAE_acc_adv])
    plot_values.append(['defence_vae',basic_KL_losses, basic_recon_losses])

    torch.cuda.empty_cache()

    print('###################################################')
    print('######   Deeper DEFENSE-VAE MNIST Model   #########')
    print('###################################################')
    

    vae_model = deeper_VAE
    net_model = Net
    current_test_name = dataset_Name+'/deeper_defence_vae'
    deepVAE_acc, deepVAE_acc_adv, deep_KL_losses, deep_recon_losses = whitebox_vae_training(test_name=current_test_name,train_end_to_end=False, 
                                    vae=vae_model, network=net_model, attack_type=attack_type, lr=0.001, 
                                    n_classes=10, batch_size=BATCH_SIZE,initial_epochs=EPOCHS, n_epochs=DEF_VAE_EPOCHS, tuning_epochs=TUNING_EPOCHS,
                                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,load_data=False,noisy=include_noise)
    results.append(['deeper_defence_vae',deepVAE_acc, deepVAE_acc_adv])
    plot_values.append(['deeper_defence_vae',deep_KL_losses, deep_recon_losses])
    
    torch.cuda.empty_cache()

    print('###################################################')
    print('###### Neuvough cell DEFENSE-VAE MNIST Model   ####')
    print('###################################################')


    vae_model = VAE_Ncell
    net_model = Net
    current_test_name = dataset_Name+'/Ncell_defence_vae'
    NcellVAE_acc, NcellVAE_acc_adv, ncell_KL_losses, ncell_recon_losses = whitebox_vae_training(test_name=current_test_name,train_end_to_end=False, 
                                    vae=vae_model, network=net_model, attack_type=attack_type, lr=0.001, 
                                    n_classes=10, batch_size=BATCH_SIZE,initial_epochs=EPOCHS, n_epochs=DEF_VAE_EPOCHS, tuning_epochs=TUNING_EPOCHS,
                                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,load_data=False,noisy=include_noise)
    results.append(['Ncell_defence_vae',NcellVAE_acc, NcellVAE_acc_adv])
    plot_values.append(['Ncell_defence_vae',ncell_KL_losses, ncell_recon_losses])

    torch.cuda.empty_cache()


    
    print('###################################################')
    print('############   NVAE MNIST Model   #################')
    print('###################################################')
    
    
    vae_model = Defence_NVAE
    net_model = Net
    current_test_name = dataset_Name+'/NVAE_defence_vae'
    #train_NVAE()
    NVAE_acc, NVAE_acc_adv, NVAE_KL_losses, NVAE_recon_losses = whitebox_vae_training(test_name=current_test_name,train_end_to_end=False, 
                                    vae=vae_model, network=net_model, attack_type=attack_type, lr=0.001, 
                                    n_classes=10, batch_size=BATCH_SIZE,initial_epochs=EPOCHS, n_epochs=DEF_VAE_EPOCHS, tuning_epochs=TUNING_EPOCHS,
                                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,load_data=LOAD_DATA,
                                    classifier_pretrained=CLASSIFIER_PRETRAINED,classifier_pre_tuned=CLASSIFIER_PRETUNED,vae_pretrained=VAE_PRETRAINED,
                                    vae_training_function=NVAE_Defense_training,nvae_params=NVAE_PARAMS,noisy=include_noise)
    results.append(['NVAE_defence_vae',NVAE_acc, NVAE_acc_adv])
    plot_values.append(['NVAE_defence_vae',NVAE_KL_losses, NVAE_recon_losses])
                         
    ######################################################

    # Create data table with results:

    result_table = pd.DataFrame(results,columns=['Model Name','Benign', 'Adversarial'])
    result_table.to_csv('../results/'+dataset_Name+'/DefenseVAE_result_table.csv')
    plot_value_table = pd.DataFrame(plot_values,columns=['Model Name','KL losses', 'Reconstruction losses'])
    plot_value_table.to_csv('../results/'+dataset_Name+'/DefenseVAE_losses.csv')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.table(cellText = result_table.values,
            rowLabels = result_table.index,
            colLabels = result_table.columns,
            loc = "center"
            )
    ax.set_title("Defense VAE Results")

    ax.axis("off")
    plt.savefig('../results/'+dataset_Name+'/DefenseVAE_Results.png')
    plt.clf()
attack_list = ['FGSM']

for attack in attack_list:

    print('###################################################')
    print('                Current Attack: '+attack+'       ')
    print('###################################################')

    attack_tag = '_'+attack 
    run_defVAE_tests('MNIST',"MNIST_Noisy"+attack_tag,attack,include_noise=True)
    generate_plots("MNIST_Noisy"+attack_tag)
    run_defVAE_tests('MNIST',"MNIST"+attack_tag,attack,include_noise=False)
    generate_plots("MNIST"+attack_tag)

    attack_list = ['HopSkipJump','CW']
    zero_shot_eval('MNIST_Noisy'+attack_tag,'MNIST',attack_list)
    zero_shot_eval('MNIST'+attack_tag,'MNIST',attack_list)

    run_defVAE_tests('FashionMNIST',"FashionMNIST_Noisy"+attack_tag,attack,include_noise=True)
    generate_plots("FashionMNIST_Noisy"+attack_tag)
    run_defVAE_tests('FashionMNIST',"FashionMNIST"+attack_tag,attack,include_noise=False)
    generate_plots("FashionMNIST"+attack_tag)

    attack_list = ['HopSkipJump','CW']
    zero_shot_eval('FashionMNIST_Noisy'+attack_tag,'FashionMNIST',attack_list)
    zero_shot_eval('FashionMNIST'+attack_tag,'FashionMNIST',attack_list)