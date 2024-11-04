import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

def generate_plots(dataset_name):

    results_path = '../results/'+dataset_name+'/'

    result_table = pd.read_csv(results_path+'DefenseVAE_result_table.csv',index_col=0)
    plot_values = pd.read_csv(results_path+'DefenseVAE_losses.csv',index_col=0)

    print(result_table)
    print(plot_values)
    final_accuracies = []
    final_kl = []
    final_recon = []
    model_names = []
    for i in range(len(plot_values)):

        model_name = plot_values.iloc[i]['Model Name']
        recon_losses = literal_eval(plot_values.iloc[i]['Reconstruction losses'])
        kl_losses = literal_eval(plot_values.iloc[i]['KL losses'])

        print("Reconstruction Losses: ")
        print(recon_losses)
        print("KL Losses: ")
        print(kl_losses)

        #plt.figure(model_name)
        ymax = max(kl_losses)*2
        ymin = 0
        ax = plt.gca()
        ax.set_ylim([ymin,ymax])
        plt.plot(range(len(recon_losses)), recon_losses,label='Reconstruction Loss')
        plt.plot(range(len(kl_losses)), kl_losses,label='KL Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(model_name+' Loss')
        plt.legend()
        plt.savefig(results_path+model_name+'_loss.png')
        plt.clf()
        model_names.append(model_name)
        max_accuracy = result_table.loc[result_table['Model Name'] == model_name]['Adversarial']
        final_accuracies.append(max_accuracy)
        final_kl.append(kl_losses[-1])
        final_recon.append(recon_losses[-1])


    #label each point with its corresponding model
    plt.title('KL Loss and Final Accuracy by Model')
    plt.xlabel('KL Loss')
    plt.ylabel('Final Accuracy')
    plt.scatter(final_kl,final_accuracies)
    for i in range(len(final_accuracies)):
        plt.annotate(model_names[i],(final_kl[i],final_accuracies[i]))
    plt.savefig(results_path+'kl_to_acc.png')
    plt.clf()

    plt.title('Reconstruction Loss and Final Accuracy by Model')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Final Accuracy')
    plt.scatter(final_recon,final_accuracies)
    for i in range(len(final_accuracies)):
        plt.annotate(model_names[i],(final_recon[i],final_accuracies[i]))
    plt.savefig(results_path+'recon_to_acc.png')


    
#generate_plots('MNIST')