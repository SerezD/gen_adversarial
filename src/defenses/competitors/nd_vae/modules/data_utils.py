import pathlib

import numpy as np
# from art.attacks.evasion import FastGradientMethod, CarliniL2Method, PixelAttack, HopSkipJump
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize


# import torchvision
# import numpy as np
# from PIL import Image
# from matplotlib import pyplot as plt
# import os
# from config import NOISE_MAX, BATCH_SIZE



# def Generate_attack_data(attack_type,classifier,x_train,y_train,x_test,y_test,eval_only=False):
#
#     print("Min: ",np.amax(x_train))
#     print("Max: ",np.amin(x_train))
#
#     # Generate labels which are the original images, duplicate rows to match the
#     # number of adversarial samples
#     # In this case, x_train is the original image, and y_train is the label
#     y = np.concatenate((x_train,x_train))
#     y = np.concatenate((y,x_train))
#     y = np.concatenate((y,x_train))
#
#     # Duplicate image classifications as well
#     y_train_adv = np.concatenate((y_train,y_train))
#     y_train_adv = np.concatenate((y_train_adv,y_train))
#     y_train_adv = np.concatenate((y_train_adv,y_train))
#
#     # Generate adversarial training samples
#     if(attack_type == 'FGSM'):
#         attack_a = FastGradientMethod(estimator=classifier, eps=0.25)
#         attack_b = FastGradientMethod(estimator=classifier, eps=0.3)
#         attack_c = FastGradientMethod(estimator=classifier, eps=0.35)
#         attack_d = FastGradientMethod(estimator=classifier, eps=0.4)
#
#         x_fgsm_a = attack_a.generate(x=x_train)
#         x_fgsm_b = attack_b.generate(x=x_train)
#         x_fgsm_c = attack_c.generate(x=x_train)
#         x_fgsm_d = attack_d.generate(x=x_train)
#
#         # Combine training samples
#         x_fgsm = np.concatenate((x_fgsm_a,x_fgsm_b))
#         x_fgsm = np.concatenate((x_fgsm,x_fgsm_c))
#         x_fgsm = np.concatenate((x_fgsm,x_fgsm_d))
#
#
#         ##### REPEAT FOR TEST DATA #######
#
#         # Generate labels which are the original images, duplicate rows to match the
#         # number of adversarial samples
#         y_fgsm_test = np.concatenate((x_test,x_test))
#         y_fgsm_test = np.concatenate((y_fgsm_test,x_test))
#         y_fgsm_test = np.concatenate((y_fgsm_test,x_test))
#
#         # Duplicate image classifications as well
#         y_test_adv = np.concatenate((y_test,y_test))
#         y_test_adv = np.concatenate((y_test_adv,y_test))
#         y_test_adv = np.concatenate((y_test_adv,y_test))
#
#         # Generate adversarial testing samples
#         #FGSM_a_test = FastGradientMethod(estimator=classifier, eps=0.25)
#         x_fgsm_a_test = attack_a.generate(x=x_test)
#         #FGSM_b_test = FastGradientMethod(estimator=classifier, eps=0.3)
#         x_fgsm_b_test = attack_b.generate(x=x_test)
#         #FGSM_c_test = FastGradientMethod(estimator=classifier, eps=0.35)
#         x_fgsm_c_test = attack_c.generate(x=x_test)
#         #FGSM_d_test = FastGradientMethod(estimator=classifier, eps=0.4)
#         x_fgsm_d_test = attack_d.generate(x=x_test)
#
#         # Combine test samples
#         x_fgsm_test = np.concatenate((x_fgsm_a_test,x_fgsm_b_test))
#         x_fgsm_test = np.concatenate((x_fgsm_test,x_fgsm_c_test))
#         x_fgsm_test = np.concatenate((x_fgsm_test,x_fgsm_d_test))
#
#
#         return x_fgsm, y, x_fgsm_test, y_fgsm_test, y_train_adv, y_test_adv
#
#     elif(attack_type == 'CW'):
#         attack_a = CarliniL2Method(classifier=classifier, max_iter=3, verbose=True, initial_const=1)
#
#
#     elif(attack_type == 'Pixel'):
#         attack_a = PixelAttack(classifier=classifier, verbose=True)
#
#     elif(attack_type == 'HopSkipJump'):
#         attack_a = HopSkipJump(classifier=classifier, max_iter=10, max_eval=1000,
#                                init_eval=50, init_size=50,
#                                batch_size=256,verbose=True)
#
#     if(eval_only == False):
#
#         print('Generating training '+attack_type+' attack: ')
#         x_attack = attack_a.generate(x=x_train)
#         print("Done.")
#     else:
#         x_attack = None
#
#     # Generate labels which are the original images, duplicate rows to match the
#     # number of adversarial samples
#     y_cw_test = np.concatenate((x_test,x_test))
#     # Duplicate image classifications as well
#     y_test_adv = np.concatenate((y_test,y_test))
#
#     # Generate adversarial test samples
#     print('Generating testing '+attack_type+' attack: ')
#     x_attack_test = attack_a.generate(x=x_test)
#     print("Done.")
#
#
#     return x_attack, y, x_attack_test, y_cw_test, y_train_adv, y_test_adv

class ImgDataset(Dataset):

    def __init__(self, train_folder, adv_folder, image_size,
                 noisy_input=False, noise_max=0.0):

        self.transforms = Compose([ToTensor(), Resize(image_size, antialias=True)])

        train_files = sorted(list(pathlib.Path(train_folder).rglob('*.jpg')) +
                             list(pathlib.Path(train_folder).rglob('*.png')))
        self.train_files = [i.absolute().as_posix() for i in train_files]

        adv_files = sorted(list(pathlib.Path(adv_folder).rglob('*.jpg')) +
                           list(pathlib.Path(adv_folder).rglob('*.png')))
        self.adv_files = [i.absolute().as_posix() for i in adv_files]

        labels_as_str = [str(i).split('/')[-2] for i in train_files]
        class_names = sorted(list(set(labels_as_str)))
        labels = np.array([class_names.index(s) for s in labels_as_str])
        self.labels = torch.LongTensor(labels)

        self.noisy_input=noisy_input
        self.noise_layer = AddGaussianNoise(0.0, noise_max)

    def __getitem__(self, index):

        x_orig = self.transforms(Image.open(self.train_files[index]).convert('RGB'))
        x = self.transforms(Image.open(self.adv_files[index]).convert('RGB'))

        y = self.labels[index]

        if self.noisy_input:
            x = self.noise_layer(x)

        sample = {'x_adv': x, 'x_orig': x_orig, 'label': y}
        return sample

    def __len__(self):
        return len(self.labels)



# class ImgDataset_Basic(Dataset):
#     def __init__(self, imgs, labels, transform=None,channel_switch=False):
#         self.data = imgs
#         self.channel_switch = channel_switch
#         self.labels = torch.LongTensor(labels)
#         self.transform = transform
#
#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.labels[index]
#
#         if self.transform:
#             # x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
#             if(self.channel_switch == True):
#                 x = x.transpose(1,2,0)
#             x = self.transform(x)
#
#
#         sample = {'x': x, 'label': y}
#         return sample
#
#     def __len__(self):
#         return len(self.data)





# def generate_adv_datasets(data_path,ART_classifier,attack_type,x_train,y_train,x_test,y_test,load_data=False,using_NVAE=False,benign=False,guided=False,save=False,noisy=False):
#    # x values are the adversarial images
#     # y values are the original (benign) images
#
#     #data_path = '../data/'+test_name+'/'
#
#     #TODO move this out into its own function
#     if not os.path.exists(data_path):
#         os.makedirs(data_path)
#
#     if(using_NVAE):
#         # Apply the necesary preproccessing for NVAE
#         train_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Pad(padding=2),
#             transforms.ToTensor(),
#             # Binarize(),
#         ])
#
#         test_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Pad(padding=2),
#             transforms.ToTensor(),
#             # Binarize(),
#         ])
#     else:
#         train_transform = transforms.ToTensor()
#         test_transform = transforms.ToTensor()
#
#
#     if((load_data == False)):
#         x_fgsm, y_fgsm, x_fgsm_test, y_fgsm_test, y_train_labels, y_test_labels = Generate_attack_data(attack_type,ART_classifier,
#                                                                                  x_train,y_train,x_test,y_test)
#         if(benign == True):
#             train_dataset = ImgDataset_Basic(x_train,y_train,transform=train_transform,channel_switch=True)
#         elif(guided == True):
#             train_dataset = ImgDataset_guided(x_fgsm,y_fgsm)
#         else:
#             train_dataset = ImgDataset(x_fgsm,y_fgsm,y_train_labels,transform=train_transform,noisy_input=noisy)
#         test_dataset = ImgDataset(x_fgsm_test,y_fgsm_test,y_test_labels,transform=test_transform,noisy_input=noisy)
#         if(save == True):
#             print('Saving Dataset...')
#             np.save(data_path+'x_fgsm.npy',x_fgsm)
#             np.save(data_path+'y_fgsm.npy',y_fgsm)
#             np.save(data_path+'x_fgsm_test.npy',x_fgsm_test)
#             np.save(data_path+'y_fgsm_test.npy',y_fgsm_test)
#             np.save(data_path+'y_train_labels.npy',y_train_labels)
#             np.save(data_path+'y_test_labels.npy',y_test_labels)
#             print('Done.')
#     else:
#         print('Loading Dataset...')
#         x_fgsm = np.load(data_path+'x_fgsm.npy')
#         y_fgsm = np.load(data_path+'y_fgsm.npy')
#         x_fgsm_test = np.load(data_path+'x_fgsm_test.npy')
#         y_fgsm_test = np.load(data_path+'y_fgsm_test.npy')
#         y_train_labels = np.load(data_path+'y_train_labels.npy')
#         y_test_labels = np.load(data_path+'y_test_labels.npy')
#         print('Done.')
#
#         train_dataset = ImgDataset(x_fgsm,y_fgsm,y_train_labels,transform=train_transform)
#         test_dataset = ImgDataset(x_fgsm_test,y_fgsm_test,y_test_labels,transform=test_transform)
#
#     return train_dataset, test_dataset
#
#
#Taken from: https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
#
#
#
# def prep_pytorch_dataset(dataset,include_noise=False,noise_max=1.0):
#     # Load the MNIST dataset and convert to basic numpy format (split into testing and training arrays)
#
#     if(dataset == 'MNIST'):
#         #THESE ARE FOR THE MNIST DATASET
#         mean = 0.1307
#         std = .3081
#         dataset = torchvision.datasets.MNIST
#     elif(dataset == 'FashionMNIST'):
#         mean = 0.5
#         std = 0.5
#         dataset = torchvision.datasets.FashionMNIST
#     # Normalization transforms for data
#     if(include_noise == True):
#         transform_norm = transforms.Compose([
#             #transforms.ToPILImage(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#             #AddGaussianNoise(0.0,noise_max)
#         ])
#     else:
#         transform_norm = transforms.Compose([
#             #transforms.ToPILImage(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])
#
#     train_set = dataset('../data', train=True, download=True, transform=transform_norm)
#     x_train = train_set.data.numpy().astype(np.float32) / 255.0
#     x_train = np.expand_dims(x_train,1)
#     y_train = train_set.targets.numpy()
#
#     test_set = dataset('../data', train=False, download=True, transform=transform_norm)
#     x_test = test_set.data.numpy().astype(np.float32) / 255.0
#     x_test = np.expand_dims(x_test,1)
#     y_test = test_set.targets.numpy()
#
#     print(x_train.dtype,y_train.dtype,x_test.dtype,y_test.dtype)
#     return x_train, y_train, x_test, y_test