
# Pre-trained Multiple Latent Variable Generative Models are good defenders against Adversarial Attacks

This is the official github repo for the paper: Pre-trained Multiple Latent Variable Generative Models are good defenders against Adversarial Attacks (Accepted at WACV 2025)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2412.03453-red)](https://arxiv.org/abs/2412.03453)
[![WACV](https://img.shields.io/badge/WACV-2025-blue)](https://openaccess.thecvf.com/content/WACV2025/html/Serez_Pre-Trained_Multiple_Latent_Variable_Generative_Models_are_Good_Defenders_Against_WACV_2025_paper.html)

## INSTALLATION

```
# Dependencies Install 
conda env create --file environment.yml
conda activate gen_adversarial

# package install (after cloning)
pip install .
```

*Note: Check the `pytorch-cuda` version in `environment.yml` to ensure it is compatible with your cuda version.*

## MLVGMS REFERENCES AND PRE-TRAINED MODELS

### StyleGAN-E4E

Used for Experiments on Celeba-A HQ - 2 classes gender classification  

paper:  [Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766)  
github: [https://github.com/omertov/encoder4editing](https://github.com/omertov/encoder4editing)  
pretrained model: [https://github.com/omertov/encoder4editing](https://github.com/omertov/encoder4editing)  

### NVAE

Used for Experiments on Celeba-A 64 - 100 classes identity classification  

paper: [NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898)  
github (official): [https://github.com/NVlabs/NVAE](https://github.com/NVlabs/NVAE)  
github (used implementation): [https://github.com/SerezD/NVAE-from-scratch](https://github.com/SerezD/NVAE-from-scratch)  
pretrained model: [https://huggingface.co/SerezD/NVAE-from-scratch](https://huggingface.co/SerezD/NVAE-from-scratch)  

### Style-Transformer

Used for Experiments on Stanford Cars 128 - 4 classes cars classification  

paper: [Style Transformer for Image Inversion and Editing](https://arxiv.org/abs/2203.07932)  
github: [https://github.com/sapphire497/style-transformer](https://github.com/sapphire497/style-transformer)  
pretrained model:  [https://github.com/sapphire497/style-transformer](https://github.com/sapphire497/style-transformer)  

## OBTAIN DATASETS

We load the used subsets for train, validation and testing at:  
[https://huggingface.co/SerezD/gen_adversarial/tree/main/datasets](https://huggingface.co/SerezD/gen_adversarial/tree/main/datasets) 

## CLASSIFIERS TRAINING AND PRE-TRAINED MODELS

For training classifiers, run: 

```
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 ./src/classifier/train.py --run_name resnet50_celeba_gender --data_path '/path/to/dataset/' --cumulative_bs 128 --epochs 50 --model_type resnet --n_classes 2 --image_size 256

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 ./src/classifier/train.py --run_name vgg11_celeba_identities --data_path '/path/to/dataset/' --cumulative_bs 256 --lr 1e-3 --epochs 200 --model_type vgg --n_classes 100 --image_size 64

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 ./src/classifier/train.py --run_name resnext50_cars_types --data_path '/path/to/dataset/' --cumulative_bs 128 --epochs 150 --model_type resnext --n_classes 4 --image_size 128
```

The pre-trained models that we used in the experiments are available at:  
[https://huggingface.co/SerezD/gen_adversarial/tree/main/classifiers](https://huggingface.co/SerezD/gen_adversarial/tree/main/classifiers)  

## COMPETITORS REFERENCES

### ADVERSARIAL-VAE

paper: [Manifold Projection for Adversarial Defense on Face Recognition](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750290.pdf)  
github: [https://github.com/nercms-mmap/A-VAE](https://github.com/nercms-mmap/A-VAE)  

We trained A-VAE on all tasks for running the experiments shown in the paper. To train run:  
```
CUDA_VISIBLE_DEVICES=0 python ./src/defenses/competitors/a_vae/train.py --path '/path/to/train/images/folder' --img_size [64,128,256]
```
where `img_size` depends on the task (ids = 64, cars = 128, gender = 256).  

The pre-trained models that we used in the experiments are available at:  
[https://huggingface.co/SerezD/gen_adversarial/tree/main/competitors](https://huggingface.co/SerezD/gen_adversarial/tree/main/competitors)  

### ND-VAE

paper: [Noisy-Defense Variational Auto-Encoder (ND-VAE): An Adversarial Defense Framework to Eliminate Adversarial Attacks](https://ieeexplore.ieee.org/document/10387596)  
github: [https://github.com/shayan223/ND-VAE](https://github.com/shayan223/ND-VAE)  

We trained ND-VAE on all tasks for running the experiments shown in the paper. To train run:  
```
CUDA_VISIBLE_DEVICES=0 python ./src/defenses/competitors/nd_vae/train_ndvae.py --images_path '/path/to/train/images/folder' --type ['celeba256', 'celeba64', 'cars128']
```

*Note: you need to generate adversarial images for training ND-VAE. To do so, check the script `./src/defenses/competitors/nd_vae/generate_fgsm_data.py`*

The pre-trained models that we used in the experiments are available at:  
[https://huggingface.co/SerezD/gen_adversarial/tree/main/competitors](https://huggingface.co/SerezD/gen_adversarial/tree/main/competitors)  

### TRADES 

paper: [TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization](https://arxiv.org/pdf/1901.08573)  
github: [https://github.com/yaodongyu/TRADES](https://github.com/yaodongyu/TRADES)  

We fine-tuned classifiers with trades on all tasks for running the experiments shown in the paper. To train run:  
```
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 ./src/defenses/competitors/trades/fine_tune_classifier.py --run_name resnet50_celeba_gender --data_path '/path/to/train/images/folder' --cumulative_bs 64 --epochs 50 --model_type resnet --n_classes 2 --beta 1.5 --resume_from '/path/to/base/classifier.pt'

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 ./src/defenses/competitors/trades/fine_tune_classifier.py --run_name vgg11_celeba_identities --data_path '/path/to/train/images/folder' --cumulative_bs 256 --epochs 50 --model_type vgg --n_classes 100 --image_size 64 --beta 1.0 --resume_from '/path/to/base/classifier.pt'

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr='localhost' --master_port=1234 ./src/defenses/competitors/trades/fine_tune_classifier.py --run_name resnext50_cars_types --data_path '/path/to/train/images/folder' --cumulative_bs 128 --epochs 50 --model_type resnext --n_classes 4 --image_size 128 --beta 8.0 --resume_from '/path/to/base/classifier.pt'
```

The pre-trained models that we used in the experiments are available at:  
[https://huggingface.co/SerezD/gen_adversarial/tree/main/competitors](https://huggingface.co/SerezD/gen_adversarial/tree/main/competitors)  

## ALPHA LEARNING EXPERIMENTS

In order to learn the best alpha parameters (Bayesian Optimization) or to try random combinations (Grid Search), 
you need to run the following: 

```
# GENERATE ADVERSARIAL DATASETS
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/create_adversarial_dataset.py --images_folder '/path/to/train/folder/' --n_samples 1024 --results_folder '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --autoencoder_path '/path/to/pretrained/mlvgm.pt' --classifier_type 'vgg-11';
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/create_adversarial_dataset.py --images_folder '/path/to/train/folder/' --n_samples 1024 --results_folder '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --autoencoder_path '/path/to/pretrained/mlvgm.pt' --classifier_type 'resnet-50';
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/create_adversarial_dataset.py --images_folder '/path/to/train/folder/' --n_samples 1024 --results_folder '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --autoencoder_path '/path/to/pretrained/mlvgm.pt' --classifier_type 'resnext-50';

# GRID SEARCH
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/grid_search.py --adv_images_path '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --classifier_type 'resnet-50' --autoencoder_path '/path/to/pretrained/mlvgm.pt' --autoencoder_name 'E4E_StyleGAN' --n_steps 512 --results_folder './results/'
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/grid_search.py --adv_images_path '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --classifier_type 'vgg-11' --autoencoder_path '/path/to/pretrained/mlvgm.pt' --autoencoder_name 'NVAE_3x8' --n_steps 512 --results_folder './results/'
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/grid_search.py --adv_images_path '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --classifier_type 'resnext-50' --autoencoder_path '/path/to/pretrained/mlvgm.pt' --autoencoder_name 'TransStyleGan' --n_steps 512 --results_folder './results/'

# BAYESIAN OPTIMIZATION
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/bayesian_optimization.py --adv_images_path '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --classifier_type 'resnet-50' --autoencoder_path '/path/to/pretrained/mlvgm.pt' --autoencoder_name 'E4E_StyleGAN' --n_optimization_steps 95 --results_folder './results/'
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/bayesian_optimization.py --adv_images_path '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --classifier_type 'vgg-11' --autoencoder_path '/path/to/pretrained/mlvgm.pt' --autoencoder_name 'NVAE_3x8' --n_optimization_steps 95 --results_folder './results/'
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/alpha_learning/bayesian_optimization.py --adv_images_path '/path/to/adversarial/generated/folder/' --classifier_path '/path/to/pretrained/classifier.pt' --classifier_type 'resnext-50' --autoencoder_path '/path/to/pretrained/mlvgm.pt'  --autoencoder_name 'TransStyleGan' --n_optimization_steps 95 --results_folder './results/'
```

## TEST DEFENSES

Once you have obtained all the pre-trained classifiers, purification autoencoders and alpha parameters you can test
a specific defense mechanism running: 

```
# BASE MODELS
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'base' --experiment 'gender' --config './configs/no_defense_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'base' --experiment 'ids' --config './configs/no_defense_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'base' --experiment 'cars' --config './configs/no_defense_cars.yaml';

# ABLATIONS 
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ablation' --experiment 'gender' --config './configs/ablation_noise_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ablation' --experiment 'gender' --config './configs/ablation_blur_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ablation' --experiment 'ids' --config './configs/ablation_noise_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ablation' --experiment 'ids' --config './configs/ablation_blur_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ablation' --experiment 'cars' --config './configs/ablation_noise_cars.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ablation' --experiment 'cars' --config './configs/ablation_blur_cars.yaml';

# COMPETITORS
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ND-VAE' --experiment 'gender' --config './configs/competitor_ndvae_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ND-VAE' --experiment 'ids' --config './configs/competitor_ndvae_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ND-VAE' --experiment 'cars' --config './configs/competitor_ndvae_cars.yaml';

TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'A-VAE' --experiment 'gender' --config './configs/competitor_avae_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'A-VAE' --experiment 'ids' --config './configs/competitor_avae_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'A-VAE' --experiment 'cars' --config './configs/competitor_avae_cars.yaml';

TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'trades' --experiment 'gender' --config './configs/competitor_trades_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'trades' --experiment 'ids' --config './configs/competitor_trades_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'trades' --experiment 'cars' --config './configs/competitor_trades_cars.yaml';

# OURS 
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_linear_no_preprocessing_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_linear_noise_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_linear_blur_gender.yaml';

TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_cosine_no_preprocessing_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_cosine_noise_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_cosine_blur_gender.yaml';

TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_learned_no_preprocessing_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_learned_noise_gender.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'gender' --config './configs/ours_learned_blur_gender.yaml';


TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_linear_no_preprocessing_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_linear_noise_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_linear_blur_ids.yaml';

TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_cosine_no_preprocessing_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_cosine_noise_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_cosine_blur_ids.yaml';

TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_learned_no_preprocessing_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_learned_noise_ids.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'ids' --config './configs/ours_learned_blur_ids.yaml';


TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_linear_no_preprocessing_cars.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_linear_noise_cars.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_linear_blur_cars.yaml';

TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_cosine_no_preprocessing_cars.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_cosine_noise_cars.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_cosine_blur_cars.yaml';

TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_learned_no_preprocessing_cars.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_learned_noise_cars.yaml';
TORCH_CUDA_ARCH_LIST=8.0 python ./src/experiments/test_defense.py --images_path '/path/to/test/subset/folder/' --defense_type 'ours' --experiment 'cars' --config './configs/ours_learned_blur_cars.yaml';
```

*Note: remember to update the configuration file, which includes paths to pretrained models and the various parameters!*

The output is a json file that indicates the success rate for each attack/image pair. 
A success rate of 100 indicates that no adversarial image has been found.

## CITATION

```
@inproceedings{serez2025pretrained,
    author    = {Serez, Dario and Cristani, Marco and Del Bue, Alessio and Murino, Vittorio and Morerio, Pietro},
    title     = {Pre-Trained Multiple Latent Variable Generative Models are Good Defenders Against Adversarial Attacks},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {6506-6516}
}
```
