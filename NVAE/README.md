# NVAE 
General Rules from N_VAE paper:
1. **Weight Normalization**: "we apply WN to any convolutional layer that is not followed by BN." In this repo we use: `torch.nn.utils.parametrizations.weight_norm`


Custom OPS contains original NVIDIA code implementing operations as:
"BN - SWISH". IT uses also the original code from the Pytorch Library.