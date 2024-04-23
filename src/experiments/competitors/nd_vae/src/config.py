




EPOCHS = 10
TUNING_EPOCHS = 10
DEF_VAE_EPOCHS = 10
BATCH_SIZE = 128
LOAD_DATA = False
CLASSIFIER_PRETRAINED = False
CLASSIFIER_PRETUNED = False
VAE_PRETRAINED = False
REDUCED_DATA = True #limits data to 300 elements for testing

NVAE_PARAMS =  {  
    'x_channels' : 1,
    'pre_proc_groups' : 2,
    'encoding_channels': 8,
    'scales': 2,
    'groups': 4,
    'cells': 4
}

NOISE_MAX = .1
USE_NOISE = True