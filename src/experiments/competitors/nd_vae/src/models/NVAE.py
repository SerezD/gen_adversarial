from tkinter import W
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
from .NVAE_utils import kl_balancer, reconstruction_loss, kl_balancer_coeff, decode_output, DiscMixLogistic
from .NVAE_utils import kl_coeff as kl_coeff_func
from tqdm import tqdm
from math import sqrt
"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# These are potentially parameters, for now they will be constants
KL_ANNEAL_PORTION = 0.3 # fraction of total training we anneal over
KL_CONST_PORTION = 0.0001
KL_CONST_COEFF = 0.0001

# Swish activation function
class Swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)


class CustomSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

def activate_swish(t):
    # The following implementation has lower memory.
    return CustomSwish.apply(t)

# Taken from https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/9
class depthwise_separable_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=5, kernels_per_layer=1,padding=0):
        super(depthwise_separable_conv, self).__init__()
        #TODO cha
        self.depthwise = nn.Conv2d(channels_in, channels_in * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=channels_in)
        self.pointwise = nn.Conv2d(channels_in * kernels_per_layer, channels_out, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



class SE_Block(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE_Block, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se

#The following 3 functions (soft_clamp,normal jit, and the normal class) are from this portion of the 
# Original NVAE paper code: https://github.com/NVlabs/NVAE/blob/9fc1a288fb831c87d93a4e2663bc30ccf9225b29/distributions.py#L27
@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)    #  5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]


@torch.jit.script
def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps

class Normal:
    def __init__(self, mu, log_sigma, temp=1.):
        self.mu = soft_clamp5(mu)
        log_sigma = soft_clamp5(log_sigma)
        self.sigma = torch.exp(log_sigma) + 1e-2      # we don't need this after soft clamp
        if temp != 1.:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)

# From the NVAE code, used as skip connection during upsampling encoder cells
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.swish = activate_swish#Swish()
        self.conv_1 = nn.Conv2d(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_2 = nn.Conv2d(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_3 = nn.Conv2d(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_4 = nn.Conv2d(C_in, C_out - 3 * (C_out // 4), 1, stride=2, padding=0, bias=True)

    def forward(self, x):
        out = self.swish(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])

        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out

#From NVAE code, used to upsample with -1 stride in decoder
class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)



#Used for skip connection for equal in/out channel cells
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Generative_Cell_NVAE(nn.Module):
    def __init__(self, in_channels,E_param,stride=1):
        super(Generative_Cell_NVAE,self).__init__()
        self.out_channels = in_channels
        self.upsample = False
        if (stride == -1):
            # self.skip = nn.Sequential(UpSample(), nn.Conv2D(in_channels, int(in_channels / channel_mult), kernel_size=1))
            self.skip = nn.Sequential(UpSample(), nn.Conv2d(in_channels, int(in_channels // 2), kernel_size=1))
            self.out_channels = in_channels // 2
            stride = 1
            self.upsample = True
            self.upsample_layer = nn.UpsamplingNearest2d(scale_factor=2)
        else:
            self.skip = Identity()
        self.in_chan = in_channels
        self.E = E_param
        self.expanded_channels = in_channels * self.E

        #Initial and final batch normalization layers
        self.bn1 = nn.BatchNorm2d(self.in_chan)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        #Batch normalization with expanded dimensions, to be used with swish activation
        self.bn_expanded1 = nn.BatchNorm2d(self.expanded_channels)
        self.bn_expanded2 = nn.BatchNorm2d(self.expanded_channels)

        #Expand channels to E * C
        self.expand = nn.Conv2d(in_channels=self.in_chan, out_channels=self.expanded_channels, kernel_size=1)

        self.bnSwish1 = Swish()

        #create 5x5 depth wise seperable convolution
        self.dep_sep_conv = depthwise_separable_conv(self.expanded_channels,self.expanded_channels,kernel_size=5,padding=2)

        self.bnSwish2 = Swish()

        #Map channels back to original size
        self.expand2 = nn.Conv2d(in_channels=self.expanded_channels, out_channels=self.out_channels, kernel_size=1)


        self.squeeze_excitation = SE_Block(self.out_channels,self.out_channels)

        self.cell = nn.ModuleList([
            self.bn1,
            #TODO consider adding swish here as well
            self.expand,
            self.bn_expanded1,
            self.bnSwish1,
            self.dep_sep_conv,
            self.bn_expanded2,
            self.bnSwish2,
            self.expand2,
            self.bn2,
            self.squeeze_excitation
        ])

        if self.upsample == True:
            self.cell = nn.ModuleList([self.upsample_layer,*self.cell])

        self.cell = nn.Sequential(*self.cell)
        
    
    def forward(self, x):
        skip_output = self.skip(x)
        output = self.cell(x)

        #TODO Check whether all cells need to be skip + 0.1 * output
        return skip_output + output




class EncCombinerCell(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super(EncCombinerCell, self).__init__()
        self.conv = nn.Conv2d(decoder_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out


# original combiner
class DecCombinerCell(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super(DecCombinerCell, self).__init__()
        self.conv = nn.Conv2d(encoder_channels + decoder_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out

#These are used for encoding (Bottom up model cell)
class Residual_Cell_NVAE(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
        super(Residual_Cell_NVAE,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        if(stride == 1):
            self.skip = Identity()
        elif(stride == 2):
            self.skip = FactorizedReduce(in_channels,out_channels)
        elif(stride == -1):
            #self.skip = nn.Sequential(UpSample(), nn.Conv2D(in_channels, int(in_channels / channel_mult), kernel_size=1))
            self.skip = nn.Sequential(UpSample(),nn.Conv2D(in_channels, int(in_channels / 2), kernel_size=1))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bnSwish1 = Swish()
        # 3x3 conv without changing channel size
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1,stride=(stride,stride))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bnSwish2 = Swish()
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),padding=1,stride=(1,1))

        self.squeeze_excitation = SE_Block(out_channels,out_channels)

        self.cell = nn.Sequential(
            self.bn1,
            self.bnSwish1,
            self.conv1,
            self.bn2,
            self.bnSwish2,
            self.conv2,
            self.squeeze_excitation
        )

    def forward(self,x):
        skip_output = self.skip(x)
        output = self.cell(x)

        #TODO Check whether all cells need to be skip + 0.1 * output
        return skip_output + output




# Used to change features after encoder pass, before being combined with h at top of decoder
class encoder_0_cell(nn.Module):
    def __init__(self,channels) -> None:
        super(encoder_0_cell,self).__init__()
        self.cell = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.ELU())
    def forward(self,x):
        return self.cell(x)


class Preproc_tower(nn.Module):
    def __init__(self,in_channels,groups,cells_per_group):
        super(Preproc_tower,self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels*(2**groups)
        self.groups = groups
        self.cells_per_group = cells_per_group
        self.groups_list = nn.ModuleList()
        current_channels = in_channels
        for g in range(groups):
            group = nn.ModuleList()
            #Only the final cell in each pre-proc block will double
            #the number of channels
            for c in range(cells_per_group):
                if(c == cells_per_group - 1):
                    group.append(Residual_Cell_NVAE(current_channels,
                                                out_channels=current_channels*2,
                                                stride=2))
                    current_channels = current_channels*2
                else:
                    group.append(Residual_Cell_NVAE(current_channels,
                                                out_channels=current_channels))
            self.groups_list.append(nn.Sequential(*group))
        #Unpack module list for sequential (with *)
        self.tower = nn.Sequential(*self.groups_list)

        if(current_channels != self.out_channels):
            raise ValueError('Error, channel output: ',current_channels,', should be:',self.out_channels)

    def forward(self,x):
        return self.tower(x)



class Postproc_tower(nn.Module):
    #NOTE Currently, blocks (number of postproc blocks) MUST equal number of preproc blocks
    def __init__(self,in_channels,blocks,cells_per_block,num_preproc_blocks):
        super(Postproc_tower,self).__init__()

        if(blocks != num_preproc_blocks):
            raise ValueError('number of postproc blocks MUST equal number of preproc blocks')

        self.in_channels = in_channels
        self.blocks = blocks
        self.cells_per_block = cells_per_block

        #We will undo the dimension increases from pre_processing
        channel_mult = 2**num_preproc_blocks
        self.tower = nn.ModuleList()
        for b in range(blocks):
            for c in range(cells_per_block):
                channels = in_channels*channel_mult
                if(c == 0):
                    #The first cell step us down toward the original scale input into the preproc tower
                    self.tower.append(Generative_Cell_NVAE(channels,E_param=2,stride=-1))
                    channel_mult = channel_mult//2
                else:
                    #The rest maintain the original data channel size
                    self.tower.append(Generative_Cell_NVAE(channels,channels))
                
        self.tower = nn.Sequential(*self.tower)
        self.out_channels = channels
    def forward(self,x):
        return self.tower(x)


class Encoder_tower(nn.Module):
    def __init__(self,in_channels,num_of_scales,groups_per_scale,cells_per_group,decoder_channels):
        super(Encoder_tower,self).__init__()
        self.in_channels = in_channels
        #Must be the input channels to the decoder, used for creating combiner cell
        self.decoder_chans = decoder_channels
        self.num_of_scales = num_of_scales
        self.groups_per_scale = groups_per_scale
        # Each scale will double channels, so final result is 2^scales channels
        self.out_channels = in_channels*(2**(num_of_scales - 1)) # -1 because last scale does not increase channel size
        self.combiner_cells = nn.ModuleList()
        #Channel multliplyer, will increase by a power of 2 with each scale
        channel_mult = 2
        in_chans = self.in_channels
        out_chans = self.in_channels*channel_mult
        self.enc_tower = nn.ModuleList()
        for s in range(num_of_scales):
            scale = nn.ModuleList()
            for g in range(groups_per_scale):
                group = nn.ModuleList()
                for c in range(cells_per_group):
                    group.append(Residual_Cell_NVAE(in_chans,
                                            out_channels=in_chans))
                scale.append(nn.Sequential(*group))

            #TODO test performance of combiners per group vs per scale
            #Add a combiner for the PREVIOUS scale (The last scale doesnt have a encCombiner, but the output of the
            # preprocess does)
            #Add combiners in reverse order by always inserting before the 0th index
            encoder_chans = in_chans
            decoder_chans = in_chans
            self.combiner_cells.insert(0,EncCombinerCell(encoder_chans,
                                        decoder_chans,encoder_chans))

            #If we arent the final scale, include a cell that increases
            #channel size to the next scale size.
            #TODO Consider increasing after EVERY scale, versus not increasing at the end of the last
            if s < (num_of_scales - 1):
            #if s < (num_of_scales):
                #chan_in = self.in_channels*channel_mult
                #chan_out = chan_in*2
                print('in chans', in_chans)
                print('out chans', out_chans)

                out_chans = in_chans * channel_mult
                #We use stride=2 on cells that increase or decrease dimensionality
                cell = Residual_Cell_NVAE(in_chans,out_chans,stride=2)
                in_chans = out_chans

                #self.enc_tower.append(cell)
                scale.append(cell)

            self.enc_tower.append(nn.Sequential(*scale))


        if(self.in_channels*(channel_mult**(num_of_scales-1)) != self.out_channels):
            raise ValueError('Error, channel output: ',self.in_channels*channel_mult,', should be:',self.out_channels)

        #The encoder tower should now be a ModuleList with each scale compiled
        #as a sequential network, while combiner_cells holds the combiner cells
        #for the ends of each scale to match with the decoder

    def forward(self,x):
        outputs = []
        outputs.append(x) #We include encoder input as the first set of latent variables
        for scale in self.enc_tower:
            x = scale(x)
            outputs.append(x)
        # Reverse outputs for the top down model
        outputs.reverse()
        return outputs
    



#This class is for ease of forward pass, and will help handle the repeated use of combiner cells
class Decoder_group(nn.Module):
    def __init__(self,in_channels,latent_size,cells_per_group,E) -> None:
        super(Decoder_group,self).__init__()
        self.in_channels = in_channels
        self.cells_per_group = cells_per_group
        self.E = E
        #TODO Once everything is running, double check that is actually the case
        #Add a decoder combiner cell between each group, creating a residual
        # connection by passing forward the first encoding output.
        self.combiner = DecCombinerCell(encoder_channels=latent_size,
                                        decoder_channels=in_channels,
                                        out_channels=in_channels)
        self.group = nn.ModuleList()
        for c in range(self.cells_per_group):
            self.group.append(Generative_Cell_NVAE(in_channels,self.E))
        self.group = nn.Sequential(*self.group)
        
    def forward(self,decoder_vars):
        x = self.group(decoder_vars)
        return self.combiner(decoder_vars,x)
        

class Decoder_tower(nn.Module):
    def __init__(self,in_channels,num_of_scales,groups_per_scale,cells_per_group,encoder_channels,num_of_preproc_blocks):
        super(Decoder_tower,self).__init__()     
        self.in_channels = in_channels
        #Must be the input channels to the encoder, used for creating combiner cell
        self.encoder_channels = encoder_channels
        #Encoder and decoder MUST share the number of scales
        self.num_of_scales = num_of_scales
        self.groups_per_scale = groups_per_scale
        self.out_channels = in_channels // (2**(num_of_scales - 1))
        #Channel multliplyer, starts as the highest value from the encoder,
        #and works its way down to the original input size with each scale
        channel_mult = 2**(num_of_scales - 1) # -1 because last scale output does not increase dims
        current_channels = in_channels
        self.post_encoder = encoder_0_cell(current_channels)
        #Trainable parameter for combining initial encoder input
        #TODO Double check the dimensions for h
        #num_of_preproc_blocks = sqrt(encoder_channels)#encoder_channels // 2
        h_scaling = 2**(num_of_scales)
        h_size = (current_channels, max(current_channels // h_scaling,4), max(current_channels // h_scaling,4))
        self.h = nn.Parameter(torch.rand(size=h_size), requires_grad=True).unsqueeze(0).to(device) # Unsqueeze to match 4-dim data

        #Initial combiner cell for first-encoding z and parameter h
        self.combiner_cells = nn.ModuleList()
        self.dec_tower = nn.ModuleList()

        for s in range(num_of_scales):
            scale = nn.ModuleList()
            if s == 0:
                out_channels = current_channels
            else:
                out_channels = current_channels // 2

            for g in range(groups_per_scale):
                group = Decoder_group(in_channels=current_channels,
                                        latent_size=current_channels,
                                        E=2,
                                        cells_per_group=cells_per_group)
                scale.append(group)
            self.combiner_cells.append(DecCombinerCell(encoder_channels=current_channels,
                                                        decoder_channels=current_channels,
                                                        out_channels=current_channels))


            # The last cell will decrease the channel size down a scale
            if s != 0:
                scale.append(Generative_Cell_NVAE(current_channels,E_param=2,stride=-1))
            #Add scale to encoding tower
            self.dec_tower.append(nn.Sequential(*scale))


            #No need to decrease beyond lowest encoding dim
            #if (s != num_of_scales - 1) and (s != 0):
            if s != 0:
                current_channels = current_channels // 2

        #Add the final combiner cell that feeds into the post=process tower
        self.combiner_cells.append(DecCombinerCell(encoder_channels=current_channels,
                                                    decoder_channels=current_channels,
                                                    out_channels=current_channels))

        if(current_channels != self.out_channels):
            raise ValueError('Error, channel output: ',current_channels,', should be:',self.out_channels)

        #Decoder tower should now be a Module list of scales
        #Combiner cells are also in expected "reverse order" (or "top down" order)
        #Generate samplers
        self.samplers = nn.ModuleList()
        decoded_chans = current_channels
        current_mult = channel_mult
        #TODO Do we want number of scale or scale+1 samplers?
        for scale in range(num_of_scales+1):
            print("SAMPLER DIMS: ", decoded_chans*current_mult)
            self.samplers.append(Sampler(decoded_chans,current_mult))
            # Top level scale will not decrease channels (top two will always be the same)
            # So we skip the first decrease in channel size
            if scale != 0:
                current_mult = current_mult // 2

    def forward(self,latent_vals,enc_combiners):
        #Plus one to include output of preprocess tower
        if(len(latent_vals) != self.num_of_scales + 1):
            print("Latent Vals: ", len(latent_vals))
            print("Number of Scales: ", self.num_of_scales)
            raise ValueError('Decoder input length must equal the number of scales + 1')
        
        #Use top encoding to generate sampler and draw sample
        z_1 = self.samplers[0](latent_vals[0])
        #Expand dims of h to match batch size
        batch_size = z_1.shape[0]
        #Do not alter self.h, each batch can be different so changes must be temporary
        h = self.h.expand(batch_size,-1,-1,-1)
        # Take the first sample, use a DecCombiner with h
        first_input = self.combiner_cells[0](z_1,h)

        #Input "latent_vals" is expected to be in correct top-down order
        output = first_input
        for scale in range(self.num_of_scales):
            scale_output = self.dec_tower[scale](output)
            combined_latent = enc_combiners[scale](latent_vals[scale+1],scale_output)
            cur_z = self.samplers[scale+1](combined_latent)
            output = self.combiner_cells[scale+1](cur_z,scale_output)

        all_p = []
        all_log_p = []
        all_q = []
        all_log_q = []
        #Extract prior and posterior data from samplers
        for s in self.samplers: #range(len(self.samplers)):
            all_p.append(s.p_dist)
            all_log_p.append(s.log_p_conv)
            all_q.append(s.q_dist)
            all_log_q.append(s.log_q_conv)
        dist_info = zip(all_q,all_p,all_log_q,all_log_p)
        #Returns a drawn sample from the bottom Z (latent space) in the tower, as well as all distribution info
        return output, dist_info



        

#TODO THIS IS WHERE NORMALIZING FLOW WILL BE APPLIED          
class Sampler(nn.Module):
    def __init__(self,in_channels,feature_mult) -> None:
        super(Sampler,self).__init__()
        self.in_channels = in_channels
        #Multiplyer for channel increase from latent -> sampler convolution
        self.feature_mult = feature_mult
        self.total_channels = in_channels*feature_mult
        self.cell = nn.Conv2d(self.total_channels, 2 * self.total_channels, kernel_size=3, padding=1, bias=True)

        # Create seperate cell for the "Prior", the decoder sampler that corresponds to each encoder sampler
        self.prior_cell = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.total_channels, 2 * self.total_channels, kernel_size=1, padding=0, bias=True))


        # Prior values
        self.mu_p = 0
        self.log_sig_p = 0
        self.p_dist = 0 # Part of "all_p"
        self.log_p_conv = 0 #part of "all_log_p"
        # Posterior values
        self.mu_q = 0
        self.log_sig_q = 0
        self.q_dist = 0 #part of "all_q"
        self.log_q_conv = 0 #part of "all_log_q"

    def forward(self,x):

        #Generate Prior
        prior = self.prior_cell(x)
        self.mu_p, self.log_sig_p = torch.chunk(prior, 2, dim=1)

        x = self.cell(x)
        #Split the resulting doubled channels of the cell into mu and sigma for the distribution
        self.mu_q, self.log_sig_q = torch.chunk(x, 2, dim=1)
        self.q_dist = Normal(self.mu_q + self.mu_p,self.log_sig_q + self.log_sig_p)
        z, _ = self.q_dist.sample()

        self.log_q_conv = self.q_dist.log_p(z)

        #Evaluate log_p(z)
        self.p_dist = Normal(self.mu_p,self.log_sig_p)
        self.log_p_conv = self.p_dist.log_p(z)
        return z



class Defence_NVAE(nn.Module):
    def __init__(self,x_channels,encoding_channels,pre_proc_groups,scales,groups,cells):
        super(Defence_NVAE,self).__init__()

        self.initial_chans = encoding_channels
        self.num_latent_scales = scales
        self.groups_per_scale = groups
        # Number of channels at the top of the encoding tower
        self.top_chans = encoding_channels * (2 ** scales - 1)
        self.samplers = [] # Will contain a sampler for each scale. 
        self.stem = nn.Conv2d(x_channels,encoding_channels,kernel_size=3,padding=1,bias=True)
        current_chans = encoding_channels

        self.pre_proc = Preproc_tower(current_chans,pre_proc_groups,cells)
        current_chans = self.pre_proc.out_channels #same as current_chans * (2**groups)

        self.encoder = Encoder_tower(current_chans,scales,groups,cells,encoding_channels)
        current_chans = self.encoder.out_channels

        self.decoder = Decoder_tower(current_chans,scales,groups,cells,encoding_channels,pre_proc_groups)
        current_chans = self.decoder.out_channels
        #TODO should I use encoding_channels or current_chans
        self.post_proc = Postproc_tower(encoding_channels,self.pre_proc.groups,cells,self.pre_proc.groups)

        #Return image to original pre-encoding dimensions times 2, one for mu, one for log_sigma
        # ADAPTED FOR MIXTURE
        num_mixtures = 10
        out_channels = int(num_mixtures + num_mixtures * 3 * x_channels)
        self.image_conditional = nn.Sequential(nn.ELU(),nn.Conv2d(encoding_channels, out_channels, 3, padding=1, bias=True))

        print("Initial channels: ",x_channels)
        print("First Encoding: ",encoding_channels)
        print("After Pre-Proc: ",self.pre_proc.out_channels)
        print("After Encoder: ",self.encoder.out_channels)
        print("After Decoder: ",self.decoder.out_channels)
        print("After Post-Proc: ",self.post_proc.out_channels)


    def loss(self,recon_x, x, log_q, log_p, kl_all, kl_diag, global_step,num_total_iter):
        alpha_i = kl_balancer_coeff(num_scales=self.num_latent_scales,
                                          groups_per_scale=self.num_latent_scales, fun='square')

        kl_coeff = kl_coeff_func(global_step, KL_ANNEAL_PORTION * num_total_iter,
                                  KL_CONST_PORTION * num_total_iter, KL_CONST_COEFF)

        # ADAPTED
        reconstructions = DiscMixLogistic(recon_x, img_channels=3, num_bits=8).log_prob(x)
        recon_loss = - torch.sum(reconstructions, dim=1)

        balanced_kl, kl_coeffs, kl_vals = kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

        nelbo_batch = recon_loss + balanced_kl
        loss = torch.mean(nelbo_batch)


        return loss, recon_loss, balanced_kl


    def forward(self,x):

        x = torch.clamp(x, 0, 1)
        x = (x * 2.) - 1

        x = self.stem(x)
        x = self.pre_proc(x)

        latent_z_vals = self.encoder(x)
        # encoder output and combiner cells have already 
        # been output in reverse order for the decoder
        decoder_output,dist_info = self.decoder(latent_z_vals,self.encoder.combiner_cells)
        x = self.post_proc(decoder_output)
        x = self.image_conditional(x)

        # Compute KL divergence across all z samplers
        # compute kl
        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.
        for q, p, log_q_conv, log_p_conv in dist_info:
            #if self.with_nf:
            #    kl_per_var = log_q_conv - log_p_conv
            #else:
            kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])

        return x, log_q, log_p, kl_all, kl_diag
        
    def generate_sample(self,x):
        logits, _, _, _, _ = self.forward(x)
        decoder = DiscMixLogistic(logits, img_channels=3, num_bits=8)
        output = decoder.sample()
        return output



'''A sample training function to test the model's generative abilities,
    not the same function used in the Defense VAE
'''

def train_NVAE(epochs=10,lr=1e-3,batch_size=128):


    """
    Create dataloaders to feed data into the neural network
    Default MNIST dataset is used and standard train/test split is performed
    """

    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        #Binarize(),
    ])

    test_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        #Binarize(),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                        transform=train_transform),#transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=test_transform),#transforms.ToTensor()),
        batch_size=1)

    """
    Initialize the network and the Adam optimizer
    """
    #mnist is a single channel
    x_channels = 1
    pre_proc_groups = 2
    encoding_channels=4
    scales=2
    groups=2
    cells=2

    net = Defence_NVAE(x_channels,encoding_channels,pre_proc_groups,scales,groups,cells).to(device)
    optimizer =  torch.optim.Adamax(net.parameters(), lr, weight_decay=1e-2, eps=1e-3)


    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    #Count steps for annealing purposes
    global_step = 0
    num_total_iter = epochs * len(train_loader)

    for epoch in range(epochs):
        print("Epoch: ",epoch+1,"/",epochs)
        for idx, data in enumerate(tqdm(train_loader), 0):

            imgs, _ = data
            #imgs = imgs.to(device)
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, log_q, log_p, kl_all, kl_diag = net(imgs)

            loss, _, _ = net.loss(out,imgs,log_q, log_p, kl_all, kl_diag, global_step,num_total_iter)

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('Epoch {}: Loss {}'.format(epoch+1, loss))



    """
    The following part takes a random image from test loader to feed into the VAE.
    Both the original image and generated image from the distribution are shown.
    """
    sample_count = 10
    image_count = 1
    net.eval()
    with torch.no_grad():
        for data in random.sample(list(test_loader), sample_count):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out = net.generate_sample(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.savefig('../results/nvae_examples/nvae_examples'+str(image_count)+'.png')
            plt.clf()
            image_count += 1


    # Save model parameters
    torch.save(net.state_dict(), '../model_parameters/NVAE_mnist.pth')
