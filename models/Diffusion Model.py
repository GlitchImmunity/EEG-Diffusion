from logging import exception
import math
from tqdm import tqdm
import torch
import torch.nn as nn
from torchmetrics import Accuracy
import pytorch_lightning as pl
from einops import rearrange

# https://arxiv.org/abs/2006.11239

class UnetUp(nn.Module):
    def __init__(self, channel_in, channel_out, use_conv=True):
        super().__init__()
        self.use_conv = False
        if use_conv:
            self.use_conv = True
            self.conv = nn.Conv1d(channel_in, channel_out, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            out = self.conv(out)
        return out

class UnetDown(nn.Module):
    def __init__(self, channel_in, channel_out, use_conv=True):
        super().__init__()
        self.use_conv = False
        if use_conv:
            self.use_conv = True
            # Kernel size 3 didn't work but this does???
            self.conv = nn.Conv1d(channel_in, channel_out, 4, 2, 1)
    
    def forward(self, x):
        out = None
        if self.use_conv:
            out = self.conv(x)
        else:
            out = nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        #print(out.shape)
        return out
        
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, activation=nn.SiLU(), groups=16, time_dimension=None, subject_dimension=None):
        super().__init__()
        
        # Define Main Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.activation1 = activation
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation2 = activation
        self.dropout = nn.Dropout(dropout)
        self.passthrough = None
        if in_channels == out_channels:
            self.passthrough = nn.Identity()
        else:
            self.passthrough = nn.Conv1d(in_channels, out_channels, 1)
            
        # Time Scale-Shift
        self.tss = False
        if time_dimension is not None:
            self.tss = True
            self.tss_act = activation
            self.tss_linear = nn.Linear(time_dimension, out_channels*2)
            
        # Subject Scale-Shift
        #self.sss = False
        #if exists(subject_dimension):
        #    self.sss = True
        #    self.sss_act = activation
        #    self.sss_linear = nn.Linear(subject_dimension, out_channels*2)
    
    def forward(self, x, time, subject=None):
        #print("resblock in:", x.shape)
        out = x
        #print("resblock in2:", out.shape)
        
        # Encode time
        time_scale = 0
        time_shift = 0
        subject_scale = 0
        subject_shift = 0
        if self.tss:
            temp = self.tss_act(time)
            temp = self.tss_linear(temp)
            temp = rearrange(temp, 'b c -> b c 1')
            time_scale_shift = temp.chunk(2, dim=1)
            time_scale, time_shift = time_scale_shift
            
        #if self.sss:
        #    temp = self.tss_act(subject)
        #    temp = self.tss_linear(temp)
        #    temp = rearrange(temp, 'b c -> b c 1')
        #    subject_scale_shift = temp.chunk(2, dim=1)
        #    subject_scale, subject_shift = subject_scale_shift
            
        # Encode subject
        
        #print("resblock in3:", out.shape)
        out = self.conv1(out)
        out = self.norm1(out)
        #out = out * (1 + time_scale + subject_scale) + time_shift + subject_scale
        out = out * (1 + time_scale) + time_shift
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        out = self.dropout(out)

        residual = self.passthrough(x)

        return residual+out
        
class ResAttention(nn.Module):
    def __init__(self, dimension, x_class_dim, sequence_length=250, heads=4, head_dimension=32):
        super().__init__()
        self.heads = heads
        
        self.norm1 = nn.BatchNorm1d(dimension)
        self.conv_q = nn.Conv1d(dimension, heads*head_dimension, 1, bias=False)
        self.conv_v = nn.Linear(x_class_dim, heads*head_dimension)
        self.conv_k = nn.Linear(x_class_dim, heads*head_dimension)
        self.attention = nn.MultiheadAttention(heads*head_dimension, heads, batch_first=True)
        self.conv_out = nn.Conv1d(heads*head_dimension, dimension, 1)
        self.norm2 = nn.BatchNorm1d(dimension)
        
    def forward(self, x, x_class):
        b, c, l = x.shape
        out = self.norm1(x)
        query = self.conv_q(out)
        query = torch.transpose(query, 1, 2)
        key = self.conv_k(x_class)
        key = key[:, None, :]
        value = self.conv_v(x_class)
        value = value[:, None, :]
        #print(query.shape, key.shape, value.shape)
        out, _ = self.attention(query, key, value)
        out = torch.transpose(out, 1, 2)
        out = self.conv_out(out)
        #print(out.shape)
        out = self.norm2(out)
        #print((out+x).shape)
        return out + x

class Unet(nn.Module):
    def __init__(self, hidden, input_channels=22, classes=4, unet_scales=(1,2,4,8)):
        super().__init__()
        self.hidden = hidden
        dims = [hidden, *map(lambda scaler: hidden * scaler, unet_scales)]
        unet_structure = list(zip(dims[:-1], dims[1:]))
        #print(unet_structure)
        
        # Encode input
        self.encode_c = nn.Conv1d(input_channels*2, hidden, 7, 1, 3)
        
        # Encode time
        time_dim = hidden*4
        self.encode_time_1 = nn.Linear(hidden, time_dim)
        self.encode_time_act = nn.SiLU()
        self.encode_time_2 = nn.Linear(time_dim, time_dim)
        
        # Encode subject
        #subjects_dim = hidden*2
        #self.encode_subject_1 = nn.Linear(subjects, subjects_dim)
        #self.encode_subject_act = nn.SiLU()
        #self.encode_subject_2 = nn.Linear(subjects_dim, subjects_dim)
        
        # Encode class
        class_dim = classes * 4
        self.encode_class_1 = nn.Linear(classes, class_dim)
        self.encode_class_act = nn.SiLU()
        self.encode_class_2 = nn.Linear(class_dim, class_dim)
        
        # Unet
        self.down_half = nn.ModuleList([])
        self.up_half = nn.ModuleList([])
        num_blocks = len(unet_structure)
        
        # Down parts of unet
        for IX, (channel_in, channel_out) in enumerate(unet_structure):
            is_last_layer = (IX == (num_blocks-1))
            #print(channel_in, channel_out)
            
            down_track = None
            if not is_last_layer:
                down_track = nn.ModuleList([ResnetBlock(channel_in, channel_in, time_dimension=time_dim), 
                                            ResnetBlock(channel_in, channel_in, time_dimension=time_dim),
                                            ResAttention(channel_in, class_dim),
                                            UnetDown(channel_in, channel_out)])
            else:
                down_track = nn.ModuleList([ResnetBlock(channel_in, channel_in, time_dimension=time_dim), 
                                            ResnetBlock(channel_in, channel_in, time_dimension=time_dim),
                                            ResAttention(channel_in, class_dim),
                                            nn.Conv1d(channel_in, channel_out,3,1,1)])
            self.down_half.append(down_track)
            
        # Valley of unet
        valley_dimension = unet_structure[-1][-1]
        #print(valley_dimension)
        self.valley_1 = ResnetBlock(valley_dimension, valley_dimension, time_dimension=time_dim)
        self.valley_attention = ResAttention(valley_dimension, class_dim)
        self.valley_2 = ResnetBlock(valley_dimension, valley_dimension, time_dimension=time_dim)
        
        # Up parts of unet
        for IX, (channel_in, channel_out) in enumerate(reversed(unet_structure)):
            is_last_layer = (IX == (num_blocks-1))
            #print(channel_in, channel_out, channel_in+channel_out)
            
            up_track = None
            if not is_last_layer:
                up_track = nn.ModuleList([ResnetBlock(channel_in+channel_out, channel_out, time_dimension=time_dim),
                                        ResnetBlock(channel_in+channel_out, channel_out, time_dimension=time_dim),
                                        ResAttention(channel_out, class_dim),
                                        UnetUp(channel_out, channel_in)])
            else:
                up_track = nn.ModuleList([ResnetBlock(channel_in+channel_out, channel_out, time_dimension=time_dim),
                                        ResnetBlock(channel_in+channel_out, channel_out, time_dimension=time_dim),
                                        ResAttention(channel_out, class_dim),
                                        nn.Conv1d(channel_out, channel_in, 3,1,1)])
            self.up_half.append(up_track)
        
        # End
        self.end_1 = ResnetBlock(hidden*2, hidden)
        self.end_2 = nn.Conv1d(hidden, input_channels*2, 1)
        
    def forward(self, x, time, x_class, subject=None):
        #print("in:", x.shape)
        #print(x.shape)
        x = torch.fft.rfft(x, dim=2, norm='ortho')[:,:,:-1]
        #print(x.shape)
        x = torch.concat([x.real, x.imag], dim=1)
        x = self.encode_c(x)
        #print("encode_c:", x.shape)
        passthrough_start = x.clone()
        
        # Encode time
        t = math.log(10000) / (self.hidden//2 - 1)
        t = torch.exp(torch.arange(self.hidden//2, device=time.device) * -t)
        t = time[:, None] * t[None, :]
        t = torch.cat((t.sin(), t.cos()), dim=-1)
        t = self.encode_time_1(t)
        t = self.encode_time_act(t)
        t = self.encode_time_2(t)
        
        # Encode subject
        #s = subject
        #if exists(subject):
        #   s = self.encode_subject_1(s)
        #    s = self.encode_subject_act(s)
        #    s = self.encode_subject_2(s)
        
        # Encode class
        c = self.encode_class_1(x_class)
        c = self.encode_class_act(c)
        c = self.encode_class_2(c)
        
        passthroughs = []
        for block1, block2, resattn, downsample in self.down_half:
            #print("down:", x.shape)
            #print()
            x = block1(x, t)
            passthroughs.append(x)
            
            x = block2(x, t)
            x = resattn(x, c)
            passthroughs.append(x)
            
            x = downsample(x)
        
        #print("valley in:", x.shape)
        x = self.valley_1(x, t)
        #print("valley middle 1:", x.shape)
        x = self.valley_attention(x, c)
        #print("valley middle 2:", x.shape)
        x = self.valley_2(x, t)
        #print("valley out:", x.shape)
        
        for block1, block2, resattn, upsample in self.up_half:
            #print()
            #print("up:", x.shape)
            x = torch.cat((x, passthroughs.pop()), dim=1)
            #print("cat1:", x.shape)
            x = block1(x, t)
            #print("cat1 after:", x.shape)

            x = torch.cat((x, passthroughs.pop()), dim=1)
            #print("cat2:", x.shape)
            x = block2(x, t)
            x = resattn(x, c)
            
            x = upsample(x)
            
        x = torch.cat((x, passthrough_start), dim=1)
        x = self.end_1(x, t)
        x = self.end_2(x)
        
        xr = x[:,:22,:]
        xc = x[:,22:,:]
        x = torch.complex(xr, xc)
        x = torch.fft.irfft(x, n=256, dim=2, norm='ortho')
        return x
        
# Orignal Diffusion Paper:
# https://arxiv.org/pdf/2006.11239.pdf#page5
# https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py
# Helper function to extract data to be of specific shape
def helper_extract(data, timesteps, shape):
    # Throws device error if not specified
    device = timesteps.device
    out = torch.gather(data, -1, timesteps.cpu())
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1))).to(timesteps.device)
        
class EEGDiffusion(pl.LightningModule):
    def __init__(
        self,
        hidden,
        lr,
        us=(1,2,4,8),
        seq_length=250,
        timesteps=500,
        sampling_timesteps=None,
        device="cuda"
        ):
        super().__init__()
        self.model = Unet(hidden, unet_scales=us)
        self.channels = 22
        self.seq_length = seq_length
        self.timesteps = timesteps
        self.lr = lr
        
        # Here is our betas: linearly increasing noise strength 
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1 - self.betas
        
        # As defined by paper
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1)
        
        # Intermediate step
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        
        # Key noise values
        # Intuition: We want to be able to produce noise that can be reversible. We do this by controlling how powerful the noise will be at different timesteps
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    # Defined by section 3.1: Foward Process
    def q_sample(self, x_start, t, noise=None):
        # Used to verify, not really needed for train/test
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get noise strength at time steps
        sqrt_alphas_cumprod_t = helper_extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = helper_extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
        # We want to ensure that a ratio between data and noise is preserved and mean stays the same
        # Apply noise
        #print(t.device)
        #print(sqrt_alphas_cumprod_t.device, x_start.device, sqrt_one_minus_alphas_cumprod_t.device, noise.device)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    # Algorithm 1
    def forward(self, x_start, t, x_class, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
    
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, x_class)
        # L2 loss used for higher sample diversity: https://arxiv.org/pdf/2111.05826.pdf
        loss = nn.functional.mse_loss(noise, predicted_noise)

        # Defining our own loss function to punish noise
        # Punishes high variance in sliding sunks of predicted noise
        #diff = noise - predicted_noise
        #v_loss = diff.unfold(dimension=2, size=16, step=8)
        #v_loss = v_loss.var(dim=3)
        #v_loss = v_loss.sum()/math.prod(v_loss.shape[:-1])/50
        #loss += vloss

        return loss
        
    @torch.no_grad()
    # Based on algorithm 4
    def p_sample(self, x, t, t_index, x_class):
        betas_t = helper_extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = helper_extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = helper_extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t, x_class) / sqrt_one_minus_alphas_cumprod_t)
    
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = helper_extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
            
    # Algorithm 2
    @torch.no_grad()
    def p_sample_loop(self, x_class, shape, initialize=None, timesteps=None):
        b, c, l = shape
        
        noisy_start = torch.randn(shape)
        noisy_start = noisy_start.to(self.device)
        if not initialize is None:
            noisy_start += initialize
        t = self.timesteps
        if not timesteps is None:
            t = timesteps
        denoise = []
        
        for i in tqdm(reversed(range(0, t)), total=t):
            noisy_start = self.p_sample(noisy_start, torch.full((b,), i, device=self.device, dtype=torch.long), i, x_class)
            denoise.append(noisy_start.cpu().numpy())
        return denoise
        
    @torch.no_grad()
    def sampling(self, x_class, batch_size=1, channels=22, sequence_length=250, initialize=None, timesteps=None):
        return self.p_sample_loop(x_class, (batch_size, channels, sequence_length), initialize, timesteps)
        
    def training_step(self, batch, batch_IX):
        data, data_class, data_subject = batch
        b = data.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        loss = self.forward(data, t, data_class)
        return loss
        
    def validation_step(self, batch, batch_IX):
        data, data_class, data_subject = batch
        b = data.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        loss = self.forward(data, t, data_class)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self): 
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
              self.opt,
              mode='min',
              factor=0.1,
              patience=5,
              verbose=True,
              min_lr=1e-8,
          )
        return [self.opt], {"optimizer": self.opt, "scheduler": self.reduce_lr_on_plateau, "monitor": "val_loss"}
    

    
    
    
    

    

    
class ResAttention_Hybrid(nn.Module):
    def __init__(self, dimension, x_class_dim=None, sequence_length=250, heads=4, head_dimension=32):
        super().__init__()
        self.heads = heads
        self.scale = heads ** -0.5
        
        self.norm1 = nn.BatchNorm1d(dimension)
        self.conv_qkv = nn.Conv1d(dimension, 3*heads*head_dimension, 1, bias=False)
        self.conv_q = nn.Conv1d(dimension, heads*head_dimension, 1, bias=False)
        self.conv_v = nn.Linear(x_class_dim, heads*head_dimension)
        self.conv_k = nn.Linear(x_class_dim, heads*head_dimension)
        self.attention = nn.MultiheadAttention(heads*head_dimension, heads, batch_first=True)
        self.conv_out = nn.Conv1d(heads*head_dimension, dimension, 1)
        self.norm2 = nn.BatchNorm1d(dimension)
        
    def forward(self, x, x_class):
        b, c, l = x.shape
        out = x
        out = self.norm1(x)
        qkv = self.conv_qkv(out)
        qkv = qkv.chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        
        query = self.conv_q(out)
        query = rearrange(query, 'b (h c) l -> b l (h c)', h=self.heads)
        key = self.conv_k(x_class)
        key = key[:, None, :]
        value = self.conv_v(x_class)
        value = value[:, None, :]
        out, _ = self.attention(query, key, value)
        query = rearrange(query, 'b l (h c) -> b (h c) l', h=self.heads)
        
        out = self.conv_out(out)
        #print(out.shape)
        out = self.norm2(out)
        #print((out+x).shape)
        return out + x

class Unet_Hybrid(nn.Module):
    def __init__(self, hidden, input_channels=22, classes=4, unet_scales=(1,2,4,8)):
        super().__init__()
        self.hidden = hidden
        dims = [hidden, *map(lambda scaler: hidden * scaler, unet_scales)]
        unet_structure = list(zip(dims[:-1], dims[1:]))
        #print(unet_structure)
        
        # Encode input
        self.encode_c = nn.Conv1d(input_channels*2, hidden, 7, 1, 3)
        
        # Encode time
        time_dim = hidden*4
        self.encode_time_1 = nn.Linear(hidden, time_dim)
        self.encode_time_act = nn.SiLU()
        self.encode_time_2 = nn.Linear(time_dim, time_dim)
        
        # Encode subject
        #subjects_dim = hidden*2
        #self.encode_subject_1 = nn.Linear(subjects, subjects_dim)
        #self.encode_subject_act = nn.SiLU()
        #self.encode_subject_2 = nn.Linear(subjects_dim, subjects_dim)
        
        # Encode class
        class_dim = classes * 4
        self.encode_class_1 = nn.Linear(classes, class_dim)
        self.encode_class_act = nn.SiLU()
        self.encode_class_2 = nn.Linear(class_dim, class_dim)
        
        # Unet
        self.down_half = nn.ModuleList([])
        self.up_half = nn.ModuleList([])
        num_blocks = len(unet_structure)
        
        # Down parts of unet
        for IX, (channel_in, channel_out) in enumerate(unet_structure):
            is_last_layer = (IX == (num_blocks-1))
            #print(channel_in, channel_out)
            
            down_track = None
            if not is_last_layer:
                down_track = nn.ModuleList([ResnetBlock(channel_in, channel_in, time_dimension=time_dim), 
                                            ResnetBlock(channel_in, channel_in, time_dimension=time_dim),
                                            ResAttention_Hybrid(channel_in, class_dim),
                                            UnetDown(channel_in, channel_out)])
            else:
                down_track = nn.ModuleList([ResnetBlock(channel_in, channel_in, time_dimension=time_dim), 
                                            ResnetBlock(channel_in, channel_in, time_dimension=time_dim),
                                            ResAttention_Hybrid(channel_in, class_dim),
                                            nn.Conv1d(channel_in, channel_out,3,1,1)])
            self.down_half.append(down_track)
            
        # Valley of unet
        valley_dimension = unet_structure[-1][-1]
        #print(valley_dimension)
        self.valley_1 = ResnetBlock(valley_dimension, valley_dimension, time_dimension=time_dim)
        self.valley_attention = ResAttention_Hybrid(valley_dimension, class_dim)
        self.valley_2 = ResnetBlock(valley_dimension, valley_dimension, time_dimension=time_dim)
        
        # Up parts of unet
        for IX, (channel_in, channel_out) in enumerate(reversed(unet_structure)):
            is_last_layer = (IX == (num_blocks-1))
            #print(channel_in, channel_out, channel_in+channel_out)
            
            up_track = None
            if not is_last_layer:
                up_track = nn.ModuleList([ResnetBlock(channel_in+channel_out, channel_out, time_dimension=time_dim),
                                        ResnetBlock(channel_in+channel_out, channel_out, time_dimension=time_dim),
                                        ResAttention_Hybrid(channel_out, class_dim),
                                        UnetUp(channel_out, channel_in)])
            else:
                up_track = nn.ModuleList([ResnetBlock(channel_in+channel_out, channel_out, time_dimension=time_dim),
                                        ResnetBlock(channel_in+channel_out, channel_out, time_dimension=time_dim),
                                        ResAttention_Hybrid(channel_out, class_dim),
                                        nn.Conv1d(channel_out, channel_in, 3,1,1)])
            self.up_half.append(up_track)
        
        # End
        self.end_1 = ResnetBlock(hidden*2, hidden)
        self.end_2 = nn.Conv1d(hidden, input_channels*2, 1)
        
    def forward(self, x, time, x_class, subject=None):
        #print("in:", x.shape)
        #print(x.shape)
        x = torch.fft.rfft(x, dim=2, norm='ortho')[:,:,:-1]
        #print(x.shape)
        x = torch.concat([x.real, x.imag], dim=1)
        x = self.encode_c(x)
        #print("encode_c:", x.shape)
        passthrough_start = x.clone()
        
        # Encode time
        t = math.log(10000) / (self.hidden//2 - 1)
        t = torch.exp(torch.arange(self.hidden//2, device=time.device) * -t)
        t = time[:, None] * t[None, :]
        t = torch.cat((t.sin(), t.cos()), dim=-1)
        t = self.encode_time_1(t)
        t = self.encode_time_act(t)
        t = self.encode_time_2(t)
        
        # Encode subject
        #s = subject
        #if exists(subject):
        #   s = self.encode_subject_1(s)
        #    s = self.encode_subject_act(s)
        #    s = self.encode_subject_2(s)
        
        # Encode class
        c = self.encode_class_1(x_class)
        c = self.encode_class_act(c)
        c = self.encode_class_2(c)
        
        passthroughs = []
        for block1, block2, resattn, downsample in self.down_half:
            #print("down:", x.shape)
            #print()
            x = block1(x, t)
            passthroughs.append(x)
            
            x = block2(x, t)
            x = resattn(x, c)
            passthroughs.append(x)
            
            x = downsample(x)
        
        #print("valley in:", x.shape)
        x = self.valley_1(x, t)
        #print("valley middle 1:", x.shape)
        x = self.valley_attention(x, c)
        #print("valley middle 2:", x.shape)
        x = self.valley_2(x, t)
        #print("valley out:", x.shape)
        
        for block1, block2, resattn, upsample in self.up_half:
            #print()
            #print("up:", x.shape)
            x = torch.cat((x, passthroughs.pop()), dim=1)
            #print("cat1:", x.shape)
            x = block1(x, t)
            #print("cat1 after:", x.shape)

            x = torch.cat((x, passthroughs.pop()), dim=1)
            #print("cat2:", x.shape)
            x = block2(x, t)
            x = resattn(x, c)
            
            x = upsample(x)
            
        x = torch.cat((x, passthrough_start), dim=1)
        x = self.end_1(x, t)
        x = self.end_2(x)
        
        xr = x[:,:22,:]
        xc = x[:,22:,:]
        x = torch.complex(xr, xc)
        x = torch.fft.irfft(x, n=256, dim=2, norm='ortho')
        return x
        
# Orignal Diffusion Paper:
# https://arxiv.org/pdf/2006.11239.pdf#page5
# https://huggingface.co/blog/annotated-diffusion
# Helper function to extract data to be of specific shape
def helper_extract(data, timesteps, shape):
    # Throws device error if not specified
    device = timesteps.device
    out = torch.gather(data, -1, timesteps.cpu())
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1))).to(timesteps.device)
        
class EEGDiffusion_Hybrid(pl.LightningModule):
    def __init__(
        self,
        hidden,
        lr,
        us=(1,2,4,8),
        seq_length=250,
        timesteps=500,
        sampling_timesteps=None,
        device="cuda"
        ):
        super().__init__()
        self.model = Unet_Hybrid(hidden, unet_scales=us)
        self.channels = 22
        self.seq_length = seq_length
        self.timesteps = timesteps
        self.lr = lr
        
        # Here is our betas: linearly increasing noise strength 
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1 - self.betas
        
        # As defined by paper
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1)
        
        # Intermediate step
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        
        # Key noise values
        # Intuition: We want to be able to produce noise that can be reversible. We do this by controlling how powerful the noise will be at different timesteps
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    # Defined by section 3.1: Foward Process
    def q_sample(self, x_start, t, noise=None):
        # Used to verify, not really needed for train/test
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get noise strength at time steps
        sqrt_alphas_cumprod_t = helper_extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = helper_extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
        # We want to ensure that a ratio between data and noise is preserved and mean stays the same
        # Apply noise
        #print(t.device)
        #print(sqrt_alphas_cumprod_t.device, x_start.device, sqrt_one_minus_alphas_cumprod_t.device, noise.device)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    # Algorithm 1
    def forward(self, x_start, t, x_class, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
    
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, x_class)
        # L2 loss used for higher sample diversity: https://arxiv.org/pdf/2111.05826.pdf
        loss = nn.functional.mse_loss(noise, predicted_noise)

        # Defining our own loss function to punish noise
        # Punishes high variance in sliding sunks of predicted noise
        #diff = noise - predicted_noise
        #v_loss = diff.unfold(dimension=2, size=16, step=8)
        #v_loss = v_loss.var(dim=3)
        #v_loss = v_loss.sum()/math.prod(v_loss.shape[:-1])/50
        #loss += vloss

        return loss
        
    @torch.no_grad()
    # Based on algorithm 4
    def p_sample(self, x, t, t_index, x_class):
        betas_t = helper_extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = helper_extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = helper_extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t, x_class) / sqrt_one_minus_alphas_cumprod_t)
    
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = helper_extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
            
    # Algorithm 2
    @torch.no_grad()
    def p_sample_loop(self, x_class, shape):
        b, c, l = shape
        
        noisy_start = torch.randn(shape)
        noisy_start = noisy_start.to(self.device)
        denoise = []
        
        for i in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps):
            noisy_start = self.p_sample(noisy_start, torch.full((b,), i, device=self.device, dtype=torch.long), i, x_class)
            denoise.append(noisy_start.cpu().numpy())
        return denoise
        
    @torch.no_grad()
    def sampling(self, x_class, batch_size=1, channels=22, sequence_length=250):
        return self.p_sample_loop(x_class, (batch_size, channels, sequence_length))
        
    def training_step(self, batch, batch_IX):
        data, data_class, data_subject = batch
        b = data.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        loss = self.forward(data, t, data_class)
        return loss
        
    def validation_step(self, batch, batch_IX):
        data, data_class, data_subject = batch
        b = data.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        loss = self.forward(data, t, data_class)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self): 
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
              self.opt,
              mode='min',
              factor=0.1,
              patience=5,
              verbose=True,
              min_lr=1e-8,
          )
        return [self.opt], {"optimizer": self.opt, "scheduler": self.reduce_lr_on_plateau, "monitor": "val_loss"}