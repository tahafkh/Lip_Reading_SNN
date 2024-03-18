import numpy as np
import torch
import torch.nn.functional as F

from utils import *
import math
from spikingjelly.activation_based import layer, rnn, neuron, functional, surrogate
from copy import deepcopy
from DCLS.construct.modules import Dcls3_1d

# New Model
class Dcls3_1_SJ(Dcls3_1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_count,
        learn_delay=True,
        stride=1,
        spatial_padding=0,
        dense_kernel_size=1,
        dilated_kernel_size=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        version='v1',
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_count,
            (stride, stride, 1),
            (*spatial_padding, 0),
            dense_kernel_size,
            dilated_kernel_size,
            groups,
            bias,
            padding_mode,  
            version,
        )
        self.learn_delay = learn_delay
        # if not self.learn_delay:
        #     torch.nn.init.constant_(self.P, dilated_kernel_size // 2)
        #     self.P.requires_grad = False
        if self.version == 'gauss':
            self.SIG.requires_grad = False
            

    def forward(self, x):
        x = x.permute(1, 2, 3, 4, 0) # [T, N, C, H, W] -> [N, C, H, W, T]
        x = F.pad(x, (self.dilated_kernel_size[0]-1, 0), mode='constant', value=0)
        x = super().forward(x)
        x = x.permute(4, 0, 1, 2, 3) # [N, C, H, W, T] -> [T, N, C, H, W]
        return x


# =======================================================================================================================================
# Two simple SNNs
# =======================================================================================================================================

class SNN1(torch.nn.Module):
	def __init__(self, n_class=4, spiking_neuron: callable = None, *args, **kwargs):
		super().__init__()

		conv = []
		conv.append(layer.MaxPool2d(4))
		conv.append(layer.Conv2d(1, 64, kernel_size=3, stride=1, bias=True))
		conv.append(layer.BatchNorm2d(64))
		conv.append(spiking_neuron(*args, **kwargs))
		conv.append(layer.Conv2d(64, 128, kernel_size=3, stride=1, bias=True))
		conv.append(layer.BatchNorm2d(128))
		conv.append(spiking_neuron(*args, **kwargs))
		conv.append(layer.AvgPool2d(2, 1))
		conv.append(layer.Conv2d(128, 128, kernel_size=3, stride=1, bias=True))
		conv.append(layer.BatchNorm2d(128))
		conv.append(spiking_neuron(*args, **kwargs))
		conv.append(layer.AvgPool2d(2, 1))
		self.down = torch.nn.Sequential(*conv)

		self.conv_fc = torch.nn.Sequential(
			layer.Flatten(),
			layer.Dropout(0.3),
			layer.Linear(25088, 256),
			spiking_neuron(*args, **kwargs),

			layer.Dropout(0.3),
			layer.Linear(256, n_class),

		)

	def forward(self, x: torch.Tensor):
		x = self.down(x)
		return self.conv_fc(x)

class SNN2(torch.nn.Module):
	def __init__(self, num_labels=4, norm_layer=None, num_init_channels=4, spiking_neuron: callable = None, *args, **kwargs):
		super().__init__()
		biais=True
		head = []
		head.append(layer.Conv2d(num_init_channels, 32, kernel_size=5, stride=1, bias=biais, padding=2))
		head.append(layer.BatchNorm2d(32))
		head.append(spiking_neuron(*args, **kwargs))

		down = []
		down.append(layer.Conv2d(32, 64, kernel_size=5, stride=2, bias=biais, padding=2))
		down.append(layer.BatchNorm2d(64))
		down.append(spiking_neuron(*args, **kwargs))

		down.append(layer.Conv2d(64, 128, kernel_size=5, stride=2, bias=biais, padding=2))
		down.append(layer.BatchNorm2d(128))
		down.append(spiking_neuron(*args, **kwargs))

		down.append(layer.Conv2d(128, 256, kernel_size=5, stride=2, bias=biais, padding=2))
		down.append(layer.BatchNorm2d(256))
		down.append(spiking_neuron(*args, **kwargs))

		down.append(layer.Conv2d(256, 512, kernel_size=5, stride=2, bias=biais, padding=2))
		down.append(layer.BatchNorm2d(512))
		down.append(spiking_neuron(*args, **kwargs))

		down.append(layer.Conv2d(512, 1024, kernel_size=5, stride=2, bias=biais, padding=2))
		down.append(layer.BatchNorm2d(1024))
		down.append(spiking_neuron(*args, **kwargs))

		self.encoder = torch.nn.Sequential(
			*head,
			*down
		)

		self.fc = torch.nn.Sequential(
			layer.Flatten(),
			layer.Dropout(0.4),
			layer.Linear(9216, 512),
			spiking_neuron(*args, **kwargs),
			layer.Dropout(0.4),
			layer.Linear(512, num_labels),

		)

	def forward(self, x: torch.Tensor):
		encoder_out=self.encoder(x)
		return self.fc(encoder_out)

# =======================================================================================================================================
# Spiking ResNet - Spiking MSTP Low Branch 
# =======================================================================================================================================

def conv3x3(in_planes, out_planes, stride=1):
	"""1x3x3 convolution with padding"""
	return layer.Conv2d(in_planes, out_planes,
					 kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)

def new_conv3x3(in_planes, out_planes, stride=1):
    return Dcls3_1_SJ(in_channels=in_planes, out_channels=out_planes, kernel_count=1,
                               stride=stride, dense_kernel_size=3, dilated_kernel_size=(3, ),
                               bias=False, groups=1, spatial_padding=(3//2, 3//2), version='v1'
                              )

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1x1 convolution with padding"""
	return layer.Conv2d(in_planes, out_planes,
					 kernel_size=(1, 1), stride=(stride, stride), bias=False)

def new_conv1x1(in_planes, out_planes, stride=1):
    return Dcls3_1_SJ(in_channels=in_planes, out_channels=out_planes, kernel_count=1,
                               stride=stride, dense_kernel_size=1, dilated_kernel_size=(3, ),
                               bias=False, groups=1, spatial_padding=(1//2, 1//2), version='v1'
                              )


class BasicBlock(torch.nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, delayed=False, stride=1, downsample=None, se=False, spiking_neuron=None,*args, **kwargs):
		super(BasicBlock, self).__init__()
		self.delayed = delayed
		final_conv3x3 = new_conv3x3 if delayed else conv3x3
		final_conv1x1 = new_conv1x1 if delayed else conv1x1
		self.conv1 = final_conv3x3(inplanes, planes, stride)
		self.bn1 = layer.BatchNorm2d(planes)
		self.spiking1 = spiking_neuron(*args, **kwargs)
		self.relu = torch.nn.ReLU(inplace=True)
		self.conv2 = final_conv3x3(planes, planes)
		self.bn2 = layer.BatchNorm2d(planes)
		self.spiking2 = spiking_neuron(*args, **kwargs)
		self.downsample = downsample
		self.stride = stride
		self.se = se
		self.inplanes=inplanes
		self.planes = planes

		if(self.se):
			self.gap = layer.AdaptiveAvgPool2d(1)
			self.conv3 = final_conv1x1(planes, planes//16)
			self.spiking3 = spiking_neuron(*args, **kwargs)
			self.conv4 = final_conv1x1(planes//16, planes)
			self.spiking4 = spiking_neuron(*args, **kwargs)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.spiking1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.spiking2(out)  

		if self.downsample is not None:
			residual = self.downsample(x)

		if(self.se):
			w = self.gap(out)
			w = self.conv3(w)
			w = self.spiking3(w)
			w = self.relu(w)
			w = self.conv4(w).sigmoid()
			w = self.spiking4(w)

			out = out * w

		out = out + residual

		return out

	def clamp_parameters(self):
		if self.delayed:
			self.conv1.clamp_parameters()
			self.conv2.clamp_parameters()
			if self.se:
				self.conv3.clamp_parameters()
				self.conv4.clamp_parameters()

class ResNet18(torch.nn.Module):
	def __init__(self, block, layers, se=False, spiking_neuron=None, *args, **kwargs):
		super(ResNet18, self).__init__()
		in_channels = 1 #kwargs['in_channels']
		self.low_rate = 1 #kwargs['low_rate']
		self.alpha = 1 #kwargs['alpha']
		self.t2s_mul = 0 #kwargs['t2s_mul']
		self.base_channel = 64 #kwargs['base_channel']
		self.inplanes = (self.base_channel + self.base_channel//self.alpha*self.t2s_mul) if self.low_rate else self.base_channel // self.alpha
		self.conv1 = layer.Conv2d(in_channels, self.base_channel // (1 if self.low_rate else self.alpha),
							   kernel_size=(7, 7),
							   stride=(2, 2), padding=(3, 3), bias=False)
		self.bn1 = layer.BatchNorm2d(self.base_channel // (1 if self.low_rate else self.alpha))
		self.spiking1 = spiking_neuron(*args, **kwargs)
		self.relu = torch.nn.ReLU(inplace=True)
		self.maxpool = layer.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.se = se
		self.layers = []
		
		self.layer1 = self._make_layer(block, self.base_channel // (1 if self.low_rate else self.alpha), layers[0], delayed=True, spiking_neuron=spiking_neuron, *args,**kwargs)
		self.layer2 = self._make_layer(block, 2 * self.base_channel // (1 if self.low_rate else self.alpha), layers[1], delayed=True, stride=2, spiking_neuron=spiking_neuron, *args,**kwargs)
		self.layer3 = self._make_layer(block, 4 * self.base_channel // (1 if self.low_rate else self.alpha), layers[2], delayed=True, stride=2, spiking_neuron=spiking_neuron, *args,**kwargs)
		self.layer4 = self._make_layer(block, 8 * self.base_channel // (1 if self.low_rate else self.alpha), layers[3], delayed=True, stride=2, spiking_neuron=spiking_neuron, *args,**kwargs)

		self.avgpool = layer.AdaptiveAvgPool2d(1)
		if self.low_rate:
			self.bn2 = torch.nn.BatchNorm1d(8*self.base_channel + 8*self.base_channel//self.alpha*self.t2s_mul)
		elif self.t2s_mul == 0:
			self.bn2 = layer.BatchNorm1d(16*self.base_channel//self.alpha)

		self.spiking2 = spiking_neuron(*args, **kwargs)

	def _make_layer(self, block, planes, blocks, delayed=False, stride=1, spiking_neuron=None, *args, **kwargs):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = torch.nn.Sequential(
				layer.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=(stride, stride), bias=False),
				layer.BatchNorm2d(planes * block.expansion),
				spiking_neuron(*args, **kwargs)
			)

		layers = []
		layers.append(block(self.inplanes, planes, delayed, stride, downsample, se=self.se, spiking_neuron=spiking_neuron, *args,**kwargs))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, delayed, se=self.se, spiking_neuron=spiking_neuron, *args,**kwargs))

		#self.inplanes += self.low_rate * block.expansion * planes // self.alpha #* self.t2s_mul
		self.inplanes = planes
		self.layers.append(layers)

		return torch.nn.Sequential(*layers)

	def forward(self, x):
		raise NotImplementedError
	
	def clamp_parameters(self):
		for layer in self.layers:
			for block in layer:
				block.clamp_parameters()

	def init_params(self):
		for m in self.modules():
			if isinstance(m, layer.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, layer.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, layer.Conv1d):
				n = m.kernel_size[0] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, layer.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

			elif isinstance(m, layer.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

			elif isinstance(m, layer.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

# I also adapted MFM but ended up not using them. It might be useful if you try to also use the high-rate branch of MSTP.
class MFM(torch.nn.Module):
	def __init__(self, in_channel, out_channel, spiking_neuron, *args, **kwargs):
		super(MFM, self).__init__()
		self.layer1 = torch.nn.Sequential(
			conv1x1(in_channel, out_channel),
			layer.BatchNorm2d(out_channel),
			spiking_neuron(*args, **kwargs)
		)
		self.local_att_layer = torch.nn.Sequential(
			conv1x1(out_channel, out_channel//4),
			layer.BatchNorm2d(out_channel//4),
			spiking_neuron(*args, **kwargs),
			conv1x1(out_channel//4, out_channel),
			layer.BatchNorm2d(out_channel),
			spiking_neuron(*args, **kwargs)
		)
		self.global_att_layer = torch.nn.Sequential(
			layer.AdaptiveAvgPool2d(1),
			conv1x1(out_channel, out_channel//4),
			layer.BatchNorm2d(out_channel//4),
			spiking_neuron(*args, **kwargs),
			conv1x1(out_channel//4, out_channel),
			layer.BatchNorm2d(out_channel),
			spiking_neuron(*args, **kwargs),
		)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x, y):
		x = torch.cat([x, y], dim=1)
		x = self.layer1(x)
		local_att = self.local_att_layer(x)
		global_att = self.global_att_layer(x)
		y = x + x * self.sigmoid(local_att + global_att)
		#y = x * self.sigmoid(local_att + global_att)
		return y


class LowRateBranch(ResNet18):
	def __init__(self, block=BasicBlock, layers=[2,2,2,2], se=False, spiking_neuron=None, n_class=5, *args,**kwargs):
		super().__init__(block, layers, se, spiking_neuron=spiking_neuron, *args,**kwargs)
		self.base_channel = 64 #kargs['base_channel']
		self.alpha = 1 #kargs['alpha']

		self.spiking_stateful_synapse = torch.nn.Sequential(
			layer.Linear(8*self.base_channel, 1024),
			spiking_neuron(*args, **kwargs),
			layer.SynapseFilter(tau=2.0, learnable=True)
		)

		self.v_cls = layer.Linear(1024, n_class)

		self.vote = layer.VotingLayer(10)
		self.dropout = torch.nn.Dropout(p=0.5)

		self.init_params()

	def forward(self, x):
		b=x.shape[1]

		x = self.conv1(x)
		x = self.bn1(x)

		x = self.spiking1(x)
		x = self.maxpool(x) # -> (T, b, 64, 22, 22)

		x = self.layer1(x) # -> (T, b, 64, 22, 22) (unchanged)
		x = self.layer2(x) # -> (T, b, 128, 11, 11)
		x = self.layer3(x) # -> (T, b, 256, 6, 6) 
		x = self.layer4(x) # -> (T, b, 512, 3, 3) 
	
		x = self.avgpool(x) # -> (T, b, 512, 1, 1)

		# we go from T, b, c, w, h  to->  T*b, w*h*c this what is done 
		x = x.view( -1, x.size(2))
		x = self.bn2(x)
		# And now from T*b, c -> T, b, w*h*c 
		x = x.view(-1 ,b, 8*64+8*64//1*0)

		x = self.dropout(x)
		x= self.spiking_stateful_synapse (x.transpose(0,1)) 
		x=x.transpose(0,1)

		x = self.dropout(x)
		x = self.v_cls(x)

		return x