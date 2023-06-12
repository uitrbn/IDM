import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out

class BSReduction(nn.Module):
    def __init__(self, input_bs, output_bs, channel, feat_size):
        super().__init__()
        self.input_bs = input_bs
        self.output_bs = output_bs
        self.channel = channel
        self.feat_size = feat_size

        self.module_list = list()

        self.channel_factor = [input_bs, 8, 1]
        for idx in range(len(self.channel_factor) - 1):
            # print('Building Layer from Channel')
            factor, next_factor = self.channel_factor[idx], self.channel_factor[idx+1]
            self.module_list.append(nn.Conv2d(channel * factor, channel * next_factor, kernel_size=3, padding=1))
            shape_feat = [channel * next_factor, feat_size[0], feat_size[1]]
            self.module_list.append(self._get_normlayer('instancenorm', shape_feat))
            self.module_list.append(self._get_activation('relu'))
            # self.module_list.append(self._get_pooling('avgpooling'))
        
        self.features = nn.Sequential(*self.module_list)
    
    def forward(self, x):
        return self.features(x)

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

class BSReduction_v2(nn.Module):
    def __init__(self, input_bs, output_bs, channel, feat_size):
        super().__init__()
        self.input_bs = input_bs
        self.output_bs = output_bs
        self.channel = channel
        self.feat_size = feat_size

        self.module_list = list()

        self.channel_factor = [input_bs, 1]
        for idx in range(len(self.channel_factor) - 1):
            # print('Building Layer from Channel')
            factor, next_factor = self.channel_factor[idx], self.channel_factor[idx+1]
            self.module_list.append(nn.Conv2d(factor, next_factor, kernel_size=1, padding=0))
            shape_feat = [next_factor, feat_size[0], feat_size[1]]
            self.module_list.append(self._get_normlayer('instancenorm', shape_feat))
            self.module_list.append(self._get_activation('relu'))
            # self.module_list.append(self._get_pooling('avgpooling'))
        
        self.features = nn.Sequential(*self.module_list)
    
    def forward(self, x):
        out = x.permute(1, 0, 2, 3)
        out = self.features(out)
        out = out.permute(1, 0, 2, 3)
        return x.mean(dim=0) + out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)


''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, upsample=False, embed=False):
        if not upsample:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            if not embed:
                out = self.classifier(out)
                return out
            else:
                return out, self.classifier(out)
        else:
            modulelist = list(self.features.modules())[1:]
            for module in modulelist[:-1]:
                x = module(x)
            out = torch.concat([x[:, :, :8, :8], x[:, :, :8, 8:], x[:, :, 8:, :8], x[:, :, 8:, 8:]], dim=0)
            out = modulelist[-1](out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def embed_channel_avg(self, x, last=-1, flatten=True):
        # import pdb; pdb.set_trace()
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)

    def embed_feature_and_logit(self, x, last=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        out = x.view(x.size(0), -1)
        for module in modulelist[last:]:
            x = module(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return out, x

    def subnet_forward(self, x, begin, last, flatten=True):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[begin:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)

    def slice_forward(self, x, slices=None):
        modulelist = list(self.features.modules())[1:]
        output_list = list()
        for module_idx, module in enumerate(modulelist):
            x = module(x)
            if module_idx in slices:
                output_list.append(x)
        return output_list

    def embed_layer_diff(self, x, last=-1, upsample=False):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        if upsample:
            x = self.upsample(x)
        return x
    
    def embed_random_mask(self, x, mask=True, cutout_ratio=0.5):
        modulelist = list(self.features.modules())[1:]
        for idx, module in enumerate(modulelist):
            if mask and 'Conv' in str(module) and idx != 0:
                assert len(x.shape) == 4
                h, w = x.shape[2], x.shape[3]
                h_cutout = int(h * cutout_ratio)
                w_cutout = int(w * cutout_ratio)
                cutout_mask = torch.ones_like(x).detach()
                h_begin = np.random.randint(0, h - h_cutout)
                w_begin = np.random.randint(0, w - w_cutout)
                cutout_mask[:, :, h_begin:h_begin+h_cutout, w_begin:w_begin+w_cutout] = 0
                # cutout_mask = 1 - cutout_mask
                x = x * cutout_mask
            x = module(x)
        out = x.view(x.size(0), -1)
        return out

    def embed_layerwise(self, x, select=[], first_layer_num=4):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:first_layer_num]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num:first_layer_num+4]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num+4:]:
            x =  module(x)
        layer_features.append(x)
        if len(select) > 0:
            layer_features = [layer_features[i] for i in select]
        return layer_features
    
    def embed_layerwise_v2_2(self, x):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:9]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[9:]:
            x =  module(x)
        layer_features.append(x)
        return layer_features

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' ConvNet '''
class ConvNet_6(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet_6, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, upsample=False, embed=False, last=100):
        # import pdb; pdb.set_trace()
        assert not upsample
        # out = self.features(x)
        # out = out.view(out.size(0), -1)
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        out = x
        if not embed:
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out
        else:
            return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def embed_channel_avg(self, x, last=-1, flatten=True):
        # import pdb; pdb.set_trace()
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)

    def embed_feature_and_logit(self, x, last=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        out = x.view(x.size(0), -1)
        for module in modulelist[last:]:
            x = module(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return out, x

    def subnet_forward(self, x, begin, last, flatten=True):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[begin:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)

    def slice_forward(self, x, slices=None):
        modulelist = list(self.features.modules())[1:]
        output_list = list()
        for module_idx, module in enumerate(modulelist):
            x = module(x)
            if module_idx in slices:
                output_list.append(x)
        return output_list

    def embed_layer_diff(self, x, last=-1, upsample=False):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        if upsample:
            x = self.upsample(x)
        return x
    
    def embed_random_mask(self, x, mask=True, cutout_ratio=0.5):
        modulelist = list(self.features.modules())[1:]
        for idx, module in enumerate(modulelist):
            if mask and 'Conv' in str(module) and idx != 0:
                assert len(x.shape) == 4
                h, w = x.shape[2], x.shape[3]
                h_cutout = int(h * cutout_ratio)
                w_cutout = int(w * cutout_ratio)
                cutout_mask = torch.ones_like(x).detach()
                h_begin = np.random.randint(0, h - h_cutout)
                w_begin = np.random.randint(0, w - w_cutout)
                cutout_mask[:, :, h_begin:h_begin+h_cutout, w_begin:w_begin+w_cutout] = 0
                # cutout_mask = 1 - cutout_mask
                x = x * cutout_mask
            x = module(x)
        out = x.view(x.size(0), -1)
        return out

    def embed_layerwise(self, x, select=[], first_layer_num=4):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:first_layer_num]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num:first_layer_num+4]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num+4:]:
            x =  module(x)
        layer_features.append(x)
        if len(select) > 0:
            layer_features = [layer_features[i] for i in select]
        return layer_features
    
    def embed_layerwise_v2_2(self, x):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:9]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[9:]:
            x =  module(x)
        layer_features.append(x)
        return layer_features

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

''' ConvNet '''
class ConvNet_Pooling(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet_Pooling, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, upsample=False, embed=False, pooling=False):
        if not upsample:
            out = self.features(x)
            if pooling:
                out = self.pooling(out)
            out = out.view(out.size(0), -1)
            if not embed:
                out = self.classifier(out)
                return out
            else:
                return out, self.classifier(out)
        else:
            modulelist = list(self.features.modules())[1:]
            for module in modulelist[:-1]:
                x = module(x)
            out = torch.concat([x[:, :, :8, :8], x[:, :, :8, 8:], x[:, :, 8:, :8], x[:, :, 8:, 8:]], dim=0)
            out = modulelist[-1](out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def embed_channel_avg(self, x, last=-1, flatten=True, pooling=False):
        # import pdb; pdb.set_trace()
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        if pooling:
            x = self.pooling(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x

    def subnet_forward(self, x, begin, last, flatten=True):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[begin:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)

    def slice_forward(self, x, slices=None):
        modulelist = list(self.features.modules())[1:]
        output_list = list()
        for module_idx, module in enumerate(modulelist):
            x = module(x)
            if module_idx in slices:
                output_list.append(x)
        return output_list

    def embed_layer_diff(self, x, last=-1, upsample=False):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        if upsample:
            x = self.upsample(x)
        return x
    
    def embed_random_mask(self, x, mask=True, cutout_ratio=0.5):
        modulelist = list(self.features.modules())[1:]
        for idx, module in enumerate(modulelist):
            if mask and 'Conv' in str(module) and idx != 0:
                assert len(x.shape) == 4
                h, w = x.shape[2], x.shape[3]
                h_cutout = int(h * cutout_ratio)
                w_cutout = int(w * cutout_ratio)
                cutout_mask = torch.ones_like(x).detach()
                h_begin = np.random.randint(0, h - h_cutout)
                w_begin = np.random.randint(0, w - w_cutout)
                cutout_mask[:, :, h_begin:h_begin+h_cutout, w_begin:w_begin+w_cutout] = 0
                # cutout_mask = 1 - cutout_mask
                x = x * cutout_mask
            x = module(x)
        out = x.view(x.size(0), -1)
        return out

    def embed_layerwise(self, x, select=[], first_layer_num=4):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:first_layer_num]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num:first_layer_num+4]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num+4:]:
            x =  module(x)
        layer_features.append(x)
        if len(select) > 0:
            layer_features = [layer_features[i] for i in select]
        return layer_features
    
    def embed_layerwise_v2_2(self, x):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:9]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[9:]:
            x =  module(x)
        layer_features.append(x)
        return layer_features

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        self.pooling = self._get_pooling(net_pooling)
        return nn.Sequential(*layers), shape_feat


''' ConvNet '''
class ConvNet_BSR(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet_BSR, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bsr = BSReduction_v2(256, 1, 128, (8, 8))

    def forward(self, x, upsample=False, embed=False):
        assert not upsample
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if not embed:
            out = self.classifier(out)
            return out
        else:
            return out, self.classifier(out)

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
    
    def forward_bsr(self, x, bsr_loc=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:bsr_loc]:
            x = module(x)
        # bsr_output = self.bsr(x.view(1, -1, x.shape[-2], x.shape[-1]))
        bsr_output = self.bsr(x)
        for module in modulelist[bsr_loc:-1]:
            x = module(x)
        x = modulelist[-1](x)
        bsr_output = modulelist[-1](bsr_output)
        logits = self.classifier(x.view(x.size(0), -1))
        bsr_logits = self.classifier(bsr_output.view(bsr_output.size(0), -1))
        return logits, bsr_logits

    def embed_channel_avg(self, x, last=-1, flatten=True, bsr=False, bsr_loc=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:bsr_loc]:
            x = module(x)
        if bsr:
            bsr_output = self.bsr(x)
        for module in modulelist[bsr_loc:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            if bsr:
                bsr_output = bsr_output.view(bsr_output.size(0), -1)
                return out, bsr_output
            return out
        else:
            if bsr:
                return x, bsr_output
            return x

    def embed_feature_and_logit(self, x, last=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        out = x.view(x.size(0), -1)
        for module in modulelist[last:]:
            x = module(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return out, x

    def subnet_forward(self, x, begin, last, flatten=True):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[begin:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)

    def slice_forward(self, x, slices=None):
        modulelist = list(self.features.modules())[1:]
        output_list = list()
        for module_idx, module in enumerate(modulelist):
            x = module(x)
            if module_idx in slices:
                output_list.append(x)
        return output_list

    def embed_layer_diff(self, x, last=-1, upsample=False):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        if upsample:
            x = self.upsample(x)
        return x
    
    def embed_random_mask(self, x, mask=True, cutout_ratio=0.5):
        modulelist = list(self.features.modules())[1:]
        for idx, module in enumerate(modulelist):
            if mask and 'Conv' in str(module) and idx != 0:
                assert len(x.shape) == 4
                h, w = x.shape[2], x.shape[3]
                h_cutout = int(h * cutout_ratio)
                w_cutout = int(w * cutout_ratio)
                cutout_mask = torch.ones_like(x).detach()
                h_begin = np.random.randint(0, h - h_cutout)
                w_begin = np.random.randint(0, w - w_cutout)
                cutout_mask[:, :, h_begin:h_begin+h_cutout, w_begin:w_begin+w_cutout] = 0
                # cutout_mask = 1 - cutout_mask
                x = x * cutout_mask
            x = module(x)
        out = x.view(x.size(0), -1)
        return out

    def embed_layerwise(self, x, select=[], first_layer_num=4):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:first_layer_num]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num:first_layer_num+4]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num+4:]:
            x =  module(x)
        layer_features.append(x)
        if len(select) > 0:
            layer_features = [layer_features[i] for i in select]
        return layer_features
    
    def embed_layerwise_v2_2(self, x):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:9]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[9:]:
            x =  module(x)
        layer_features.append(x)
        return layer_features

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' ConvNet '''
class ConvNet_F_C(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, upsample=False, embed=False):
        if not upsample:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            if not embed:
                out = self.classifier(out)
                return out
            else:
                return out, self.classifier(out)
        else:
            modulelist = list(self.features.modules())[1:]
            for module in modulelist[:-1]:
                x = module(x)
            out = torch.concat([x[:, :, :8, :8], x[:, :, :8, 8:], x[:, :, 8:, :8], x[:, :, 8:, 8:]], dim=0)
            out = modulelist[-1](out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def embed_channel_avg(self, x, last=-1, flatten=True):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' ConvNet with two kinds of normalization layers '''
class ConvNet_GBN(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet_GBN, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, upsample=False, embed=False):
        if not upsample:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            if not embed:
                out = self.classifier(out)
                return out
            else:
                return out, self.classifier(out)
        else:
            modulelist = list(self.features.modules())[1:]
            for module in modulelist[:-1]:
                x = module(x)
            out = torch.concat([x[:, :, :8, :8], x[:, :, :8, 8:], x[:, :, 8:, :8], x[:, :, 8:, 8:]], dim=0)
            out = modulelist[-1](out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def embed_channel_avg(self, x, last=-1, flatten=True, bn_layer=False):
        if not bn_layer:
            modulelist = list(self.features.modules())[1:]
            for module in modulelist[:last]:
                x = module(x)
            if flatten:
                out = x.view(x.size(0), -1)
                return out
            else:
                return x
        else:
            modulelist = list(self.features.modules())[1:]
            for module_idx, module in enumerate(modulelist[:last]):
                if module_idx in [1, 5, 9]:
                    x = self.bn_layers[module_idx//4](x)
                else:
                    x = module(x)
            if flatten:
                out = x.view(x.size(0), -1)
                return out
            else:
                return x

    def embed_feature_and_logit(self, x, last=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        out = x.view(x.size(0), -1)
        for module in modulelist[last:]:
            x = module(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return out, x

    def subnet_forward(self, x, begin, last, flatten=True):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[begin:last]:
            x = module(x)
        if flatten:
            out = x.view(x.size(0), -1)
            return out
        else:
            return x
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)

    def slice_forward(self, x, slices=None):
        modulelist = list(self.features.modules())[1:]
        output_list = list()
        for module_idx, module in enumerate(modulelist):
            x = module(x)
            if module_idx in slices:
                output_list.append(x)
        return output_list

    def embed_layer_diff(self, x, last=-1, upsample=False):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        if upsample:
            x = self.upsample(x)
        return x
    
    def embed_random_mask(self, x, mask=True, cutout_ratio=0.5):
        modulelist = list(self.features.modules())[1:]
        for idx, module in enumerate(modulelist):
            if mask and 'Conv' in str(module) and idx != 0:
                assert len(x.shape) == 4
                h, w = x.shape[2], x.shape[3]
                h_cutout = int(h * cutout_ratio)
                w_cutout = int(w * cutout_ratio)
                cutout_mask = torch.ones_like(x).detach()
                h_begin = np.random.randint(0, h - h_cutout)
                w_begin = np.random.randint(0, w - w_cutout)
                cutout_mask[:, :, h_begin:h_begin+h_cutout, w_begin:w_begin+w_cutout] = 0
                # cutout_mask = 1 - cutout_mask
                x = x * cutout_mask
            x = module(x)
        out = x.view(x.size(0), -1)
        return out

    def embed_layerwise(self, x, select=[], first_layer_num=4):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:first_layer_num]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num:first_layer_num+4]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num+4:]:
            x =  module(x)
        layer_features.append(x)
        if len(select) > 0:
            layer_features = [layer_features[i] for i in select]
        return layer_features
    
    def embed_layerwise_v2_2(self, x):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:9]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[9:]:
            x =  module(x)
        layer_features.append(x)
        return layer_features

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        self.bn_layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
                self.bn_layers += [self._get_normlayer('batchnorm', shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        self.bn_layers = nn.ModuleList(self.bn_layers)
        return nn.Sequential(*layers), shape_feat



from torch.autograd import Function
class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        # ctx.save_for_backward(input_, alpha_)
        ctx.save_for_backward(input_)
        ctx.alpha_ = alpha_
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _ = ctx.saved_tensors
        alpha_ = ctx.alpha_
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None

revgrad = RevGrad.apply

''' ConvNet_Adv '''
class ConvNet_Adv(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32), fc_num=100):
        super(ConvNet_Adv, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        # self.classifier = nn.Linear(num_feat, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, num_classes)
        )

        self.classifier_list = list()
        self.fc_num = fc_num
        for i in range(self.fc_num):
            self.classifier_list.append(
                nn.Sequential(
                    nn.Linear(2048, 1000),
                    nn.BatchNorm1d(1000, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(1000, 1000),
                    nn.BatchNorm1d(1000, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(1000, num_classes)
                )
            )

    def forward(self, x, adv=False, adv_factor=1):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if adv:
            out = revgrad(out, adv_factor)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def embed_channel_avg(self, x, last=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        out = x.view(x.size(0), -1)
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        return out
    
    def multi_fc_forward(self, x, fc_index):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier_list[fc_index](out)
        return out

    def embed_layer_diff(self, x, last=-1, upsample=False):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        if upsample:
            x = self.upsample(x)
        return x
    
    def embed_random_mask(self, x, mask=True, cutout_ratio=0.5):
        modulelist = list(self.features.modules())[1:]
        for idx, module in enumerate(modulelist):
            if mask and 'Conv' in str(module) and idx != 0:
                assert len(x.shape) == 4
                h, w = x.shape[2], x.shape[3]
                h_cutout = int(h * cutout_ratio)
                w_cutout = int(w * cutout_ratio)
                cutout_mask = torch.ones_like(x).detach()
                h_begin = np.random.randint(0, h - h_cutout)
                w_begin = np.random.randint(0, w - w_cutout)
                cutout_mask[:, :, h_begin:h_begin+h_cutout, w_begin:w_begin+w_cutout] = 0
                # cutout_mask = 1 - cutout_mask
                x = x * cutout_mask
            x = module(x)
        out = x.view(x.size(0), -1)
        return out

    def embed_layerwise(self, x, select=[], first_layer_num=4):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:first_layer_num]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num:first_layer_num+4]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num+4:]:
            x =  module(x)
        layer_features.append(x)
        if len(select) > 0:
            layer_features = [layer_features[i] for i in select]
        return layer_features
    
    def embed_layerwise_v2_2(self, x):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:9]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[9:]:
            x =  module(x)
        layer_features.append(x)
        return layer_features

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat



''' ConvNet '''
class ConvNet_fc(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32), fc_dim=128):
        super(ConvNet_fc, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.fc = nn.Linear(num_feat, fc_dim)
        self.classifier = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def embed_channel_avg(self, x, last=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        out = x.view(x.size(0), -1)
        # out = self.features(x)
        # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        return out

    def embed_with_output(self, x, last=-1):
        modulelist = list(self.features.modules())[1:]
        for module in modulelist[:last]:
            x = module(x)
        feat = x.view(x.size(0), -1)
        out = self.classifier(self.fc(feat))
        return out

    def embed_layerwise(self, x, select=[], first_layer_num=4):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:first_layer_num]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num:first_layer_num+4]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[first_layer_num+4:]:
            x =  module(x)
        layer_features.append(x)
        if len(select) > 0:
            layer_features = [layer_features[i] for i in select]
        return layer_features
    
    def embed_layerwise_v2_2(self, x):
        modulelist = list(self.features.modules())[1:]
        layer_features = list()
        for module in modulelist[:9]:
            x = module(x)
        layer_features.append(x)
        for module in modulelist[9:]:
            x =  module(x)
        layer_features.append(x)
        return layer_features

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' LeNet '''
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x



''' AlexNet '''
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


''' AlexNetBN '''
class AlexNetBN(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNetBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel==1 and ic==0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm=='instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)
def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')
def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)
def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)
def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)

class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2), # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel==1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        return out

def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # if num_classes == 200:
        #     self.classifier = nn.Linear(512*block.expansion, num_classes)
        # else:
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def embed_by_layer(self, x, layer_idx=100, last=100):
        layer_idx = min(layer_idx, last)
        # modulelist = list(self.features.modules())[1:]
        # for module in modulelist[:last]:
        #     x = module(x)
        # out = x.view(x.size(0), -1)
        # # out = self.features(x)
        # # out = out.mean(dim=2).mean(dim=2).view(out.size(0), -1)
        # return out
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if layer_idx == 1:
            return out.view(out.size(0), -1)
        out = self.layer2(out)
        if layer_idx == 2:
            return out.view(out.size(0), -1)
        out = self.layer3(out)
        if layer_idx == 3:
            return out.view(out.size(0), -1)
        out = self.layer4(out)
        if layer_idx == 4:
            return out.view(out.size(0), -1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18BN(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet50(channel, num_classes):
    return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet101(channel, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes)

def ResNet152(channel, num_classes):
    return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes)

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
#         super(BasicBlock, self).__init__()
#         # self.bn1 = nn.BatchNorm2d(in_planes)
#         self.bn1 = nn.GroupNorm(in_planes, in_planes, affine=True)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.Insta(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.droprate = dropRate
#         self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                                padding=0, bias=False) or None
#     def forward(self, x):
#         if not self.equalInOut:
#             x = self.relu1(self.bn1(x))
#         else:
#             out = self.relu1(self.bn1(x))
#         out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#         out = self.conv2(out)
#         return torch.add(x if self.equalInOut else self.convShortcut(x), out)

# class NetworkBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
#         super(NetworkBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
#     def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
#         layers = []
#         for i in range(int(nb_layers)):
#             layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         return self.layer(x)

# class WideResNet(nn.Module):
#     def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
#         super(WideResNet, self).__init__()
#         nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert((depth - 4) % 6 == 0)
#         n = (depth - 4) / 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         # self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.bn1 = nn.GroupNorm(nChannels[3], nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.nChannels)
#         return self.fc(out)