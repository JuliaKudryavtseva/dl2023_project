import torch
import torch.nn as nn
import math
import torch.nn.functional as F

    
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(channels//reduction, channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.attention(x))
        
    
class Flatten(nn.Module):      
    def forward(self, inp): return inp.view(inp.size(0), -1)

    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        return x
    

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size)
        self.mp = nn.AdaptiveMaxPool2d(output_size)
        
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)
        
            
def custom_head(num_classes=1000, num_feat=1024, ps=0.5):

    return nn.Sequential(
        Flatten(),
        nn.BatchNorm1d(num_features=num_feat),
        nn.Dropout(p=ps/2),
        nn.Linear(in_features=num_feat, out_features=num_feat // 2, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=num_feat // 2),
        nn.Dropout(p=ps),
        nn.Linear(in_features=num_feat // 2, out_features=num_classes, bias=True),
    )


# class Upsample(nn.Module):
#     def __init__(self, scale):
#         super(Upsample, self).__init__()
#         self.scale = scale

#     def forward(self, x):
#         return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
    

def conv_bn_act(inp, oup, kernel_size, stride=1, groups=1, bias=True, eps=1e-3, momentum=0.01):
    return nn.Sequential(
        SamePadConv2d(inp, oup, kernel_size, stride, groups=groups, bias=bias),
        nn.BatchNorm2d(oup, eps, momentum),
        Swish()
    )


class SamePadConv2d(nn.Conv2d):

    def __init__(self, inp, oup, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(inp, oup, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)


class Flattener:
    def __init__(self):
        self.flattened_module = []
    
    def flat(self, module):
        flattened_module = []
        childrens = list(module.children())
        for children in childrens:
            flattened_module.append(children)
        return flattened_module
    
    def __call__(self, module):
        childrens = list(module.children())
        for children in childrens:
            if len(self.flat(children))==0:
                self.flattened_module.append(children)
            else:
                self.__call__(children) 
        return self.flattened_module