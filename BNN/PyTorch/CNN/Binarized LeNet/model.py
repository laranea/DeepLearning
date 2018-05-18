import torch
import torch.nn.functional as F


## Binary Modules

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


class BinarizeLinear(torch.nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = torch.nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out

class BinarizeConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = torch.nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out



## Model

class LeNet_Binarized(torch.nn.Module):
    def __init__(self):
        super(LeNet_Binarized,self).__init__()
        self.ratio_infl=3
        ## Repacing ReLU with hardtanh
        self.convnet=torch.nn.Sequential(
                        BinarizeConv2d(1,6*self.ratio_infl,kernel_size=5),
                        torch.nn.MaxPool2d(kernel_size=2, stride=2),
                        torch.nn.BatchNorm2d(6*self.ratio_infl),
                        torch.nn.Hardtanh(inplace=True),

                        BinarizeConv2d(6*self.ratio_infl,16*self.ratio_infl,kernel_size=5),
                        torch.nn.MaxPool2d(kernel_size=2,stride=2),
                        torch.nn.BatchNorm2d(16*self.ratio_infl),
                        torch.nn.Hardtanh(),

                        BinarizeConv2d(16*self.ratio_infl,120, kernel_size=5),
                        torch.nn.BatchNorm2d(120),
                        torch.nn.Hardtanh()
                        )

        self.classifier=torch.nn.Sequential(
                        BinarizeLinear(120,84*self.ratio_infl),
                        torch.nn.BatchNorm1d(84*self.ratio_infl),
                        torch.nn.Hardtanh(),

                        BinarizeLinear(84*self.ratio_infl,10),
                        torch.nn.BatchNorm1d(10),
                        torch.nn.LogSoftmax()
                        )


    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1, 120)
        x = self.classifier(x)
        return x
