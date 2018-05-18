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

class CNN_Binarized(torch.nn.Module):

        def __init__(self):
            super(CNN_Binarized,self).__init__()
            self.ratio_infl=3

            self.features=torch.nn.Sequential(
                            BinarizeConv2d(1,10*self.ratio_infl,kernel_size=5),
                            torch.nn.MaxPool2d(kernel_size=2),
                            torch.nn.BatchNorm2d(10*self.ratio_infl),
                            torch.nn.Hardtanh(inplace=True),

                            BinarizeConv2d(10*self.ratio_infl,20, kernel_size=5),
                            torch.nn.MaxPool2d(kernel_size=2),
                            torch.nn.BatchNorm2d(20),
                            torch.nn.Hardtanh(inplace=True)
                            )

            self.classifier=torch.nn.Sequential(
                            BinarizeLinear(320,50),
                            torch.nn.BatchNorm1d(50),
                            torch.nn.Hardtanh(),

                            BinarizeLinear(50,10),
                            torch.nn.BatchNorm1d(10),
                            torch.nn.LogSoftmax()
                            )

        def forward(self,x):
#            in_size=x.size(0)
            x=self.features(x)
            x=x.view(-1,320)
            x=self.classifier(x)
            return x
