import torch



## Binarized Modules

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)



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




## Defining the Model


class MLP_Binarized(torch.nn.Module):
    def __init__(self):
        super(MLP_Binarized, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 520*self.infl_ratio)
        self.ht1 = torch.nn.Hardtanh()
        self.bn1 = torch.nn.BatchNorm1d(520*self.infl_ratio)
        self.fc2 = BinarizeLinear(520*self.infl_ratio, 320*self.infl_ratio)
        self.ht2 = torch.nn.Hardtanh()
        self.bn2 = torch.nn.BatchNorm1d(320*self.infl_ratio)
        self.fc3 = BinarizeLinear(320*self.infl_ratio, 240*self.infl_ratio)
        self.ht3 = torch.nn.Hardtanh()
        self.bn3 = torch.nn.BatchNorm1d(240*self.infl_ratio)
        self.fc4 = BinarizeLinear(240*self.infl_ratio, 120*self.infl_ratio)
        self.ht4 = torch.nn.Hardtanh()
        self.bn4 = torch.nn.BatchNorm1d(120*self.infl_ratio)
        self.fc5 = torch.nn.Linear(120*self.infl_ratio,10)
        self.logsoftmax=torch.nn.LogSoftmax()
        self.drop=torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.ht1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.ht2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.ht3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.ht4(x)
        x = self.fc5(x)
        return self.logsoftmax(x)
