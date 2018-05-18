import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools
import pickle
import model
## Load Data

batch_size=128

train_dataset=datasets.MNIST(root='./../mnist/',train=True, download=True, transform=transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()]))
test_dataset=datasets.MNIST(root='./../mnist', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()]))

train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


model=model.LeNet_Binarized()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)


def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output=model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            train_loss=loss.data[0]
            epoch_dec=batch_idx/len(train_loader)
            final_epoch=(epoch-1+(int(epoch_dec*100)/100))

            import numpy
            with open('./train_loss_lenet_binarized.txt', 'a') as lossfile, open('./epochs_lenet_binarized.txt', 'a') as epochfile:
                lossfile.write(str(train_loss.numpy())+"\n")
                epochfile.write(str(final_epoch)+"\n")



def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#    target_names=['Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', 'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9']
#    cnf_matrix=confusion_matrix(target, pred)
#    print(cnf_matrix)
#    report=classification_report(target,pred)
#    print(report)
#    plt.figure()
#    plot_confusion_matrix(cnf_matrix, classes=target_names,title='Confusion matrix')
#    plt.show()
    accuracy=100*(float(correct)/len(test_loader.dataset))
    return test_loss, accuracy#, cnf_matrix, report





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



## Train and Test

for epoch in range(1, 501):
    train(epoch)
    test_loss, test_accuracy=test()
    with open('./test_bin_lenet.txt','a') as testfile:
        testfile.write("Epoch:"+str(epoch)+"\n")
        testfile.write("Test Loss:"+str(test_loss.numpy())+"\n")
        testfile.write("Test Accuracy:"+str(test_accuracy)+"\n")
print("Finished Training!!!\n")

## Evaluations
#test_loss,test_accuracy, cnf_matrix, report =test()
#test_loss, test_accuracy=test()
#with open('./test_bin_lenet.txt','a') as testfile:
#    testfile.write("Test Loss:"+str(test_loss.numpy())+"\n")
#    testfile.write("Test Accuracy:"+str(test_accuracy)+"\n")
 #   testfile.write("Confusion Matrix:"+str(cnf_matrix)+"\n")
#    testfile.write("Classification Report:"+str(report)+"\n")
#testfile.close()
#print('Succesfully wrote the evaluation results to file!!!')





## Save the model

filename = './binarized_lenet.pickle'
pickle.dump(model, open(filename, 'wb'))

##Loading Model
#try:
#    loaded_model = pickle.load(open(filename,'rb'))
#    print('Model Loaded Succesfully!!\n')
#except:
#    print("ERROR Loading!!\n")
