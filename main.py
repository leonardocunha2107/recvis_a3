import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import torch
import numpy as np
import json
from IPython import display
import matplotlib.pyplot as plt
SEP='-'*40

def train(model,logger,train_loader,optimizer,use_cuda):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        logger.update({'train_loss':loss})

        with torch.no_grad():
            pred = output.data.max(1, keepdim=True)[1]
            logger.update({'train_acc': pred.eq(target.data.view_as(pred)).cpu().sum()/len(data)})


def validation(model,logger,val_loader,use_cuda):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            logger.update({'val_loss':validation_loss})

        validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    logger.update({'val_acc':100. * correct / len(val_loader.dataset)})
    
if __name__=='__main__':
        # Training settings

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Data initialization and loading
    from data import data_transforms

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                            transform=data_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                            transform=data_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    from model import Net
    model = Net()
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    logger=Jimmy('Charles',['loss','acc'],'logs')


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        validation(model,logger,val_loader)
        logger.close_epoch()
        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        #torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
    logger.close()