import pytorch_nndct
import torch
import torchvision
import torch.nn as nn
import argparse
import os
from torchvision import datasets, models, transforms
import random
from tqdm import tqdm
parser = argparse.ArgumentParser()

parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')

args, _ = parser.parse_known_args()

def load_data(train=True,
              data_dir='data/CIFAR10',
              batch_size=16,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='efficientnetv2',
              **kwargs):
    valdir = os.path.join(data_dir, 'cifar10-python')
    print(valdir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if model_name =='inceptionv3':
        size = 299
        resize = 299
    else:
        size = 224
        resize = 224
    dataset = torchvision.datasets.CIFAR10(root=valdir, 
                    train=False, 
                    download=False,
                    transform = transforms.Compose([
                         transforms.Resize(224),
                          transforms.ToTensor(),
                         normalize]
                            ))
    if subset_len:
        assert subset_len <= len(dataset)
        if sample_method == 'random':
            dataset = torch.utils.data.Subset(
                dataset, random.sample(range(0, len(dataset)), subset_len))
        else:
            dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return data_loader

def evaluate_model(model, criterion = nn.CrossEntropyLoss() , dataloader=load_data() ):
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, leave=False):
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    eval_acc = running_corrects.double() / len(dataloader.dataset)

    print(f' Acc: {eval_acc:.4f}')

    return eval_acc


testloader = load_data(subset_len=args.subset_len)
model = torch.jit.load('/efficientnet/quantize_result/EfficientNet_int.pt')
acc= evaluate_model(model, dataloader=testloader)