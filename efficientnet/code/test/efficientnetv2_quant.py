# Simple quantization of effficienet architecture

from tqdm import tqdm
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
import os
import re
import sys
import argparse
import time
import pdb
import random
import os.path as osp

# load quant apis
from pytorch_nndct.apis import torch_quantizer

import torch
import torchvision
import torchvision.transforms as transforms
sys.path.append('./code/model/')
# from efficientnet import efficientnet_v2_s


# device = torch.device("cuda")
device = torch.device("cpu")


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="data/CIFAR10/",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="./float/efficientnetv2.pth",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=8,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode',
                    default='calib',
                    choices=['float', 'calib', 'test'],
                    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune',
                    dest='fast_finetune',
                    action='store_true',
                    help='fast finetune model before calibration')
parser.add_argument('--deploy',
                    dest='deploy',
                    action='store_true',
                    help='export xmodel for deployment')

parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')


parser.add_argument(
    '--output_dir',
    default='quantized',
    help='Directory to save qat result.')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--device', default='gpu',
                    choices=['gpu', 'cpu'], help='assign runtime device')
args, _ = parser.parse_known_args()


def load_data(train=True,
              data_dir='data/CIFAR10',
              batch_size=32,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='efficientnetv2',
              **kwargs):

    # prepare data
    # random.seed(12345)
    traindir = osp.join(data_dir, 'cifar10-python')
    valdir = osp.join(data_dir, 'cifar10-python')
    train_sampler = None
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if model_name == 'inception_v3':
        size = 299
        resize = 299
    else:
        size = 224
        resize = 224
    if train:
        dataset = torchvision.datasets.CIFAR10(
            root=traindir,
            train=True,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if subset_len:
            assert subset_len <= len(dataset)
            if sample_method == 'random':
                dataset = torch.utils.data.Subset(
                    dataset, random.sample(range(0, len(dataset)), subset_len))
            else:
                dataset = torch.utils.data.Subset(
                    dataset, list(range(subset_len)))
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **kwargs)
    else:
        dataset = torchvision.datasets.CIFAR10(root=valdir,
                                               train=False,
                                               download=False,
                                               transform=transforms.Compose([
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
                dataset = torch.utils.data.Subset(
                    dataset, list(range(subset_len)))
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return data_loader, train_sampler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
      for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, val_loader, loss_fn, device):

    model.eval()
    model = model.to(device)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    Loss = 0
    for iteraction, (images, labels) in tqdm(
            enumerate(val_loader), total=len(val_loader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        Loss += loss.item()
        total += images.size(0)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        if (iteraction != 0 and iteraction % 1000 == 0):
            print('image_size=%d,\t top1=%.1f,\t top5=%.1f' %
                  (images.size(2), top1.avg, top5.avg))
    return top1.avg, top5.avg, Loss/total


def quantization(title='optimize',
                 model_name='', file_path=''):
    data_dir = args.data_dir
    quant_mode = args.quant_mode
    #  finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    config_file = args.config_file

    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(
            r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    if deploy:
        args.device = 'cpu'

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if file_path:
        model = torch.load(file_path, map_location = device)
        print("=== Load pretrained model ===")

    input = torch.randn([batch_size, 3, 224, 224]).to(device)
    if quant_mode == 'float':
        quant_model = model
    else:
        quantizer = torch_quantizer(
            quant_mode, model, (input), device=device, quant_config_file=config_file, output_dir=args.output_dir)

        quant_model = quantizer.quant_model
    # to get loss value after evaluation
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    val_loader, _ = load_data(
        subset_len=subset_len,
        train=False,
        batch_size=batch_size,
        sample_method='random',
        data_dir=data_dir,
        model_name=model_name)

    acc1_gen, acc5_gen, loss_gen = evaluate(
        quant_model, val_loader, loss_fn, device)

    # logging accuracy
    print('loss: %g' % (loss_gen))
    print('top-1 / top-5 accuracy: %.1f / %.1f' % (acc1_gen, acc5_gen))

    # handle quantization result
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_xmodel(deploy_check=True)
        quantizer.export_torch_script()
        quantizer.export_onnx_model(dynamic_batch=True)


if __name__ == '__main__':

    model_name = 'efficientnetv2'
    file_path = args.model_dir

    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = ' quantization'
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += ' with optimization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(
        title=title,
        model_name=model_name,
        file_path=file_path)

    print("-------- End of {} test ".format(model_name))
