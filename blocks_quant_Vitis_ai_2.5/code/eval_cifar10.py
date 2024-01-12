# Evaluates the model in test set of CIFAR10 dataset
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from tqdm import tqdm
import os
import re
import sys
import argparse
import time
import pdb
import random

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

sys.path.append('./code/model/')


# from efficientnet import efficientnet_v2_s

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="data/CIFAR10/",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_name',
    default='efficientnetv2',
    choices=['efficientnetv2', 'inceptionv3', 'resnet50']
)
parser.add_argument(
    '--model_path',
    default="./float/efficientnetv2.pth"
)
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument(
    '--data_type',
    default='float32',
    choices=['float32', 'float16'])
parser.add_argument(
    '--device',
    default='gpu',
    choices=['gpu', 'cpu'])
parser.add_argument('--export_float_onnx_model',
                    action='store_true',
                    help='export float onnx model')
args, _ = parser.parse_known_args()


if args.device == 'cpu':
    device = torch.device("cpu")
    print('Set device to CPU')
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(train=True,
              data_dir='data/CIFAR10',
              batch_size=32,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='efficientnetv2',
              **kwargs):
    valdir = os.path.join(data_dir, 'cifar10-python')
    print(valdir)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if model_name == 'inceptionv3':
        size = 299
        resize = 299
    else:
        size = 224
        resize = 224
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
            dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return data_loader


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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def evaluate(model, val_loader, loss_fn, model_type, data_type='float32'):

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    Loss = 0
    for iteraction, (images, labels) in tqdm(
            enumerate(val_loader), total=len(val_loader)):
        if model_type == 'pth':
            images = images.to(device)
            outputs = model(images)
        else:  # model_type == 'onnx'
            if data_type == 'float16':
                images = images.half()
            outputs = model.run({'input': images.cpu().numpy()})[0]
            outputs = torch.tensor(np.array(outputs))
        outputs = outputs.to(device)
        labels = labels.to(device)
        total += images.size(0)
        acc1, acc5 = accuracy(outputs.to(torch.float32), labels, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        if (iteraction != 0 and iteraction % 1000 == 0):
            print('image_size=%d,\t top1=%.1f,\t top5=%.1f' %
                  (images.size(2), top1.avg, top5.avg))
    return top1.avg, top5.avg, Loss / total


def main(model_name=''):
    data_dir = args.data_dir
    batch_size = args.batch_size
    subset_len = args.subset_len
    print("=== Load pretrained model ===")

    print('Model name:', model_name)
    model_type = args.model_path.split('.')[-1]
    print('Model type:', model_type)
    if model_type not in ['onnx', 'pth']:
        print(
            f'[Error] Evaluating model type \"{model_type} \" is not supported yet')
        return

    print('Loading model:', args.model_path)
    if model_type == 'pth':
        model = torch.load(args.model_path)
        model = model.to(device)
        model.eval()
    else:
        import migraphx
        print('Evaluating onnx model using AMD MIGRAPHX')
        model = migraphx.parse_onnx(args.model_path)
        model.compile(migraphx.get_target("gpu"))

    if args.export_float_onnx_model:
        print('Exporting float onnx model')
        input = torch.randn([1, 3, 299, 299])
        model = model.cpu()
        model.eval()
        model_name = 'efficientnetv2'
        torch.onnx.export(model, input, '{}_fp32.onnx'.format(model_name), input_names=['input'], output_names=[
                          'output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        model = model.cuda().half().eval()
        torch.onnx.export(model, input.cuda().half(), '{}_fp16.onnx'.format(model_name), input_names=[
                          'input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        return

    # to get loss value after evaluation
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    val_loader = load_data(
        subset_len=subset_len,
        train=False,
        batch_size=batch_size,
        sample_method='',
        data_dir=data_dir,
        model_name=model_name)

    acc1_gen, acc5_gen, loss_gen = evaluate(
        model, val_loader, loss_fn, model_type, args.data_type)

    # logging accuracy
    print('top-1 / top-5 accuracy: %.1f / %.1f' % (acc1_gen, acc5_gen))


if __name__ == '__main__':

    model_name = args.model_name
    print("-------- Start {} test ".format(model_name))

    main(model_name=args.model_name)

    print("-------- End of {} test ".format(model_name))
