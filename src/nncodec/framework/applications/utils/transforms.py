'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

"""
Defines data and model transforms.
"""

import torch
import copy
import torch.nn as nn
from torch.functional import F
import cv2 as cv
import numpy as np
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

MDL_TRAFOS = [
    "model_transform_ImageNet_to_CIFAR100",
    "LSA"
]

DATA_TRAFOS = [
    "transforms_tef_model_zoo",
    "transforms_pyt_model_zoo"
]

transforms_pyt_model_zoo = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def transforms_tef_model_zoo(filename, label, image_size=224):

    img = cv.imread(filename.numpy().decode()).astype(np.float32)

    resize = 256
    if image_size > 224:
        resize = image_size

    # Resize
    height, width, _ = img.shape
    new_height = height * resize // min(img.shape[:2])
    new_width = width * resize // min(img.shape[:2])
    img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)

    # Crop
    height, width, _ = img.shape
    startx = width // 2 - (image_size // 2)
    starty = height // 2 - (image_size // 2)
    img = img[starty:starty + image_size, startx:startx + image_size]
    assert img.shape[0] == image_size and img.shape[1] == image_size, (img.shape, height, width)

    # BGR to RGB
    img = img[:, :, ::-1]

    return img, label

def model_transform_ImageNet_to_CIFAR100(original_model):
    if original_model.__class__.__name__ == 'ResNet':
        if hasattr(original_model, "fc"):
            classifier_in = original_model.fc.weight.shape[1]
            original_model.fc = nn.Linear(classifier_in, 100)
    elif 'MobileNet' in original_model.__class__.__name__ or\
            'EfficientNet' in original_model.__class__.__name__:
        idx = 1 if not 'V3' in original_model.__class__.__name__ else 3
        if hasattr(original_model, "classifier"):
            classifier_in = original_model.classifier[idx].weight.shape[1]
            original_model.classifier[idx] = nn.Linear(classifier_in, 100)
    return original_model

class ScaledConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.weight_scaling = nn.Parameter(torch.ones_like(torch.Tensor(out_channels, 1, 1, 1)))
        # self.reset_parameters()

    def reset_parameters(self):
        # The if condition is added so that the super call in init does not reset_parameters as well.
        if hasattr(self, 'weight_scaling'):
            nn.init.normal_(self.weight_scaling, 1, 0)#1e-5)
            super().reset_parameters()

    def forward(self, input):
        major, minor, _ = map(int, torch.__version__.split("."))
        if major > 1 or (major == 1 and minor > 7):
            return self._conv_forward(input, self.weight_scaling * self.weight, self.bias)
        else:
            return self._conv_forward(input, self.weight_scaling * self.weight)

class ScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.weight_scaling = nn.Parameter(torch.ones_like(torch.Tensor(out_features, 1)))
        # self.reset_parameters()

    def reset_parameters(self):
        # The if condition is added so that the super call in init does not reset_parameters as well.
            if hasattr(self, 'weight_scaling'):
                nn.init.normal_(self.weight_scaling, 1, 0)#1e-5)
                super().reset_parameters()

    def forward(self, input):
        return F.linear(input, self.weight_scaling * self.weight, self.bias)

class LSA:
    def __init__(self, original_model):
        self.mdl = copy.deepcopy(original_model)

    def update_conv2d(self, m, parent):
        lsa_update = ScaledConv2d(m[1].in_channels, m[1].out_channels, m[1].kernel_size, m[1].stride,
                                  m[1].padding, m[1].dilation, m[1].groups, None, m[1].padding_mode)
        lsa_update.weight, lsa_update.bias = m[1].weight, m[1].bias
        setattr(parent, m[0], lsa_update)

    def update_linear(self, m, parent):
        lsa_update = ScaledLinear(m[1].in_features, m[1].out_features)
        lsa_update.weight, lsa_update.bias = m[1].weight, m[1].bias
        setattr(parent, m[0], lsa_update)

    def add_lsa_params_recursive(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.weight.requires_grad:
                self.update_conv2d((name, child), module)
            elif isinstance(child, nn.Linear) and child.weight.requires_grad:
                self.update_linear((name, child), module)
            elif len(list(child.children())) > 0:
                self.add_lsa_params_recursive(child)

    def add_lsa_params(self):
        self.add_lsa_params_recursive(self.mdl)
        return self.mdl


def split_datasets(testset, trainset, num_partitions: int, batch_size: int, num_workers: int, val_ratio: float = 0.1):

    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2 ** 32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #
    # g = torch.Generator()
    # g.manual_seed(0)

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(909)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,)
                       # worker_init_fn=seed_worker)#, generator=g)
        )
        if val_ratio == 0:
            valloaders.append(DataLoader(testset, batch_size=batch_size))
        else:
            valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,)
                       # worker_init_fn=seed_worker)#, generator=g)
            )

    # create dataloader for the test set
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader
