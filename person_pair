import torch
import torch.nn as nn
import numpy as np
from einops import repeat, rearrange
from timm import create_model
from utils.transformer import Transformer

ViT_imagenet = create_model('vit_large_patch16_224', pretrained=False, num_classes=1024)
ViT_imagenet_1 = create_model('vit_large_patch16_224', pretrained=False, num_classes=1024)
ViT_imagenet_2 = create_model('vit_large_patch16_224', pretrained=False, num_classes=1024)
print("---success load pretrain ViT---")
ViT_dict = ViT_imagenet.state_dict()
pretrained_model = torch.load(r'/xxx/model/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet.load_state_dict(ViT_dict)

ViT_dict = ViT_imagenet_1.state_dict()
pretrained_model = torch.load(r'/xxx/model/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet_1.load_state_dict(ViT_dict)

ViT_dict = ViT_imagenet_2.state_dict()
pretrained_model = torch.load(r'/xxx/model/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet_2.load_state_dict(ViT_dict)

SIZE = 512

class person_pair(nn.Module):
    def __init__(self, num_classes=6):
        super(person_pair, self).__init__()

        self.pair = ViT_imagenet_1
        self.person_a = ViT_imagenet
        self.person_b = self.person_a
        self.scene = ViT_imagenet_2
        self.bboxes = nn.Linear(10, SIZE)

        self.fc_pair = nn.Linear(1024, SIZE)
        self.fc_A = nn.Linear(1024, SIZE)
        self.fc_B = nn.Linear(1024, SIZE)
        self.fc_scene = nn.Linear(1024, SIZE)

    def forward(self, pair, person_a, person_b, bbox, full_im):
        personPair = self.pair(pair)
        person_1 = self.person_a(person_a)
        person_2 = self.person_b(person_b)
        box = self.bboxes(bbox)
        scene = self.scene(full_im)

        personPair = self.fc_pair(personPair)
        person_1 = self.fc_A(person_1)
        person_2 = self.fc_B(person_2)
        scene = self.fc_scene(scene)

        return personPair, person_1, person_2, box, scene
