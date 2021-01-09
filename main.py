import torch
import torchvision
from torch import nn
from torch import optim
from PIL import Image
from torchvision import transforms
from torchvision import models
from torchvision.utils import save_image
import numpy as np
from functools import reduce
import argparse
import os

def create_vgg19(dict_path='vgg19-dcbb9e9d.pth'):
    model = torchvision.models.vgg19(pretrained=False)
    if not os.path.exists(dict_path):
        os.system('wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
    else:
        model.load_state_dict(torch.load(dict_path))
    return model

def load_image(image_name, loader):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.style_features = [0, 5, 10, 19, 28]
        self.content_features = [20]
        self.model = create_vgg19('vgg19-dcbb9e9d.pth').features[:29]

    def forward(self, x):
        style_features = []
        content_features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if layer_num in self.style_features:
                style_features.append(x)

            if layer_num in self.content_features:
                content_features.append(x)

        return style_features, content_features

class TotalLoss(nn.Module):
    def __init__(self, style_targets, content_targets, alpha, beta):
        super(TotalLoss, self).__init__()
        self.style_targets = style_targets
        self.content_targets = content_targets
        self.gram_style_targets = [GramMatrix(f) for f in self.style_targets]

    def forward(self, style_features, content_features):
        content_loss = [nn.functional.mse_loss(c, t) for c, t in zip(content_features, self.content_targets)]
        style_loss = [nn.functional.mse_loss(GramMatrix(s), t) for s, t in zip(style_features, self.gram_style_targets)]
        total_loss = alpha * sum(content_loss) + beta * sum(style_loss)
        return total_loss

def GramMatrix(frame):
    batch_size, channel, height, width = frame.size()
    gram = frame.view(channel, height * width).mm(
        frame.view(channel, height * width).t()
    )
    return gram

def train(style_features, content_features, generated_image, model, learning_rate, total_steps, alpha, beta):
    optimizer = optim.Adam([generated_image], lr=learning_rate)
    criterion = TotalLoss(style_features, content_features, alpha, beta)
    for step in range(total_steps):
        generated_style_features, generated_content_features = model(generated_image)

        style_loss = 0
        content_loss = 0

        total_loss = criterion(generated_style_features, generated_content_features)
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if step % 20 == 0:
            print('Epoch {}, Loss: {}'.format(step, total_loss.detach().cpu().item()))
            save_image(generated_image, 'data/generated_image'+str(step)+'.png')


image_size = 500
loader = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[], std=[]),
            ]
        )
model= VGG().eval()

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--content', default='img/original_image.png', help='content image')
parser.add_argument('-s', '--style', default='img/style_image.png', help='style image')

args = parser.parse_args()
original_image_name = args.content
style_image_name = args.style

original_image = load_image(original_image_name, loader)
style_image = load_image(style_image_name, loader)
generated_image = original_image.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 1e-3
alpha = 1
beta = 0.05

_, original_features = model(original_image)
style_features, _ = model(style_image)

train(style_features, original_features, generated_image, model, learning_rate, total_steps, alpha, beta)

