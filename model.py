import torch
import torchvision

class Vgg19(torch.nn.Module):
    def __init__(self, content_indices=[21], style_indices=[1, 6, 11, 20, 29], show_progress=False, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True, progress=show_progress).features
        self.content_indices = content_indices
        self.style_indices = style_indices
        self.features = sorted(self.content_indices + self.style_indices)

        start_index = 0
        for i in range(len(self.features)):
            current_slice = torch.nn.Sequential()
            while start_index <= self.features[i]:
                current_slice.add_module(str(start_index), vgg_pretrained_features[start_index])
                start_index += 1

            setattr(self, 'slice'+str(i+1), current_slice)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, inputs):
        result = inputs
        content_features = []
        style_features = []
        for i in range(len(self.features)):
            result = getattr(self, 'slice'+str(i+1))(result)

            if self.features[i] in self.content_indices:
                content_features.append(result)

            if self.features[i] in self.style_indices:
                style_features.append(result)

        return (content_features, style_features)

