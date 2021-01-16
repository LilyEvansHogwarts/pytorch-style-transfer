import torch
import torchvision
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def GramMatrix(x, should_normalize=True):
    batch_size, ch, h, w = x.size()
    features = x.view(batch_size, ch, h*w)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram 

# content_loss function
def calculate_content_loss(content_targets, content_features):
    content_loss = 0.0
    for t, f in zip(content_targets, content_features):
        content_loss += torch.nn.MSELoss(reduction='mean')(t, f)
    return content_loss

# style_loss function
def calculate_style_loss(style_targets, style_features):
    style_loss = 0.0
    for t, f in zip(style_targets, style_features):
        style_loss += torch.nn.MSELoss(reduction='sum')(t, GramMatrix(f))
    return style_loss

# tv_loss function
def calculate_tv_loss(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


def build_loss(content_targets, style_targets, model, optimizing_img, content_weight, style_weight, tv_weight):
    content_features, style_features = model(optimizing_img)
    content_loss = calculate_content_loss(content_targets, content_features)
    style_loss = calculate_style_loss(style_targets, style_features)
    tv_loss = calculate_tv_loss(optimizing_img)
    total_loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss
    return total_loss, content_loss, style_loss, tv_loss

def make_tuning_step(content_targets, style_targets, model, optimizer, content_weight, style_weight, tv_weight):
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(content_targets, style_targets, model, optimizing_img, content_weight, style_weight, tv_weight)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return total_loss, content_loss, style_loss, tv_loss
    return tuning_step

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'{img_path} does not exist.')

    img = cv.imread(img_path)[:, :, ::-1] # convert BGR into RGB
    
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * new_height / current_height)
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32)
    img /= 255.0
    return img

def prepare_img(img_path, target_shape, device):
    img = load_image(img_path, target_shape=target_shape)

    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Lambda(lambda x: x.mul(255)),
                                                torchvision.transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL),
    ])

    img = transform(img).to(device).unsqueeze(0)
    return img

def prepare_save_image(optimizing_img):
    output_img = optimizing_img.squeeze(axis=0).detach().cpu().numpy()
    output_img = np.moveaxis(output_img, 0, 2) # convert (ch, h, w) to (h, w, ch), convert RGB to BGR
    output_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    output_img = np.clip(output_img, 0, 255).astype('uint8') # image.mode is BGR
    return output_img
    
def save_and_maybe_display(img_id, optimizing_img, img_path, saving_freq, num_of_iterations, img_format, should_display=False):
    if not should_display and img_id < num_of_iterations-1 and img_id % saving_freq:
        return

    output_img = optimizing_img.squeeze(axis=0).detach().cpu().numpy()
    output_img = np.moveaxis(output_img, 0, 2) # convert (ch, h, w) to (h, w, ch)
    output_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    output_img = np.clip(output_img, 0, 255).astype('uint8') # RGB

    if img_id == num_of_iterations-1 or img_id % saving_freq == 0:
        file_name = os.path.join(img_path, str(img_id).zfill(img_format[0]) + img_format[1])
        cv.imwrite(file_name, output_img[:, :, ::-1]) # convert RGB to BGR, and save the image
        if should_display:
            plt.imshow(output_img)
            plt.show()

