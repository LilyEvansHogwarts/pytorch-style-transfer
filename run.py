import torch
from model import Vgg19
from utils import *

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Vgg19().to(device).eval()
    
    content_image = prepare_img(config['content_img_path'], config['height'], device)
    style_image = prepare_img(config['style_img_path'], config['height'], device)

    content_targets, _ = model(content_image)
    _, style_targets = model(style_image)
    style_targets = [GramMatrix(s) for s in style_targets]

    optimizing_img = torch.autograd.Variable(content_image, requires_grad=True)

    # optimizer = torch.optim.LBFGS((optimizing_img, ), max_iter=1000, line_search_fn='strong_wolfe')
    optimizer = torch.optim.Adam((optimizing_img, ), lr=1e1)
    
    tuning_step = make_tuning_step(content_targets, style_targets, model, optimizer, config['content_weight'], config['style_weight'], config['tv_weight'])

    for cnt in range(config['iteration']):
        total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
        print(f'Epochs: {cnt} | total_loss: {total_loss.item():12.4f} | content_loss: {content_loss.item() * config["content_weight"]:12.4f} | style_loss: {style_loss.item() * config["style_weight"]:12.4f}| tv_loss: {tv_loss.item() * config["tv_weight"]:12.4f}')
        save_and_maybe_display(cnt, optimizing_img, config['output_img_path'], config['saving_freq'], config['iteration'], config['img_format'], should_display=True)

    



config = {}
config['content_img_path'] = 'data/content_images/figures.jpg'
config['style_img_path'] = 'data/style_images/candy.jpg'
config['height'] = 500

config['content_weight'] = 1e5
config['style_weight'] = 3e4
config['tv_weight'] = 1e0

config['iteration'] = 2000
config['output_img_path'] = 'data/output_images'
config['saving_freq'] = 100
config['img_format'] = (4, '.jpg')

train(config)
