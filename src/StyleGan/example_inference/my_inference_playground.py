from argparse import Namespace

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from src.StyleGan.models.hyperstyle import HyperStyle


def run_inversion(inputs, net, opts):

    y_hat, latent, weights_deltas, codes = None, None, None, None

    for iter in range(opts.n_iters_per_batch):

        y_hat, latent, weights_deltas, codes, _ = net.forward(inputs, y_hat=y_hat, codes=codes,
                                                              weights_deltas=weights_deltas, return_latents=True,
                                                              resize=opts.resize_outputs, randomize_noise=False,
                                                              return_weight_deltas_and_codes=True)

        # resize input to 256 before feeding into next iteration
        y_hat = net.face_pool(y_hat)

    return y_hat, latent, weights_deltas


def load_model(checkpoint_path: str, encoder_path: str, device: str = 'cuda'):

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    opts = ckpt['opts']
    opts['checkpoint_path'] = checkpoint_path
    opts['load_w_encoder'] = True
    opts['w_encoder_checkpoint_path'] = encoder_path
    opts = Namespace(**opts)

    net = HyperStyle(opts)
    net.eval()
    net.to(device)

    return net, opts


def run_inference(config: dict):

    # load model
    net, opts = load_model(config['model_path'], config['w_encoder_path'])
    print('Model successfully loaded!')

    img_transforms = config['transform']

    # load batch of 4 images
    original_images = torch.empty((0, 3, 256, 256), device='cuda:0')

    images_path = config["image_path"]
    b = len(images_path)
    for path in images_path:
        img = Image.open(path).convert("RGB").resize((256, 256))
        img = img_transforms(img).unsqueeze(0)
        original_images = torch.cat((original_images, img.to('cuda:0')), dim=0)

    initial_inversion, _ = net.get_initial_inversion(original_images, resize=True)

    # adjust weights for each image
    opts.n_iters_per_batch = 4
    opts.resize_outputs = False  # generate outputs at full resolution

    with torch.no_grad():
        result_batch, _, w_deltas = run_inversion(original_images, net, opts)

    # reconstruction after obtaining weights
    final_inversion, _ = net.get_inversion(original_images, w_deltas, resize=True)

    # display images
    all_images = torch.cat((original_images, initial_inversion, result_batch, final_inversion), dim=0)
    all_images = (all_images + 1) / 2.
    all_images = make_grid(all_images, nrow=b).permute(1, 2, 0).cpu().numpy()

    plt.imshow(all_images)
    plt.axis(False)
    plt.title("ROWS = [original, initial_inv, computed_inv, passed_inv]")
    plt.show()


if __name__ == '__main__':

    afhq_wild_config = {
        "model_path": "/media/dserez/runs/stylegan2/inversions/hyperstyle_inverter_afhq_wild.pt",
        "w_encoder_path": "/media/dserez/runs/stylegan2/inversions/encoder_afhq_wild.pt",
        "image_path": ["./images/cheetah.jpg",
                       "./images/fox.jpg",
                       "./images/lion.jpg",
                       "./images/tiger.jpg"],
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }

    run_inference(afhq_wild_config)