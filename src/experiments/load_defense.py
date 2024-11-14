from argparse import Namespace

import torch
import yaml

from src.attacks.untargeted import DeepFool, CW, AutoAttack
from src.defenses.competitors.a_vae.model import StyledGenerator
from src.defenses.competitors.a_vae.purification_model import AVaeDefenseModel
from src.defenses.competitors.nd_vae.purification_model import NDVaeDefenseModel
from src.defenses.competitors.nd_vae.modules.models.NVAE import Defence_NVAE
from src.defenses.ours.models import CelebaGenderClassifier, CelebaIdentityClassifier, E4EStyleGanDefenseModel, \
    NVAEDefenseModel, CarsTypeClassifier, TransStyleGanDefenseModel
from src.defenses.ablations.models import GaussianNoiseDefenseModel, GaussianBlurDefenseModel
from src.defenses.wrappers import EoTWrapper

# TODO we should increase C in C&W attack to achieve perturbations with an higher value
# TODO remove args.bounds_l2
def load(args):
    """
    :return updated args, defense model
    """

    # load params from yaml
    with open(args.config, 'r', encoding='utf-8') as stream:
        d_params = Namespace(**yaml.safe_load(stream))

    # load stuff depending on dataset
    if args.experiment == 'gender':

        # attack parameters
        args.image_size = 256
        args.bounds_l2 = (0.5, 1.0, 2.0, 4.0)

        args.attacks = {
            'deepfool': DeepFool(num_classes=2, overshoot=0.02, max_iter=128),
            # 'c&w': CW(c=16., kappa=0.05, steps=8192, lr=1e-3),  # SUBMISSION TIME!
            'c&w': CW(c=16., kappa=0.02, steps=512, lr=1e-3),  # CR TIME !
            'autoattack': AutoAttack()  # TODO update hps
        }

        base_classifier = CelebaGenderClassifier(d_params.classifier_path, args.device)

        # only used for "ours"
        hl_instance = E4EStyleGanDefenseModel

    elif args.experiment == 'ids':

        # attack parameters
        args.image_size = 64
        args.bounds_l2 = (0.1, 0.5, 1.0, 2.0)

        args.attacks = {
            'deepfool': DeepFool(num_classes=8, overshoot=0.02, max_iter=128),
            # 'c&w': CW(c=8., kappa=0.05, steps=8192, lr=1e-3),  # SUBMISSION TIME!
            'c&w': CW(c=8., kappa=0.02, steps=512, lr=1e-3),  # CR TIME !
            'autoattack': AutoAttack()  # TODO update hps
        }

        base_classifier = CelebaIdentityClassifier(d_params.classifier_path, args.device)

        # only used for "ours"
        hl_instance = NVAEDefenseModel

    elif args.experiment == 'cars':

        # attack parameters
        args.image_size = 128
        args.bounds_l2 = (0.5, 1.0, 2.0, 4.0)

        args.attacks = {
            'deepfool': DeepFool(num_classes=4, overshoot=0.02, max_iter=128),
            # 'c&w': CW(c=8., kappa=0.02, steps=8192, lr=1e-3),  # SUBMISSION TIME!
            'c&w': CW(c=12., kappa=0.02, steps=512, lr=1e-3),  # CR TIME !
            'autoattack': AutoAttack()  # TODO update hps
        }

        base_classifier = CarsTypeClassifier(d_params.classifier_path, args.device)

        # only used for "ours"
        hl_instance = TransStyleGanDefenseModel
    else:
        raise NotImplementedError

    # build defense model itself
    if args.defense_type == 'base':

        # no purification
        defense_model = base_classifier
        defense_model.get_purified = lambda x: x

    elif args.defense_type == 'ablation':

        # apply only some gaussian noise or blur
        if d_params.type == 'noise':
            defense_model = GaussianNoiseDefenseModel(base_classifier, args.bounds_l2[-1])
        else:
            defense_model = GaussianBlurDefenseModel(base_classifier)

        defense_model = EoTWrapper(defense_model, args.eot_steps)
        defense_model.get_purified = lambda x: defense_model.model.purify(x)

    elif args.defense_type == 'A-VAE':

        a_vae = StyledGenerator(args.image_size)
        state_dict = torch.load(d_params.autoencoder_path, map_location='cpu')
        a_vae.load_state_dict(state_dict)
        a_vae.to(args.device).eval()

        defense_model = AVaeDefenseModel(base_classifier, a_vae, d_params.kernel_size)

        defense_model = EoTWrapper(defense_model, args.eot_steps)
        defense_model.get_purified = lambda x: defense_model.model.purify(x)

    elif args.defense_type == 'ND-VAE':

        nd_vae = Defence_NVAE(d_params.x_channels,
                              d_params.encoding_channels,
                              d_params.pre_proc_groups,
                              d_params.scales,
                              d_params.groups,
                              d_params.cells,
                              args.image_size)

        state_dict = torch.load(d_params.autoencoder_path, map_location='cpu')
        nd_vae.load_state_dict(state_dict)
        nd_vae.to(args.device).eval()

        defense_model = NDVaeDefenseModel(base_classifier, nd_vae, d_params.noise_std)

        defense_model = EoTWrapper(defense_model, args.eot_steps)
        defense_model.get_purified = lambda x: defense_model.model.purify(x)

    elif args.defense_type == 'trades':

        # no purification, simply load robust classifier
        defense_model = base_classifier
        defense_model.get_purified = lambda x: x

    elif args.defense_type == 'ours':

        defense_model = hl_instance(base_classifier,
                                    d_params.autoencoder_path,
                                    d_params.interpolation_alphas,
                                    d_params.alpha_attenuation,
                                    d_params.initial_noise_eps,
                                    d_params.gaussian_blur_input,
                                    args.device)
        defense_model = EoTWrapper(defense_model, args.eot_steps)
        defense_model.get_purified = lambda x: defense_model.model(x, preds_only=False)[-1]
    else:
        raise NotImplementedError

    return args, defense_model
