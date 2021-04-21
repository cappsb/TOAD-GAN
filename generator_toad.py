import os
from shutil import copyfile
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))  # uncomment if opening form other dir

from config import get_arguments, post_config
from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mariokart.special_mariokart_downsampling import special_mariokart_downsampling
from mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mariokart.tokens import REPLACE_TOKENS as MARIOKART_REPLACE_TOKENS
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from mario.special_mario_downsampling import special_mario_downsampling


from megaman.level_image_gen import LevelImageGen as MegaManLevelGen
from megaman.tokens import REPLACE_TOKENS as MEGAMAN_REPLACE_TOKENS
from megaman.tokens import TOKEN_GROUPS as MEGAMAN_TOKEN_GROUPS
from megaman.special_megaman_downsampling import special_megaman_downsampling

from generate_noise import generate_spatial_noise
from models import load_trained_pyramid
from loderunner.special_loderunner_downsampling import special_loderunner_downsampling
from loderunner.level_image_gen import LevelImageGen as LoderunnerLevelGen
from loderunner.tokens import REPLACE_TOKENS as LODERUNNER_REPLACE_TOKENS
from loderunner.tokens import TOKEN_GROUPS as LODERUNNER_TOKEN_GROUPS


def generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=None, scale_v=1.0, scale_h=1.0,
                     current_scale=0, gen_start_scale=0, num_samples=50, render_images=True, save_tensors=False,
                     save_dir="random_samples"):
    """
    Generate samples given a pretrained TOAD-GAN (generators, noise_maps, reals, noise_amplitudes).
    Uses namespace "opt" that needs to be parsed.
    "in_s" can be used as a starting image in any scale set with "current_scale".
    "gen_start_scale" sets the scale generation is to be started in.
    "num_samples" is the number of different samples to be generated.
    "render_images" defines if images are to be rendered (takes space and time if many samples are generated).
    "save_tensors" defines if tensors are to be saved (can be needed for token insertion *experimental*).
    "save_dir" is the path the samples are saved in.
    """

    # Holds images generated in current scale
    images_cur = []

    # Check which game we are using for token groups
    if opt.game == 'mario':
        token_groups = MARIO_TOKEN_GROUPS
    elif opt.game == 'mariokart':
        token_groups = MARIOKART_TOKEN_GROUPS
    elif opt.game == 'megaman':
        token_groups = MEGAMAN_TOKEN_GROUPS
    elif opt.game == 'loderunner':
        token_groups = LODERUNNER_TOKEN_GROUPS
    else:
        token_groups = []
        NameError("name of --game not recognized. Supported: mario, mariokart, megaman")
    
    # Main sampling loop
    for G, Z_opt, noise_amp in zip(generators, noise_maps, noise_amplitudes):
        
        if current_scale >= len(generators):
            break  # if we do not start at current_scale=0 we need this

        logger.info("Generating samples at scale {}", current_scale)

        # Padding (should be chosen according to what was trained with)
        n_pad = int(1*opt.num_layer)
        if not opt.pad_with_noise:
            m = nn.ZeroPad2d(int(n_pad))  # pad with zeros
        else:
            m = nn.ReflectionPad2d(int(n_pad))  # pad with reflected noise

        # Calculate shapes to generate
        if 0 < gen_start_scale <= current_scale:  # Special case! Can have a wildly different shape through in_s
            scale_v = in_s.shape[-2] / (noise_maps[gen_start_scale-1].shape[-2] - n_pad * 2)
            scale_h = in_s.shape[-1] / (noise_maps[gen_start_scale-1].shape[-1] - n_pad * 2)
            nzx = (Z_opt.shape[-2] - n_pad * 2) * scale_v
            nzy = (Z_opt.shape[-1] - n_pad * 2) * scale_h
        else:
            nzx = (Z_opt.shape[-2] - n_pad * 2) * scale_v
            nzy = (Z_opt.shape[-1] - n_pad * 2) * scale_h

        # Save list of images of previous scale and clear current images
        images_prev = images_cur
        images_cur = []

        # Token insertion (Experimental feature! Generator needs to be trained with it)
        if current_scale < (opt.token_insert + 1):
            channels = len(token_groups)
            if in_s is not None and in_s.shape[1] != channels:
                old_in_s = in_s
                in_s = token_to_group(in_s, opt.token_list, token_groups)
        else:
            channels = len(opt.token_list)
            if in_s is not None and in_s.shape[1] != channels:
                old_in_s = in_s
                in_s = group_to_token(in_s, opt.token_list, token_groups)

        # If in_s is none or filled with zeros reshape to correct size with channels
        if in_s is None:
            in_s = torch.zeros(reals[0].shape[0], channels, *reals[0].shape[2:]).to(opt.device)
        elif in_s.sum() == 0:
            in_s = torch.zeros(1, channels, *in_s.shape[-2:]).to(opt.device)
        print(channels)
        # Generate num_samples samples in current scale
        # for n in tqdm(range(0, 1)):


        # Get noise image
        # z_curr = generate_spatial_noise([1, channels, int(round(nzx)), int(round(nzy))], device=opt.device) #replace with z_curr = latent_vector
        z_curr = torch.randn([1, channels, int(round(nzx)), int(round(nzy))], device=opt.device)
        # z_curr = torch.zeros([1, channels, int(round(nzx)), int(round(nzy))], device=opt.device)
        # print(len(z_curr), len(z_curr[0]), len(z_curr[0][0]), len(z_curr[0][0][0]))
        # print(z_curr[0][0][0])
        # quit()
        z_curr = m(z_curr)

        # Set up previous image I_prev
        if (not images_prev) or current_scale == 0:  # if there is no "previous" image
            I_prev = in_s
        else:
            I_prev = images_prev[0]

            # Transform to token groups if there is token insertion
            if current_scale == (opt.token_insert + 1):
                I_prev = group_to_token(I_prev, opt.token_list, token_groups)

        I_prev = interpolate(I_prev, [int(round(nzx)), int(round(nzy))], mode='bilinear', align_corners=False)
        I_prev = m(I_prev)

        # We take the optimized noise map Z_opt as an input if we start generating on later scales
        if current_scale < gen_start_scale:
            z_curr = Z_opt

        # Define correct token list (dependent on token insertion)
        if opt.token_insert >= 0 and z_curr.shape[1] == len(token_groups):
            token_list = [list(group.keys())[0] for group in token_groups]
        else:
            token_list = opt.token_list

        ###########
        # Generate!
        ###########
        z_in = noise_amp * z_curr + I_prev
        I_curr = G(z_in.detach(), I_prev, temperature=1)

        # Save all scales
        # if True:
        # Save scale 0 and last scale
        # if current_scale == 0 or current_scale == len(reals) - 1:
        # Save only last scale
        if current_scale == len(reals) - 1:
            dir2save = opt.out_ + '/' + save_dir

            # Make directories
            try:
                os.makedirs(dir2save, exist_ok=True)
                if render_images:
                    os.makedirs("%s/img" % dir2save, exist_ok=True)
                if save_tensors:
                    os.makedirs("%s/torch" % dir2save, exist_ok=True)
                os.makedirs("%s/txt" % dir2save, exist_ok=True)
            except OSError:
                pass

            # Convert to ascii level
            level = one_hot_to_ascii_level(I_curr.detach(), token_list)

            # Render and save level image
            if render_images:
                img = opt.ImgGen.render(level)
                img.save("%s/img/%d_sc%d.png" % (dir2save, 0, current_scale))

            # Save level txt
            with open("%s/txt/%d_sc%d.txt" % (dir2save, 0, current_scale), "w") as f:
                f.writelines(level)

            # Save torch tensor
            if save_tensors:
                torch.save(I_curr, "%s/torch/%d_sc%d.pt" % (dir2save, 0, current_scale))

        # Append current image
        images_cur.append(I_curr)

        # Go to next scale
        current_scale += 1

    return I_curr.detach()  # return last generated image (usually unused)


if __name__ == '__main__':
    # NOTICE: The "output" dir is where the generator is located as with main.py, even though it is the "input" here

    # Parse arguments
    parse = get_arguments()
    parse.add_argument("--out_", help="folder containing generator files")
    parse.add_argument("--scale_v", type=float, help="vertical scale factor", default=1.0)
    parse.add_argument("--scale_h", type=float, help="horizontal scale factor", default=1.0)
    parse.add_argument("--gen_start_scale", type=int, help="scale to start generating in", default=0)
    parse.add_argument("--num_samples", type=int, help="number of samples to be generated", default=1)
    parse.add_argument("--make_mario_samples", action="store_true", help="make 1000 samples for each mario generator"
                                                                         "specified in the code.", default=False)
    parse.add_argument("--seed_mariokart_road", action="store_true", help="seed mariokart generators with a road image",
                       default=False)
    parse.add_argument("--token_insert_experiment", action="store_true", help="make token insert experiment "
                                                                              "(experimental!)", default=False)
    opt = parse.parse_args()

    if (not opt.out_) and (not opt.make_mario_samples):
        parse.error('--out_ is required (--make_mario_samples experiment is the exception)')

    opt = post_config(opt)

    
    # Code to make samples for given generator
    token_insertion = True if opt.token_insert and opt.token_insert_experiment else False

    # Init game specific inputs
    replace_tokens = {}
    sprite_path = opt.game + '/sprites'
    if opt.game == 'mario':
        opt.ImgGen = MarioLevelGen(sprite_path)
        replace_tokens = MARIO_REPLACE_TOKENS
        downsample = special_mario_downsampling

    elif opt.game == 'mariokart':
        opt.ImgGen = MariokartLevelGen(sprite_path)
        replace_tokens = MARIOKART_REPLACE_TOKENS
        downsample = special_mariokart_downsampling


    elif opt.game == 'loderunner':
        opt.ImgGen = LoderunnerLevelGen(sprite_path)
        replace_tokens = LODERUNNER_REPLACE_TOKENS
        downsample = special_loderunner_downsampling


    elif opt.game == 'megaman':
        opt.ImgGen = MegaManLevelGen(sprite_path)
        replace_tokens = MEGAMAN_REPLACE_TOKENS
        downsample = special_megaman_downsampling

    else:
        NameError("name of --game not recognized. Supported: mario, mariokart, megaman")

    # Load level
    real = read_level(opt, None, replace_tokens).to(opt.device)
    # Load Generator
    generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(opt)


    # Get input shape for in_s
    real_down = downsample(1, [[opt.scale_v, opt.scale_h]], real, opt.token_list)
    real_down = real_down[0]
    in_s = torch.zeros_like(real_down, device=opt.device)
    prefix = "arbitrary"

    # Directory name
    s_dir_name = "%s_random_samples_v%.5f_h%.5f_st%d" % (prefix, opt.scale_v, opt.scale_h, opt.gen_start_scale)

    generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=in_s,
                        scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)


