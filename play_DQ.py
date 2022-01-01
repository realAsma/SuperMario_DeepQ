import yaml
import argparse
import time

import torch
import torch.optim as optim

from game_DQ import MarioGame

from models import DeepQ

from utils import get_best_metric, save_checkpoint, AverageMeter, write_best_metric
import os
import copy
import math

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from math import cos
from math import pi
import random

parser = argparse.ArgumentParser(description='CS7643 deep_pipes')
parser.add_argument('--config', default='./configs/config_DeepQ.yaml')
parser.add_argument('--device', default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


def validate(epoch, env, n_games=1):
    iter_time = AverageMeter()
    rewards = AverageMeter()
    num_steps = AverageMeter()
    env.model.eval()

    for idx in range(n_games):
        start = time.time()
        reward_game, _, steps_game, _ = env.game(epsilon=0.01)
        iter_time.update(time.time() - start)
        rewards.update(reward_game)
        num_steps.update(steps_game)

        print(f'Validate Game : [{idx + 1}]/[{n_games}]\t'
              f'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
              f'reward : {rewards.val:.3f} ({rewards.avg:.3f})\t '
              f'Total steps : {steps_game} ({num_steps.avg:.3f})')
    return rewards.avg, num_steps.avg


def main():
    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    model = DeepQ(state_len=args.image_size, action_len=args.action_len, info_len=args.info_len,
                  model_comp=args.model_comp,
                  device=args.device).to(args.device)
    rewards_max = float('-inf')
    model_max = None
    #model_path = './checkpoints/DeepQ_lr_1e-05_batch_128_downsample_2_skipframe_4_best.pth.tar'
    model_path = './checkpoints/DeepQ_lr_0.0001_batch_128_downsample_2_skipframe_4_0.pth.tar'

    comment = f'lr_{args.lr}_batch_{args.batch_size}' \
              f'_downsample_{args.downsample}_skipframe_{args.skip_frame_cnt}'

    model.load_state_dict(torch.load(model_path, map_location=args.device)['state_dict'])
    env = MarioGame(model=model, optimizer=None, criterion=None, args=args, version=1)
    rewards_val_epoch, steps_val = validate(0, env, n_games=10)
    print("validation done.....")

# import matplotlib.pyplot as plt
# state_t[0].requires_grad,state_t[1].requires_grad = True, True
# loss = criterion(state_t, action_t, reward_t, state_next_t)
# optimizer.zero_grad()
# loss.backward()
# plt.imshow(torch.norm(s[1].grad, dim =0, keepdim=True))
if __name__ == '__main__':
    main()
