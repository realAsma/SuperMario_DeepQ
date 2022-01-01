
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


def train(epoch, env, epsilon=0.5, n_games=1):
    iter_time = AverageMeter()
    rewards = AverageMeter()
    num_steps = AverageMeter()
    loss = AverageMeter()
    env.model.train()

    for idx in range(n_games):
        start = time.time()
        reward_game, loss_game, steps_game, total_steps = env.game(epsilon)
        iter_time.update(time.time() - start)
        rewards.update(reward_game)
        num_steps.update(steps_game)
        loss.update(loss_game)

        print(f'Game {epoch}: [{idx + 1}][{n_games}]\tTime {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
              f'Loss : {loss_game:.4f}) ({loss.avg:.4f})\treward : {rewards.val:.3f} ({rewards.avg:.3f})\t '
              f'Eps steps : {steps_game} ({num_steps.avg:.1f})\tTotal steps : {total_steps}')
        writer.add_scalars(f'Train Eps Reward ', {'Train eps reward': reward_game}, total_steps)
    return rewards.avg, loss.avg, num_steps.avg, total_steps


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

def adjust_epsilon_cos(idx):
    if idx <= args.warmup:
        return 1.0
    else:
        return max(args.epsilon ** idx, args.min_epsilon)

def main():
    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    if args.model == "DeepQ":
        model = DeepQ(state_len=args.image_size, action_len=args.action_len, info_len=args.info_len,
                      model_comp=args.model_comp,
                      device=args.device).to(args.device)
        optimizer = optim.Adam(model.QSA_online.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = model.loss
        scheduler = None, None

    print(model)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.LoadBest:
        model.load_state_dict(torch.load(args.BestModelPath, map_location=args.device)['state_dict'])
        optimizer.load_state_dict(torch.load(args.BestModelPath, map_location=args.device)['optimizer'])
        print("Resuming from epoch : " + str(args.epoch_start))
        for g in optimizer.param_groups:
            g['lr'] = args.lr
    env = MarioGame(model=model, optimizer=optimizer, criterion=criterion, args=args)

    rewards_max, config_max = -1, -1  # get_best_metric(args)
    model_max = None
    comment = f'lr_{args.lr}_batch_{args.batch_size}' \
              f'_downsample_{args.downsample}_skipframe_{args.skip_frame_cnt}'
    global writer
    writer = SummaryWriter(comment=comment)

    for epoch in range(args.epoch_start, args.epochs):
        epsilon = adjust_epsilon_cos(epoch)
        print(f'Game : [{epoch}]/[{args.epochs}]\tepsilon : {epsilon : .2f}')

        # train loop
        rewards_train_epoch, loss, steps_train, total_steps = train(epoch, env, epsilon, n_games=32)
        rewards_val_epoch, steps_val = validate(epoch, env, n_games=5)
        print("validation done.....")

        writer.add_scalars(f'reward ', {'reward train': rewards_train_epoch, 'reward val': rewards_val_epoch},
                           total_steps)
        writer.add_scalars(f'Steps ', {'steps train': steps_train, 'steps val': steps_val}, epoch)
        writer.add_scalars(f'Epsilon ', {'epsilon': epsilon}, epoch)
        writer.add_scalars(f'Loss ', {'Loss ': loss}, epoch)

        writer.flush()
        config['epoch'] = epoch
        if rewards_val_epoch > rewards_max:
            rewards_max, config_max, model_max = rewards_val_epoch, config, copy.deepcopy(model)
        save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'Reward': rewards_max,
                         'config': config},
                        epoch%10==0,
                        filename=os.path.join(args.saveloc, args.model + '_' + comment + '_' + str(epoch) + '.pth.tar'))

    writer.close()
    print('Best rReward: {:.4f}'.format(rewards_max))
    print('Model Config: ' + str(config_max))
    write_best_metric(args, rewards_max=rewards_max, config_max=config_max)

    if model_max:
        save_checkpoint({'state_dict': model_max.state_dict(), 'optimizer': optimizer.state_dict()
                            , 'Reward': rewards_max, 'config': config_max},
                        True, filename=os.path.join(args.saveloc, args.model + '_' + comment + '_best.pth.tar'))


if __name__ == '__main__':
    main()
