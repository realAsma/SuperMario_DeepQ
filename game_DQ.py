import yaml
import argparse

from collections import deque
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from models import DeepQ
from collections.abc import Iterable
from utils import AverageMeter
import torch
import random
import copy
import numpy as np
import torch.nn.functional as F

class MarioGame:

    def __init__(self, model=DeepQ, optimizer=None, criterion=None,
                 render=False,world=1, stage=1, version=1, args=None):
        assert 1 <= world <= 8
        assert 1 <= stage <= 4
        assert 0 <= version <= 3
        env_s = f'SuperMarioBros-{world}-{stage}-v{version}'
        env = gym_super_mario_bros.make(env_s)
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        #self.env = JoypadSpace(env, self.action_space)
        self.action_space = ['NOOP', 'right', 'right A', 'right B', 'right A B', 'A', 'left']

        self.render = args.render
        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion

        self.args = args

        self.device = self.args.device
        self.total_steps = 0

        self.batch_size = self.args.batch_size
        self.replay_buf = deque(maxlen=self.args.replay_buffer_size)

        # Action space is  Discrete(7)
        # Action means  ['NOOP', 'right', 'right A', 'right B', 'right A B', 'A', 'left']
        # State shape is  (240, 256, 3)

    def game(self, epsilon=0):
        done = False
        info = {'x_pos': 40, 'y_pos': 79, 'time': 400}
        reward = 0
        reward_eps = 0
        state = self.env.reset()
        step = 0
        loss_eps = AverageMeter()
        stopEps = StopNonProEps(step_thres=self.args.step_thres)
        action = None

        while not done:
            state_prev, _, _ = self.preprocess(state=state, action_prev=action, info=info)
            action = self.model(state_prev, epsilon)
            state, reward, done, info = self.step(action)
            reward_eps += reward

            if self.model.training:
                # samples sars from replay buffer, for now samples if deque size > batch_size
                # This might lead to earlier samples to be sampled much more than later samples and shift the distribution
                reward, done = stopEps.action(reward, done, info)
                state_next, reward, ndone = self.preprocess(state=state, reward=reward, action_prev= action, done=done, info = info)
                self.buffer_update(state_prev, action, reward, ndone, state_next)

                if step % self.batch_size == 0 and len(self.replay_buf) >= self.batch_size*16:
                    state_t, action_t, reward_t, ndone_t, state_next_t = self.sample()
                    # Calculate loss and optimize the models
                    #   using a tuple to pass actor and critic for gradient clipping
                    loss_batch = self.model_update(state_t, action_t, reward_t, ndone_t, state_next_t)
                    loss_eps.update(loss_batch)
                    if (step // self.args.batch_size) % self.args.print_every == 0:
                        print(f"\tStep [{step}]/XXX "
                                  f"Loss : {loss_batch:.4f}"
                                  f"\tTotal rewards : {reward_eps}\tepsilon : {epsilon:.2f}")
                        with torch.no_grad():
                            qsa = self.model.QSA_target((state_t[0][0:5, :], state_t[1][0:5, :]))
                        print("\tQSA:\n\t"+str(qsa[0])+"\n\t"+str(qsa[1])+"\n\t"+str(qsa[2])+"\n\t"+str(qsa[3]))
                        print("\tHistorgam of last batch actions: " +
                              str(torch.histc(action_t.float(), bins=self.args.action_len, min=0, max=self.args.action_len)))

                if self.total_steps % self.args.update_target_every == 0 or done:
                    # only actor is needed for the target network, avoid copying the replay buffer
                    self.model.QSA_target = copy.deepcopy(self.model.QSA_online)
                    self.model.QSA_target.eval()
                    #self.copy_counter = 0
                self.total_steps += 1

            #if self.render:
            #    self.env.render()
            step += 1

        return reward_eps, loss_eps.avg, step, self.total_steps

    def model_update(self, state_t, action_t, reward_t, done_t, state_next_t):
        loss = self.criterion(state_t, action_t, reward_t, done_t, state_next_t)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.QSA_online.parameters(), self.args.clip_grad)
        self.optimizer.step()
        return loss.item()

    def step(self, action):
        reward_eps = 0
        for i in range(self.args.skip_frame_cnt):
            state, reward, done, info = self.env.step(action)
            reward_eps += float(reward)
            if self.render:
                self.env.render()
            if done:
                return state, reward, done, info
        return state, reward_eps, done, info

    def reset(self):
        self.env.reset()

    def stop(self):
        self.env.close()

    def preprocess(self, state, reward=0, action_prev=None, info = None, done=False):

        ################################################
        # function to format to the environment variables appropriately so that it can be fed in to NN
        ###############################################
        # state tensor size is now 1x1x120x128

        state_processed = torch.FloatTensor(np.mean(state[::self.args.downsample, ::self.args.downsample, :], axis=2)/128.0-1).unsqueeze(dim=0)

        action_tensor = torch.zeros((len(self.action_space),))
        if action_prev:
            action_tensor[action_prev] = 1.0
        info_tensor = torch.zeros((3,))

        if info:
            info_tensor[0] = info['x_pos']/120.0-1
            info_tensor[1] = info['y_pos']/128.0-1
            info_tensor[2] = info['time']/200-1

        action_info = torch.cat((action_tensor, info_tensor)).unsqueeze(dim = 0)

        # normalize reward, reward varies from -15 to 15
        # Game objective is to move as far as right as possible, increasing the penatly for deatch bu done_mult
        if done:
            reward += -15*self.args.die_mult

        reward_processed = torch.unsqueeze(torch.tensor(reward)/(15*(1+self.args.die_mult)*self.args.skip_frame_cnt), dim=0).to(self.device)

        ndone_processed = torch.tensor(not done).unsqueeze(dim=0).to(self.device)

        return [state_processed, action_info], reward_processed, ndone_processed

    def buffer_update(self, state_t, action_t, reward_t, done_t, state_tp1):
        item = (state_t, action_t, reward_t, done_t, state_tp1)
        self.replay_buf.append(item)

    def sample(self):
        if len(self.replay_buf) < self.batch_size:
            return None, None, None, None
        rand_indices = random.sample(range(len(self.replay_buf)), k=self.batch_size)
        state = torch.cat([self.replay_buf[i][0][0] for i in rand_indices], dim=0), torch.cat([self.replay_buf[i][0][1] for i in rand_indices], dim=0)
        action = torch.unsqueeze(torch.LongTensor([self.replay_buf[i][1] for i in rand_indices]), dim = 1).to(self.device)
        reward = torch.cat([self.replay_buf[i][2] for i in rand_indices], dim=0)
        done = torch.cat([self.replay_buf[i][3] for i in rand_indices], dim=0)
        state_next = torch.cat([self.replay_buf[i][4][0] for i in rand_indices], dim=0), torch.cat([self.replay_buf[i][4][1] for i in rand_indices], dim=0)
        return state, action, reward, done, state_next


class StopNonProEps:
    def __init__(self, step_thres):
        self.xpos = -1
        self.count = 0
        self.step_thres = step_thres

    def action(self, reward, done, info):
        if self.xpos == info['x_pos']:
            self.count += 1
            if self.count == self.step_thres:
                reward = -15
                done = True
        else:
            self.count = 0
            self.xpos = info['x_pos']
        return reward, done



def main():
    parser = argparse.ArgumentParser(description='CS7643 deep_pipes')
    parser.add_argument('--config', default='./configs/config_ActorCritic.yaml')
    parser.add_argument('--device', default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    MG = MarioGame(args=args)
    state = MG.env.reset()
    done = False
    action = 1
    count = 1
    while not done:
        i = 0
        while i < count and not done:
            state, reward, done, info = MG.action_wrapper(action)
            MG.env.render()
            i += 1
    print(reward)


def main2():
    env_s = f'SuperMarioBros-1-1-v1'
    env = gym_super_mario_bros.make(env_s)
    env = JoypadSpace(env,[["NOOP"], ["right"], ["left"], ["A"]])
    # Action means  ['NOOP', 'right', 'right A', 'right B', 'right A B', 'A', 'left']
    state = env.reset()

    done = False
    count = 1
    while not done:
        i = 0
        while i < count and not done:
            state, reward, done, info = env.step(action)
            env.render()
            i += 1
    print(reward)

if __name__ == '__main__':
    main2()
