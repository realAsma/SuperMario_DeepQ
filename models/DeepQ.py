import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import copy


class DeepQ(nn.Module):
    def __init__(self, gamma=0.99,  model_comp = 'FC', device = None,
                 state_len = 240*256//4, action_len = 4, info_len = 3):
        super(DeepQ, self).__init__()
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.gamma = gamma

        self.num_Actions = action_len

        self.indices = torch.unsqueeze(torch.LongTensor(range(0, self.num_Actions)), dim=0).to(self.device)
        if model_comp == 'FC':
            self.QSA_online = ModelFC(in_features=state_len + action_len + info_len, out_features=action_len).to(self.device)
            self.QSA_target = copy.deepcopy(self.QSA_online)
        else:
            self.QSA_online = ModelNetL4(state_features=32*13*12, action_info_len=action_len+info_len,out_features=action_len).to(self.device)
            self.QSA_target = copy.deepcopy(self.QSA_online)

        self.QSA_target.eval()


        ##############################################
        #    To Do: Learning Rate Scheduling         #
        ##############################################

    def forward(self, state, epsilon=0):
        if random.random() < epsilon:
            return random.randint(0, self.num_Actions-1)
        with torch.no_grad():
            return self.QSA_target(state).argmax().item()

    @staticmethod
    def entropy(policy):
        return -torch.sum(policy * torch.log(policy))/policy.shape[0]

    def loss(self, state_t, action_mask, reward_t, ndone_t, state_tp1):
        # Critic optimize
        with torch.no_grad():
            Q_target = reward_t + self.gamma * ndone_t * self.QSA_target(state_tp1).max(dim=1)[0]
        Q_obs = self.QSA_online(state_t)[action_mask == self.indices]
        return F.mse_loss(Q_obs, Q_target)


class ModelNet(nn.Module):
    def __init__(self, out_features=7):
        super(ModelNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=7, stride=2), nn.ReLU(),
                    nn.Flatten(), nn.Linear(in_features=4*57*61, out_features=out_features))

    def forward(self, states):
        return self.net(states)


class ModelFC(nn.Module):
    def __init__(self, in_features=120*128, out_features=7):
        super(ModelFC, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_features=in_features, out_features=256), nn.ReLU(),
                                 nn.Linear(in_features=256, out_features=out_features))
        self.net[2].weight.data.uniform_(-3e-3, 3e-3)
        self.net[2].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        states = torch.cat((torch.flatten(states[0],start_dim=1), states[1]),dim=1)
        return self.net(states)


class ModelNetL4(nn.Module):
    def __init__(self, state_features = 32*29*27, action_info_len = 11, out_features=7):
        super(ModelNetL4, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4), nn.ReLU(),
                                 nn.Conv2d(in_channels=16, out_channels=32,kernel_size=4, stride=2),
                                 nn.Flatten())
        #for 128 x 120 -> after cnn -> 5824
        self.fc = nn.Sequential(nn.Linear(in_features= 5824 + action_info_len, out_features=256), nn.ReLU(),
                                 nn.Linear(in_features=256, out_features=out_features))
        self.fc[2].weight.data.uniform_(-3e-3, 3e-3)
        self.fc[2].bias.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, states):
        px_state = self.cnn(states[0].unsqueeze(dim=1))
        return self.fc(torch.cat((px_state, states[1]), dim=1))



