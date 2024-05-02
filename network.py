
from torch.distributions import MultivariateNormal
from torch import nn
import torch
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNetwork, self).__init__()

        self.cov_var = torch.full(size=[out_dim], fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
    def get_action(self, x):
        mean = self.forward(x)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def get_prob_action(self, x, actions):
        """Calculates the probability of taking a certain action in current network state"""
        mean = self.forward(x)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(actions)
        return log_probs

class ValueNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ValueNetwork, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
        