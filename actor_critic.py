#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, exploration_param=0.05, device="cpu"):
        super(ActorCritic, self).__init__()
        # output of actor in [0, 1]
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.ReLU(),
                nn.Linear(32, 1)
                )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_var = torch.full((action_dim,), exploration_param**2).to(self.device)
        self.random_action = False


    def forward(self, state):
        value = self.critic(state)
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        # print(f'{action_mean}   {cov_mat}')
        dist = MultivariateNormal(action_mean, cov_mat)
        if not self.random_action:
            action = action_mean
        else:
            action = dist.sample()
        action_logprobs = dist.log_prob(action)
        return action.detach(), action_logprobs, value, action_mean

    def evaluate(self, state, action):
        _, _, value, action_mean = self.forward(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        # print(f'{action_mean}   {cov_mat}')
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action.unsqueeze(1))
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(value), dist_entropy

