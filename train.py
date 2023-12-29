import torch
import numpy as np
from ppo_agent import PPO
import os
import json



class Storage:
    def __init__(self):
        self.states = []
        self.actions = []
        self.true_actions = []
        self.logprobs = []
        self.old_logprobs = []
        self.values = []
        self.rewards = []
        self.returns = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.old_logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.returns.claer()
        self.true_actions.clear()

# 定义训练参数
state_dim = 150
state_length = 4
action_dim = 1
exploration_param = 0.99
lr = 1e-6
betas = (0.9, 0.999)
gamma = 0.99
ppo_epoch = 4
ppo_clip = 0.2
use_gae = False
observation_dim = 150
action_dim = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ppo_agent = PPO(state_dim, state_length, action_dim, exploration_param, lr, betas, gamma, ppo_epoch, ppo_clip, use_gae)

data_folder = './testbed_dataset_chunk_0'
done = 0
total=len(os.listdir(data_folder))
num_epochs = 10
no_epoch = 0
for epoch in range(num_epochs):  
    files_nums = 0
    no_epoch += 1
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)   
        with open(file_path, 'r') as f:
            episode_data = json.load(f)
            observations = torch.tensor(episode_data['observations']).to(device)
            actions = torch.tensor(episode_data['bandwidth_predictions']).to(device)
            rewards = torch.tensor([a + b if not (np.isnan(a) or np.isnan(b)) else 0 for a, b in zip(episode_data['video_quality'], episode_data['audio_quality'])]).to(device)

            length = len(observations)
            version_size = 2
            total_policy_loss = 0
            total_value_loss = 0

            for i in range(length - version_size):
                storage = Storage()
                for j in range(version_size):
                    action_pred = ppo_agent.select_action(observations[i + j], storage, actions[i + j])
                    storage.true_actions.append(actions[i + j])
                    storage.returns.append(rewards[i + j]) 
                policy_loss, value_loss = ppo_agent.update(storage)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                # print(i)
            for i in range(1, version_size - 1):
                storage = Storage()
                for j in range(version_size - i):                        
                    action_pred = ppo_agent.select_action(observations[length - version_size + j], storage, actions[length - version_size + j])
                    storage.true_actions.append(actions[length - version_size + j])
                    storage.returns.append(rewards[length - version_size + j])    
                policy_loss, value_loss = ppo_agent.update(storage)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
            files_nums += 1

            print(f'epoch: {no_epoch}   done_json_files: {files_nums}')
            print(f'average_policy_loss:{total_policy_loss / length}   average_value_loss{total_value_loss / length}')

    ppo_agent.save_model('./saved_models/') 
