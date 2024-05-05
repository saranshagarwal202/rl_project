from torch import nn
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import numpy as np
from tqdm import tqdm
from json import loads, dumps
import os
import gymnasium as gym
from network import PolicyNetwork


class Network(nn.Module):
    def __init__(self, in_dim) -> None:
        super(Network, self).__init__()

        self.l1 = nn.Linear(in_dim, 256)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(256, 256)
        self.a2 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.l3 = nn.Linear(256, 256)
        self.a3 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.l4 = nn.Linear(256, 1)
        

        # self.seq = nn.Sequential(
        #     nn.Linear(in_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.drop1(x)
        x = self.l3(x)
        x = self.drop2(x)
        x = self.a3(x)
        x= self.l4(x)
        return x

class Data_generator(torch.utils.data.Dataset):
    def __init__(self, env, stage=100, n_models=5, T_len=50, total_T=int(24000/50)) -> None:
        super(Data_generator).__init__()

        self.env = gym.make(env)
        self.T_len = T_len
        self.n_models = n_models
        self.state_dim = self.env.observation_space.shape[0]
        self.total_T = total_T
        # initialize policy
        self.policy = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.policy.load_state_dict(state_dict=torch.load(f"data/{self.env.spec.id}/policy_{stage}.pt"))

        # make data
        self.make_trajectories()

        # make indices
        indices1 = np.random.choice(len(self.data), size=(
            len(self.data), n_models), replace=True)
        indices2 = np.random.choice(len(self.data), size=(
            len(self.data), n_models), replace=True)

        # check to insure same trajectory is not paired together
        check = True
        while check:
            common_pairs = (indices1 == indices2).any(axis=1).nonzero()
            for idx in common_pairs[0]:
                indices1[idx, :] = np.random.choice(
                    len(self.data), size=n_models)
            check = (indices1 == indices2).any()

        # indices shape = (2, len(data), 5)
        self.indices = torch.tensor([indices1, indices2], dtype=torch.int32)

    def make_trajectories(self):
        self.data = []
        T=0
        pbar = tqdm(total=self.total_T, desc="Generating data")
        while T < self.total_T:
            state, _ = self.env.reset()
            done = False
            rewards = 0
            ep_states = []
            while not done:
                ep_states.append(state)
                action = self.policy(torch.tensor(state, dtype=torch.float32)).detach().numpy()
                n_state, reward, done, truncated, _ = self.env.step(action)
                rewards+=reward
                state=n_state
            if len(ep_states)>50:
                self.data.append([ep_states, rewards])
                T+=1
                pbar.update()
        
        pbar.close()
    
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        X = torch.zeros(size=(2, self.T_len, self.n_models, self.state_dim))
        Y = torch.zeros((self.n_models))
        # these index slices represent training pairs for each of n_model (default 5) reward networks
        index_slice_1 = self.indices[0, index, :]
        index_slice_2 = self.indices[1, index, :]
        for i in range(self.n_models):
            T1 = self.data[index_slice_1[i]]
            T2 = self.data[index_slice_2[i]]

            # choosing random slice of 50 states
            idx1 = np.random.choice(len(T1[0])-self.T_len, 1)[0]
            idx2 = np.random.choice(len(T2[0])-self.T_len, 1)[0]
            X[0, :, i, :] = torch.tensor(
                np.array(T1[0][idx1: idx1+self.T_len], dtype=np.float32))
            X[1, :, i, :] = torch.tensor(
                np.array(T2[0][idx2: idx2+self.T_len], dtype=np.float32))
            # setting labels as logits, 1 representing trajectory with higher rewards
            Y[i] = int(T2[1] > T1[1])

        return X, Y

class Data(torch.utils.data.Dataset):
    def __init__(self, env, stage=3, data_ratio=0.8, data_type='train', test_indices=[], n_models=5, T_len=50) -> None:
        """
        parameters:
            data_file: the location of json data file
            n_models: number of parallel reward models being trained.
        """
        super(Data).__init__()
        data = []
        self.T_len = T_len
        self.n_models = n_models
        data_dir = f"data/{env}/states_bank"
        for file in sorted(os.listdir(data_dir)):
            f = open(os.path.join(data_dir, file), 'r')
            data.extend(loads(f.read()))
            f.close()

        self.state_dim = len(data[0][0][0])
        
        # select data based upon stage and print its best and mean reward
        if stage<3:
            items = (len(data)//3)*stage
            data = data[:items]
        # sorting data to rank from worst to best
        data.sort(key=lambda x: x[1])

        # make sets of 50 t lenin each state
        self.data = torch.zeros((6000//T_len, T_len, self.state_dim))
        self.data_rewards = torch.zeros((6000//T_len, T_len))
        i = 0
        data_i = 0
        while i<self.data.shape[0]:
            #pick random set of 50 states from this T
            
            idx = np.random.choice(max(len(data[data_i][0])-self.T_len, 1), 1)[0]
            self.data_rewards[i] = torch.tensor(np.array(data[data_i][1][idx:idx+T_len], dtype=np.float32))
            self.data[i, :, :] = torch.tensor(np.array(data[data_i][0][idx:idx+T_len], dtype=np.float32))
            i+=1
            data_i+=1
            if data_i==len(data):
                data_i=0

        del data

        if data_type=='train':
            indices = np.random.choice(len(self.data), len(self.data), replace=False)
            indices = indices[:min(len(indices), 6000)]
            print(f"Using {len(indices)} for training +test")
            self.train_indices = indices[:int(len(self.data)*data_ratio)]
            self.test_indices = indices[int(len(self.data)*data_ratio):]
            # self.data = [self.data[i] for i in self.train_indices]
            self.data = self.data[self.train_indices]
        else:
            # self.data = [self.data[i] for i in test_indices]
            self.data = self.data[test_indices]

        self.data = torch.reshape(self.data, (self.data.shape[0]*self.data.shape[1], self.data.shape[2]))
        self.data_rewards = torch.reshape(self.data_rewards, (self.data_rewards.shape[0]*self.data_rewards.shape[1],))
        # rews = [d[1] for d in self.data]
        print(f"PPO mean reward: {self.data_rewards.mean()}")
        print(f"PPO max reward: {self.data_rewards.max()}")

        # creating index pairs on which each model will train
        # prob of sampling lower trajectories is less compared to trajectories generated later in training with better ppo models
        self.indices1 = np.random.choice(len(self.data), size=(
            len(self.data), n_models), replace=True)
        self.indices2 = np.random.choice(len(self.data), size=(
            len(self.data), n_models), replace=True)

        # check to insure same trajectory is not paired together
        check = True
        while check:
            common_pairs = (self.indices1 == self.indices2).any(axis=1).nonzero()
            for idx in common_pairs[0]:
                self.indices1[idx, :] = np.random.choice(
                    len(self.data), size=n_models)
            check = (self.indices1 == self.indices2).any()

        # indices shape = (2, len(data), 5)
        # self.indices = torch.tensor([indices1, indices2], dtype=torch.int32)

    def __len__(self):
        return self.indices1.shape[0]

    def __getitem__(self, index):
        X1 = torch.zeros((self.state_dim, self.n_models))
        X2 = torch.zeros((self.state_dim, self.n_models))
        for i in range(self.n_models):
            if self.data_rewards[self.indices1[index]][i]>self.data_rewards[self.indices2[index]][i]:
                X1[:, i] = self.data[self.indices1[index][i]]
                X2[:, i] = self.data[self.indices2[index][i]]
            else:
                X1[:, i] = self.data[self.indices2[index][i]]
                X2[:, i] = self.data[self.indices1[index][i]]
        
        return X1, X2



    # def __getitem__(self, index):
    #     X = torch.zeros(size=(2, self.T_len, self.n_models, self.state_dim))
    #     # these index slices represent training pairs for each of n_model (default 5) reward networks
    #     index_slice_1 = self.indices[0, index, :]
    #     index_slice_2 = self.indices[1, index, :]
    #     for i in range(self.n_models):
    #         T1 = self.data[index_slice_1[i]]
    #         T2 = self.data[index_slice_2[i]]

    #         # choosing random slice of 50 states
    #         # 0th index is traj with greater reward always
    #         if T1[1]>T2[1]:
    #             idx1 = np.random.choice(max(len(T1[0])-self.T_len, 1), 1)[0]
    #             idx2 = np.random.choice(max(len(T2[0])-self.T_len, 1), 1)[0]
    #             X[0, :, i, :] = torch.tensor(
    #                 np.array(T1[0][idx1: idx1+self.T_len], dtype=np.float32))
    #             X[1, :, i, :] = torch.tensor(
    #                 np.array(T2[0][idx2: idx2+self.T_len], dtype=np.float32))
    #         else:
    #             idx1 = np.random.choice(max(len(T2[0])-self.T_len, 1), 1)[0]
    #             idx2 = np.random.choice(max(len(T1[0])-self.T_len, 1), 1)[0]
    #             X[0, :, i, :] = torch.tensor(
    #                 np.array(T2[0][idx1: idx1+self.T_len], dtype=np.float32))
    #             X[1, :, i, :] = torch.tensor(
    #                 np.array(T1[0][idx2: idx2+self.T_len], dtype=np.float32))

    #     return X

    def reset_index(self):
        """reset indices"""
        # creating index pairs on which each model will train
        indices1 = np.random.choice(len(self.data), size=(
            len(self.data), self.n_models), replace=True)
        indices2 = np.random.choice(len(self.data), size=(
            len(self.data), self.n_models), replace=True)

        # check to insure same trajectory is not paired together
        check = True
        while check:
            common_pairs = (indices1 == indices2).any(axis=1).nonzero()
            for idx in common_pairs[0]:
                indices1[idx, :] = np.random.choice(
                    len(self.data), size=self.n_models)
            check = (indices1 == indices2).any()

        # indices shape = (2, len(data), 5)
        self.indices = torch.tensor([indices1, indices2], dtype=torch.int32)


class Reward():
    def __init__(self, state_dim, env, lr=1e-4,
                 n_iter=10000, batch_size=64, stage=3,
                 n_models=5, T_len=50, mode='train') -> None:

        self.lr = lr
        self.n_iter = n_iter
        self.n_models = n_models
        self.T_len = T_len
        self.env = env
        self.batch_size = batch_size

        # generate 5 reward networks
        self.device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.reward_model = []
        self.reward_model_std_params = {}
        self.optimizers = []
        self.history = []

        if mode == 'train':
            for i in range(n_models):
                self.reward_model.append(Network(state_dim).to(self.device))
                self.reward_model_std_params[f"reward_model_{i}"] = {'min': 0, 'max': 0}
                self.optimizers.append(torch.optim.Adam(self.reward_model[i].parameters(), lr=self.lr))
            # train data
            self.train_data = Data(env, stage=stage, n_models=n_models, T_len=T_len)
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True,
                num_workers=os.cpu_count()-1, persistent_workers=True,
                prefetch_factor=7)
            
            # test data
            self.test_data = Data(env, stage=stage, data_ratio=0.2, data_type='test', test_indices=self.train_data.test_indices, n_models=n_models, T_len=T_len)
            self.test_dataloader = torch.utils.data.DataLoader(
                self.test_data, batch_size=batch_size, shuffle=True,
                num_workers=os.cpu_count()-1, persistent_workers=True,
                prefetch_factor=7)
            

            self.loss_fn = nn.BCEWithLogitsLoss()

        elif mode == "train_continue":
            for i in range(n_models):
                self.reward_model.append(Network(state_dim).to(self.device))
                self.optimizers.append(torch.optim.Adam(self.reward_model[i].parameters(), lr=self.lr))
            self.train_data = Data(env, n_models=n_models, T_len=T_len)
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True, 
                num_workers=os.cpu_count()-1, persistent_workers=True,
                prefetch_factor=7)
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.load()
            f = open(f"data/{self.env}/reward_network/history.json", 'r')
            self.history = loads(f.read())
            f.close()
            self.n_iter -=len(self.history)

        else:
            for i in range(n_models):
                self.reward_model.append(Network(state_dim).to(self.device))

            self.load()

    def learn(self):

        self.history = []
        self.prev_losses = [10000 for i in range(self.n_models)]
        pbar = tqdm(total=self.n_iter)
        for epochs in range(self.n_iter):
            train_losses = [0 for i in range(self.n_models)]
            test_losses = [0 for i in range(self.n_models)]
            for X1, X2 in self.train_dataloader:

                # X1 = torch.reshape(X1, (X1.shape[0]*X1.shape[1], X1.shape[2], X1.shape[3]))
                # X2 = torch.reshape(X2, (X2.shape[0]*X2.shape[1], X2.shape[2], X2.shape[3]))
                # flipping Y with prob 0.1
                # random_tensor = torch.rand(Y.size())
                # mask = random_tensor < 0.01
                # Y = torch.where(mask, 1 - Y, Y)
                
                # reshape X and Y
                # curr_batch_size = X1.shape[0]

                # X = torch.reshape(X, (X.shape[0]*2*X.shape[2], X.shape[3], X.shape[4]))
                # Y = torch.reshape(Y, (Y.shape[0]*2, Y.shape[2]))
                # preds = torch.zeros(Y.shape)
                
                # training 5 models one by one
                for i, model in enumerate(self.reward_model):
                    model.train()
                    # preds = torch.zeros((Y.shape[0], 1), device=self.device)
                    preds1 = model(X1[:, :, i].squeeze().to(self.device))
                    preds2 = model(X2[:, :, i].squeeze().to(self.device))
                    self.reward_model_std_params[f'reward_model_{i}']['min'] = min(self.reward_model_std_params[f'reward_model_{i}']['min'], preds1.min().item())
                    self.reward_model_std_params[f'reward_model_{i}']['max'] = max(self.reward_model_std_params[f'reward_model_{i}']['max'], preds1.max().item())
                    # preds = preds.reshape(curr_batch_size, self.T_len)
                    # preds = preds.mean(axis=2)

                    # preds = preds.sum(axis=1)
                    # y = torch.zeros((preds.shape[0], 2))
                    # y[:, 0] = Y[:,i]
                    # y[:, 1] = 1-Y[:,i]
                    # y = y.reshape_as(preds)
                    # loss = self.loss_fn(preds, y.to(self.device)) # add loss here
                    # To do: recheck loss calculation formula
                    # y = Y[:, i].to(self.device, dtype=torch.int32)
                    # Tj = torch.exp(preds[:, 0]) # higher reward
                    # Ti = torch.exp(preds[:, 1]) # lower reward
                    # check to see loss does not go nan
                    loss = -(torch.log(torch.exp(preds1)/(torch.exp(preds1)+torch.exp(preds2))).mean())
                    l2_reg = 0.001*sum(param.norm()**2 for param in model.parameters() if param.requires_grad)
                    loss = loss+l2_reg
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()
                    train_losses[i]+= loss.item()
            
            for X1, X2 in self.test_dataloader:

                # flipping Y with prob 0.1
                # random_tensor = torch.rand(Y.size())
                # mask = random_tensor < 0.01
                # Y = torch.where(mask, 1 - Y, Y)
                
                # reshape X and Y
                # curr_batch_size = X1.shape[0]
                # X1 = torch.reshape(X1, (X1.shape[0]*X1.shape[1], X1.shape[2], X1.shape[3]))
                # X2 = torch.reshape(X2, (X2.shape[0]*X2.shape[1], X2.shape[2], X2.shape[3]))
                # flipping Y with prob 0.1
                # random_tensor = torch.rand(Y.size())
                # mask = random_tensor < 0.01
                # Y = torch.where(mask, 1 - Y, Y)
                
                # reshape X and Y
                # curr_batch_size = X1.shape[0]

                # X = torch.reshape(X, (X.shape[0]*2*X.shape[2], X.shape[3], X.shape[4]))
                # Y = torch.reshape(Y, (Y.shape[0]*2, Y.shape[2]))
                # preds = torch.zeros(Y.shape)
                
                # training 5 models one by one
                for i, model in enumerate(self.reward_model):
                    model.eval()
                    # preds = torch.zeros((Y.shape[0], 1), device=self.device)
                    preds1 = model(X1[:, :, i].squeeze().to(self.device))
                    preds2 = model(X2[:, :, i].squeeze().to(self.device))
                    self.reward_model_std_params[f'reward_model_{i}']['min'] = min(self.reward_model_std_params[f'reward_model_{i}']['min'], preds1.min().item())
                    self.reward_model_std_params[f'reward_model_{i}']['max'] = max(self.reward_model_std_params[f'reward_model_{i}']['max'], preds1.max().item())
                    # preds = preds.reshape(curr_batch_size, self.T_len)
                    # preds = preds.mean(axis=2)

                    # preds = preds.sum(axis=1)
                    # y = torch.zeros((preds.shape[0], 2))
                    # y[:, 0] = Y[:,i]
                    # y[:, 1] = 1-Y[:,i]
                    # y = y.reshape_as(preds)
                    # loss = self.loss_fn(preds, y.to(self.device)) # add loss here
                    # To do: recheck loss calculation formula
                    # y = Y[:, i].to(self.device, dtype=torch.int32)
                    # Tj = torch.exp(preds[:, 0]) # higher reward
                    # Ti = torch.exp(preds[:, 1]) # lower reward
                    # check to see loss does not go nan
                    loss = -(torch.log(torch.exp(preds1)/(torch.exp(preds1)+torch.exp(preds2))).mean())
                    l2_reg = 0.001*sum(param.norm()**2 for param in model.parameters() if param.requires_grad)
                    loss = loss+l2_reg
                    test_losses[i]+= loss.item()


            train_losses = [round(lo/len(self.train_dataloader), 2) for lo in train_losses]
            test_losses = [round(lo/len(self.test_dataloader), 2) for lo in test_losses]
            self.history.append((train_losses, test_losses))
            pbar.set_postfix_str(f"train: {train_losses}, test: {test_losses}") 
            pbar.update()
            if epochs%10==0:
                self.save_checkpoint(test_losses)
            
            # self.train_data.reset_index()
        pbar.close()
        # saving history and reward models
        self.save_checkpoint(test_losses)

    def save_checkpoint(self, losses):
        try:
            f = open(f"data/{self.env}/reward_network/history.json", 'w')
        except FileNotFoundError:
            os.makedirs(f"data/{self.env}/reward_network/")
            f = open(f"data/{self.env}/reward_network/history.json", 'w')

        f.write(dumps(self.history))
        f.close()
        
        for i, model in enumerate(self.reward_model):
            if losses[i]<self.prev_losses[i]:
                torch.save(model.state_dict(), f"data/{self.env}/reward_network/reward_model_{i}.pt")
                self.prev_losses[i]=losses[i]
        
        f = open(f"data/{self.env}/reward_network/reward_model_std_params.json", 'w')
        f.write(dumps(self.reward_model_std_params))
        f.close()

    def load(self):
        for i, model in enumerate(self.reward_model):
            model.load_state_dict(torch.load(f"data/{self.env}/reward_network/reward_model_{i}.pt"))
        f = open(f"data/{self.env}/reward_network/reward_model_std_params.json", 'r')
        self.reward_model_std_params = loads(f.read())
        f.close()
    
    def get_reward(self, X):
        """X shape is (batch_size, n_models, state_space_dim)"""
        rewards = []
        for i, model in enumerate(self.reward_model):
            model.eval()
            rew = F.sigmoid(model(X.squeeze().to(self.device))).detach().cpu()
            # standardizing reward
            # rew = (rew-self.reward_model_std_params[f"reward_model_{i}"]['min'])/(self.reward_model_std_params[f"reward_model_{i}"]['max']-self.reward_model_std_params[f"reward_model_{i}"]['min'])
            rewards.append(rew)
        return sum(rewards)/len(rewards)



if __name__ == "__main__":

    reward = Reward(state_dim=11, env="Hopper-v4", n_iter=3000, lr=1e-4, stage=1, mode='train')
    reward.learn()
    print("Done")