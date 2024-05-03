from torch import nn
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import numpy as np
from tqdm import tqdm
from json import loads, dumps
import os


class Network(nn.Module):
    def __init__(self, in_dim) -> None:
        super(Network, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.seq(x)


class Data(torch.utils.data.Dataset):
    def __init__(self, env, n_models=5, T_len=50) -> None:
        """
        parameters:
            data_file: the location of json data file
            n_models: number of parallel reward models being trained.
        """
        super(Data).__init__()
        self.data = []
        self.T_len = T_len
        self.n_models = n_models
        data_dir = f"data/{env}/states_bank"
        for file in os.listdir(data_dir):
            f = open(os.path.join(data_dir, file), 'r')
            self.data.extend(loads(f.read()))
            f.close()

        self.state_dim = len(self.data[0][0][0])

        # creating index pairs on which each model will train
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

    def __len__(self):
        return self.indices.shape[1]

    def __getitem__(self, index):
        X = torch.zeros(size=(2, self.T_len, self.n_models, self.state_dim))
        Y = torch.zeros((2, self.n_models))
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
            Y[0, i] = T1[1] >= T2[1]
            Y[1, i] = T2[1] > T1[1]

        return X, Y


class Reward():
    def __init__(self, state_dim, env, lr=1e-4,
                 n_iter=10000, batch_size=64, 
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
        self.optimizers = []
        self.history = []

        if mode == 'train':
            for i in range(n_models):
                self.reward_model.append(Network(state_dim).to(self.device))
                self.optimizers.append(torch.optim.Adam(self.reward_model[i].parameters(), lr=self.lr))
            train_data = Data(env, n_models=n_models, T_len=T_len)
            self.train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=True)
            self.loss_fn = nn.BCEWithLogitsLoss()

        elif mode == "train_continue":
            for i in range(n_models):
                self.reward_model.append(Network(state_dim).to(self.device))
                self.optimizers.append(torch.optim.Adam(self.reward_model[i].parameters(), lr=self.lr))
            train_data = Data(env, n_models=n_models, T_len=T_len)
            self.train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1, persistent_workers=True)
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
        pbar = tqdm(total=self.n_iter)
        for epochs in range(self.n_iter):
            losses = [0 for i in range(self.n_models)]
            for X, Y in self.train_dataloader:
                
                # reshape X and Y
                curr_batch_size = X.shape[0]
                X = torch.reshape(X, (X.shape[0]*2*X.shape[2], X.shape[3], X.shape[4]))
                Y = torch.reshape(Y, (Y.shape[0]*2, Y.shape[2]))
                # preds = torch.zeros(Y.shape)
                
                # training 5 models one by one
                for i, model in enumerate(self.reward_model):
                    # preds = torch.zeros((Y.shape[0], 1), device=self.device)
                    preds = model(X[:, i, :].squeeze().to(self.device))
                    preds = preds.reshape(curr_batch_size*2, self.T_len)
                    preds = preds.sum(axis=1)

                    loss = self.loss_fn(preds, Y[:, i].to(self.device)) # add loss here
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()
                    losses[i]+= loss.item()
            losses = [round(lo/len(self.train_dataloader), 3) for lo in losses]
            self.history.append(losses)
            pbar.set_postfix_str(f"Loss: {losses}")
            pbar.update()
            if epochs%10==0:
                self.save_checkpoint(self.history)
        
        # saving history and reward models
        self.save_checkpoint(self.history)

    def save_checkpoint(self, history):
        try:
            f = open(f"data/{self.env}/reward_network/history.json", 'w')
        except FileNotFoundError:
            os.makedirs(f"data/{self.env}/reward_network/")
            f = open(f"data/{self.env}/reward_network/history.json", 'w')

        f.write(dumps(history))
        f.close()
        
        for i, model in enumerate(self.reward_model):
            torch.save(model.state_dict(), f"data/{self.env}/reward_network/reward_model_{i}.pt")

    def load(self):
        for i, model in enumerate(self.reward_model):
            model.load_state_dict(torch.load(f"data/{self.env}/reward_network/reward_model_{i}.pt"))
    
    def get_reward(self, X):
        """X shape is (batch_size, n_models, state_space_dim)"""
        rewards = []
        for i, model in enumerate(self.reward_model):
            rewards.append(model(X.squeeze().to(self.device)).detach().cpu())
        
        return sum(rewards)/len(rewards)



if __name__ == "__main__":

    reward = Reward(state_dim=11, env="Hopper-v4", n_iter=10000, lr=1e-4, mode='train')
    reward.learn()
    # for X, Y in reward.train_dataloader:
    #     X = torch.reshape(X, (128, 5, 50, 11))[:, :, 0, :].squeeze()
    #     break
    # reward = Reward(state_dim=11, env="Hopper-v4", n_iter=10, mode='test')
    # reward.get_reward(X)
    print("Done")