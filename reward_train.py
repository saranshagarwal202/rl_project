from torch import nn
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import numpy as np
from tqdm import tqdm
from json import loads
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
    def __init__(self, data_dir, n_models=5, T_len=50) -> None:
        """
        parameters:
            data_file: the location of json data file
            n_models: number of parallel reward models being trained.
        """
        super(Data).__init__()
        self.data = []
        self.T_len = T_len
        self.n_models = n_models
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
        X = torch.zeros(size=(2, self.n_models, self.T_len, self.state_dim))
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
            X[0, i, :, :] = torch.tensor(
                np.array(T1[0][idx1: idx1+self.T_len], dtype=np.float32))
            X[1, i, :, :] = torch.tensor(
                np.array(T2[0][idx2: idx2+self.T_len], dtype=np.float32))
            # setting labels as logits, 1 representing trajectory with higher rewards
            Y[0, i] = T1[1] >= T2[1]
            Y[1, i] = T2[1] > T1[1]

        return X, Y


class Reward():
    def __init__(self, state_dim, data_dir='', lr=1e-4, discount=0.99,
                 n_iter=10000, batch_size=64, n_models=5, T_len=50, mode='train') -> None:

        self.lr = lr
        self.discount = discount
        self.n_iter = n_iter
        self.n_models = n_models
        self.T_len = T_len

        # generate 5 reward networks
        self.device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.reward = []
        self.optimizers = []
        for i in range(n_models):
            self.reward.append(Network(state_dim).to(self.device))
            self.optimizers.append(torch.optim.Adam(self.reward[i].parameters(), lr=self.lr))

        if mode == 'train':
            train_data = Data(data_dir, n_models=n_models, T_len=T_len)
            self.train_dataloader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=True)
            self.loss_fn = nn.BCEWithLogitsLoss()

    def learn(self):

        for epochs in range(self.n_iter):
            pbar = tqdm(total=len(self.train_dataloader), desc=f"Epoch: {epochs}/{self.n_iter}")
            for X, Y in self.train_dataloader:
                
                # reshape X and Y
                X = torch.reshape(X, (X.shape[0]*2, X.shape[2], X.shape[3], X.shape[4]))
                Y = torch.reshape(Y, (Y.shape[0]*2, Y.shape[2]))
                # preds = torch.zeros(Y.shape)
                
                # training 5 models one by one
                losses = []
                for i, model in enumerate(self.reward):
                    preds = torch.zeros((Y.shape[0], 1), device=self.device)
                    for j in range(self.T_len):
                        # Pass each of 50 states one by one through models
                        preds += model(X[:, i, j, :].squeeze().to(self.device))
                    loss = self.loss_fn(preds.squeeze(), Y[:, i].to(self.device)) # add loss here
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()
                    losses.append(round(loss.item(), 3))
                pbar.set_postfix_str(f"Loss: {losses}")
                pbar.update()

                






if __name__ == "__main__":
    # train_data = Data("data/Hopper-v4/states_bank")
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_data, batch_size=64, shuffle=True)

    # sample = next(iter(train_dataloader))
    # print(sample)
    # for X, Y in train_dataloader:
    #     print(X)
    #     print(Y)
    reward = Reward(state_dim=11, data_dir="data/Hopper-v4/states_bank")
    reward.learn()

    print("Done")