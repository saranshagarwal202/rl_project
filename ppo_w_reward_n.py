import torch
from torch.optim import Adam
from network import PolicyNetwork, ValueNetwork
from reward_train import Reward
import argparse
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import json
import os


class PPO():
    def __init__(self, env, lr, discount, clip, time_steps_per_batch, time_steps_per_trajectory,
                 total_timesteps, learning_iterations) -> None:
        """
        Parameters:
            env: Gymnasium environment
            lr: Learning rate to be used in optimizer
            time_steps_per_batch: total number of timesteps produced by multiple trajectories in a batch
            time_steps_per_trajectory: Max timesteps to run a trajectory (To insure we are not stuck in a infinite trajectory).
            total_timesteps: Total timesteps to train the model
            learning_iterations: number of iterations to take for each batch of data.
        """

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.time_steps_per_batch = time_steps_per_batch
        self.time_steps_per_trajectory = time_steps_per_trajectory
        self.total_timesteps = total_timesteps
        self.discount = discount
        self.learning_iterations = learning_iterations
        self.clip = clip

        # 1) Initialize policy network and value network
        self.policy = PolicyNetwork(self.obs_dim, self.act_dim)
        self.value = ValueNetwork(self.obs_dim, 1)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.value_optim = Adam(self.value.parameters(), lr=lr)

        self.device = "cpu"
        self.acc_device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.reward_net = Reward(self.obs_dim, env=self.env.spec.id, mode='test') 

    def collect_trajectories(self):
        states = []
        actions = []
        log_probabilities = []
        rewards = []
        mean_rewards = []
        traj_length = []
        t = 0
        while t < self.time_steps_per_batch:
            state, _ = self.env.reset()
            done = False
            ep_states = []
            ep_rewards = []
            for episode_t in range(self.time_steps_per_trajectory):
                ep_states.append(state)

                act, log_prob = self.policy.get_action(
                    torch.tensor(state, dtype=torch.float32))
                state, rew, done, _, _ = self.env.step(act.numpy())

                ep_rewards.append(rew)
                actions.append(act.detach().numpy())
                log_probabilities.append(log_prob.detach().cpu().numpy())
                t += 1

                if done:
                    break

            states.extend(ep_states)
            traj_length.append(episode_t + 1)
            t_r = self.reward_net.get_reward(torch.tensor(ep_states, dtype=torch.float32, device=self.acc_device))
            rewards.append(t_r + torch.tensor(ep_rewards, dtype=torch.float32).unsqueeze(1))
            mean_rewards.append(sum(ep_rewards))
        return torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32), \
               torch.tensor(np.array(actions)), \
               torch.tensor(np.array(log_probabilities, dtype=np.float32)), \
               rewards, \
               torch.tensor(np.array(traj_length, dtype=np.int32)), \
               sum(mean_rewards) / len(mean_rewards)

    def calculate_advantage(self, states, rewards: list):

        V = self.value(states).squeeze()

        # Calculating rewards to go
        Gt = torch.zeros(len(states))
        pos = 0
        for i, T_rewards in enumerate(rewards):
            discounts = torch.zeros(
                len(T_rewards), device=self.device).fill_(self.discount)
            discount_power = torch.linspace(
                start=0, end=len(T_rewards),
                steps=len(T_rewards), dtype=torch.int32, device=self.device)
            discounts = discounts ** discount_power

            for j in range(len(T_rewards)):
                Gt[pos] = (T_rewards[j:] * discounts[:len(T_rewards) - j]).sum()
                pos += 1

        return Gt - V.detach(), Gt

    def learn(self):
        t = 0
        prev_mean_reward = 0
        pbar = tqdm(total=self.total_timesteps)
        while t < self.total_timesteps:
            states, actions, log_prob, rewards, T_lengths, mean_reward = self.collect_trajectories()

            t += T_lengths.sum()

            self.save_rewards(rewards, T_lengths, t,
                              mean_reward, prev_mean_reward)

            At, Gt = self.calculate_advantage(states, rewards)
            At = (At - At.mean()) / (At.std() + 1e-10)

            # Update policy by PPO-clip objective
            for i in range(self.learning_iterations):
                V = self.value(states).squeeze()
                curr_log_prob = self.policy.get_prob_action(states, actions)

                ratios = torch.exp(curr_log_prob - log_prob)
                surr1 = ratios * At
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * At

                policy_loss = (-torch.min(surr1, surr2)).mean()
                value_loss = torch.nn.MSELoss()(V, Gt)

                self.policy_optim.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.policy_optim.step()

                self.value_optim.zero_grad()
                value_loss.backward()
                self.value_optim.step()

            pbar.set_postfix_str(
                f"Mean reward: {round(mean_reward, 3)}, Mean Gt:{Gt.mean() / T_lengths.shape[0]}, "
                f"Avg Episodic Length: {round((T_lengths.sum() / T_lengths.shape[0]).item())}")
            pbar.update(n=int(T_lengths.sum()))
            prev_mean_reward = mean_reward
        pbar.close()

    def save_rewards(self, rewards, T_lengths, curr_t, curr_mean_r, prev_mean_r):
        # Save rewards
        try:
            data = {}
            for i, l in enumerate(T_lengths):
                data[(l + curr_t).item()] = rewards[i].sum().item()
            f = open(f"data/{self.env.spec.id}/ppo_with_reward_fn/rewards/{curr_t.item()}.json", 'w')
            f.write(json.dumps(data))
            f.close()

        except FileNotFoundError:
            os.makedirs(f"data/{self.env.spec.id}/ppo_with_reward_fn/rewards")

        # Save model
        if curr_mean_r > prev_mean_r:
            torch.save(self.policy.state_dict(),
                       f"data/{self.env.spec.id}/ppo_with_reward_fn/policy.pt")
            torch.save(self.value.state_dict(),
                       f"data/{self.env.spec.id}/ppo_with_reward_fn/value.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='Hopper-v4')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--time-steps-per-batch", default=4096, type=int)
    parser.add_argument("--time-steps-per-trajectory", default=400, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--total-timesteps", default=500_000, type=int)
    parser.add_argument("--learning-iterations", default=10, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--clip", default=0.2, type=str)
    parser.add_argument("--checkpoint-n", default=5, type=int)
    parser.add_argument("--checkpoint-t-len", default=1000, type=int)
    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--model-path", default="", type=str)
    args = parser.parse_args()

    if args.mode == 'train':
        # Make the environment
        env = gym.make(args.env)
        if os.path.exists(f"data/{env.spec.id}/ppo_with_reward_fn/"):
            import shutil
            shutil.rmtree(f"data/{env.spec.id}/ppo_with_reward_fn/")
            del shutil

        # Setting seeds
        torch.manual_seed(args.seed)
        kwargs = {
            "env": env,
            "lr": args.lr,
            "discount": args.discount,
            "clip": args.clip,
            "time_steps_per_batch": args.time_steps_per_batch,
            "time_steps_per_trajectory": args.time_steps_per_trajectory,
            "total_timesteps": args.total_timesteps,
            "learning_iterations": args.learning_iterations
        }

        agent = PPO(**kwargs)
        agent.learn()

        agent.policy.load_state_dict(torch.load(f"data/{args.env}/ppo_with_reward_fn/policy.pt"))
    else:
        env = gym.make(args.env, render_mode="human")
        kwargs = {
            "env": env,
            "lr": args.lr,
            "discount": args.discount,
            "clip": args.clip,
            "time_steps_per_batch": args.time_steps_per_batch,
            "time_steps_per_trajectory": args.time_steps_per_trajectory,
            "total_timesteps": args.total_timesteps,
            "learning_iterations": args.learning_iterations
        }
        agent = PPO(**kwargs)
        agent.policy.load_state_dict(torch.load(args.model_path))

    env = gym.make(args.env, render_mode="human")
    state, _ = env.reset(seed=args.seed)
    for t in range(1000):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = agent.policy(state)[0].detach().numpy()
        n_state, reward, terminated, truncated, _ = env.step(action)
        print(reward)
        done = terminated or truncated
        state = n_state
        if done:
            break
