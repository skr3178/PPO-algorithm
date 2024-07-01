# All codes is from cleanRL repo:
# I added my notes from understanding and experimenting

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# class Agent(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#         self.network = nn.Sequential(
#             layer_init(nn.Conv2d(4, 32, 8, stride=4)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(32, 64, 4, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(64, 64, 3, stride=1)),
#             nn.ReLU(),
#             nn.Flatten(),
#             layer_init(nn.Linear(64 * 7 * 7, 512)),
#             nn.ReLU(),
#         )
#         # self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
#         self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
#         self.critic = layer_init(nn.Linear(512, 1), std=1)
#
#
#     def get_value(self, x):
#         return self.critic(self.network(x / 255.0))
#
#     def get_action_and_value(self, x, action=None):
#         hidden = self.network(x / 255.0)
#         logits = self.actor(hidden)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def create_network():
    return nn.Sequential(
        layer_init(nn.Conv2d(4, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )

def create_actor(envs):
    return layer_init(nn.Linear(512, envs.action_space.n), std=0.01)

def create_critic():
    return layer_init(nn.Linear(512, 1), std=1)

def get_value(network, critic, x):
    return critic(network(x / 255.0))

def get_action_and_value(network, actor, critic, x, action=None):
    hidden = network(x / 255.0)
    logits = actor(hidden)
    probs = Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), critic(hidden)

# Usage example:
# Assuming `envs` is available in the context

# Create a random input tensor
# x = torch.randn(1, 4, 84, 84)


# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--exp_name", type=str, default="ppo_atari")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--track", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="cleanRL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--capture_video", action="store_true", default=False)
    parser.add_argument("--env_id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total_timesteps", type=int, default=10000000) #10)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128) #5)
    parser.add_argument("--anneal_lr", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--norm_adv", action="store_true", default=True)
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--clip_vloss", action="store_true", default=True)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--minibatch_size", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=0)
    args, _ = parser.parse_known_args()
    return vars(args)

args2 = parse_args()

class ArgsObject:
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)


# Create an object with attributes from the args dictionary
args = ArgsObject(args2)

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk



env = gym.make('CartPole-v1')


args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
)
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        # self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)


    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


# agent = Agent(envs).to(device)
agent = Agent(env).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# ALGO Logic: Storage setup
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset(seed=args.seed)
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(args.num_envs).to(device)


# for iteration in range(1, args.num_iterations + 1): #9765
for iteration in range(1, 1 + 1):  # 9765
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        # print("iteration", iteration)
        frac = 1.0 - (iteration - 1.0) / args.num_iterations  # ranges from 1, 0.5, ... 0.00010240655401949628
        # print("frac", frac)
        lrnow = frac * args.learning_rate
        # print("lrnow", lrnow)  # ranges from 0.00025 ... 2.560163850487407e-08
        optimizer.param_groups[0]["lr"] = lrnow

    # for step in range(0, args.num_steps): #128
    for step in range(0, 1):  # 128
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)


    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)): #127, 126, 125, 124, 123, 122, 121, 120. .... 0
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
                # nextnonterminal tensor([1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0') nextvalues tensor([[-0.0006, -0.0033, -0.0006, -0.0006, -0.0006, -0.0006, -0.0098, -0.0006]],
                #        device='cuda:0')
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
                # all values: nextnonterminal tensor([1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0') nextvalues tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0')


            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t] #temporal difference error # tensor([-0.0033,  0.0071, -0.0033, -0.0033, -0.0033, -0.0033, -0.0007, -0.0033],
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) # reshape to (-1, 4, 84, 84)
    b_logprobs = logprobs.reshape(-1) # torch.Size([1024])||torch.Size([128, 8])
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape) # torch.Size([1024])||torch.Size([128, 8]) ||
    b_advantages = advantages.reshape(-1) # torch.Size([1024])||torch.Size([128, 8])
    b_returns = returns.reshape(-1) # torch.Size([1024])||torch.Size([128, 8])
    b_values = values.reshape(-1) # torch.Size([1024])||torch.Size([128, 8])

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size) # array([   0,    1,    2, ..., 1021, 1022, 1023])
    clipfracs = []
    # for epoch in range(args.update_epochs): #4
    for epoch in range(1):
        np.random.shuffle(b_inds) # is this shuffling same as randomness
        for start in range(0, args.batch_size, args.minibatch_size): # 0, 256, 512, 768
            end = start + args.minibatch_size # [start, end: 0, 256; 256, 512; 512, 768; 768, 1024]
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds]) # torch.Size([256, 4, 84, 84]), torch.Size([256])

            # action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) # clamp pushes or forces the elements within the tesnor bet 0.9/-0.9. helps to stabilize the training process.
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() #  The value loss is calculated using the mean squared error between the predicted values and the actual returns.

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

envs.close()
writer.close()