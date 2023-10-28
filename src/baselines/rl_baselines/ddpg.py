import argparse
import datetime
import os
import pprint
from src.IVRECore import IVRE
from src.IVRE import SymbolicIVRE
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Interactive_Blicket_DDPG")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int,
                        nargs="*", default=[512, 512, 512])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=320)
    parser.add_argument("--update-per-step", type=int, default=0.01)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--training-num", type=int, default=32)
    parser.add_argument("--test-num", type=int, default=1000)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="test the performance",
    )
    return parser.parse_args()


def train_ddpg(args=get_args()):
    env = SymbolicIVRE(IVRE())
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    args.noise_clip = args.noise_clip * args.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))

    if args.training_num > 1:
        train_envs = SubprocVectorEnv(
            [lambda: SymbolicIVRE(IVRE()) for _ in range(args.training_num)]
        )
    else:
        train_envs = SymbolicIVRE(IVRE())

    test_envs = SymbolicIVRE(IVRE())

    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)
    # model
    net_a = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = Actor(
        net_a, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        estimation_step=args.n_step,
        action_space=env.action_space,
    )
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(
            args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(
        policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_ddpg'
    log_path = os.path.join(args.logdir, args.task, "ddpg", log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        save_fn=save_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    )
    pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(
        n_episode=args.test_num, render=args.render)
    print(
        f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


def test_ddpg(args=get_args()):
    env = SymbolicIVRE(IVRE())
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    args.noise_clip = args.noise_clip * args.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))

    test_envs = SymbolicIVRE(IVRE())
    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)
    # model
    net_a = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = Actor(
        net_a, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(
            args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # collector

    # Let's watch its performance!
    policy.eval()
    test_collector = Collector(policy, test_envs)
    for i in range(1):
        test_collector.reset()

        result = test_collector.collect(n_episode=10000, render=args.render)
        # print(result["lens"])
        # x = [x for x in range(1, 11)]
        # sns.barplot(x="day", y="total_bill", data=list(map(lambda x: int(x), result["lens"])))
        lens_true = []
        for i in range(len(result["rews"])):
            if result["rews"][i] > 0:
                lens_true.append(result["lens"][i])
        # sns.set_theme(style="darkgrid")
        # sns.histplot(data=lens_true, bins=10, binwidth=0.9, discrete=True, stat="percent")
        # plt.xticks(list(range(11)))
        # plt.xlabel("Steps")
        # plt.savefig("./test.pdf")
        # print(lens_true)
        # from collections import Counter
        # print(Counter(lens_true))
        print(
            f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean(),}, accuracy: {(result["rews"] > 0).mean()}'
        )


if __name__ == "__main__":
    args = get_args()
    if args.test:
        test_ddpg(args)
    else:
        train_ddpg(args)
