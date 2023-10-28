import argparse
import datetime
import os
import torch
from torch import nn
from src.IVRECore import IVRE, SymbolicIVRE
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy, DDPGPolicy
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import Net, Recurrent, MLP
from tianshou.utils.net.continuous import Actor
import torchvision.models as models
import torchvision.transforms as transforms
from einops import rearrange, repeat

ModuleType = Type[nn.Module]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="BlicketVisualRNN")
    # parser.add_argument('--model', type=str, required=True)
    parser.add_argument("--model", type=str, default="DDPG")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--stack-num", type=int, default=10)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=160)
    parser.add_argument("--update-per-step", type=int, default=0.00625)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


class CriticStack(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, 1, hidden_sizes, device=self.device)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        s = torch.as_tensor(
            s,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )
        if a is not None:
            a = torch.as_tensor(
                a,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        logits, h = self.preprocess(s, a)
        logits = self.last(logits)
        return logits


class ConvNet_RNN(nn.Module):
    def __init__(
        self,
        policyNet,
        device: Optional[Union[str, int, torch.device]] = None,
        critic=False,
    ) -> None:
        super().__init__()
        self.input_dim = 0
        self.critic = critic
        self.device = device
        self.output_dim = 512
        model = models.resnet18(pretrained=True)
        for para in model.parameters():
            para.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=48, bias=True), nn.Sigmoid()
        )

        self.model = model.to(self.device)
        self.policyNet = policyNet
        self.normalize = torch.nn.Sequential(
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def preproces(self, img):
        img = self.normalize(img)
        return img

    def forward(self, s, a=None, info=0) -> torch.Tensor:
        # img.shape = batch, 10, w, h, 3
        # belief.shape = batch, 48
        if self.device:
            s = torch.tensor(s, device=self.device, dtype=torch.float32)

        """Mapping: s -> flatten -> logits.

        In the evaluation mode, s should be with shape ``[bsz, dim]``; in the
        training mode, s should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """

        if len(s.shape) == 2:
            s = s.unsqueeze(1)

        batch_size = s.shape[0]
        s_len = s.shape[1]
        size = 10 * 120 * 160 * 3

        if self.critic:
            a = repeat(a, "b d -> b s d", s=s_len)
            s = torch.cat((s, a), dim=2)

        img, belief = s[:, :, :size], s[:, :, size:]
        img = rearrange(
            img, "b len (p c w h) -> (b len p) c w h", c=3, w=120, h=160, p=10
        )
        img = self.preproces(img)
        embedding = self.model(img)
        embedding = rearrange(
            embedding, "(b len p) c -> b len (p c)", b=batch_size, len=s_len
        )

        observation = torch.cat((embedding, belief), dim=2)
        print(observation.shape)
        action = self.policyNet(observation)

        return action  # type: ignore


def test_ddpg(args=get_args()):
    env = SymbolicIVRE(IVRE())
    args.state_shape = 48 * 10 + 48
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

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
    lstm_a = Recurrent(
        layer_num=args.layer_num,
        state_shape=args.state_shape,
        action_shape=512,
        hidden_layer_size=args.hidden_size,
        device=args.device,
    )
    net_a = ConvNet_RNN(lstm_a, device=args.device)
    # net_a = ConvNet_RNN(w, h, c, args.state_shape, args.action_shape, device=args.device)
    actor = Actor(
        net_a, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    lstm_c = Recurrent(
        layer_num=args.layer_num,
        state_shape=np.prod(args.state_shape) + np.prod(args.action_shape),
        action_shape=512,
        hidden_layer_size=args.hidden_size,
        device=args.device,
    )

    net_c = ConvNet_RNN(lstm_c, device=args.device, critic=True)
    critic = CriticStack(net_c, device=args.device).to(args.device)
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
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    test_collector = Collector(policy, test_envs)
    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    for i in range(10):
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        print(
            f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean(),}, accuracy: {(result["rews"] > 0).mean()}'
        )


def test_td3(args=get_args()):
    env = SymbolicIVRE(IVRE())
    args.state_shape = 48 * 10 + 48
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    args.policy_noise = args.policy_noise * args.max_action
    args.noise_clip = args.noise_clip * args.max_action

    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

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
    lstm_a = Recurrent(
        layer_num=args.layer_num,
        state_shape=args.state_shape,
        action_shape=512,
        hidden_layer_size=args.hidden_size,
        device=args.device,
    )
    net_a = ConvNet_RNN(lstm_a, device=args.device)
    # net_a = ConvNet_RNN(w, h, c, args.state_shape, args.action_shape, device=args.device)
    actor = Actor(
        net_a, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    # mlp_c1 = Net(
    #     args.state_shape,
    #     args.action_shape,
    #     hidden_sizes=args.hidden_sizes,
    #     concat=True,
    #     device=args.device
    # )
    # mlp_c2 = Net(
    #     args.state_shape,
    #     args.action_shape,
    #     hidden_sizes=args.hidden_sizes,
    #     concat=True,
    #     device=args.device
    # )

    # net_c1 = ConvNet_RNN(mlp_c1, device=args.device)
    # critic1 = Critic(net_c1, device=args.device).to(args.device)
    lstm_c1 = Recurrent(
        layer_num=args.layer_num,
        state_shape=np.prod(args.state_shape) + np.prod(args.action_shape),
        action_shape=512,
        hidden_layer_size=args.hidden_size,
        device=args.device,
    )

    net_c1 = ConvNet_RNN(lstm_c1, device=args.device, critic=True)
    critic1 = CriticStack(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    lstm_c2 = Recurrent(
        layer_num=args.layer_num,
        state_shape=np.prod(args.state_shape) + np.prod(args.action_shape),
        action_shape=512,
        hidden_layer_size=args.hidden_size,
        device=args.device,
    )

    net_c2 = ConvNet_RNN(lstm_c2, device=args.device, critic=True)
    critic2 = CriticStack(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    # net_c2 = ConvNet_RNN(mlp_c2, device=args.device)
    # critic2 = Critic(net_c2, device=args.device).to(args.device)

    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    test_collector = Collector(policy, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_td3'
    log_path = os.path.join(args.logdir, args.task, "td3", log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    for i in range(10):
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        print(
            f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean(),}, accuracy: {(result["rews"] > 0).mean()}'
        )


if __name__ == "__main__":
    args = get_args()
    if args.model == "TD3":
        test_td3(args)
    elif args.model == "DDPG":
        test_ddpg(args)
    else:
        raise NotImplementedError
    # args = get_args()
    # env = SymbolicIVRE(IVRE())
    # args.state_shape = 48*10 + 48
    # args.action_shape = env.action_space.shape
    # args.max_action = env.action_space.high[0]
    # args.exploration_noise = args.exploration_noise * args.max_action
    # print("Observations shape:", args.state_shape)
    # print("Actions shape:", args.action_shape)
    # print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # train_envs = SymbolicIVRE(IVRE())
    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)
    # model
    # mlp = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device).to(args.device)
    # net_a = ConvNet_RNN(mlp, device=args.device).to(args.device)
    # # net_a = ConvNet_RNN(w, h, c, args.state_shape, args.action_shape, device=args.device)
    # c = torch.rand((16, 10*120*160*3+48))
