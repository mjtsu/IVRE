from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import Net
from einops import rearrange
import torchvision.models as models
import torchvision.transforms as transforms
from src.const import (
    IMG_HEIGHT,
    IMG_WIDTH,
    MAX_TRIAL,
    MAX_TRIAL,
    UNIQUE_OBJ_CNT,
    OBJ_MAX_BLICKET
)
from einops import rearrange

ModuleType = Type[nn.Module]

class ProcessLSTM(nn.Module):
    def __init__(
        self,
        layer_num: int,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]],
        hidden_sizes: Union[int, Sequence[int]],
        device: Union[str, int, torch.device] = "cpu",
        concat: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_sizes[0],
            hidden_size=hidden_sizes[0],
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc1 = nn.Linear(UNIQUE_OBJ_CNT, hidden_sizes[0])
        self.net = Net(
            state_shape=OBJ_MAX_BLICKET + hidden_sizes[0],
            action_shape=action_shape,
            hidden_sizes=hidden_sizes[1:],
            device=device,
            concat=concat,
        )
        self.concat = concat
        self.output_dim = self.net.output_dim

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: s -> flatten -> logits.
        In the evaluation mode, s should be with shape ``[bsz, dim]``; in the
        training mode, s should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """
        blicket_num = s[:, :OBJ_MAX_BLICKET]
        s = s[:, OBJ_MAX_BLICKET:]
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)
        blicket_num = torch.as_tensor(
            blicket_num, device=self.device, dtype=torch.float32
        )

        if not self.concat:
            s = rearrange(s, "b (len d) -> b len d", len=(MAX_TRIAL + 1))
            history, last_belief = s[:, :-1, :], s[:, -1, :]
            history = self.fc1(history)
            self.nn.flatten_parameters()
            _, (hidden, _) = self.nn(history)
            hidden = hidden[-1]
            embedding = torch.cat((hidden, blicket_num), dim=-1)

        else:
            obs = s[:, : (MAX_TRIAL + 1) * UNIQUE_OBJ_CNT]
            action = s[:, (MAX_TRIAL + 1) * UNIQUE_OBJ_CNT:]
            obs = rearrange(obs, "b (len d) -> b len d", len=(MAX_TRIAL + 1))
            history, last_belief = obs[:, :-1, :], obs[:, -1, :]
            history = self.fc1(history)
            self.nn.flatten_parameters()
            _, (hidden, _) = self.nn(history)
            hidden = hidden[-1]
            embedding = torch.cat((hidden, blicket_num, action), dim=-1)

        logits, state = self.net(embedding, state)
        return logits, state


class ConvNet(nn.Module):
    def __init__(
        self, policyNet, device: Optional[Union[str, int, torch.device]] = None
    ) -> None:
        super().__init__()
        self.input_dim = 0
        self.device = device
        self.output_dim = 512
        model = models.resnet18(pretrained=True)

        model.conv1 = nn.Conv2d(
            3 * 19, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        model.fc = nn.Identity()

        self.model = model.to(self.device)
        self.policyNet = policyNet
        self.normalize = torch.nn.Sequential(
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def preproces(self, img):
        img = img / 255.0
        img = self.normalize(img)
        return img

    def forward(self, s, info=0) -> torch.Tensor:
        # img.shape = batch, 10, w, h, 3
        # belief.shape = batch, 48
        if self.device:
            s = torch.tensor(s, device=self.device, dtype=torch.float32)

        batch_size = s.shape[0]
        size = (MAX_TRIAL + UNIQUE_OBJ_CNT - 1) * IMG_HEIGHT * IMG_WIDTH * 3

        img, info = s[:, :size], s[:, size:]
        img = rearrange(
            img, "b (p h w c) -> (b p) c h w", c=3, h=IMG_HEIGHT, w=IMG_WIDTH, p=19
        )

        img = self.preproces(img)

        img = rearrange(img, "(b p) c h w -> b (p c) h w",
                        c=3, h=224, w=224, p=19)

        embedding = self.model(img)
        embedding = rearrange(embedding, "(b d) c -> b (d c)", b=batch_size)
        observation = torch.cat((embedding, info), dim=1)

        action = self.policyNet(observation)
        return action  # type: ignore
