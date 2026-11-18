from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import nn


class BaseRewardLoss(nn.Module, ABC):
    def __init__(self, name: str, weighting: float):
        super().__init__()
        self.name = name
        self.weighting = weighting

    @staticmethod
    def freeze_parameters(parameters: Iterable[torch.nn.Parameter]) -> None:
        for parameter in parameters:
            parameter.requires_grad_(False)

    @abstractmethod
    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_text_features(self, prompt: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        image_features = self.get_image_features(image)
        text_features = self.get_text_features(prompt)
        return self.weighting * self.compute_loss(image_features, text_features)