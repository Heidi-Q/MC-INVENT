from abc import ABC, abstractmethod
from typing import Tuple

import re
import numpy as np
import torch
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import LearningStrategyConfiguration
from collections import deque


class BaseLearningStrategy(ABC):
    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger):
        self.critic_model = critic_model
        self.optimizer = optimizer
        self._configuration = configuration
        self._running_mode_enum = GenerativeModelRegimeEnum()
        self._logger = logger
        self._disable_prior_gradients()
        ### My code
        self.average_score_buffer = deque(maxlen=5)
        self.average_actor_nlls_buffer = deque(maxlen=5)

    def log_message(self, message: str):
        self._logger.log_message(message)

    # ### Raw code
    # TODO: Return the loss as well.
    def run(self, scaffold_batch: np.ndarray, decorator_batch: np.ndarray,
            score: torch.Tensor, actor_nlls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:#, step:int
        loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls = \
            self._calculate_loss(scaffold_batch, decorator_batch, score, actor_nlls)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return negative_actor_nlls, negative_critic_nlls, augmented_nlls

    # ### My code each 5 steps update average
    # def run(self, scaffold_batch: np.ndarray, decorator_batch: np.ndarray,
    #         score: torch.Tensor, actor_nlls: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     # if len(score) != 20:#len(self.average_score_buffer[-1]):
    #     #     score = np.pad(score, (0, 20 - len(score)), mode='constant', constant_values=0)
    #     # if len(actor_nlls) != 20:
    #     #     actor_nlls = torch.nn.functional.pad(actor_nlls, (0, 20 - len(actor_nlls)))
    #     try:
    #         self.average_score_buffer.append(score)
    #         self.average_actor_nlls_buffer.append(actor_nlls)
    #         average_score = np.mean(self.average_score_buffer, axis=0)
    #         recent_actor_nlls = torch.stack(tuple(self.average_actor_nlls_buffer))
    #         average_actor_nlls = torch.mean(recent_actor_nlls, dim=0)
    #         loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls = \
    #             self._calculate_loss(scaffold_batch, decorator_batch, average_score, average_actor_nlls)
    #     except:
    #         loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls = \
    #                     self._calculate_loss(scaffold_batch, decorator_batch, score, actor_nlls)
    #
    #
    #     if step % 5 == 0:
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         self.average_score_buffer.clear()
    #         self.average_actor_nlls_buffer.clear()
    #
    #     return negative_actor_nlls, negative_critic_nlls, augmented_nlls

    ### My code every step update average of 5 steps
    # def run(self, scaffold_batch: np.ndarray, decorator_batch: np.ndarray,
    #         score: torch.Tensor, actor_nlls: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     # if len(score) != 20:#len(self.average_score_buffer[-1]):
    #     #     score = np.pad(score, (0, 20 - len(score)), mode='constant', constant_values=0)
    #     # if len(actor_nlls) != 20:
    #     #     actor_nlls = torch.nn.functional.pad(actor_nlls, (0, 20 - len(actor_nlls)))
    #     try:
    #         self.average_score_buffer.append(score)
    #         self.average_actor_nlls_buffer.append(actor_nlls)
    #         average_score = np.mean(self.average_score_buffer, axis=0)
    #         recent_actor_nlls = torch.stack(tuple(self.average_actor_nlls_buffer))
    #         average_actor_nlls = torch.mean(recent_actor_nlls, dim=0)
    #         loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls = \
    #             self._calculate_loss(scaffold_batch, decorator_batch, average_score, average_actor_nlls)
    #     except:
    #         loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls = \
    #                     self._calculate_loss(scaffold_batch, decorator_batch, score, actor_nlls)
    #
    #
    #     if step % 5 >= 0:
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         self.average_score_buffer.clear()
    #         self.average_actor_nlls_buffer.clear()
    #
    #     return negative_actor_nlls, negative_critic_nlls, augmented_nlls

    @abstractmethod
    def _calculate_loss(self, scaffold_batch, decorator_batch, score, actor_nlls):
        raise NotImplementedError("_calculate_loss method is not implemented")

    def _to_tensor(self, array, use_cuda=True):
        if torch.cuda.is_available() and use_cuda:
            return torch.tensor(array, device=torch.device("cuda"))
        return torch.tensor(array, device=torch.device("cpu"))

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        self.critic_model.set_mode(self._running_mode_enum.INFERENCE)
        for param in self.critic_model.network.parameters():
            param.requires_grad = False
