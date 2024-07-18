# simpo_trainer.py
from trl import DPOTrainer
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn


class SimPOTrainer(DPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Pass all other arguments using **kwargs
        training_args = kwargs["args"]
        self.gamma = training_args.gamma

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - gamma_logratios

        if self.loss_type == "sigmoid":
            original_losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            original_losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = (
            self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        )
        rejected_rewards = (
            self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        )

        # Calculate uncertainty
        uncertainty = torch.abs(pi_logratios)

        # Calculate preference strength
        preference_strength = torch.sigmoid(pi_logratios)

        # Compute base weights
        base_weights = uncertainty * preference_strength

        # Map base weights to [-1, 1] range
        min_weight, max_weight = base_weights.min(), base_weights.max()
        normalized_weights = (
            2 * (base_weights - min_weight) / (max_weight - min_weight) - 1
        )

        # Define variance (you can adjust this value)
        variance = 1e-20

        # Scale normalized weights by variance and add 1
        final_weights = 1 + normalized_weights * variance

        # Normalize final weights to sum to 1
        final_weights = final_weights / final_weights.sum()

        # Apply weights to losses
        weighted_losses = original_losses * 1

        return (
            weighted_losses,
            original_losses,
            final_weights,
            uncertainty,
            preference_strength,
            chosen_rewards,
            rejected_rewards,
        )

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps_1, all_logps_2 = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        all_logps = all_logps_1 / all_logps_2

        # all_logps_1 = all_logps_1.float()
        # all_logps_2 = all_logps_2.float()
        # all_logps = all_logps.float()

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        # print(f"all_logps {all_logps}")
        # print(f"type(all_logps) {type(all_logps)}")

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            all_logps_1,
            all_logps_2,
        )

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            all_logps_1,
            all_logps_2,
        ) = self.concatenated_forward(model, batch)

        (
            weighted_losses,
            original_losses,
            weights,
            uncertainty,
            margin,
            chosen_rewards,
            rejected_rewards,
        ) = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}original_losses"] = original_losses.mean().cpu()
        metrics[f"{prefix}weighted_losses"] = weighted_losses.mean().cpu()
        metrics[f"{prefix}weights"] = weights.mean().cpu()
        metrics[f"{prefix}uncertainty"] = uncertainty.mean().cpu()
        metrics[f"{prefix}margin"] = margin.mean().cpu()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards - rejected_rewards).mean().cpu()
        )

        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = (
            policy_rejected_logits.detach().mean().cpu()
        )
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        metrics[f"{prefix}all_logps_1"] = all_logps_1.detach().float().mean().cpu()
        metrics[f"{prefix}all_logps_2"] = all_logps_2.detach().float().mean().cpu()

        return weighted_losses.mean(), metrics
