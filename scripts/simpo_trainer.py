# simpo_trainer.py
from trl import DPOTrainer
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn


class SimPOTrainer(DPOTrainer):

    def __init__(self, **kwargs):
        """
        Initialize a SimPOTrainer instance.
        
        This constructor initializes the SimPOTrainer by passing all keyword arguments to its superclass and retrieving
        the training arguments from the provided kwargs. It extracts the 'gamma' parameter from the training arguments,
        which is used for scaling loss computations during the training process.
        
        Parameters:
            **kwargs (dict): A dictionary of keyword arguments that must include the key 'args'. The value associated with
                'args' should be an object containing a 'gamma' attribute.
        """
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
        """
        Compute the SimPO loss and related metrics for chosen and rejected policy log-probabilities.
        
        This method computes the loss for a training step by processing the log-probabilities of chosen
        and rejected policies. It first calculates the log ratio between the chosen and rejected log-probabilities,
        adjusts it using a gamma offset, and then applies the specified loss function (either "sigmoid" or "hinge")
        to compute the original loss. In addition, it computes rewards for both chosen and rejected policies, measures
        uncertainty as the absolute value of the log ratio, and calculates preference strength using the sigmoid of
        the log ratio. Base weights are then derived from the uncertainty and preference strength, mapped to a range
        of [-1, 1], scaled by a small variance, and normalized so that their sum equals 1. The final weighted loss is
        obtained by applying these weights to the original loss.
        
        Parameters:
            policy_chosen_logps (torch.FloatTensor): Log-probabilities of the chosen policies.
            policy_rejected_logps (torch.FloatTensor): Log-probabilities of the rejected policies.
        
        Returns:
            tuple: A tuple containing:
                - weighted_losses (torch.FloatTensor): Losses after applying the computed weights.
                - original_losses (torch.FloatTensor): Losses computed using the selected loss function ("sigmoid" or "hinge").
                - final_weights (torch.FloatTensor): Normalized weights mapped to the range [-1, 1] and scaled by a variance factor.
                - uncertainty (torch.FloatTensor): The absolute difference between chosen and rejected policy log-probabilities.
                - preference_strength (torch.FloatTensor): The preference strength computed as the sigmoid of the log ratio.
                - chosen_rewards (torch.FloatTensor): Rewards computed for the chosen policies.
                - rejected_rewards (torch.FloatTensor): Rewards computed for the rejected policies.
        
        Raises:
            ValueError: If the loss type specified in self.loss_type is neither "sigmoid" nor "hinge".
        """
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
        """
        Perform a single forward pass by concatenating chosen and rejected inputs.
        
        This method first concatenates inputs from both chosen and rejected examples using
        the `concatenated_inputs` helper, which also adapts the inputs for encoder-decoder models
        (if applicable). It then performs a single forward pass through the given model to obtain
        logits. Log probabilities are computed via `get_batch_logps`, and the resulting tensors are
        split into components corresponding to chosen and rejected examples. This approach avoids
        multiple forward passes, offering performance benefits with Fully Sharded Data Parallel (FSDP).
        
        Parameters:
            model (nn.Module): The model to evaluate.
            batch (Dict[str, Union[List, torch.LongTensor]]): A dictionary containing the batch data.
                It must include:
                    - "chosen_labels": Tensor used to determine the split between chosen and rejected examples.
                    - Other keys required by `self.concatenated_inputs`, such as "concatenated_input_ids",
                      "concatenated_attention_mask", "concatenated_labels", and for encoder-decoder models,
                      optionally "concatenated_decoder_input_ids".
        
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                A tuple containing:
                    - chosen_logps (torch.FloatTensor): Log probabilities for chosen examples.
                    - rejected_logps (torch.FloatTensor): Log probabilities for rejected examples.
                    - chosen_logits (torch.FloatTensor): Logits corresponding to chosen examples.
                    - rejected_logits (torch.FloatTensor): Logits corresponding to rejected examples.
                    - all_logps_1 (torch.FloatTensor): The first component tensor from the log probability calculation.
                    - all_logps_2 (torch.FloatTensor): The second component tensor from the log probability calculation.
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
        """
        Compute loss and performance metrics for a given batch during training or evaluation.
        
        This method performs a forward pass on the batch by concatenating inputs for both chosen and rejected policies
        using the `concatenated_forward` call, then computes various loss components and reward-based metrics via
        the `simpo_loss` method. It aggregates the mean values of the computed metrics (e.g., losses, weights, uncertainty,
        margins, rewards, log probabilities, and logits) and prepends a prefix ("eval_") to metric keys when in evaluation mode.
        
        Parameters:
            model (nn.Module): The neural network model used to process the batch.
            batch (Dict[str, Union[List, torch.LongTensor]]): A dictionary containing the batch data required for the forward pass.
            train_eval (Literal["train", "eval"], optional): Mode indicator. Set to "train" for training and "eval" for evaluation.
                Keys in the returned metrics dict will be prefixed with "eval_" when in evaluation mode. Defaults to "train".
        
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - A torch.Tensor representing the mean of the weighted losses for the batch.
                - A dictionary mapping metric names (with optional prefix) to their corresponding mean values (all computed on CPU):
                    * "{prefix}original_losses": Mean of the original losses from the loss computation.
                    * "{prefix}weighted_losses": Mean of the weighted losses.
                    * "{prefix}weights": Mean of the final computed weights.
                    * "{prefix}uncertainty": Mean uncertainty derived from the loss computation.
                    * "{prefix}margin": Mean preference margin.
                    * "{prefix}rewards/chosen": Mean reward for chosen policies.
                    * "{prefix}rewards/rejected": Mean reward for rejected policies.
                    * "{prefix}rewards/accuracies": Mean binary accuracy where rewards from chosen policies exceed those from rejected ones.
                    * "{prefix}rewards/margins": Mean difference between chosen and rejected rewards.
                    * "{prefix}logps/rejected": Mean log probability for rejected policies.
                    * "{prefix}logps/chosen": Mean log probability for chosen policies.
                    * "{prefix}logits/rejected": Mean logits for rejected policies.
                    * "{prefix}logits/chosen": Mean logits for chosen policies.
                    * "{prefix}all_logps_1": Mean value of the first additional log probability tensor.
                    * "{prefix}all_logps_2": Mean value of the second additional log probability tensor.
        
        Examples:
            >>> mean_loss, metrics = trainer.get_batch_loss_metrics(model, batch, "train")
            >>> print("Mean Loss:", mean_loss)
            >>> for key, value in metrics.items():
            ...     print(f"{key}: {value}")
        """
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
