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
        
        This constructor calls the superclass (DPOTrainer) constructor with all provided keyword arguments,
        and then retrieves the training arguments from the "args" key in kwargs. It extracts the "gamma" attribute
        from these training arguments, which is used to configure the trainer's behavior.
        
        Parameters:
            **kwargs (dict): Arbitrary keyword arguments for trainer initialization. Must include an "args" key
                             whose value is an object containing a "gamma" attribute.
        
        Raises:
            KeyError: If the "args" key is not present in the provided kwargs.
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
        Compute the SimPO loss and associated metrics based on chosen and rejected policy log probabilities.
        
        This method calculates the difference between the chosen and rejected log probabilities (pi_logratios), adjusts it with a gamma factor, and computes the loss using either a sigmoid-based or hinge-based function depending on the configured loss type. In addition to the loss calculations, it derives auxiliary metrics:
        - **uncertainty**: The absolute difference of the log probabilities.
        - **preference_strength**: The sigmoid of the log probability differences, indicating the relative preference strength.
        - **chosen_rewards** and **rejected_rewards**: Scaled versions of the respective policy log probabilities.
        - **final_weights**: Computed by normalizing and scaling the base weights (derived from the product of uncertainty and preference strength) to ensure they sum to 1.
        
        Parameters:
            policy_chosen_logps (torch.FloatTensor): Log probabilities associated with the chosen policy.
            policy_rejected_logps (torch.FloatTensor): Log probabilities associated with the rejected policy.
        
        Returns:
            tuple: A 7-tuple containing:
                - weighted_losses (torch.FloatTensor): The loss values after applying the weight scaling.
                - original_losses (torch.FloatTensor): The raw loss values computed using the selected loss type.
                - final_weights (torch.FloatTensor): The normalized weights derived from uncertainty and preference strength.
                - uncertainty (torch.FloatTensor): The absolute difference between chosen and rejected log probabilities.
                - preference_strength (torch.FloatTensor): The preference strength computed as the sigmoid of the log probability difference.
                - chosen_rewards (torch.FloatTensor): Scaled rewards computed from the chosen policy log probabilities.
                - rejected_rewards (torch.FloatTensor): Scaled rewards computed from the rejected policy log probabilities.
        
        Raises:
            ValueError: If an unknown loss type is specified (expected "sigmoid" or "hinge").
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
        Performs a single forward pass on a concatenated batch of inputs for both chosen and rejected samples.
        
        This method concatenates the inputs (and corresponding labels) for chosen and rejected samples into one batch,
        allowing the model to process them in a single forward pass, which is especially beneficial for FSDP training.
        After obtaining the logits from the model, it computes log probabilities using a custom method and then splits
        the combined results into parts corresponding to the chosen and rejected samples.
        
        Parameters:
            model (nn.Module): The model to run the forward pass.
            batch (Dict[str, Union[List, torch.LongTensor]]): A dictionary containing input data for the batch. It must
                include the key "chosen_labels" to determine the split between chosen and rejected samples, as well as other
                necessary keys for creating the concatenated batch. For encoder-decoder models, additional keys such as
                "concatenated_decoder_input_ids" may be expected.
        
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                A tuple containing:
                    - chosen_logps (torch.FloatTensor): Log probabilities for the chosen samples.
                    - rejected_logps (torch.FloatTensor): Log probabilities for the rejected samples.
                    - chosen_logits (torch.FloatTensor): Model logits for the chosen samples.
                    - rejected_logits (torch.FloatTensor): Model logits for the rejected samples.
                    - all_logps_1 (torch.FloatTensor): First component of computed log probabilities from the entire batch.
                    - all_logps_2 (torch.FloatTensor): Second component of computed log probabilities from the entire batch.
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
        Compute the SimPO loss and aggregate various performance metrics for a given batch during training or evaluation.
        
        This method performs a forward pass on the provided batch using the supplied model via the
        `concatenated_forward` function, and then computes the SimPO loss and additional metrics using
        the `simpo_loss` method. The computed metrics include overall losses, weights, uncertainty,
        reward statistics, log probabilities, and logits for both chosen and rejected policies. A prefix
        ("eval_" if in evaluation mode) is applied to each metric key to distinguish between training and
        evaluation metrics.
        
        Parameters:
            model (nn.Module): The model used to process the input batch.
            batch (Dict[str, Union[List, torch.LongTensor]]): A dictionary containing the inputs required
                by the model. This batch should include all necessary tokens/features for processing.
            train_eval (Literal["train", "eval"], optional): Indicates whether the batch is processed in
                training mode ("train") or evaluation mode ("eval"). Default is "train".
        
        Returns:
            Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
                A tuple where:
                    - The first element is the mean of the weighted SimPO loss for the batch.
                    - The second element is a dictionary containing metrics including:
                        * "<prefix>original_losses": Mean of the original unweighted losses.
                        * "<prefix>weighted_losses": Mean of the weighted losses.
                        * "<prefix>weights": Mean of the normalized weights applied to the losses.
                        * "<prefix>uncertainty": Mean uncertainty computed from the log probabilities.
                        * "<prefix>margin": Mean margin (difference) used in the loss computation.
                        * "<prefix>rewards/chosen": Mean reward for the chosen policy.
                        * "<prefix>rewards/rejected": Mean reward for the rejected policy.
                        * "<prefix>rewards/accuracies": Mean reward accuracy, where accuracy is calculated as
                          the fraction of instances with chosen reward greater than rejected reward.
                        * "<prefix>rewards/margins": Mean difference between chosen and rejected rewards.
                        * "<prefix>logps/chosen": Mean log probability for chosen actions.
                        * "<prefix>logps/rejected": Mean log probability for rejected actions.
                        * "<prefix>logits/chosen": Mean logit for chosen actions.
                        * "<prefix>logits/rejected": Mean logit for rejected actions.
                        * "<prefix>all_logps_1": Mean of the first auxiliary log probabilities tensor.
                        * "<prefix>all_logps_2": Mean of the second auxiliary log probabilities tensor.
                Note: The "<prefix>" is "eval_" when train_eval is "eval", and is an empty string for training.
        
        Usage Example:
            loss, metrics = trainer.get_batch_loss_metrics(model, batch, train_eval="train")
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
