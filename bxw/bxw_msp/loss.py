import mindspore
from mindspore import ops, nn, Tensor

def _log_prob(value, mean=None, sd=None):
        """
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            mean (Tensor): The mean of the distribution. Default: self.mean.
            sd (Tensor): The standard deviation of the distribution. Default: self.sd.

        Returns:
            Tensor: The log probability of the value.
        """

        # Compute the log probability
        unnormalized_log_prob = -0.5 * (ops.sqrt((value - mean) / sd))
        coff = -0.5 * ops.Log()(2 * ops.Pi())
        neg_normalization = coff - ops.Log(sd)
        return unnormalized_log_prob + neg_normalization


class SAMSVAE_ELBOLossModule():
    def __init__(
        self,
        local_variables=["z_basal"],
        perturbation_plated_variables=["E", "mask"],
    ):
        super().__init__()
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )
        variables = local_variables + perturbation_plated_variables
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables

    def loss_fn(self, guide_dists, model_dists, model_samples, D, X, D_obs_counts):
        # print(D.requires_grad,D_obs_counts.requires_grad, X.requires_grad)  False False False
        loss_terms = {}
        loss_terms["reconstruction"] = model_dists["p_x"].log_prob(X).sum(-1)
        for k in guide_dists.keys():
            var_key = k[2:]  # drop 'q_'(去掉前缀q_，得到Z_basal、E、mask)
            loss_term = model_dists[f"p_{var_key}"].log_prob(model_samples[var_key])
            loss_term = loss_term - guide_dists[f"q_{var_key}"].log_prob(
                model_samples[var_key]
            )
            if var_key in self.perturbation_plated_variables:   # (如果是E、mask，则重加权)
                loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                    D, D_obs_counts, loss_term
                )
            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term
        batch_elbo = sum([v for k, v in loss_terms.items()])
        loss = -batch_elbo.mean()
        metrics = {
            f"loss_term_{k}": -v.mean() for k, v in loss_terms.items()
        }
        return loss, metrics

    def _compute_reweighted_perturbation_plated_loss_term(
        self, conditioning_variable, total_obs_per_condition, loss_term
    ):
        condition_nonzero = (conditioning_variable != 0).type(mindspore.float32)
        obs_scaling = 1 / total_obs_per_condition
        obs_scaling[ops.isinf(Tensor(obs_scaling,mindspore.float64))] = 0
        obs_scaling = obs_scaling.reshape(1, -1)
        rw_condition_nonzero = condition_nonzero * obs_scaling
        rw_loss_term = ops.matmul(rw_condition_nonzero, loss_term)
        return rw_loss_term
    