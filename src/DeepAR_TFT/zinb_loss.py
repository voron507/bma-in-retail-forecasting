import torch
import torch.nn.functional as F
from pytorch_forecasting.metrics.base_metrics._base_metrics import DistributionLoss
from torch.distributions import NegativeBinomial, Bernoulli, Distribution

class ZINBDistribution(Distribution):
    def __init__(self, mu, alpha, gate):
        super().__init__(validate_args=False)
        self.mu = mu
        self.alpha = alpha
        self.gate = gate  # zero inflation prob

        total_count = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu)
        p = p.clamp(1e-5, 1 - 1e-5)
        self.nb = NegativeBinomial(total_count=total_count, probs=p)
        self.zero_dist = Bernoulli(gate)

    def log_prob(self, y):
        y = y.long()

        # NB log-prob
        nb_lp = self.nb.log_prob(y)
        nb_lp_0 = self.nb.log_prob(torch.zeros_like(y))
        
        # Zero inflation mixture
        log_prob_zero = torch.logaddexp(torch.log(self.gate + 1e-8), torch.log(1 - self.gate + 1e-8) + nb_lp_0)
        log_prob_nonzero = torch.log(1 - self.gate + 1e-8) + nb_lp
        return torch.where(y == 0, log_prob_zero, log_prob_nonzero)
    
    def sample(self, sample_shape=torch.Size()):
        # Sample NB and zero-inflation mask
        nb_sample = self.nb.sample(sample_shape)
        zeros = self.zero_dist.sample(sample_shape).bool()

        # zero-inflation: with prob gate produce 0, else NB
        return torch.where(zeros, torch.zeros_like(nb_sample), nb_sample)
    
    @property
    def mean(self):
        return (1 - self.gate) * self.mu


class ZINBLoss(DistributionLoss):
    output_size = 3  # mu, alpha, gate
    distribution_arguments = ["mu", "alpha", "gate"]

    def __init__(self, alpha_reg: float = 1e-4, gate_reg: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        # Make L1 & L2 tunable
        self.alpha_reg = alpha_reg
        self.gate_reg = gate_reg

    def distribution(self, parameters, **kwargs):
        mu = torch.exp(parameters[..., 0])
        #mu = F.softplus(parameters[..., 0]) + 1e-3
        mu = torch.clamp(mu, 1e-5, 1e8)
        alpha = F.softplus(parameters[..., 1]) + 1e-3
        alpha = torch.clamp(alpha, 1e-4, 40.0)
        gate = parameters[..., 2].sigmoid() * 0.999 + 0.0005 

        return ZINBDistribution(mu, alpha, gate)
    
    def loss(self, y_pred, target):
        # 1. Unpack the 'target' tuple
        if isinstance(target, (list, tuple)):
            y_true = target[0]
        else:
            y_true = target

        # 2. Get the distribution
        dist = self.map_x_to_distribution(y_pred)
        
        # 3. Calculate Negative Log Likelihood
        # Shape: [Batch, Horizon]
        nll = -dist.log_prob(y_true)
        
        return nll
    
    def map_x_to_distribution(self, x: torch.Tensor):
        return self.distribution(x)
    
    def map_x_to_prediction(self, x: torch.Tensor):
        dist = self.distribution(x)
        return dist.mean

    # def forward(self, y_pred, target, **kwargs):

    #     if isinstance(y_pred, dict):
    #         y_pred = y_pred["prediction"]

    #     # 1. Unpack the 'target' tuple
    #     mask = None
    #     if isinstance(target, (list, tuple)):
    #         y_true = target[0]
    #         mask = target[1].float()
    #     else:
    #         y_true = target

    #     # 2. Get the distribution
    #     dist = self.distribution(y_pred)
        
    #     # 3. Get the 2D tensor of per-point losses
    #     nll = -dist.log_prob(y_true)

    #     # 4. Apply the mask
    #     if mask is not None:
    #         # Ensure mask has the same ndim as nll
    #         if nll.ndim > mask.ndim:
    #             mask = mask.unsqueeze(-1)
            
    #         nll = (nll * mask).sum() / (mask.sum() + 1e-8)
    #     else:
    #         nll = nll.mean()

        # # 5. Add regularization
        # alpha = F.softplus(y_pred[..., 1]) + 1e-3
        # safe_alpha = torch.clamp(alpha, min=1e-4)
        # #gate = y_pred[..., 2].sigmoid() * 0.999 + 0.0005
        # alpha_penalty = self.alpha_reg * (alpha ** 2).mean() # L2 penalty to fight overdispersion
        # #gate_penalty = self.gate_reg * gate.mean()

        # return nll #+ alpha_penalty #+ gate_penalty


# Evaluation metrics
def crps_from_samples(y_true, samples, mask=None, batch_size=1000):
    """
    Calculates CRPS using a batched approach.

    Args:
        y_true (torch.tensor): True values of y
        samples (torch.tensor): Samples predicted.
        mask (torch.tensor): True values mask.

    Returns:
        crps (float): CRPS.
    """
    # Ensure inputs are on the same device
    device = samples.device
    if y_true.device != device:
        y_true = y_true.to(device)
    if mask is not None and mask.device != device:
        mask = mask.to(device)

    num_series = y_true.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)

    # Process in batches
    for i in range(0, num_series, batch_size):
        end = min(i + batch_size, num_series)
        
        # Slice the batch
        y_b = y_true[i:end]       # (B, H)
        s_b = samples[i:end]      # (B, H, Samples)
        
        # Term 1: |X - y|
        term1 = torch.mean(torch.abs(s_b - y_b.unsqueeze(-1)), dim=-1)
        
        # Term 2: 0.5 * |X - X'|
        s_diff = torch.abs(s_b.unsqueeze(-1) - s_b.unsqueeze(-2))
        term2 = 0.5 * torch.mean(s_diff, dim=(-1, -2))
        
        batch_crps = term1 - term2
        
        if mask is not None:
            m_b = mask[i:end]
            total_loss += torch.sum(batch_crps * m_b)
            total_weight += torch.sum(m_b)
        else:
            total_loss += torch.sum(batch_crps)
            total_weight += batch_crps.numel()

    # Avoid division by zero
    if total_weight == 0:
        return torch.tensor(0.0, device=device)
        
    return total_loss / total_weight


def crps_from_quantiles(preds, target, weights=None, quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]):
    """
    Deprecated.
    Calculates the Weighted Average Quantile Loss (CRPS Proxy) with mask.

    Args:
        preds (torch.Tensor): Shape [Batch, Horizon, Quantiles]
        target (torch.Tensor): Shape [Batch, Horizon]
        weights (torch.Tensor, optional): Shape [Batch, Horizon]. Defaults to 1.0.
        quantiles (list, optional): List of quantiles used in the model. 
                                    Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].

    Returns:
        float: The scalar CRPS (averaged over all quantiles).
    """

    # 1. Handle Weights and shape
    if weights is None:
        weights = torch.ones_like(target)

    # 2. CRITICAL SHAPE FIX (The "Bulletproof" Logic)
    if preds.ndim != 3:
        raise ValueError(f"Predictions must be 3D [Batch, Horizon, Quantiles], got {preds.shape}")
        
    n_batch, n_horizon, n_quantiles = preds.shape

    # 3. Calculate Pinball Loss
    loss_sum = 0.0
    
    for i, q in enumerate(quantiles):
        # Slice the specific quantile forecast -> [Batch, Horizon]
        pred_q = preds[..., i]
        
        # Calculate Error
        errors = target - pred_q
        
        # Pinball Logic: max(q * error, (q-1) * error)
        q_loss = torch.max((q - 1) * errors, q * errors)
        
        # Apply Weights (Masking)
        # Sum of Weighted Loss / Sum of Weights
        weighted_q_loss = (q_loss * weights).sum() / (weights.sum() + 1e-8)
        loss_sum += weighted_q_loss

    # 4. Average over quantiles
    crps = loss_sum / len(quantiles)
    
    return crps.item()


def calculate_pointwise_log_likelihood(y_val, mask_val, parameters):
    """
    Calculates the Log-Probability of the true data given the predicted parameters.

    Args:
        y_val (np.array): True target validation values.
        mask_val (np.array): Mask for the true target validation values.
        parameters (np.array): Mu, Alpha, Gate for each observation.
    
    Returns: 
        valid_log_lik (np.array): array of shape (N_series * Horizon).
    """
    # --- 1. Convert to torch for ZINB ---
    y_true_tensor = torch.tensor(y_val)
    mask_tensor = torch.tensor(mask_val).float()
    
    # Unpack parameters (Batch, Time, 3)
    mu = torch.tensor(parameters[..., 0])
    alpha = torch.tensor(parameters[..., 1])
    gate = torch.tensor(parameters[..., 2])

    # --- 2. Instantiate custom Distribution ---
    dist = ZINBDistribution(mu, alpha, gate)
    
    # --- 3. Calculate Log Probability ---
    log_lik = dist.log_prob(y_true_tensor) # (N_series, Horizon)

    # --- 4. Apply the mask and sum only valid log-likelihoods ---
    masked_log_lik = log_lik * mask_tensor
    
    # --- 5. Flatten for ArviZ ---
    masked_log_lik_flat = masked_log_lik.numpy().flatten()
    mask_flat = mask_tensor.numpy().flatten()

    valid_log_lik = masked_log_lik_flat[mask_flat == 1]

    return valid_log_lik


if __name__ == "__main__":
    pass