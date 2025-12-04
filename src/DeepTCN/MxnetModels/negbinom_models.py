from mxnet import nd
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
import logging

# --- Set up logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def NegativeBinomialLoss(mu, alpha, y, mask=None, softZero=1e-6):
    """
    Calculates the NLL of a Negative Binomial distribution.
    This loss works on the unscaled counts (y).

    Args:
        mu: (batch_size, horizon) - predicted mean (must be > 0)
        alpha: (batch_size, horizon) - predicted dispersion (must be > 0)
        y: (batch_size, horizon) - raw, unscaled true counts
        mask:  (batch, horizon) mask for the future values

    Returns:
        res (floar): Negative log likelihood for Negative Binomial Distribution.
    """
    mu_stable = nd.maximum(mu, softZero)
    alpha_stable = nd.clip(alpha, softZero, 40.0)

    # --- 1. Use more stable parametrization ---
    r = 1.0 / alpha_stable
    theta = 1.0 / (1.0 + alpha_stable * mu_stable)
    log_theta = nd.log(theta + softZero)
    log_1m_theta = nd.log(1.0 - theta + softZero)

    # --- 2. Calculate the log-likelihood using gaammaln ---
    p1 = nd.gammaln(y + r + softZero)
    p2 = nd.gammaln(r + softZero)
    p3 = nd.gammaln(y + 1.0)
    p4 = r * log_theta
    p5 = y * log_1m_theta
    log_nb = p1 - p2 - p3 + p4 + p5
    nll = -log_nb

    if mask is not None:
        mask = mask.astype('float32')
        nll = nll * mask
        res = nd.sum(nll) / (nd.sum(mask) + softZero)
    
    else:
        res = nd.mean(nll)

    return res

def ZeroInflatedNegativeBinomialLoss(mu, alpha, gate, y, mask=None, alpha_reg: float = 1e-4, gate_reg: float = 1e-4, softZero=1e-6):
    """
    Zero-Inflated Negative Binomial NLL.

    Args:
        mu:    (batch, horizon) predicted mean (>0)
        alpha: (batch, horizon) predicted dispersion (>0)
        gate:  (batch, horizon) probability of extra zeros in (0,1)
        y:     (batch, horizon) raw, unscaled true counts
        mask:  (batch, horizon) mask for the future values
        alpha_reg (float) Multiplier for the alpha regularization term.
        gate_reg (float) Multiplier for the gate regularization term.
    """
    # Stabilize parameters
    mu_stable = nd.clip(mu + 1e-3, 1e-4, 1e8)
    alpha_stable = nd.clip(alpha, 1e-4, 40.0)
    gate_stable = nd.clip(gate * 0.999 + 0.0005, 0.0005, 0.9995)

    # NB parameters: same as in NegativeBinomialLoss
    r = 1.0 / alpha_stable
    theta = 1.0 / (1.0 + alpha_stable * mu_stable)
    log_theta = nd.log(theta + softZero)
    log_1m_theta = nd.log(1.0 - theta + softZero)

    # Ensure y >= 0
    y = nd.maximum(y, 0.0)

    # NB log pmf for general y
    p1 = nd.gammaln(y + r + softZero)
    p2 = nd.gammaln(r + softZero)
    p3 = nd.gammaln(y + 1.0)
    p4 = r * log_theta
    p5 = y * log_1m_theta
    log_nb_y = p1 - p2 - p3 + p4 + p5  # log P_NB(y | mu, alpha)

    # NB log pmf for y = 0
    y0 = nd.zeros_like(y)
    p1_0 = nd.gammaln(y0 + r + softZero)
    p2_0 = nd.gammaln(r + softZero)
    p3_0 = nd.gammaln(y0 + 1.0)
    # p4_0 is same r*log_theta
    p4_0 = p4
    p5_0 = y0 * log_1m_theta
    log_nb_0 = p1_0 - p2_0 - p3_0 + p4_0 + p5_0  # log P_NB(0 | mu, alpha)

    is_zero = (y == 0)

    # log [ π + (1-π) * P_NB(0) ]
    log_prob_zero = nd.log(
        gate_stable + (1.0 - gate_stable) * nd.exp(log_nb_0) + softZero
    )

    # log [ (1-π) * P_NB(y>0) ] = log(1-π) + log P_NB(y)
    log_prob_nonzero = nd.log(1.0 - gate_stable + softZero) + log_nb_y

    log_prob = nd.where(is_zero, log_prob_zero, log_prob_nonzero)

    nll = -log_prob

    # Apply mask if exists
    if mask is not None:
        mask = mask.astype('float32')
        nll_masked = nll * mask
        nll_mean = nd.sum(nll_masked) / (nd.sum(mask) + softZero)
    else:
        nll_mean = nd.mean(nll)

    # Apply Alpha Penalty
    log_alpha_stable = nd.log(alpha_stable + softZero)
    alpha_penalty = alpha_reg * nd.mean(log_alpha_stable ** 2) # L2 penalty

    # Apply gate penalty
    gate_penalty = gate_reg * nd.mean(gate_stable)

    res = nll_mean + alpha_penalty + gate_penalty

    return res



class MultiQuantileLoss(gloss.Loss):
    """
    Calculates the sum of Quantile Losses for multiple quantiles.
    L = Sum_q [ q * (y - y_pred)+ + (1-q) * (y_pred - y)+ ]
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9], weight=None, batch_axis=0, **kwargs):
        super(MultiQuantileLoss, self).__init__(weight, batch_axis, **kwargs)
        self.quantiles = quantiles

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """
        pred: (Batch, Horizon, Num_Quantiles)
        label: (Batch, Horizon)
        sample_weight: (Batch, Horizon) - This is your mask
        """
        # Reshape label to (Batch, Horizon, 1) to broadcast against quantiles
        label = label.expand_dims(axis=-1) # (B, H, 1)
        
        loss_total = 0.0
        
        # Iterate over quantiles (q_idx corresponds to the last dim of pred)
        for i, q in enumerate(self.quantiles):
            # Extract prediction for this specific quantile
            pred_q = F.slice_axis(pred, axis=-1, begin=i, end=i+1) # (B, H, 1)
            
            # Calculate element-wise difference
            diff = pred_q - label
            
            # Standard Quantile Loss Formula
            # mask_over = (pred < label) -> implies underprediction
            # Loss = q * (y - y_hat) if y > y_hat else (1-q) * (y_hat - y)
            loss_q = F.where(diff < 0, 
                             (1 - q) * F.abs(diff),  # Over-prediction penalty
                             q * F.abs(diff))        # Under-prediction penalty
            
            loss_total = loss_total + loss_q

        # Apply Masking (sample_weight)
        if sample_weight is not None:
            # Expand mask to (B, H, 1)
            mask = sample_weight.expand_dims(axis=-1)
            loss_total = loss_total * mask
            # Return mean over valid points
            return F.sum(loss_total) / (F.sum(mask) + 1e-8)
            
        return F.mean(loss_total)


class ResidualTCN(nn.Block):
    """
    The residual blocks of the TCN model: Current residual.
    Slightly modified to add padding to accept all inputs.
    """
    def __init__(self, d, n_residue=35, k=3, **kwargs):
        super(ResidualTCN, self).__init__(**kwargs)
        self.padding = (k - 1) * d
        self.conv1 = nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d, padding=self.padding)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d, padding=self.padding)
        self.bn2 = nn.BatchNorm()
        
    def forward(self, x):
        # Block 1
        out = self.conv1(x)
        out = out[:, :, :-self.padding]
        out = nd.relu(self.bn1(out))

        # Block 2
        out = self.conv2(out)
        out = out[:, :, :-self.padding]
        out = self.bn2(out)
        return nd.relu(out + x)


class futureResidual(nn.HybridBlock):
    """
    The residual blocks of the TCN model: Future residual.
    Severely modified to accept all inputs and additional Dropout added for regularization.
    """
    def __init__(self, in_channels_real=0, in_channels_cat=0, out_channels=64, dropout=0.2, **kwargs):
        super(futureResidual, self).__init__(**kwargs)
        self.total_in = in_channels_real + in_channels_cat
        self.fc1 = nn.Dense(out_channels, flatten=False)
        self.bn1 = nn.BatchNorm(axis=2)
        self.fc2 = nn.Dense(out_channels, flatten=False)
        self.bn2 = nn.BatchNorm(axis=2)
        self.dropout_l = nn.Dropout(dropout)
        logger.debug(f"futureResidual block initialized with {self.total_in} input channels, {out_channels} output channels.")
        
    def hybrid_forward(self, F, lagX, xCat, xReal):
        decoderInput = F.concat(xCat, xReal, dim=2)
        out = F.relu(self.bn1(self.fc1(decoderInput)))
        out = self.dropout_l(out)
        out = self.bn2(self.fc2(out))
        out = F.relu(out)
        lagX_broadcast = F.broadcast_like(lagX, out, lhs_axes=(1,), rhs_axes=(1,))
        return F.relu(F.concat(lagX_broadcast, out, dim=2))


class TCN(nn.Block):
    """
    The core model from DeepTCN paper.
    Severely modified to accept all inputs and additional regularization.
    """
    def __init__( self, cat_features_info: dict, n_real_features: int, 
                 inputSize=12, outputSize=12, num_channels=32, dilations=[1,2,4], 
                 embed_dim=10, dropout=0.2, quantiles=[0.1, 0.5, 0.9], **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.dilations = dilations
        self.encoder = nn.Sequential()
        self.outputLayer= nn.Sequential()
        self.cat_features_info = cat_features_info
        self.n_real_features = n_real_features
        self.embeddings = {}
        self.encoder_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(axis=2)
        # self.quantiles = quantiles
        # num_outputs = len(quantiles)

        with self.name_scope():
            total_embed_dim = 0
            for i, (col, cardinality) in enumerate(cat_features_info.items()):
                col_embed_dim = min(embed_dim, (cardinality + 1) // 2)
                logger.debug(f"Embedding for {col}: cardinality={cardinality}, dim={col_embed_dim}")

                # Register embedding layer to the block
                setattr(self, f"embed_{col}", nn.Embedding(cardinality, col_embed_dim))
                self.embeddings[i] = getattr(self, f"embed_{col}")
                total_embed_dim += col_embed_dim
            logger.debug(f"Total embedding dimension: {total_embed_dim}")
            logger.debug(f"Total real features: {n_real_features}")

            # Total number of input features
            n_input_features = 1 + total_embed_dim + n_real_features
            logger.debug(f"Total input features: {n_input_features}")

            # Input convolution to map features to num_channels
            self.input_conv = nn.Conv1D(in_channels=n_input_features, channels=num_channels, kernel_size=1)

            # nResidue is the total number of features for the TCN
            logger.debug(f"TCN nResidue (channels) set to: {num_channels}")

            for d in self.dilations:
                self.encoder.add(ResidualTCN(d=d, n_residue=num_channels, k=3))
            
            self.n_future_cat = total_embed_dim
            self.n_future_real = n_real_features
            decoder_out_channels = 64

            self.decoder = (futureResidual(in_channels_real = self.n_future_real,
                                           in_channels_cat=self.n_future_cat,
                                           out_channels=decoder_out_channels, dropout=dropout))

            # Base architecture
            output_layer_in_dim = num_channels + decoder_out_channels
            self.outputLayer.add(nn.Dense(output_layer_in_dim, in_units=output_layer_in_dim, flatten=False))
            self.outputLayer.add(nn.BatchNorm(axis=2))
            self.outputLayer.add(nn.Swish())
            self.outputLayer.add(nn.Dropout(dropout))
            # self.output_quantiles = nn.Dense(num_outputs, in_units=output_layer_in_dim, flatten=False)

            self.mu = nn.Dense(1, in_units=output_layer_in_dim, flatten=False, activation='softrelu')
            self.alpha = nn.Dense(1, in_units=output_layer_in_dim, flatten=False, activation=None)
            self.gate = nn.Dense(1, in_units=output_layer_in_dim, flatten=False, activation='sigmoid', bias_initializer=init.Constant(0.42))

    def forward(self, xNum, xCat, xReal):
        # Embed categorical features
        embedded_cats = []
        for i in range(xCat.shape[2]):
            embed_layer = self.embeddings[i]
            embedded_cats.append(embed_layer(xCat[:, :, i]))
        embedConcat = nd.concat(*embedded_cats, dim=2)

        # Split historical and future features
        embedTrain = embedConcat[:, :self.inputSize, :]
        realTrain = xReal[:, :self.inputSize, :]
        xNum_scaled = nd.log1p(xNum)

        # Create Encoder input
        inputSeries = nd.concat(xNum_scaled, realTrain, embedTrain, dim=2)
        inputSeries = nd.transpose(inputSeries, axes=(0,2,1))
        inputSeries = self.input_conv(inputSeries)

        for subTCN in self.encoder:
            inputSeries = subTCN(inputSeries)
            inputSeries = self.encoder_dropout(inputSeries)
        
        # Create Decoder input
        output = inputSeries[:, :, -1:]
        output = nd.transpose(output, axes=(0,2,1))
        output = self.layer_norm(output)
        # Get future covariates
        embedTest = embedConcat[:, self.inputSize:, :]
        realTest = xReal[:, self.inputSize:, :]

        # Run decoder & Output Layer
        decoder_output = self.decoder(output, embedTest, realTest)
        output = self.outputLayer(decoder_output)
        mu = self.mu(output)
        raw_alpha = self.alpha(output)
        alpha = nd.clip(nd.exp(raw_alpha), a_min=1e-4, a_max=1e4)
        gate = self.gate(output)
        return mu, alpha, gate

        # predictions = self.output_quantiles(output)

        # return predictions


if __name__ == "__main__":
    pass