import mxnet as mx
from mxnet import nd
import numpy as np
from sklearn import preprocessing


def DLPreprocess(dt, catFeatureList=None, numericFeatureList=None):
    """
    Function for data preprocess.
    (From the original DeepTCN paper - Not applied).
    """
    # label encoding of the categorical features
    labelEncList = []
    if catFeatureList is not None:
        for categoryFeature in catFeatureList:
            labelEnc = preprocessing.LabelEncoder()
            labelEnc.fit(dt.loc[:, categoryFeature])
            labelEncList.append(labelEnc)
            dt.loc[:, categoryFeature] = labelEnc.transform(dt.loc[:, categoryFeature])
    # numeric feature normalization
    if numericFeatureList is not None:
        dt[numericFeatureList] = preprocessing.scale(dt[numericFeatureList])
    return dt, labelEncList


def crps_from_samples_mx(y_true, samples, mask=None, ctx=mx.cpu(), batch_size=32):
    """
    Calculates the sample-based CRPS approximation in batches to avoid OOM.
    CRPS = E|X - y| - 0.5 * E|X - X'|
    """
    if mask is not None:
        mask_f = mask.astype('float32')
        if mask_f.ndim == 3:
            mask_f = nd.squeeze(mask_f, axis=-1)
    else:
        mask_f = None

    N = y_true.shape[0]
    total_crps_sum = 0.0
    total_valid_points = 0.0
    
    # --- Batched Processing ---
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        
        # Slice batch
        y_batch = y_true[i:end_idx]      # (B, T) or (B, T, 1)
        s_batch = samples[i:end_idx]     # (B, T, S)
        m_batch = mask_f[i:end_idx] if mask_f is not None else None # (B, T)
        
        # Reshape y for broadcasting calculations
        if y_batch.ndim == 2:
            y = y_batch.expand_dims(axis=-1) # (B, T, 1)
        else:
            y = y_batch

        # --- TERM 1: E|X - y| ---
        term1 = nd.mean(nd.abs(s_batch - y), axis=-1) # Result is (B, T)

        # --- TERM 2: 0.5 * E|X - X'| ---
        s_x = s_batch.expand_dims(axis=-1)       # (B, T, S, 1)
        s_x_prime = s_batch.expand_dims(axis=-2) # (B, T, 1, S)
        
        # Calculate absolute difference
        abs_diff = nd.abs(s_x - s_x_prime) # (B, T, S, S)
        term2 = 0.5 * nd.mean(abs_diff, axis=(-1, -2)) # Result is (B, T)

        # --- Combine ---
        crps_batch = term1 - term2 # (B, T)

        # --- Accumulate ---
        if m_batch is not None:
            current_sum = nd.sum(crps_batch * m_batch).asscalar()
            current_count = nd.sum(m_batch).asscalar()
            total_crps_sum += current_sum
            total_valid_points += current_count
        else:
            total_crps_sum += nd.sum(crps_batch).asscalar()
            total_valid_points += crps_batch.size

    # Avoid division by zero
    if total_valid_points == 0:
        return 0.0
        
    return total_crps_sum / total_valid_points


def calculate_pointwise_log_likelihood_mx(y_val, mask_val, mu, alpha, gate, ctx=mx.cpu(), softZero=1e-6):
    """
    Calculates the pointwise Log-Probability of the true data using MXNet operations.

    Args:
        y_val (np.array): True target values.
        mask_val (np.array): Mask for valid values.
        mu (np.array): Predicted mean (stabilized).
        alpha (np.array): Predicted dispersion (stabilized).
        gate (np.array): Predicted gate probability (stabilized).
        ctx (mx.Context): Context to perform calculations (default CPU).
    
    Returns: 
        valid_log_lik (np.array): Flattened array of valid log-likelihoods.
    """
    # --- 1. Convert Numpy to MXNet NDArray ---
    y = nd.array(y_val, dtype='float32', ctx=ctx)
    mask = nd.array(mask_val, dtype='float32', ctx=ctx)
    mu_stable = nd.array(mu, dtype='float32', ctx=ctx)
    alpha_stable = nd.array(alpha, dtype='float32', ctx=ctx)
    gate_stable = nd.array(gate, dtype='float32', ctx=ctx)

    # --- 2. ZINB Parameters (Same as Loss) ---
    r = 1.0 / (alpha_stable + softZero)
    theta = 1.0 / (1.0 + alpha_stable * mu_stable)
    log_theta = nd.log(theta + softZero)
    log_1m_theta = nd.log(1.0 - theta + softZero)

    # Ensure y >= 0 (sanity check)
    y = nd.maximum(y, 0.0)

    # --- 3. NB log pmf for general y ---
    p1 = nd.gammaln(y + r + softZero)
    p2 = nd.gammaln(r + softZero)
    p3 = nd.gammaln(y + 1.0)
    p4 = r * log_theta
    p5 = y * log_1m_theta
    
    # log P_NB(y | mu, alpha)
    log_nb_y = p1 - p2 - p3 + p4 + p5

    # --- 4. NB log pmf for y = 0 ---
    y0 = nd.zeros_like(y)
    p1_0 = nd.gammaln(y0 + r + softZero)
    p3_0 = nd.gammaln(y0 + 1.0) # gammaln(1) = 0
    p5_0 = y0 * log_1m_theta    # 0
    
    # log P_NB(0 | mu, alpha)
    log_nb_0 = p1_0 - p2 - p3_0 + p4 + p5_0

    # --- 5. Zero-Inflation Mixture Logic ---
    is_zero = (y == 0)

    # Case A: y = 0
    # log [ π + (1-π) * P_NB(0) ]
    prob_zero_mixture = gate_stable + (1.0 - gate_stable) * nd.exp(log_nb_0)
    log_prob_zero = nd.log(prob_zero_mixture + softZero)

    # Case B: y > 0
    # log [ (1-π) * P_NB(y) ] = log(1-π) + log P_NB(y)
    log_prob_nonzero = nd.log(1.0 - gate_stable + softZero) + log_nb_y

    # Combine
    log_prob = nd.where(is_zero, log_prob_zero, log_prob_nonzero)

    # --- 6. Masking and Flattening ---
    log_prob_flat = log_prob.reshape(-1)
    mask_flat = mask.reshape(-1)
    
    # Convert back to numpy to apply boolean indexing
    log_prob_np = log_prob_flat.asnumpy()
    mask_np = mask_flat.asnumpy()
    
    # Return only valid log-likelihoods
    valid_log_lik = log_prob_np[mask_np == 1]

    return valid_log_lik


def SMAPE(yPred, yTrue):
    """
    Function for evaluation.
    (From the original DeepTCN paper - Not applied)
    """
    assert len(yPred) == len(yTrue)
    denominator = (np.abs(yTrue) + np.abs(yPred))
    diff = np.abs(yTrue - yPred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)


def ND(yPred, yTrue):
    """
    Function for evaluation.
    (From the original DeepTCN paper - Not applied)
    """
    assert len(yPred) == len(yTrue)
    demoninator = np.sum(np.abs(yTrue))
    diff = np.sum(np.abs(yTrue - yPred))
    return 1.0*diff/demoninator


def RMSLE(yPred, yTrue):
    """
    Function for evaluation.
    (From the original DeepTCN paper - Not applied)
    """
    assert len(yPred) == len(yTrue)
    assert len(yTrue) == len(yPred)
    return np.sqrt(np.mean((np.log(1+yPred) - np.log(1+yTrue))**2))

def NRMSE(yPred, yTrue):
    """
    Function for evaluation.
    (From the original DeepTCN paper - Not applied)
    """
    assert len(yPred) == len(yTrue)
    denominator = np.mean(yTrue)
    diff = np.sqrt(np.mean(((yPred-yTrue)**2)))
    return diff/denominator

def rhoRisk(yPred, yTrue, rho):
    """
    The robust quantile loss (pinball loss), normalized by the mean of absolute true values.
    Function from the original paper, modified to work correctly with Negative Binomial Loss.
    """
    assert yPred.shape == yTrue.shape
    residual = yTrue - yPred
    
    # Calculate pinball loss
    loss = (rho * residual.clip(min=0)) + ((1 - rho) * -residual.clip(max=0))
    
    # Normalize with a stable denominator
    denominator = np.mean(np.abs(yTrue)) + 1e-6 # 1e-6 to avoid division by zero
    
    # Return normalized loss, scaled by 100
    return 100 * np.mean(loss) / denominator


def rhoRisk2(yPred, yTrue, rho):
    """
    Deprecated.
    Function from the original paper, modified to work correctly with Negative Binomial Loss.
    """
    assert len(yPred) == len(yTrue)
    
    diff1 = (yTrue - yPred) * rho * (yTrue >= yPred)
    diff2 = (yPred - yTrue) * (1 - rho) * (yTrue < yPred)
    
    numerator = 2 * (np.sum(diff1) + np.sum(diff2))
    denominator = np.sum(np.abs(yTrue)) + 1e-6 # Fixed denominator
    
    return numerator / denominator

if __name__ == "__main__":
    pass