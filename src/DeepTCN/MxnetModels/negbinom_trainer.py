import config
import logging
import numpy as np
import pandas as pd

from scipy.stats import nbinom
from tqdm import tqdm

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxboard import SummaryWriter

from DeepTCN.utils import rhoRisk
from DeepTCN.MxnetModels.negbinom_models import NegativeBinomialLoss, ZeroInflatedNegativeBinomialLoss, MultiQuantileLoss


# --- Set up logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Define global variables ---
MODELS_DIR = config.MODELS_DIR


class nnTrainer(object):
    def __init__(self, nnModel, modelCtx, dataCtx):
        self.modelCtx = modelCtx
        self.dataCtx = dataCtx
        self.model = nnModel

    def saveCheckpoint(self, nnModel, mark, testAuc, fold):
        savePath = MODELS_DIR / 'DeepTCN' / fold
        savePath.mkdir(parents=True, exist_ok=True)
        filename = savePath / "mark_{:s}_metrics_{:.3f}.param".format(mark, testAuc)
        nnModel.save_parameters(str(filename))
        return filename

    def predict(self, nnModel, testX, testCat, testReal, batchSize=3000):
        """
        Predict method from the original DeepTCN paper.
        Modified to accept all covariates, and predict 3 outputs. 
        """
        test_dataset = gluon.data.ArrayDataset(testX, testCat, testReal)
        test_data_loader = gluon.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

        mu_list, alpha_list, gate_list = [], [], []

        for data, cat_data, real_data in test_data_loader:
            data = nd.array(data, dtype='float32', ctx=self.dataCtx)
            cat_data = nd.array(cat_data, dtype='float32', ctx=self.dataCtx)
            real_data = nd.array(real_data, dtype='float32', ctx=self.dataCtx)
            out = nnModel(data, cat_data, real_data)

            if isinstance(out, (list, tuple)) and len(out) == 3:
                mu, alpha, gate = out
                gate_list.append(gate.as_in_context(mx.cpu()))
            else:
                mu, alpha = out

            mu_list.append(mu.as_in_context(mx.cpu()))
            alpha_list.append(alpha.as_in_context(mx.cpu()))

        predMu = nd.concat(*mu_list, dim=0)
        predAlpha = nd.concat(*alpha_list, dim=0)

        if gate_list:
            predGate = nd.concat(*gate_list, dim=0)
            return predMu, predAlpha, predGate
        
        return predMu, predAlpha
    
    
    def probEvaluator(self, nnModel, testX, testCat, testReal, testY, testMaskFuture, lossFunc, 
                  alpha_reg=0.0, gate_reg=0.0):
        """
        Evaluator method from the original DeepTCN paper.
        Severely modified to work with Negative Binomial Loss. 
        """
        out = self.predict(nnModel, testX, testCat, testReal)
        if isinstance(out, (list, tuple)) and len(out) == 3:
            mu_pred, alpha, gate = out
            has_gate = True
        else:
            mu_pred, alpha = out
            gate = None
            has_gate = False

        validTrue = testY
        mu_pred_ctx = mu_pred.as_in_context(self.modelCtx)
        alpha_ctx = alpha.as_in_context(self.modelCtx)
        label_ctx = nd.array(validTrue, dtype='float32', ctx=self.dataCtx)
        mask_ctx = nd.array(testMaskFuture, dtype='float32', ctx=self.dataCtx)

        validPredMu_np = mu_pred.asnumpy()
        validPredAlpha_np = alpha.asnumpy()
        if has_gate:
            gate_ctx = gate.as_in_context(self.modelCtx)
            loss = nd.mean(lossFunc(mu_pred_ctx, alpha_ctx, gate_ctx, label_ctx, mask_ctx, alpha_reg=alpha_reg, gate_reg=gate_reg)).asscalar()
            validPredGate_np = gate.asnumpy()
        else:
            loss = nd.mean(lossFunc(mu_pred_ctx, alpha_ctx, label_ctx, mask_ctx)).asscalar()
            validPredGate_np = np.zeros_like(validPredMu_np)
        
        # Reshape to (N*L, 1) for metric calculation
        validTrue_flat = validTrue.reshape(-1, 1)
        validPredMu_flat = validPredMu_np.reshape(-1, 1)
        validPredAlpha_flat = validPredAlpha_np.reshape(-1, 1)
        validPredGate_flat = validPredGate_np.reshape(-1, 1)

        # Convert mu, alpha into n, p for nbinom
        n = 1.0 / (validPredAlpha_flat + 1e-6) 
        p = 1.0 / (1.0 + validPredAlpha_flat * validPredMu_flat)

        # Define quantiles to calculate
        quantiles = [0.5, 0.9]
        validPredQ = {}

        for q in quantiles:
            # Calculate the residual probability q' to map to standard NB
            # q' = (q - gate) / (1 - gate)
            p_star = (q - validPredGate_flat) / (1.0 - validPredGate_flat + 1e-6)
            p_star = np.clip(p_star, 0.0, 1.0) # Clip to ensure 0-1 range for PPF
            
            # Determine if quantile is 0: if q <= gate, then quantile is 0
            is_zero_q = validPredGate_flat >= q
            
            # Calculate PPF on the residual probability for the NB part
            validPredQ_nb = nbinom.ppf(p_star, n, p)
            
            # Apply ZINB rule: 0 if q <= gate, otherwise use the NB PPF
            validPredQ[q] = np.where(is_zero_q, 0.0, validPredQ_nb)

        validPredQ50 = validPredQ[0.5]
        validPredQ90 = validPredQ[0.9]

        # Calculate rho-risks on the raw counts
        rho50 = rhoRisk(validPredQ50.reshape(-1, 1), validTrue_flat, 0.5)
        rho90 = rhoRisk(validPredQ90.reshape(-1, 1), validTrue_flat, 0.9)

        return loss, rho50, rho90


    def fit(self, mark, trainX, trainCat, trainReal, trainY, trainMaskFuture, testX, testCat, testReal, testY, testMaskFuture, paramsDict, fold, tb_logger=None):
        """
        Fit method from the original DeepTCN paper.
        Severely modified to acccept additional covariates, better regularization through weight decay included, 
        more stable gradients through clipping added, and more informative and logical Early Stopping Framework implemented.
        TensorBoard added.
        """
        epochs = paramsDict['epochs']
        esEpochs = paramsDict['esEpochs']
        evalCriteria = paramsDict['evalCriteria']

        batchSize = paramsDict['batchSize']
        learningRate = paramsDict['learningRate']

        optimizer = paramsDict['optimizer']
        initializer = paramsDict['initializer']

        # Regularization
        weight_decay = paramsDict.get('weightDecay', 0.0)
        clip_gradient = paramsDict.get('clipGradient', 1.0)
        alpha_reg = paramsDict.get('alphaReg', 0.0)
        gate_reg = paramsDict.get('gateReg', 0.0)

        # The loss
        lossFunc = ZeroInflatedNegativeBinomialLoss
        # The model initialization
        self.model.collect_params().initialize(initializer, ctx=self.modelCtx)
        # The trainer
        optimizer_params = {
            'learning_rate': learningRate,
            'clip_gradient': clip_gradient, 
            'wd': weight_decay
        }

        trainer = gluon.Trainer(
            self.model.collect_params(), 
            optimizer=optimizer, 
            optimizer_params=optimizer_params
        )

        nSamples = trainX.shape[0]
        # Create training data loader
        train_dataset = gluon.data.ArrayDataset(trainX, trainCat, trainReal, trainY, trainMaskFuture)
        train_data_loader = gluon.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

        # Keep the metrics history
        history = {'train_loss': [], 'valid_loss': [], 'valid_rho50': [], 'valid_rho90': []}

        # The early stopping framework
        bestValidMetric = 999999. if evalCriteria =='min' else 0
        patience_counter = 0
        best_model_path = None

        try:
            for e in tqdm(range(epochs), desc=f"Epochs (Fold: {fold})"):
                cumLoss = 0.
            
                for data, cat_data, real_data, y_true, mask in train_data_loader:
                    data = nd.array(data, dtype='float32', ctx=self.dataCtx)
                    cat_data = nd.array(cat_data, dtype='float32', ctx=self.dataCtx)
                    real_data = nd.array(real_data, dtype='float32', ctx=self.dataCtx)
                    label = nd.array(y_true, dtype='float32', ctx=self.dataCtx)
                    mask = nd.array(mask, dtype='float32', ctx=self.dataCtx)

                    with autograd.record():
                        out = self.model(data, cat_data, real_data)
                        if isinstance(out, (list, tuple)) and len(out) == 3:
                            mu, alpha, gate = out
                            loss = lossFunc(mu, alpha, gate, label, mask, alpha_reg, gate_reg)
                        else:
                            mu, alpha = out
                            loss = lossFunc(mu, alpha, label, mask)

                    loss.backward()
                    trainer.step(batch_size=data.shape[0], ignore_stale_grad=True)
                    cumLoss += nd.sum(loss).asscalar()
            
                avg_train_loss = cumLoss / nSamples
                history['train_loss'].append(avg_train_loss)

                if tb_logger:
                    tb_logger.add_scalar(tag=f'loss/train_loss', value=avg_train_loss, global_step=e)

                if testX is not None and len(testX) > 0:
                    testLoss, rho50, rho90 = self.probEvaluator(self.model, testX, testCat, testReal, testY, testMaskFuture, lossFunc, alpha_reg=alpha_reg, gate_reg=gate_reg)
                    history['valid_loss'].append(testLoss)
                    history['valid_rho50'].append(rho50)
                    history['valid_rho90'].append(rho90)

                    if tb_logger:
                        tb_logger.add_scalar(tag=f'loss/valid_loss', value=testLoss, global_step=e)
                        tb_logger.add_scalar(tag=f'metrics/valid_rho50', value=rho50, global_step=e)
                        tb_logger.add_scalar(tag=f'metrics/valid_rho90', value=rho90, global_step=e)

                    logger.debug(f"Epoch {e+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Valid Loss: {testLoss:.4f}, Valid Rho50: {rho50:.4f}, Valid Rho90: {rho90:.4f}")

                    current_metric = testLoss
                    if(current_metric < bestValidMetric):
                        bestValidMetric = current_metric
                        best_model_path = self.saveCheckpoint(self.model, mark, testLoss, fold)
                        patience_counter = 0
                        logger.info(f"  -> New best model found (Valid Loss: {current_metric:.4f}). Saving to {best_model_path}")
                    else:
                        patience_counter += 1

                    if (esEpochs > 0) and (patience_counter >= esEpochs):
                        logger.info(f"--- Early stopping triggered after {patience_counter} epochs with no improvement. ---")
                        break
                else:
                    # No validation data
                    logger.info(f"Epoch {e+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

            if best_model_path is None:
                # This happens if training finishes without improvement, or esEpochs <= 0, or no validation set
                best_model_path = self.saveCheckpoint(self.model, mark, -1, fold)
                logger.info(f"--- Training finished. Saving final model to {best_model_path} ---")
        
        finally:
            if tb_logger:
                tb_logger.close()
                    
        return history, best_model_path

if __name__ == "__main__":
    pass