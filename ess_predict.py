import argparse

from utils import visualization
from utils.visualization import mean_confidence_interval
from constants import MODELS
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from scipy.special import logsumexp


def make_plots(args):
    configs = [args.bow, args.binary_bow, args.boc]
    models = ['{}{}'.format(model, config) for (model, config) in zip(MODELS, configs)]
    precision, recall, logp, kldiv, log_densities, params = visualization.load_results(models)
    visualization.plot_precision(precision, models)
    visualization.plot_recall(recall, models)
    visualization.plot_logp(logp, models)
    visualization.plot_kldiv(kldiv, models)
    for model in models:
        visualization.hist_densities(log_densities[model], model)
        visualization.hist_param(params[model].reshape(-1), model)

def evaluate(models):
    """
    Evaluate accuracy.
    """
    log_densities_models = []
    log_densities, _, ground_truth = visualization.load_test_results(models)
    for model in models:
        print(log_densities[model].shape)
        input()
        log_densities_models.append(np.expand_dims(log_densities[model], axis=1))
        # ll_precisions_models.append(ll_precisions[model])
        # ll_recalls_models.append(ll_recalls[model])

    log_densities_models = np.concatenate(log_densities_models, axis=1)

    predictions, ess = predict_ess(log_densities_models)

    # predict_ll_ensemble

    # predict_ll_ess

    if np.isnan(ess).any():
        print(np.where(np.isnan(ess)))
    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    print("[ESS] f1: {} | accuracy: {} | precision: {}, recall: {}".format(f1, accuracy, precision, recall))
    # print("[LL] precision: {}| recall: {}".\
    #     format(mean_confidence_interval(ll_precisions_models),
    #     mean_confidence_interval(ll_recalls_models)))
    # return f1, accuracy, precision, recall, ll_precisions, ll_recalls

def predict_ess(log_densities_models):
    """
    Predict the class of the inputs via effective sample size (ESS)
    """
    ess = _evaluate_ess(log_densities_models)
    ess_thres = _find_threshold(ess)
    predictions = np.zeros_like(ess).astype(int)
    predictions[ess < ess_thres] = 1
    #if np.isnan(log_density).any():
    #    print(inputs[np.where(np.isnan(log_density))])
    #print(self.threshold)
    return list(predictions), ess

def predict_ll_ensemble(log_densities_models):
    """
    Predict the class of the inputs
    """
    N_models = log_densities_models.shape[1]
    thres_models = _find_threshold(log_densities_models)
    print(thres_models.shape)
    predictions = np.zeros_like(log_densities_models[:,0]).astype(int)
    print(predictions.shape)
    predictions_models = np.zeros_like(log_densities_models).astype(int)
    predictions_models[log_densities_models < thres_models] = 1
    predictions[predictions_models.sum(axis=1)/N_models > 0.5] = 1
    #if np.isnan(log_density).any():
    #    print(inputs[np.where(np.isnan(log_density))])
    #print(self.threshold)
    return list(predictions), log_densities_models

def _find_threshold(nparray):
    threshold = np.nanpercentile(nparray, 10, axis=0)
    return threshold

def _evaluate_ess(log_densities_models):
    # log_densities: N_samples * N_models
    # output log(ess)
    log_w = log_densities_models - logsumexp(log_densities_models, axis=1)
    log_ess = -1.0 * logsumexp(2 * log_w, axis=1)
    return log_ess


if __name__ == '__main__':
    models_filenames = [
        '',
        '',
        ''
    ]
    evaluate(models_filenames)
