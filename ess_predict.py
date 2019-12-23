import argparse

from utils import visualization
from utils.visualization import mean_confidence_interval
from constants import MODELS
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from scipy.special import logsumexp
from scipy.stats import sem


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

def evaluate(prefix, models):
    """
    Evaluate accuracy.
    """
    log_densities_models = []
    log_densities, _, ground_truth = visualization.load_test_results(prefix, models)
    for model in models:
        #print(log_densities[model].shape)
        #input()
        log_densities_models.append(np.nan_to_num(np.expand_dims(log_densities[model], axis=1)))
        #, nan=-1.7976931348623157e+30))
        # ll_precisions_models.append(ll_precisions[model])
        # ll_recalls_models.append(ll_recalls[model])

    log_densities_models = np.concatenate(log_densities_models, axis=1)
    #print(log_densities_models)
    print("N_test_samples * N_models: {}".format(log_densities_models.shape))
    # predict_sem_ensemble
    predictions, _ = predict_sem(log_densities_models)
    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    print("[SEM] f1: {} | accuracy: {} | precision: {}, recall: {}".format(f1, accuracy, precision, recall))
    # predict_ess_ensemble
    predictions, _ = predict_ess(log_densities_models)
    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    print("[ESS] f1: {} | accuracy: {} | precision: {}, recall: {}".format(f1, accuracy, precision, recall))
    # predict_ll_ensemble
    predictions, _ = predict_ll_ensemble(log_densities_models)
    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    print("[LL] f1: {} | accuracy: {} | precision: {}, recall: {}".format(f1, accuracy, precision, recall))

    # predict_ll_sem
    predictions, _ = predict_ll_sem(log_densities_models)
    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    print("[LL_SEM] f1: {} | accuracy: {} | precision: {}, recall: {}".format(f1, accuracy, precision, recall))

    # predict_ll_ess
    predictions, _ = predict_ll_ess(log_densities_models)
    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    print("[LL_ESS] f1: {} | accuracy: {} | precision: {}, recall: {}".format(f1, accuracy, precision, recall))

def predict_sem(log_densities_models):
    """
    Predict the class of the inputs via effective sample size (ESS)
    """
    neg_sem_score =  -sem(log_densities_models, axis=1)
    thres = _find_threshold(neg_sem_score)
    predictions = np.zeros_like(neg_sem_score).astype(int)
    predictions[neg_sem_score < thres] = 1
    print('negative sem threshold: {}'.format(thres))
    return list(predictions), neg_sem_score

def predict_ess(log_densities_models):
    """
    Predict the class of the inputs via effective sample size (ESS)
    """
    ess = _evaluate_ess(log_densities_models)
    ess_thres = _find_threshold(ess)
    predictions = np.zeros_like(ess).astype(int)
    predictions[ess < ess_thres] = 1
    print('log(ess) threshold: {}'.format(ess_thres))
    return list(predictions), ess

def predict_ll_ensemble(log_densities_models):
    """
    Predict the class of the inputs
    """
    N_models = log_densities_models.shape[1]
    thres_models = _find_threshold(log_densities_models)
    predictions = np.zeros_like(log_densities_models[:,0]).astype(int)
    predictions_models = np.zeros_like(log_densities_models).astype(int)
    predictions_models[log_densities_models < thres_models] = 1
    predictions[predictions_models.sum(axis=1)/N_models >= 0.5] = 1
    return list(predictions), None

def predict_ll_sem(log_densities_models):
    """
    Predict the class of the inputs
    """
    neg_sem_score =  np.expand_dims(-sem(log_densities_models, axis=1), axis=1)
    N_models = log_densities_models.shape[1]
    ll_sem = log_densities_models - 35*np.exp(neg_sem_score)
    thres_models = _find_threshold(log_densities_models)
    predictions = np.zeros_like(log_densities_models[:,0]).astype(int)
    predictions_models = np.zeros_like(log_densities_models).astype(int)
    predictions_models[ll_sem < thres_models] = 1
    predictions[predictions_models.sum(axis=1)/N_models >= 0.5] = 1
    return list(predictions), ll_sem

def predict_ll_ess(log_densities_models):
    """
    Predict the class of the inputs
    """
    ess = _evaluate_ess(log_densities_models)
    N_models = log_densities_models.shape[1]
    ll_ess = log_densities_models - 5.0*(ess)
    thres_models = _find_threshold(log_densities_models)
    predictions = np.zeros_like(log_densities_models[:,0]).astype(int)
    predictions_models = np.zeros_like(log_densities_models).astype(int)
    predictions_models[ll_ess < thres_models] = 1
    predictions[predictions_models.sum(axis=1)/N_models >= 0.5] = 1
    return list(predictions), ll_ess

def _find_threshold(nparray):
    threshold = np.nanpercentile(nparray, 10, axis=0)
    return threshold

def _evaluate_ess(log_densities_models):
    # log_densities: N_samples * N_models
    # output log(ess)
    log_w = log_densities_models - \
            logsumexp(log_densities_models, axis=1, keepdims=True)
    #print(log_w)
    #print(np.sum(np.exp(log_w), axis=1))
    log_ess = -1.0 * logsumexp(2 * log_w, axis=1, keepdims=True)
    #print(log_ess)
    #print(np.exp(log_ess))
    return log_ess


if __name__ == '__main__':
    prefix = "/data1/yrli/vae-anomaly-detector/results/test"
    models = [
        #'boc00/2019_12_22_12_40/epoch_950-f1_0.6780715396578538',
        #'boc00/2019_12_22_12_40/epoch_760-f1_0.6811145510835913',
        'boc00/2019_12_22_12_40/epoch_870-f1_0.6842105263157894',
        'boc00/2019_12_22_12_40/epoch_880-f1_0.6801242236024845',
        'boc00/2019_12_22_12_40/epoch_900-f1_0.6832298136645962',
        'boc01/2019_12_22_13_15/epoch_940-f1_0.7069767441860465',
        'boc01/2019_12_22_13_15/epoch_870-f1_0.70625',
        'boc01/2019_12_22_13_15/epoch_980-f1_0.7069767441860465',
        'boc02/2019_12_22_13_59/epoch_1000-f1_0.7120743034055728',
        'boc02/2019_12_22_13_59/epoch_870-f1_0.7082683307332294',
        'boc02/2019_12_22_13_59/epoch_920-f1_0.7107692307692307',
        'boc03/2019_12_22_14_42/epoch_1000-f1_0.6944444444444444',
        'boc03/2019_12_22_14_42/epoch_330-f1_0.6924265842349305',
        'boc03/2019_12_22_14_42/epoch_600-f1_0.6850998463901691',
        'boc04/2019_12_22_15_16/epoch_830-f1_0.6861538461538461',
        'boc04/2019_12_22_15_16/epoch_910-f1_0.6810477657935285',
        'boc04/2019_12_22_15_16/epoch_990-f1_0.6862442040185471',
        #'boc05/2019_12_22_19_53/epoch_1000-f1_0.6948356807511739',
        #'boc05/2019_12_22_19_53/epoch_800-f1_0.7069767441860465',
        #'boc05/2019_12_22_19_53/epoch_950-f1_0.712962962962963',
        #'boc06/2019_12_22_19_53/epoch_1000-f1_0.6936236391912909',
        #'boc06/2019_12_22_19_53/epoch_870-f1_0.691131498470948',
        #'boc06/2019_12_22_19_53/epoch_940-f1_0.6941896024464832',
        'boc07/2019_12_22_19_53/epoch_790-f1_0.687211093990755',
        'boc07/2019_12_22_19_53/epoch_910-f1_0.6749611197511665',
        'boc07/2019_12_22_19_53/epoch_990-f1_0.6738794435857804',
        #'boc08/2019_12_22_19_54/epoch_770-f1_0.6992366412213741',
        #'boc08/2019_12_22_19_54/epoch_950-f1_0.6984615384615385',
        #'boc08/2019_12_22_19_54/epoch_970-f1_0.6945736434108528',
        #'boc09/2019_12_22_19_55/epoch_690-f1_0.709480122324159',
        #'boc09/2019_12_22_19_55/epoch_860-f1_0.7058823529411764',
        #'boc09/2019_12_22_19_55/epoch_970-f1_0.6965944272445821',
        #'boc00/2019_12_22_00_25/epoch_940-f1_0.7372429550647371',
        #'boc00/2019_12_22_00_25/epoch_750-f1_0.7350037965072134',
        #'boc00/2019_12_22_00_25/epoch_820-f1_0.734351145038168',
        #'boc01/2019_12_22_03_37/epoch_650-f1_0.6947852760736197',
        #'boc01/2019_12_22_03_37/epoch_950-f1_0.7126436781609196',
        #'boc02/2019_12_22_03_37/epoch_650-f1_0.7215865751334859',
        #'boc02/2019_12_22_03_37/epoch_1000-f1_0.7208588957055215',
        #'boc03/2019_12_22_07_51/epoch_650-f1_0.7174409748667174',
        #'boc03/2019_12_22_07_51/epoch_1000-f1_0.7249042145593869',
        #'boc04/2019_12_22_07_51/epoch_650-f1_0.7304747320061254',
        #'boc04/2019_12_22_07_51/epoch_850-f1_0.7335375191424196'
    ]
    evaluate(prefix, models)
