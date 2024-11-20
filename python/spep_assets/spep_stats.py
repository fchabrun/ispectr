"""
Created Nov 20 by Chabrun F
"""


def get_bootstrap_metric_ci(groundtruths, preds, metric, bootstraps=1000, alpha=.05):
    from sklearn import metrics
    from sklearn import utils
    import numpy as np

    assert alpha * bootstraps % 1 == 0, "Must choose alpha and bootstraps such as alpha * bootstraps % 1 == 0"
    if metric == "accuracy":
        base_metric = ((preds >= .5) * 1 == groundtruths).sum() / len(groundtruths)
    elif metric == "roc_auc":
        base_metric = metrics.roc_auc_score(groundtruths, preds)
    elif metric == "f1":
        base_metric = metrics.f1_score(groundtruths, preds >= .5)
    else:
        assert False, f"Unknown {metric=}"
    results = []
    for seed in range(bootstraps):
        # re bootstrap our test set
        bootstrap_groundtruths, bootstrap_preds = utils.resample(groundtruths, preds, replace=True, n_samples=None, random_state=seed, stratify=groundtruths)
        # bootstrap_idxes = np.random.RandomState(seed=seed).choice(len(groundtruths), size=len(groundtruths), replace=True)
        # bootstrap_groundtruths = groundtruths[bootstrap_idxes]
        # bootstrap_preds = preds[bootstrap_idxes]
        # compute our model's accuracy for this bootstrap
        if metric == "accuracy":
            model_metric = ((bootstrap_preds >= .5) * 1 == bootstrap_groundtruths).sum() / len(bootstrap_groundtruths)
        elif metric == "roc_auc":
            model_metric = metrics.roc_auc_score(bootstrap_groundtruths, bootstrap_preds)
        elif metric == "f1":
            model_metric = metrics.f1_score(bootstrap_groundtruths, bootstrap_preds >= .5)
        else:
            assert False, f"Unknown {metric=}"
        results.append(model_metric)
    # compute final accuracy
    results = np.sort(results)
    lo_value, hi_value = results[int(alpha * bootstraps)], results[int((1 - alpha) * bootstraps)]
    mean_value = np.mean(results)
    metric_name = "Accuracy" if metric == "accuracy" else "ROC-AUC" if metric == "roc_auc" else "F1" if metric == "f1" else "(unknown metric)"
    if metric == "accuracy":
        base_metric, mean_value, lo_value, hi_value = [f"{100 * metric_value:.1f}" for metric_value in [base_metric, mean_value, lo_value, hi_value, ]]
    elif metric == "roc_auc":
        base_metric, mean_value, lo_value, hi_value = [f"{metric_value:.2f}" for metric_value in [base_metric, mean_value, lo_value, hi_value, ]]
    elif metric == "f1":
        base_metric, mean_value, lo_value, hi_value = [f"{metric_value:.2f}" for metric_value in [base_metric, mean_value, lo_value, hi_value, ]]
    else:
        assert False, f"Unknown {metric=}"
    print(f"Base {metric_name}: {base_metric} // bootstrap {metric_name}: {mean_value} ({100 * (1 - alpha):.0f}%CI: {lo_value}-{hi_value})")
