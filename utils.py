import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch 

'''
Returns AUC and AP scores given true and false scores
'''
def get_score(pscore, nscore):
    ntp = pscore.size(0)
    ntn = nscore.size(0)

    score = torch.cat([pscore, nscore]).numpy()
    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    ap = average_precision_score(labels, score)
    auc = roc_auc_score(labels, score)

    return [auc, ap]


'''
Returns the threshold that achieves optimal TPR and FPR
(Can be tweaked to return better results for FPR if desired)

Does this by checking where the TPR and FPR cross each other
as the threshold changes (abs of TPR-(1-FPR))

Please do this on TRAIN data, not TEST -- you cheater
'''
def get_optimal_cutoff(pscore, nscore, fw=0.5):
    ntp = pscore.size(0)
    ntn = nscore.size(0)

    tw = 1-fw

    score = torch.cat([pscore, nscore]).numpy()
    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    fpr, tpr, th = roc_curve(labels, score)
    fn = np.abs(tw*tpr-fw*(1-fpr))
    best = np.argmin(fn, 0)

    print("Optimal cutoff %0.4f achieves TPR: %0.2f FPR: %0.2f on train data" 
        % (th[best], tpr[best], fpr[best]))
    return th[best]