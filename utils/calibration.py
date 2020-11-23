import numpy as np


def brier_score(labels, probs):
    probs[np.arange(len(probs)), labels] -= 1
    return np.sqrt((probs ** 2).sum(1)).mean(0)


def classification_calibration(labels, probs, bins=10, tag=""):
    brier = brier_score(labels, probs.copy())

    preds = np.argmax(probs, axis=1)
    total = labels.shape[0]
    probs = np.max(probs, axis=1)
    lower = 0.0
    increment = 1.0 / bins
    upper = increment
    accs = np.zeros([bins + 1], dtype=np.float32)
    gaps = np.zeros([bins + 1], dtype=np.float32)
    confs = np.arange(0.0, 1.01, increment)
    ECE = 0.0
    for i in range(bins):
        ind1 = probs >= lower
        ind2 = probs < upper
        ind = np.where(np.logical_and(ind1, ind2))[0]
        lprobs = probs[ind]
        lpreds = preds[ind]
        llabels = labels[ind]
        acc = np.mean(np.asarray(llabels == lpreds, dtype=np.float32))
        prob = np.mean(lprobs)
        if np.isnan(acc):
            acc = 0.0
            prob = 0.0
        ECE += np.abs(acc - prob) * float(lprobs.shape[0])
        gaps[i] = np.abs(acc - prob)
        accs[i] = acc
        upper += increment
        lower += increment
    ECE /= np.float(total)
    MCE = np.max(np.abs(gaps))

    accs[-1] = 1.0

    print(f'ECE {tag}: ' + str(np.round(ECE * 100.0, 2)) + '\n')
    print(f'MCE {tag}: ' + str(np.round(MCE * 100.0, 2)) + '\n')
    print(f'Brier score {tag}: ' + str(np.round(brier * 100., 2)) + '\n')