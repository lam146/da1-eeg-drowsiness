"""Refine public EEG assistance using cached features and train-only scaling."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from develop_pilot_model import session_weights
from try_cui_transfer import SPECTRAL, score


def predict(source, training, testing, columns, fraction, separate):
    sw, tw = session_weights(source), session_weights(training)
    if separate and fraction:
        source_scaler = StandardScaler().fit(source[columns], sample_weight=sw)
        target_scaler = StandardScaler().fit(training[columns], sample_weight=tw)
        sx, tx = source_scaler.transform(source[columns]), target_scaler.transform(training[columns])
        test_x = target_scaler.transform(testing[columns])
    else:
        joined = pd.concat([training[columns], source[columns]])
        weights = np.r_[(1 - fraction) * tw, fraction * sw / sw.sum() * tw.sum()]
        scaler = StandardScaler().fit(joined, sample_weight=weights)
        sx, tx = scaler.transform(source[columns]), scaler.transform(training[columns])
        test_x = scaler.transform(testing[columns])
    values = np.vstack([tx, sx])
    labels = np.r_[training.label, source.label]
    weights = np.r_[(1 - fraction) * tw, fraction * sw / sw.sum() * tw.sum()]
    included = weights > 0
    model = LogisticRegression(C=.1, max_iter=3000)
    model.fit(values[included], labels[included], sample_weight=weights[included])
    return model.predict_proba(test_x)[:, 1]


def run(folder):
    source = pd.read_csv(folder / "public_fp1_features.csv")
    target = pd.read_csv(folder / "target_features_PROVISIONAL.csv", dtype={"date": str})
    columns = SPECTRAL + [name for name in source if name.startswith(("ar_", "sample_entropy_"))]
    candidates = {"target_only": (0., False)}
    for name, separate in [("joint", False), ("separate", True)]:
        for fraction in (.1, .25, .5, 1.):
            candidates[f"{name}_{fraction}"] = (fraction, separate)
    rows, predictions, selection = [], [], {}
    for branch, features in [("spectral", SPECTRAL), ("spectral_ar_entropy_relative", columns)]:
        for person in sorted(target.subject_id.unique()):
            train, test = target.loc[target.subject_id.ne(person)], target.loc[target.subject_id.eq(person)]
            inner_splits = list(LeaveOneGroupOut().split(train, groups=train.subject_id + "/" + train.date))
            ranking = {}
            for name, (fraction, separate) in candidates.items():
                pred = np.empty(len(train), dtype=int)
                for left, right in inner_splits:
                    prob = predict(source, train.iloc[left], train.iloc[right], features, fraction, separate)
                    pred[right] = prob >= .5
                ranking[name] = float(balanced_accuracy_score(train.label, pred))
            chosen = max(ranking, key=ranking.get)
            selection[f"{branch}/{person}"] = dict(chosen=chosen, training_day_scores=ranking)
            for name, (fraction, separate) in candidates.items():
                prob = predict(source, train, test, features, fraction, separate)
                for identifier in ([name, "selected_on_training_days"] if name == chosen else [name]):
                    rows.append(dict(branch=branch, model=identifier, person=person, **score(test.label, prob)))
                    predictions.append(test[["subject_id", "session_id", "window_start_ms", "label"]].assign(
                        branch=branch, model=identifier, probability=prob, predicted=(prob >= .5).astype(int)))
            print(branch, person, chosen, flush=True)
    folds = pd.DataFrame(rows)
    summary = folds.groupby(["branch", "model"])[["balanced_accuracy", "awake_recall", "drowsy_recall"]].mean()
    folds.to_csv(folder / "tuned_folds.csv", index=False)
    summary.to_csv(folder / "tuned_comparison.csv", encoding="utf-8-sig")
    pd.concat(predictions).to_csv(folder / "tuned_predictions_PROVISIONAL.csv", index=False)
    (folder / "tuning_selection.json").write_text(json.dumps(dict(selection=selection, features=columns,
        limits="Development only. Source-domain scaling uses public data; target scaling fits only target training rows. Held-out person never fits scaler. No absolute power or amplitude features in this refinement."), indent=2), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder)
