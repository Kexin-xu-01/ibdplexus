"""
Train Random Forest classifiers (CD vs UC) on slide-level Virchow2 embeddings.

Runs 5-fold patient-level CV on both prism2_base and prism2_diagnostic embeddings.

Inputs
------
- /home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv
- /home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/
      20x_224px_0px_overlap/prism2_base/          (.h5 files, 2560-d)
      20x_224px_0px_overlap/prism2_diagnostic/     (.h5 files, 3072-d)

Outputs  (all under /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/)
--------
- {model}_fold_metrics.csv        per-fold AUC, AP, accuracy, F1, confusion matrix
- {model}_slide_predictions.csv   per-slide predicted label and prob_uc
- {model}_summary.json            aggregated statistics
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
SPLITS_CSV = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv'
EMB_DIRS = {
    'prism2_base': ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
                    '20x_224px_0px_overlap/prism2_base'),
    'prism2_diagnostic': ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
                          '20x_224px_0px_overlap/prism2_diagnostic'),
}
OUT_DIR = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/02_04_imaging_allsites/results'

# ── RF hyperparameters ────────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)

# Label encoding: CD=0, UC=1; positive class = UC
LABEL_MAP = {"Crohn's disease": 0, "Ulcerative colitis": 1}


def load_embeddings(slide_ids, emb_dir):
    X, valid_ids = [], []
    for sid in slide_ids:
        path = os.path.join(emb_dir, f'{sid}.h5')
        if os.path.exists(path):
            with h5py.File(path, 'r') as h:
                X.append(h['features'][:])
            valid_ids.append(sid)
        else:
            print(f"  WARNING: missing embedding for {sid}")
    return np.array(X, dtype=np.float32), valid_ids


def run_cv(model_name, emb_dir, slides_df, out_dir=OUT_DIR):
    print(f"\n{'='*60}")
    print(f"  {model_name}  |  CD=0, UC=1, positive class=UC")
    print(f"{'='*60}")

    id2label = dict(zip(slides_df['slide_id'], slides_df['label']))
    id2diag  = dict(zip(slides_df['slide_id'], slides_df['diagnosis']))

    fold_results, all_preds = [], []

    for fold in range(5):
        train_df = slides_df[slides_df['fold'] != fold]
        val_df   = slides_df[slides_df['fold'] == fold]

        X_train, train_ids = load_embeddings(train_df['slide_id'].tolist(), emb_dir)
        X_val,   val_ids   = load_embeddings(val_df['slide_id'].tolist(),   emb_dir)

        y_train = np.array([id2label[s] for s in train_ids])
        y_val   = np.array([id2label[s] for s in val_ids])

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)

        proba   = clf.predict_proba(X_val)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]      # P(UC)
        y_pred  = clf.predict(X_val)

        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)

        fold_res = dict(
            fold=fold,
            n_train=int(len(y_train)), n_val=int(len(y_val)),
            auc=round(auc, 4), ap=round(ap, 4),
            accuracy=round(cr['accuracy'], 4),
            cd_precision=round(cr['CD']['precision'], 4),
            cd_recall=round(cr['CD']['recall'], 4),
            cd_f1=round(cr['CD']['f1-score'], 4),
            uc_precision=round(cr['UC']['precision'], 4),
            uc_recall=round(cr['UC']['recall'], 4),
            uc_f1=round(cr['UC']['f1-score'], 4),
            tn=int(cm[0, 0]), fp=int(cm[0, 1]),
            fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        )
        fold_results.append(fold_res)
        print(f"  Fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr['accuracy']:.4f}  "
              f"CD-F1={cr['CD']['f1-score']:.4f}  UC-F1={cr['UC']['f1-score']:.4f}")

        for sid, yt, yp, ys in zip(val_ids, y_val, y_pred, y_score):
            all_preds.append(dict(
                slide_id=sid, fold=fold, diagnosis=id2diag[sid],
                true_label=int(yt), pred_label=int(yp),
                prob_uc=round(float(ys), 5)))

    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]

    print(f"\n  AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  AP:       {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    pd.DataFrame(fold_results).to_csv(
        os.path.join(out_dir, f'{model_name}_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(out_dir, f'{model_name}_slide_predictions.csv'), index=False)

    summary = dict(
        model=model_name,
        mean_auc=round(float(np.mean(aucs)), 4),
        std_auc=round(float(np.std(aucs)), 4),
        mean_ap=round(float(np.mean(aps)), 4),
        std_ap=round(float(np.std(aps)), 4),
        mean_acc=round(float(np.mean(accs)), 4),
        std_acc=round(float(np.std(accs)), 4),
        folds=fold_results,
        label_encoding={'CD': 0, 'UC': 1},
        positive_class='UC',
        score_column='prob_uc',
        rf_params=RF_PARAMS,
    )
    with open(os.path.join(out_dir, f'{model_name}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--emb_base_dir", type=str, default=EMB_DIRS["prism2_base"],
                   help="Directory containing prism2_base .h5 files.")
    p.add_argument("--emb_diag_dir", type=str, default=EMB_DIRS["prism2_diagnostic"],
                   help="Directory containing prism2_diagnostic .h5 files.")
    p.add_argument("--out_dir", type=str, default=OUT_DIR,
                   help="Output directory for results.")
    args = p.parse_args()

    emb_dirs = {
        "prism2_base":       args.emb_base_dir,
        "prism2_diagnostic": args.emb_diag_dir,
    }
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    slides_df = pd.read_csv(SPLITS_CSV)
    slides_df['label'] = slides_df['diagnosis'].map(LABEL_MAP)
    print(f"Loaded {len(slides_df)} slides")
    print(slides_df['diagnosis'].value_counts().to_string())

    results = {}
    for model_name, emb_dir in emb_dirs.items():
        results[model_name] = run_cv(model_name, emb_dir, slides_df, out_dir)

    print('\n\n=== FINAL COMPARISON ===')
    print(f"{'Model':<22} {'AUC':<20} {'AP':<20} {'Accuracy'}")
    for name, s in results.items():
        print(f"{name:<22} {s['mean_auc']:.4f} ± {s['std_auc']:.4f}   "
              f"{s['mean_ap']:.4f} ± {s['std_ap']:.4f}   "
              f"{s['mean_acc']:.4f} ± {s['std_acc']:.4f}")
    print(f"\nResults saved to {out_dir}/")


if __name__ == '__main__':
    main()
