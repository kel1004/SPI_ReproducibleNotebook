# -*- coding: utf-8 -*-
"""Reproduce the Day-28 OULAD audit in G1_Theme2_S4_ConsentTransparency.docx.

Usage
-----
    python G1_Theme2_S4_Analysis.py --data-dir data/oulad
    python G1_Theme2_S4_Analysis.py --download

The script expects the six OULAD CSV files in --data-dir. With --download it
fetches the CC BY 4.0 UCI release (ID 349) and extracts it there instead.

Outputs
-------
    G1_Theme2_S4_Results.json   all numbers reported in Tables I-IV
    console                     human-readable table summaries
"""
from __future__ import annotations

import argparse
import json
import math
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

UCI_URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
FILES = ["studentInfo.csv", "studentRegistration.csv", "studentVle.csv",
         "studentAssessment.csv", "assessments.csv", "vle.csv"]
KEYS = ["code_module", "code_presentation", "id_student"]


def ensure_data(data_dir: str, download: bool) -> str:
    os.makedirs(data_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(data_dir, f)) for f in FILES):
        return data_dir
    if not download:
        raise FileNotFoundError(
            "Missing OULAD CSV files in %s. Re-run with --download." % data_dir)
    zpath = os.path.join(data_dir, "oulad.zip")
    print("Downloading", UCI_URL)
    urllib.request.urlretrieve(UCI_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in FILES:
                with z.open(name) as src, open(os.path.join(data_dir, base), "wb") as dst:
                    dst.write(src.read())
    return data_dir


def load(data_dir: str):
    # OULAD uses "?" for missing values (notably score and imd_band).
    read = lambda f: pd.read_csv(os.path.join(data_dir, f), low_memory=False,
                                 na_values=["?"])
    return (read("studentInfo.csv"), read("studentRegistration.csv"),
            read("studentVle.csv"), read("studentAssessment.csv"),
            read("assessments.csv"), read("vle.csv"))


def build_cohort(st, reg, sv, sa, ass, vle):
    reg = reg.copy()
    reg["dr"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    reg["du"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
    df = st.merge(reg, on=KEYS, how="inner")
    log = {"raw": int(len(df))}
    df = df[df["dr"].notna()].copy()
    log["missing_registration_date"] = log["raw"] - len(df)
    before = len(df)
    df = df[df["dr"] < 28].copy()
    log["late_registration"] = before - len(df)
    before = len(df)
    df = df[~(df["du"] < 28)].copy()
    log["pre_cutoff_withdrawal"] = before - len(df)
    log["cohort"] = int(len(df))
    log["unique_learners"] = int(df["id_student"].nunique())
    df["atrisk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    log["at_risk_n"] = int(df["atrisk"].sum())
    log["at_risk_pct"] = round(float(df["atrisk"].mean() * 100), 2)

    keys = df[KEYS].drop_duplicates()
    sv2 = sv[sv["date"] <= 28].copy()
    sv2["week"] = (sv2["date"] // 7).clip(lower=0)
    sv2 = sv2.merge(keys, on=KEYS, how="inner")
    log["vle_rows_cohort"] = int(len(sv2))
    log["clicks_cohort"] = int(sv2["sum_click"].sum())

    total = sv2.groupby(KEYS)["sum_click"].sum().rename("total_clicks").reset_index()
    weeks = sv2.groupby(KEYS)["week"].nunique().rename("active_weeks").reset_index()
    sv3 = sv2.merge(vle[["id_site", "code_module", "code_presentation", "activity_type"]],
                    on=["id_site", "code_module", "code_presentation"], how="left")
    acts = sv3.pivot_table(index=KEYS, columns="activity_type", values="sum_click",
                           aggfunc="sum", fill_value=0).reset_index()
    actcols = [c for c in acts.columns if c not in KEYS]

    sa2 = sa[sa["date_submitted"] <= 28].merge(
        ass[["id_assessment", "code_module", "code_presentation"]],
        on="id_assessment", how="left")
    sa2 = sa2.merge(keys, on=KEYS, how="inner")
    log["assessment_rows_cohort"] = int(len(sa2))
    grade = (sa2.groupby(KEYS)["score"].mean()
             .rename("early_grade").reset_index())

    df = (df.merge(total, on=KEYS, how="left")
            .merge(weeks, on=KEYS, how="left")
            .merge(grade, on=KEYS, how="left")
            .merge(acts, on=KEYS, how="left"))
    df[["total_clicks", "active_weeks"]] = df[["total_clicks", "active_weeks"]].fillna(0)
    for c in actcols:
        df[c] = df[c].fillna(0)
    log["grade_n"] = int(df["early_grade"].notna().sum())
    log["imd_missing"] = int(df["imd_band"].isna().sum())
    log["imd_missing_pct"] = round(float(df["imd_band"].isna().mean() * 100), 2)
    log["known_imd"] = int(df["imd_band"].notna().sum())
    log["activity_types"] = len(actcols)
    return df, log, actcols


def describe(df, col, value):
    d = df[df[col] == value] if value is not None else df
    q = d["total_clicks"].quantile([.25, .5, .75]).values
    g = d["early_grade"].dropna()
    qg = g.quantile([.25, .5, .75]).values
    return {
        "n": int(len(d)), "share_pct": round(100 * len(d) / len(df), 2),
        "clicks_mean": round(float(d["total_clicks"].mean()), 1),
        "clicks_sd": round(float(d["total_clicks"].std()), 1),
        "clicks_median": float(q[1]), "clicks_iqr": float(q[2] - q[0]),
        "grade_mean": round(float(g.mean()), 1), "grade_sd": round(float(g.std()), 1),
        "grade_median": float(qg[1]), "grade_iqr": float(qg[2] - qg[0]),
        "grade_n": int(len(g)),
        "risk_n": int(d["atrisk"].sum()),
        "risk_pct": round(float(100 * d["atrisk"].mean()), 2),
    }


def test(df, col, a, b):
    ct = pd.crosstab(df[col], df["atrisk"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = int(ct.values.sum())
    v = math.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    A = df[df[col] == a]["atrisk"]
    B = df[df[col] == b]["atrisk"]
    odds = (A.sum() * (len(B) - B.sum())) / ((len(A) - A.sum()) * B.sum())
    se = math.sqrt(1 / A.sum() + 1 / (len(A) - A.sum()) +
                   1 / B.sum() + 1 / (len(B) - B.sum()))
    return {
        "n": n, "chi2": round(float(chi2), 2), "df": int(dof),
        "p": float(p), "cramers_v": round(float(v), 4),
        "odds_ratio": round(float(odds), 3),
        "ci_low": round(float(math.exp(math.log(odds) - 1.96 * se)), 3),
        "ci_high": round(float(math.exp(math.log(odds) + 1.96 * se)), 3),
        "rate_a_pct": round(float(A.mean() * 100), 2),
        "rate_b_pct": round(float(B.mean() * 100), 2),
    }


def model_audit(df, actcols):
    df = df.copy()
    df["disability"] = df["disability"].map({"Y": True, "N": False}).astype(object)
    num_full = actcols + ["early_grade", "total_clicks", "active_weeks",
                          "num_of_prev_attempts", "studied_credits"]
    cat_full = ["gender", "region", "highest_education", "imd_band",
                "age_band", "disability"]
    num_min = ["early_grade", "total_clicks", "active_weeks"]
    cat_min = []
    num_dem = []
    cat_dem = ["imd_band", "disability", "age_band", "gender"]

    def pipeline(num, cat):
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("oh", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat),
        ])
        return Pipeline([
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=42)),
        ])

    def run(num, cat, name):
        X = df[num + cat]
        y = df["atrisk"].values
        groups = df["id_student"].values
        oof = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            model = pipeline(num, cat)
            model.fit(X.iloc[tr], y[tr])
            oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        pred = (oof >= 0.5).astype(int)

        def sub(mask):
            tn, fp, fn, tp = confusion_matrix(y[mask], pred[mask], labels=[0, 1]).ravel()
            return {"n": int(mask.sum()),
                    "fpr": round(float(fp / (fp + tn) * 100), 1),
                    "fnr": round(float(fn / (fn + tp) * 100), 1),
                    "tpr": round(float(tp / (tp + fn) * 100), 1)}

        low = df["imd_band"] == "0-10%"
        high = df["imd_band"] == "90-100%"
        dis_y = df["disability"] == True  # noqa: E712
        dis_n = df["disability"] == False  # noqa: E712
        a, b, c, d = sub(low), sub(high), sub(dis_y), sub(dis_n)
        eod = abs(a["tpr"] - b["tpr"]) / 100
        aod = 0.5 * (abs(a["fpr"] - b["fpr"]) + abs(a["tpr"] - b["tpr"])) / 100
        eod_d = abs(c["tpr"] - d["tpr"]) / 100
        aod_d = 0.5 * (abs(c["fpr"] - d["fpr"]) + abs(c["tpr"] - d["tpr"])) / 100
        return {
            "model": name, "inputs": len(num) + len(cat),
            "auc": round(float(roc_auc_score(y, oof)), 3),
            "accuracy_pct": round(float(accuracy_score(y, pred) * 100), 1),
            "imd_low": a, "imd_high": b, "disability_y": c, "disability_n": d,
            "eod_imd": round(float(eod), 3), "aod_imd": round(float(aod), 3),
            "eod_disability": round(float(eod_d), 3),
            "aod_disability": round(float(aod_d), 3),
        }

    return [run(num_full, cat_full, "Full"),
            run(num_min, cat_min, "Minimized"),
            run(num_dem, cat_dem, "Demographic")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/oulad")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="G1_Theme2_S4_Results.json")
    args = ap.parse_args()
    data_dir = ensure_data(args.data_dir, args.download)
    st, reg, sv, sa, ass, vle = load(data_dir)

    df, log, actcols = build_cohort(st, reg, sv, sa, ass, vle)
    results = {"cleaning_log": log, "tables": {}, "tests": {}, "models": {}}
    results["tables"]["all"] = describe(df, None, None)
    results["tables"]["imd_0_10"] = describe(df, "imd_band", "0-10%")
    results["tables"]["imd_40_50"] = describe(df, "imd_band", "40-50%")
    results["tables"]["imd_90_100"] = describe(df, "imd_band", "90-100%")
    results["tables"]["disability_y"] = describe(df, "disability", "Y")
    results["tables"]["disability_n"] = describe(df, "disability", "N")
    results["tables"]["age_0_35"] = describe(df, "age_band", "0-35")
    df["age_35_plus"] = df["age_band"].isin(["35-55", "55<="])
    results["tables"]["age_35_plus"] = describe(df, "age_35_plus", True)
    results["tables"]["male"] = describe(df, "gender", "M")
    results["tables"]["female"] = describe(df, "gender", "F")
    results["tests"]["imd"] = test(df, "imd_band", "0-10%", "90-100%")
    results["tests"]["disability"] = test(df, "disability", "Y", "N")
    results["tests"]["gender"] = test(df, "gender", "M", "F")
    results["models"] = model_audit(df, actcols)
    results["imd_rates"] = {
        str(k): {"n": int(v["n"]), "risk_pct": round(float(v["risk"] * 100), 1)}
        for k, v in df.groupby("imd_band")["atrisk"].agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "risk"}).iterrows()
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results, indent=1))
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()
