# SPI Reproducible Notebook

**Consent and Transparency Standards for Learning Analytics in Philippine HEIs**  
Group 1 · Theme 2 · Aningat, Dumalagan, Magallanes  
Social and Professional Issues · Technological University of the Philippines

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kel1004/SPI_ReproducibleNotebook/v1.0.0?filepath=G1_Theme2_S4_Analysis.ipynb)

## One-click run

Click the Binder badge, wait for the environment to build, then choose **Run → Run All Cells**.

- First launch: about 5–10 minutes while Binder builds and caches the image.
- Analysis: about 2–4 minutes; the notebook downloads the 45 MB OULAD archive.

## What it reproduces

The notebook runs `G1_Theme2_S4_Analysis.py`, the same script supplied with the S4 paper, and reproduces:

| Output | Content |
|---|---|
| Table I | Day-28 cleaning log and cohort counts |
| Table II | Descriptive statistics by subgroup |
| Table III | Chi-square tests, odds ratios, Cramér's V |
| Table IV | Full, minimized, and demographic-only model audit |

Key values:

- Cohort: **27,528 registrations**, **24,835 learners**, **12,150 at risk (44.14%)**.
- IMD 0–10% vs 90–100%: **56.06% vs 34.87%**, OR **2.384** (95% CI 2.123–2.677).
- Disability Y vs N: **53.79% vs 43.13%**, OR **1.535** (95% CI 1.416–1.665).
- Full model: AUC **0.777**, AOD **0.221**.
- Minimized model: AUC **0.725**, AOD **0.086**.
- Demographic-only model: AUC **0.588**, AOD **0.870**.

The final cell asserts that the generated `results.json` matches `expected_results.json` exactly.

## Files

| File | Purpose |
|---|---|
| `G1_Theme2_S4_Analysis.ipynb` | Reproducible notebook; runs the script and displays Tables I–IV |
| `G1_Theme2_S4_Analysis.py` | Tested analysis pipeline |
| `expected_results.json` | Published aggregate results for exact comparison |
| `requirements.txt` | Pinned analysis dependencies |
| `runtime.txt` | Pins Python 3.13 for Binder |

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `G1_Theme2_S4_Analysis.ipynb` and run all cells.

Or run the script directly:

```bash
python G1_Theme2_S4_Analysis.py --download --data-dir data/oulad --out results.json
```

## Reproducibility notes

- Python and direct dependencies are pinned.
- The Binder badge points to tag `v1.0.0`, not the moving `HEAD`.
- The notebook computes every table from OULAD; it does not hard-code table values.
- OULAD is public, de-identified, and licensed CC BY 4.0 (UCI Machine Learning Repository ID 349).
- No re-identification was attempted; only aggregate results are reported.
- OULAD is a United Kingdom distance-learning dataset. The paper uses it as a design stress test, not as a sample of Filipino students.

## Citation

Aningat, C. R. M., Dumalagan, C. J. C., & Magallanes, A. C. *Consent and Transparency Standards for Learning Analytics in Philippine HEIs* (S4 full draft). Social and Professional Issues, Technological University of the Philippines, 2026.
