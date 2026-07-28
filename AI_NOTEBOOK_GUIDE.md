# AI Reference Guide — Defense-Detection Project Report Notebook

*Read this whole file before editing anything.* It is the reference an AI needs to safely
extend the project report notebook **`Defense_Detection_Project_Report.ipynb`** (a Google
Colab notebook that documents a final-year research project in chronological order).

> You were pointed here by a handoff prompt; the **specific changes to make** are in that
> prompt. This file gives you the background and the rules. **The most important rule is §3:
> you do not edit the `.ipynb` — you edit `build_notebook.py` and rebuild.**

---

## 1. Project overview

The project builds, in the **ns-3** simulator, a **black-hole attack** on the **OLSR**
routing protocol and **four defenses** against it, all behind one interface
(`OlsrDefenseStrategy`):

- **Watchdog** — cross-layer promiscuous watchdog (Baiad et al., 2014).
- **FPNT** — Fuzzy-Petri-Net trust, propagated via TC (Tan et al., 2015).
- **DCFM / GCOP** — graph-colouring contradiction rules + fictitious-node injection
  (Schweitzer et al., 2025 — the supervisor's own paper).
- **TRUST** — trust-based forward monitoring (Adnane et al., 2013); studied in two
  calibrations (Trust 1.0 vs 2.0).

The **research goal** is *passive detection*: from OLSR control traffic alone, can a
machine-learning model tell **whether** a defense is running and **which** one? The central
finding is a **detectability ↔ efficacy tradeoff**: a better, gentler defense leaves fewer
traces and is *harder* to detect passively.

**Team:** Oded Ofek, Hananel Kadron. **Supervisors:** Nadav Schweitzer (first author of the
DCFM paper) and Dror Mughaz. **Institution:** Jerusalem College of Technology. The notebook
is written in **English**.

---

## 2. The two code repositories

| Repo | Visibility | Branch | In the notebook |
|---|---|---|---|
| **NS-3** — `hananelk26/manet-olsr-project` (attack, defenses, simulations) | **public** | `master` | **clickable links** via `ref()` |
| **ML** — `hananelk26/ML-for-NS3` (learning code + results) | **private** | `main` | **paths only** (no links) via `refml()` |

The ML side has **two pipelines** (both over one feature universe: **95 Core + 33 V2 = 128**):

- **Campaign 1 — the `defense_ml` package** (earlier, June–early July): the publication-grade
  pipeline. The observability ladder 95→67→58→18, the DCFM phantom analysis, Trust 1.0/2.0,
  transfer/open-set, the tradeoff thesis. Path root: `defense_ml/defense_ml_project/`.
- **Campaign 2 — `defense_detection_v4.py`** (later, git 2026-07-19): a leaner pipeline run on
  the *normalised* 128-feature dataset (the 32/29/26/76-feature experiments). Its scripts and
  result trees live under `scripts_for_all_128/`, organised into per-step folders — see §7.1.

> Because the ML repo is **private**, GitHub blob links to it return 404 for any reader.
> That is why ML files are shown as repo-relative **paths**, not links.

---

## 3. How the notebook is built — **the single most important rule**

**The `.ipynb` is generated. Never hand-edit it. Edit `build_notebook.py` and rebuild:**

```bash
python build_notebook.py        # writes Defense_Detection_Project_Report.ipynb
```

`build_notebook.py` builds the notebook from a list of cells declared with two helpers:

- `md("...")` — a Markdown cell (plain string).
- `md(f"...")` — a Markdown cell that interpolates the link helpers below. **In an f-string,
  any literal brace must be doubled** (`{{` / `}}`), e.g. `{{fpnt,dcfm}}`, `∈ {{0,1}}`.
- `code("...")` — a code cell (rarely used; the report is prose).

### Link helpers (defined near the top of `build_notebook.py`)

- `ref(path, label=None)` → a **clickable** NS-3 GitHub link. `path` is repo-relative, e.g.
  `ref("src/olsr/model/olsr-defense-gcop.cc")`. Spaces are `%20`-encoded automatically.
- `refml(path, label=None)` → an ML file as **code text + full path in parentheses** (no
  link, because the repo is private). Example output: `` `config.py` (`defense_ml/…/config.py`) ``.
  If `label` is omitted (or equals `path`) it prints the path once.
- `DML` — a constant = `"defense_ml/defense_ml_project"`, prepended for Campaign-1 files:
  `refml(DML + "/defense_ml/config.py", "config.py")`.

### Two more conventions baked into the script

- **Anchors** are written `<a id="step-7" name="step-7"></a>` — **both** `id` and `name`
  (Colab scrolls via `name`; GitHub/Jupyter via `id`). Every step, guide section, and
  reference target has one. Links use `[text](#step-7)`.
- **Epistemic markers**: `[VERIFIED]` (confirmed by a script's output) vs `[HYPOTHESIS]`
  (plausible but untested — must not be reported as a result). Preserve this distinction.

---

## 4. Notebook structure (7 parts, 33 steps)

Chronological. Each step is one `md(f"...")` cell.

| Part | Steps | Topic |
|---|---|---|
| **I — Foundations** | 1–5 | Kick-off; OLSR/AODV study; literature + features; the attack; the defense interface |
| **II — Defense Implementation** | 6–18 | The 4 defenses built & validated; harness bugs; F1–F5 methodology; attack fix; TRUST added |
| **III — ML Campaign 1 (`defense_ml`)** | 19–24 | Dataset/task/leak-free pipeline; FPNT=TC-size artifact; DCFM=holographic phantom; the ladder 95→67→18; the tradeoff thesis; transfer/open-set/audit |
| **IV — The Transition** | 25 | DCFM realigned to the paper + feature normalisation |
| **V — ML Campaign 2 (`defense_detection_v4`)** | 26–33 | 128-feature schema; dataset gen; Exp 1/2/3 (32/29/26/76 features); the normalisation hypothesis; Exp 2b DCFM-cluster ablation; Step 33 — normalisation leak confirmed from source, transfer test, and the final a-priori 21-feature set |
| **VI — Synthesis** | — | Open questions; planned full-scale campaign |
| **VII — Annotated Source-File Guide** | — | Per-file explanations (NS-3 + both ML pipelines); then References; File Index |

### Per-step template (follow it exactly for new steps)

```python
md(f"""
<a id="step-N" name="step-N"></a>
## Step N — <short title>
**Date:** YYYY-MM-DD

### <Motivation / problem>
...why this step happened...

### <What was done>
...the work...

### <Result / outcome>
...what came out, with [VERIFIED]/[HYPOTHESIS] where relevant...

### Sources
- <NS-3 files via {{ref(...)}}, ML files via {{refml(...)}}>
- Outputs: <result files this run produced>
""")
```

Also present and kept in sync: the **Table of Contents** (near the top), the **References**
list, and the **File Index** (which links NS-3 files and lists ML files + result trees).

---

## 5. Conventions for adding new content

1. **Place chronologically.** Insert the new step where its date belongs, not necessarily at
   the end. Most new project progress is dated *after* the current material, so it usually goes
   at the end of the relevant Part (or a new Part) — but check the date.

2. **If you insert in the middle, renumber the later steps.** The safest way (used before) is a
   one-off regex transformer over `build_notebook.py` that shifts step numbers by `+k`,
   **high-number-first** to avoid collisions, with digit-boundary lookahead:

   ```python
   import re, io
   p = "build_notebook.py"; k = 1          # how many steps you inserted
   s = io.open(p, encoding="utf-8").read()
   for n in range(32, LAST_UNCHANGED, -1):  # e.g. range(32, 24, -1) to shift 25..32 by +k
       m = n + k
       s = re.sub(r'Step %d(?!\d)'  % n, 'Step %d'  % m, s)   # display headers, [Step N], prose
       s = re.sub(r'step-%d(?!\d)'  % n, 'step-%d'  % m, s)   # anchors + (#step-N) links
   io.open(p, "w", encoding="utf-8").write(s)
   ```

   Then manually fix any `[Part X](#step-M)` **labels** whose part changed, and rebuild.

3. **Links:** NS-3 files → `ref()` (clickable). ML files → `refml()` (path only). If the ML repo
   is ever made public, flip `refml` to emit a link (one function change) — until then, paths.

4. **Update the TOC and File Index** to match new steps / new files / new result outputs.

5. **Provenance rule:** prefer **dated session logs and git commit messages** over recollection;
   where they disagree, follow the logs and note the disagreement in *Open Questions*. Mark every
   claim `[VERIFIED]` or `[HYPOTHESIS]`.

6. Keep the notebook **English**; keep the em-dash step-title style `## Step N — Title`.

---

## 6. Validate after every rebuild

Run this right after `python build_notebook.py`. It must print **no broken links** and
**steps sequential**:

```python
import json, re
nb = json.load(open('Defense_Detection_Project_Report.ipynb', encoding='utf-8'))
body = ''.join(''.join(c['source']) for c in nb['cells'])
anchors = set(re.findall(r'<a id="([^"]+)"', body))
hrefs   = re.findall(r'\]\(#([^)]+)\)', body)
steps   = re.findall(r'## Step (\d+) —', body)
print('cells:', len(nb['cells']))
print('BROKEN internal links:', sorted(set(h for h in hrefs if h not in anchors)))
print('steps sequential 1..N:', steps == [str(i) for i in range(1, len(steps)+1)])
print('stray "{" leaks (f-string bugs):', body.count('{ref(') + body.count('{refml('))
```

Expected: `BROKEN internal links: []`, `steps sequential 1..N: True`, `stray "{" leaks: 1`.
The single known leak is a pre-existing one — a `{ref("run_simulations.sh")}` inside a plain
`md("...")` cell that should be `md(f"...")`. It is harmless (it renders literally) and unrelated
to recent edits; leave it unless you are specifically fixing it. If the count rises **above 1**,
a new f-string had an un-doubled literal brace — fix it in `build_notebook.py`.

---

## 7. Where things live

Working directory (`…/Report Generation/`):

| Path | What |
|---|---|
| `build_notebook.py` | **The source of truth** — edit this, then rebuild |
| `Defense_Detection_Project_Report.ipynb` | The generated report (upload to Colab) |
| `AI_NOTEBOOK_GUIDE.md` | This reference guide |
| `md_files/` | The 11 dated NS-3 session logs the report was reconstructed from |
| `md_files_of_ML/` | The two ML session logs (`CLAUDE.md`, `RESEARCH_SUMMARY.md`) |

The two repos may also be cloned locally for deep reads (NS-3 is public; ML is private and must
be cloned by the owner). Datasets and most `results/` trees are large / git-ignored / partly on a
collaborator's machine — so cite files by path even when you cannot open them.

### 7.1 Campaign-2 layout — `scripts_for_all_128/` (ML repo)

Reorganised (2026-07-26) from a flat folder into **per-step folders that mirror the notebook**.
Each step folder holds both its scripts and its result tree, so a step is self-contained:

| Folder | Notebook step | Contents |
|---|---|---|
| `step28_exp1_baseline_32/` | Step 28 | `run_all_defenses.sh`, `verify_features.py`, `rank_importance.py`, `per_defense_tables.py`, `results_run1/` |
| `step29_exp2_ablation_26/` | Step 29 | `diagnose_leakage.py`, `run_behavioral.sh`, `compare_runs.py`, `three_metrics.py`, `results_run2_behavioral/` |
| `step30_exp3_expansion_76/` | Step 30 | `scan_core95.py`, `build_features76.py`, `run_expanded.sh`, `compare_26_vs_76.py`, `features_clean.txt`, `results_run3_expanded/` |
| `Step_32_Experiment_Between_26_and_29_Features/` | Step 32 | `run_drop4.sh`, `compare_drop4.py`, `step32_dcfm_summary.csv`, one `Run_*` sub-folder per single/pair ablation (each with its own `results_run4_drop4/`) |
| `Step_33_Defense_Independent_Normalization_Features/` | Step 33 | `classify_apriori.py`, `run_apriori21.sh`, `compare_apriori.py`, `transfer_test.py`, `features_apriori_lenient.txt`, `results_run9_apriori21/` (+ `transfer_test/`) |
| `_tools/` | — | `table.py` (quick AUC/MCC/TPR table for any results dir) |

Conventions for this tree:

- **Paths inside the scripts are absolute** (`/mnt/d/Hananel/ML-for-NS3/...` in `.py`,
  `$REPO/...` in `.sh`), so the scripts run correctly from anywhere and cross-step comparisons
  (e.g. `compare_runs.py`, `three_metrics.py`) resolve without `../` juggling. If the repo is
  ever moved, update those constants.
- **git tracks only the small text outputs** (`.csv` / `.json` / `.tex`); heavy binaries are
  git-ignored via `**/final_model.pkl` and `**/figures/*.png`. When committing new runs, verify
  no `.pkl`/`.png` are staged before committing.
- The Step-32 `Run_With_29_Features_Without_<X>` sub-folders are **28-feature** runs (the "29"
  is the drop3 starting point); `Run_Without_<X>_AND_<Y>` are **27-feature** runs.
