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
>
> ⚠️ **The header comment in `build_notebook.py` contradicts this.** Lines 8–9 describe the ML
> repo as *public, branch `main`* and define `REPO_ML = "https://github.com/hananelk26/ML-for-NS3/blob/main"`.
> That constant is **dead — zero uses anywhere in the file**, and `refml()` still emits a path
> only. The behaviour matches this table; only the comment is stale. Do not "fix" `refml()` to
> use `REPO_ML` on the strength of that comment.

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

## 4. Notebook structure (7 parts, 40 steps)

Chronological. Each step is one `md(f"...")` cell.

| Part | Steps | Topic |
|---|---|---|
| **I — Foundations** | 1–6 | Kick-off; OLSR/AODV study; literature + features; the attack; the defense interface; the comment to *Scientific Reports* |
| **II — Defense Implementation** | 7–19 | The 4 defenses built & validated; harness bugs; F1–F5 methodology; attack fix; TRUST added |
| **III — ML Campaign 1 (`defense_ml`)** | 20–25 | Dataset/task/leak-free pipeline; FPNT=TC-size artifact; DCFM=holographic phantom; the ladder 95→67→18; the tradeoff thesis; transfer/open-set/audit |
| **IV — The Transition** | 26 | DCFM realigned to the paper + feature normalisation |
| **V — ML Campaign 2 (`defense_detection_v4`)** | 27–40 | 128-feature schema; dataset gen; Exp 1/2/3 (32/29/26/76 features); the normalisation hypothesis; Exp 2b DCFM-cluster ablation; Step 34 — normalisation leak confirmed from source, transfer test, and the final a-priori 21-feature set; Step 35 — generalisation experiments (mobility transfer, cross-defense matrix, LODO) **specified**; Step 36 — cross-defense intersection, the Step-35 prediction overturned; Step 37 — the normalisation hypothesis **measured** on a paired un-normalised DCFM dataset; Step 38 — traffic load varied (one CBR flow instead of three), 33/32/27 sets, the control-plane signature shown invariant and the static↔mobile inversion reversed; **Steps 39–40 — two defense realignments to their source papers** (Watchdog → Baiad 2014/2016 + Marti 2000; TRUST → Adnane 2013), each with an oracle-column sampling defect found in the process. They are not ML steps and sit at the end of Part V only because that is where the chronology put them |
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

   > ⚠️ **The regex silently misses plural references.** `Step %d` does not match `Steps 28–33`,
   > `Steps 8, 12, 14`, or `Steps 3 and 17` — the literal is `Step` + space, and these have
   > `Steps` + space. The *anchor* in `[Steps 28–33](#step-28)` **is** rewritten by the
   > `step-%d` pass, so you end up with correct links carrying wrong labels — and the §6
   > validator cannot see it, because the link resolves. Before shifting, inventory them:
   >
   > ```bash
   > grep -n -o -E "Steps [0-9]+[^)]{0,18}" build_notebook.py | sort -u
   > ```
   >
   > and fix each by hand afterwards. After rebuilding, verify with:
   >
   > ```python
   > bad = [(t,a) for t,a in re.findall(r'\[Steps? (\d+)[^\]]*\]\(#step-(\d+)\)', body) if t != a]
   > ```
   >
   > Also re-check the **TOC list numbers** (they are plain markdown ordinals, untouched by the
   > regex) and the **Part-range table in §4 of this guide**.

3. **Links:** NS-3 files → `ref()` (clickable). ML files → `refml()` (path only). If the ML repo
   is ever made public, flip `refml` to emit a link (one function change) — until then, paths.

4. **Update the TOC and File Index** to match new steps / new files / new result outputs.

5. **Provenance rule:** prefer **dated session logs and git commit messages** over recollection;
   where they disagree, follow the logs and note the disagreement in *Open Questions*. Mark every
   claim `[VERIFIED]` or `[HYPOTHESIS]`.

6. Keep the notebook **English**; keep the em-dash step-title style `## Step N — Title`.

7. **Two traps when documenting or reproducing a `--features-file` run** (both found the hard
   way in Step 37):

   - **The `DataPacketRate` alias only works via the preset.** `resolve_features()` substitutes
     `MacDataPacketRate` for `DataPacketRate` *only* on the `metrics32` branch. Pass an explicit
     `--features` / `--features-file` list against a schema that uses the other name and the
     feature is **dropped with a warning only** — you silently get 26 features where you asked
     for 27, and the run still succeeds. Always diff the list against `head -1` of the CSV first.
   - **List order changes results.** `FeatureSelector`'s correlation pruning is a deterministic
     greedy keep-first over column order, so a re-ordered list can prune a different member of a
     correlated pair. A feature list meant to reproduce a `--drop-features` baseline must be in
     **`METRICS` order**, not alphabetical.

   Related: the emitter writes `L_pdr` to both `PacketDeliveryRatio` and `RxTxPacketRatio`, and
   `1 - L_pdr` to `PacketLossRatio`; `MidMessageRate` and `HnaMessageRate` are identically zero.
   `FeatureSelector` removes all five in-fold, so a nominal 27-feature set is **effectively 23**.
   Quote nominal counts, but footnote the effective one.

8. **Three more traps found in Step 38** (one-flow dataset):

   - **The raw `CoreAndV2` header emits `DataPacketRate` twice** — once in Core group A, once in
     V2 — both from `m_dataPackets / dur`. `pandas` mangles the second to `DataPacketRate.1`, so
     the raw CSV has **133 columns, not 128**. The values are identical, so nothing is wrong
     numerically, but `--feature-set all` on raw data carries a perfectly collinear pair. The
     normalised schema has no collision (Core is `DataPacketRatePerFlow` there).
   - **`run_config.json` misreports `feature_set`.** It records `"metrics32"` for **every** run,
     including `--features-file` runs with 33 or 27 names, because `cfg.feature_set` is never
     updated by `resolve_features()`. `n_base_features_used` is correct — audit that field, not
     `feature_set`.
   - **Effective dimension must be measured on the engineered matrix, not the base features.**
     Squares, cubes and ratios of a pruned base feature routinely survive selection: the 27-set
     on the one-flow static data keeps 8 of 27 base features but **26** engineered columns.
     `run_config.json` reports only `n_total_features` (before selection), so the real figure has
     to be computed by importing v4 and calling its own `engineer_features()` + `FeatureSelector`
     — see `step_36_.../preflight.py`.

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

Expected (as of Step 40): `cells: 61`, `BROKEN internal links: []`,
`steps sequential 1..N: True` with `N = 40`, `stray "{" leaks: 1`.
The single known leak is a pre-existing one — a `{ref("run_simulations.sh")}` inside a plain
`md("...")` cell that should be `md(f"...")`. It is harmless (it renders literally) and unrelated
to recent edits; leave it unless you are specifically fixing it. If the count rises **above 1**,
a new f-string had an un-doubled literal brace — fix it in `build_notebook.py`.

> ⚠️ **The block above cannot be pasted into a Windows `cmd.exe` one-liner as written.** Two
> characters break it: `<` in `<a id="..."` is a redirection operator, and the `"` in `[^"]+`
> closes the quoted argument. Replace them with their hex escapes (`\x3c`, `\x22`) and the em
> dash with `\u2014` — the console can mangle a pasted `—`, and that failure is **silent**:
> the step regex matches nothing, `N` comes out `0`, and `steps sequential` reports `True`
> against an empty list. Working one-liner:
>
> ```cmd
> python -c "import json,re;nb=json.load(open('Defense_Detection_Project_Report.ipynb',encoding='utf-8'));body=''.join(''.join(c['source']) for c in nb['cells']);a=set(re.findall('\x3ca id=\x22([^\x22]+)\x22',body));h=re.findall('\\]\\(#([^)]+)\\)',body);s=re.findall('## Step (\\d+) \\u2014',body);print('cells:',len(nb['cells']));print('BROKEN links:',sorted(set(x for x in h if x not in a)));print('sequential:',s==[str(i) for i in range(1,len(s)+1)],'N =',len(s));print('stray braces:',body.count('{ref(')+body.count('{refml('));print('label mismatch:',[(t,q) for t,q in re.findall('\\[Steps? (\\d+)[^\\]]*\\]\\(#step-(\\d+)\\)',body) if t!=q])"
> ```
>
> Also set `PYTHONUTF8=1` before `python build_notebook.py` on Windows; the file contains em
> dashes and other non-ASCII text, and the default console encoding can raise
> `UnicodeEncodeError`. Running the build and the validator from WSL avoids all of this.

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
| `step28_exp1_baseline_32/` | Step 29 | `run_all_defenses.sh`, `verify_features.py`, `rank_importance.py`, `per_defense_tables.py`, `results_run1/` |
| `step29_exp2_ablation_26/` | Step 30 | `diagnose_leakage.py`, `run_behavioral.sh`, `compare_runs.py`, `three_metrics.py`, `results_run2_behavioral/` |
| `step30_exp3_expansion_76/` | Step 31 | `scan_core95.py`, `build_features76.py`, `run_expanded.sh`, `compare_26_vs_76.py`, `features_clean.txt`, `results_run3_expanded/` |
| `Step_32_Experiment_Between_26_and_29_Features/` | Step 33 | `run_drop4.sh`, `compare_drop4.py`, `step32_dcfm_summary.csv`, one `Run_*` sub-folder per single/pair ablation (each with its own `results_run4_drop4/`) |
| `Step_33_Defense_Independent_Normalization_Features/` | Step 34 | `classify_apriori.py`, `run_apriori21.sh`, `compare_apriori.py`, `transfer_test.py`, `features_apriori_lenient.txt`, `results_run9_apriori21/` (+ `transfer_test/`) |
| `Step_34_Cross_Defense_Intersection/` | Step 36 | `step34_lodo.py` and a `features/` dir (incl. `features_27_step32.txt`, the canonical 27-set); five result trees — `results_rf/`, `results_rf_noleak/`, `results_27/`, `results_27_rf/`, `results_27_rf_PERMUTED/` |
| `step_35_dcfm_non_normalized/` | Step 37 | `preflight.py`, `run_dcfm_nonorm.sh`, `compare_normalized_vs_raw.py`, `features_27.txt`, `features_32.txt`, `feature_name_map.csv`, `results_run_nonorm_27/`, `results_run_nonorm_32/`, `preflight_report/`, `comparison/` |
| `step_36_dcfm_non_normalized_1ch_18msgs/` | Step 38 | `probe_1ch.py`, `preflight.py`, `run_all.sh`, `compare_1ch_vs_3ch.py`, `compare_accuracy_prev_vs_cur.py`, `PREDICTIONS.md`; `33_features/`, `32_features/`, `27_features/` (each with its `features_NN.txt`, `results_static/`, `results_mobile/`); `preflight_report/`, `comparison/`, `logs/` |
| `step31_normalization_test/` | — | Ad-hoc normalisation-leak probe (`test_normalization_leakage.py`, `step31_run.log` at the tree root). Not tied to a notebook step; predates the per-step reorganisation |
| `_tools/` | — | `table.py` (quick AUC/MCC/TPR table for any results dir) |

Conventions for this tree:

- **Paths inside the scripts are absolute** (`/mnt/d/Hananel/ML-for-NS3/...` in `.py`,
  `$REPO/...` in `.sh`), so the scripts run correctly from anywhere and cross-step comparisons
  (e.g. `compare_runs.py`, `three_metrics.py`) resolve without `../` juggling. If the repo is
  ever moved, update those constants.
- **git tracks only the small text outputs** (`.csv` / `.json` / `.tex`); heavy binaries are
  git-ignored via `**/final_model.pkl` and `**/figures/*.png`. When committing new runs, verify
  no `.pkl`/`.png` are staged before committing.
- The Step-33 (folder `Step_32_…`) `Run_With_29_Features_Without_<X>` sub-folders are **28-feature** runs (the "29"
  is the drop3 starting point); `Run_Without_<X>_AND_<Y>` are **27-feature** runs.
- **Step 35 adds no folder here, and its code may not exist.** The step is written as a change
  to `defense_detection_v4.py` itself (section `[9] Transfer experiments`, flags
  `--transfer-mobility` / `--transfer-defense` / `--lodo`), with outputs under the v4 default
  results root `defense_ml/defense_ml_project/results/30_schema33/paper_v4/transfer/` — inside
  the *Campaign-1* tree, because `paper_v4/` has always been v4's default output directory.
>
  ⚠️ **Verified absent (2026-08-05).** The current `defense_detection_v4.py` has sections
  `[1]`–`[8]` only. There is **no** section `[9]`, no transfer/LODO flags, and `--defense`
  still defaults to `"fpnt"` rather than `None`. The repository owner confirmed this file is
  the current one. So Step 35 documents work that is not in the pipeline, and Step 36 — which
  did run LODO — used its own standalone script (`Step_34_Cross_Defense_Intersection/`)
  instead. **Do not cite Step 35's flags as available, and do not assume `paper_v4/transfer/`
  is populated.** Resolve against git history before relying on either.
