#!/usr/bin/env python3
"""
Builds the chronological project report as a Colab notebook (.ipynb).

Edit the CELLS list below and re-run:  python build_notebook.py
Output: Defense_Detection_Project_Report.ipynb

NS-3 links resolve to the real public repository (below). ML links use the REPO_ML
placeholder until that repository's URL is supplied; a single find & replace of REPO_ML
then resolves every ML link.
"""

import json
from pathlib import Path

# NS-3 repository (attack, defenses, interface, simulations) — public, branch: master.
REPO = "https://github.com/hananelk26/manet-olsr-project/blob/master"
# ML repository (learning code + results) — public, branch: main.
REPO_ML = "https://github.com/hananelk26/ML-for-NS3/blob/main"
# defense_ml package root inside the ML repo (Campaign 1 pipeline).
DML = "defense_ml/defense_ml_project"
# Campaign-2 Step-34 folder (cross-defense LODO experiment).
S34 = "scripts_for_all_128/Step_34_Cross_Defense_Intersection"
S35 = "scripts_for_all_128/step_35_dcfm_non_normalized"
# Campaign-2 Step-38 folder (un-normalised DCFM, one CBR flow of 18 messages).
S36 = "scripts_for_all_128/step_36_dcfm_non_normalized_1ch_18msgs"
OUT = Path(__file__).parent / "Defense_Detection_Project_Report.ipynb"

# --------------------------------------------------------------------------
# Reference helpers
# --------------------------------------------------------------------------

def ref(path: str, label: str | None = None) -> str:
    """Markdown link to a file in the NS-3 repository (spaces %20-encoded in the URL)."""
    return f"[`{label or path}`]({REPO}/{path.replace(' ', '%20')})"


def refml(path: str, label: str | None = None) -> str:
    """Reference to a file in the **private** ML repository. NOT a hyperlink — a GitHub
    blob URL to a private repo 404s for any reader without access — so this renders the
    (optional) short name plus the full repo-relative path in parentheses, as code text.
    When only a path is given it is shown once."""
    if label is None or label == path:
        return f"`{path}`"
    return f"`{label}` (`{path}`)"


# --- NS-3: attack + protocol + interface ---
ATTACK      = "src/olsr/model/olsr-routing-protocol.cc"
ATTACK_H    = "src/olsr/model/olsr-routing-protocol.h"
IFACE       = "src/olsr/model/olsr-defense-strategy.h"
IFACE_CC    = "src/olsr/model/olsr-defense-strategy.cc"

# --- NS-3: current defense implementations (one file per defense) ---
D_WATCHDOG  = "src/olsr/model/olsr-watchdog-defense.cc"
D_WATCH_H   = "src/olsr/model/olsr-watchdog-defense.h"
D_FPNT      = "files for all defenses/FPNT/olsr-defense-fpnt.cc"
D_FPNT_H    = "files for all defenses/FPNT/olsr-defense-fpnt.h"
D_GCOP      = "src/olsr/model/olsr-defense-gcop.cc"
D_GCOP_H    = "src/olsr/model/olsr-defense-gcop.h"
D_TRUST     = "files for all defenses/Trust/olsr-trust-defense.cc"
D_TRUST_H   = "files for all defenses/Trust/olsr-trust-defense.h"

# --- NS-3: feature schema + per-defense swap folder ---
FEATURES    = "scratch/olsr_window_features.h"
SWAP        = "files for all defenses"

# --- ML repository ---
ML_PIPE     = "defense_detection_v4.py"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ==========================================================================
# TITLE
# ==========================================================================
md(f"""
# Detection and Classification of Black-Hole Defense Mechanisms in OLSR-Based MANETs

### A Chronological Project Report

---

**Institution:** Jerusalem College of Technology
**Students:** Oded Ofek, Hananel Kadron
**Supervisors:** Nadav Schweitzer, Dror Mughaz
**Simulator:** ns-3 (`ns-3-dev`, v3.45)
**Routing protocol:** OLSR (RFC 3626)
**Project period:** October 2025 – July 2026
**Report compiled:** July 2026

---

> **Note on supervision and sources.** The DCFM/GCOP defense implemented in this
> project is based on Schweitzer et al. (2025) — of which the project supervisor,
> **Nadav Schweitzer**, is the first author. The 33-feature "V2" schema used in the
> final machine-learning stage was supplied directly by him. This is recorded here
> for transparency: it explains both the depth of access to the algorithm's intent
> and the choice of DCFM as one of the four defenses under study.
""")

# ==========================================================================
# ABSTRACT
# ==========================================================================
md("""
## Abstract

Mobile Ad-hoc Networks (MANETs) running the Optimized Link State Routing (OLSR)
protocol are vulnerable to the **black-hole attack**, in which a malicious node
attracts routing traffic by falsifying its topological centrality and then silently
discards the data packets that transit through it. A substantial body of literature
proposes countermeasures, but published defenses are rarely accompanied by a
maintained implementation for a current simulator.

This project asks a question that inverts the usual framing. Rather than *"how do we
defend an OLSR network against a black-hole attack?"*, we ask: **"given only passively
observable traffic from an OLSR network, can a machine-learning model determine whether
a black-hole defense is active — and if so, which one?"** The question matters from an
attacker's perspective (reconnaissance before committing to an attack strategy) and
from a defender's perspective (a defense whose presence is trivially detectable is a
defense whose evasion can be planned).

To answer it, we implemented in ns-3 a four-mechanism black-hole attack against OLSR,
a pluggable defense-strategy interface, and **four independent defenses** drawn from
the literature: a cross-layer **Watchdog** (Baiad et al., 2014), a Fuzzy-Petri-Net
trust mechanism **FPNT** (Tan et al., 2015), the graph-colouring contradiction
mechanism **DCFM/GCOP** (Schweitzer et al., 2025), and a fourth trust-based defense
(**TRUST2**). Each defense was validated to demonstrate that it genuinely recovers the
Packet Delivery Ratio (PDR) lost to the attack. A 128-feature schema was then designed,
and a labelled dataset was generated by running, per defense and per mobility regime,
2,000 simulations that each emit four feature vectors — one per measurement window
(baseline / attack-only / defense-only / defense+attack).

The central scientific finding of the project is **not** the classifier's accuracy but
the analysis of why early accuracy figures were untrustworthy. Initial runs produced
ROC-AUC values up to 1.0000; permutation-importance analysis showed these were driven
by **single features encoding simulation-configuration constants rather than network
behaviour** — that is, label leakage. Three methodological results follow, and are the
principal contributions of this report:

1. **ROC-AUC conceals the magnitude of leakage.** Removing six leaking features cost
   FPNT/mobile only −0.16 ROC-AUC, but collapsed TPR@1%FPR by a factor of 5.6
   (0.9985 → 0.1778). Operational metrics, not ranking metrics, expose leakage.
2. **Univariate leakage screening is insufficient when a pipeline engineers ratios and
   products.** A feature whose univariate AUC is *exactly* 0.500 — chance — carried 31%
   of the importance in a model achieving AUC 1.0000, by interacting with a second
   weakly-informative feature to reconstruct a removed leak.
3. **Mobility degrades passive detectability across defenses** — with the notable and
   still-unexplained exception of DCFM, where mobility makes detection *easier*.

A root-cause hypothesis attributing the leakage to the **normalisation denominators**
rather than to any individual feature is stated precisely, together with the test that
would confirm it. That hypothesis is **explicitly not yet tested**, and is marked as
such throughout.
""")

# ==========================================================================
# HOW TO READ
# ==========================================================================
md(f"""
## How to Read This Report

### Structure

The report is **chronological**. Each step carries the date on which the work was done,
the problem that motivated it, what was done, the outcome, and links to the relevant
source files. Where a step's outcome was later revised, the revision appears at its own
date rather than being retro-fitted into the earlier entry — the point of the document
is to record the path, including the wrong turns, which in this project were often the
most informative part.

### Epistemic status markers

The machine-learning analysis distinguishes rigorously between what was measured and
what was inferred. Two markers are used and are load-bearing:

| Marker | Meaning |
|---|---|
| **[VERIFIED]** | Confirmed by direct output of a script that was executed. Reproducible from the artefacts. |
| **[HYPOTHESIS]** | A mechanistic explanation consistent with the evidence but **not yet tested**. Must not be reported as a finding. |

### Source-code references

Each step links only to the **current** implementation of the component it concerns —
not to every historical revision. The linked set is deliberately small:

- the black-hole attack,
- the defense-strategy interface,
- one implementation file per defense,
- the feature schema,
- the simulation harnesses.

The machine-learning section is referenced at finer granularity, because that is where
the project's analytical contribution lies.

> **Link status.** NS-3 file references are **clickable links** to the public repository
> [`hananelk26/manet-olsr-project`]({REPO.rsplit("/blob/", 1)[0]}) (branch `master`). The ML
> repository `hananelk26/ML-for-NS3` (branch `main`) is **private**, so its files are given
> as **repo-relative paths in code font, not links** — a private-repo URL would return 404
> for any reader. The ML work spans **two pipelines** — the earlier `defense_ml` package
> ([Part III](#step-20)) and the later `defense_detection_v4` ([Part V](#step-27)).

### Provenance of this report

This report was reconstructed from four sources: eleven dated session logs written during
the project (`md_files/`), the running project journal, the project's own README and
source code, and its **git commit history** (≈ 33 project commits dated
2026-04-15 → 2026-07-17). **Where the contemporaneous logs and later recollection
disagree, this report follows the logs**, and records the disagreement in
[Open Questions](#open-questions) rather than silently choosing a version.
""")

# ==========================================================================
# TOC
# ==========================================================================
md("""
## Table of Contents

**Part I — Foundations (Oct 2025 – Feb 2026)**
1. [Kick-off and problem definition](#step-1) — 2025-10-26
2. [Protocol study: OLSR and AODV](#step-2) — Nov 2025
3. [Literature review and feature harvesting](#step-3) — 2025-11-17, 2025-11-26
4. [Implementing the black-hole attack; selecting three defense families](#step-4) — 2025-12-03
5. [The defense-strategy interface](#step-5)
6. [A published defense that cannot work, and the comment to *Scientific Reports*](#step-6) — 2026-02-18

**Part II — Defense implementation and validation (Feb – May 2026)**

7. [Two working defenses: Watchdog and FPNT](#step-7) — 2026-02-18
8. [DCFM/GCOP: algorithm correctness](#step-8) — 2026-04-16
9. [DCFM/GCOP: the MAC-saturation artefact and penalty tuning](#step-9) — 2026-04-17
10. [Watchdog: hardening and the four-phase harness](#step-10) — 2026-04-19
11. [Watchdog: two false-positive bugs](#step-11) — 2026-04-22
12. [Watchdog: risk analysis and algorithm design](#step-12) — 2026-04-23
13. [Watchdog: multi-topology evaluation](#step-13) — 2026-04-25
14. [The propagation-loss root-cause bug](#step-14) — 2026-04-29
15. [DCFM/GCOP: milestone and deviations from the paper](#step-15) — 2026-04-29
16. [Fixing the evaluation methodology: F1–F5](#step-16) — 2026-05-01
17. [Watchdog: harness parity and calibration](#step-17) — 2026-05-03
18. [Correcting the attack; supervisor review](#step-18) — 2026-05-06, 2026-05-13, 2026-05-18
19. [Adding the fourth defense: TRUST2](#step-19)

**Part III — Machine Learning, Campaign 1: the `defense_ml` pipeline (Jun – early Jul 2026)**

20. [The first ML campaign: dataset, task, and a leak-free pipeline](#step-20) — 2026-06-14 → 07-02
21. [Baseline detection, the red flag, and FPNT as a single artifact](#step-21)
22. [DCFM: a broad, "holographic" phantom signature](#step-22) — 2026-06-23 → 25
23. [The observability ladder 95 → 67 → 18, and the external observer](#step-23)
24. [Two planes, and the central tradeoff thesis](#step-24)
25. [Advanced experiments: generalisation, novelty, defending the method](#step-25)

**Part IV — The Transition (late Jun – mid Jul 2026)**

26. [DCFM realignment to the paper; the 24 June feature run](#step-26) — 2026-06-24 → 07-10

**Part V — Machine Learning, Campaign 2: the `defense_detection_v4` pipeline (Jul 2026)**

27. [The 128-feature schema](#step-27)
28. [Dataset generation](#step-28) — 2026-07-13
29. [Experiment 1 — baseline on 32 features; leakage discovered](#step-29)
30. [Experiment 2 — univariate screen and ablation to 26 features](#step-30)
31. [Experiment 3 — expansion to 76 features; leakage returns](#step-31) — 2026-07-19
32. [Root-cause analysis: the normalisation hypothesis](#step-32)
33. [Experiment 2b — single-feature ablation of the DCFM cluster](#step-33) — 2026-07-24
34. [Defense-independent features: normalisation leak, transfer, and the final 21-feature set](#step-34) — 2026-07-27
35. [Generalisation as a pipeline capability: mobility transfer, cross-defense matrix, and LODO](#step-35) — 2026-07-29
36. [Cross-defense intersection: detecting an unseen defense; the Step-35 prediction overturned](#step-36) — 2026-08-04
37. [The normalisation hypothesis measured: DCFM re-run on un-normalised features](#step-37) — 2026-08-05
38. [Traffic load as a variable: DCFM un-normalised with one CBR flow](#step-38) — 2026-08-09
39. [Realigning the Watchdog defense to its source papers, and a measurement bug](#step-39) — 2026-08-19

**Part VI — Synthesis**

40. [Open questions](#open-questions)
41. [Planned full-scale campaign](#full-campaign)

**Part VII — Annotated Source-File Guide**

42. [How the four windows are measured — the `Enabled` cold-start](#guide-coldstart)
43. [`src/olsr/model/` — protocol core, interface, defenses](#guide-model)
44. [`scratch/` — feature schema and simulations](#guide-scratch)
45. [`files for all defenses/` — the per-defense swap sets](#guide-swap)
46. [Repository-root batch scripts](#guide-scripts)
47. [Reproducing the dataset — the exact commands](#guide-repro)

**Reference**

48. [References](#references)
49. [File index](#file-index)
""")

# ==========================================================================
# PART I
# ==========================================================================
md("""
---
# Part I — Foundations
*October 2025 – February 2026*
---
""")

md("""
<a id="step-1" name="step-1"></a>
## Step 1 — Kick-off and problem definition
**Date:** 2025-10-26

### Context
Project kick-off meeting. Present: students Oded Ofek and Hananel Kadron; supervisors
Nadav Schweitzer and Dror Mughaz.

### What was defined
The central goal was set: develop an **AI model capable of identifying which defense
mechanisms are active** in a mobile ad-hoc network (MANET). This framing — detecting
the defense rather than the attack — is what distinguishes the project from the bulk of
the MANET-security literature, and it drove every subsequent decision, including the
deliberate choice (Step 13) to keep one defense entirely passive so that classifying it
would be genuinely difficult.

### Tasks assigned
1. **Protocol study.** Independent study of the operating principles and message
   structure of the **AODV** and **OLSR** routing protocols.
2. **Literature review.** Locate and read academic work on intrusion-detection systems
   (IDS) in MANETs, focused on AODV and OLSR (search terms: *"MANET IDS AODV"*,
   *"MANET IDS OLSR"*). The stated primary objective was **not** to catalogue defenses
   but to **map the features** researchers used to detect attacks — since the model
   under construction would need to detect the presence of those very features.
3. **Environment.** Begin familiarisation with the **ns-3** simulator.

### Outcome
Scope, method, and the two-protocol reading list fixed. OLSR was subsequently chosen as
the sole target protocol.
""")

md("""
<a id="step-2" name="step-2"></a>
## Step 2 — Protocol study: OLSR and AODV
**Date:** November 2025

### OLSR (RFC 3626) — the target protocol
OLSR is a **proactive** link-state protocol for networks with no routers, in which each
node reaches only nearby nodes by radio. Rather than flooding every message through
every neighbour, OLSR gives every node a full view of the topology and a routing table.
Its operation rests on four stages:

1. **HELLO messages** (every 2 s) — each node advertises its neighbour list. After two
   HELLO rounds a node knows both its 1-hop and its 2-hop neighbours: any address in a
   received HELLO that is not in the node's own neighbour list must belong to a
   neighbour-of-a-neighbour.
2. **MPR selection** — each node elects a minimal subset of its 1-hop neighbours
   (**Multi-Point Relays**) sufficient to reach all of its 2-hop neighbours. Each node
   tracks both the MPRs it selected and the set of nodes that selected *it* (its **MPR
   selector set**, which is not itself advertised in HELLO).
3. **TC messages** (every 5 s) — a node with a non-empty MPR selector set advertises
   who selected it. Only MPRs re-broadcast TC messages, so a TC reaches the whole
   network at a fraction of the cost of a flood.
4. **Routing-table computation** — from the accumulated TC messages, every node
   computes a next hop for every destination.

**Why this structure is attackable.** MPR election is a *popularity contest decided on
self-reported data*. A node that lies about its neighbourhood wins the election. Every
attack and every defense in this project operates on that fact.

### AODV — studied for contrast
AODV is **reactive**: routes are discovered on demand via `RREQ`/`RREP` flooding, torn
down via `RERR`, and freshness is arbitrated by destination sequence numbers (DSN).
AODV was studied for comparison and to inform the feature taxonomy (Step 3), but was
not implemented.

### Outcome
A working model of *why* OLSR is vulnerable, and the vocabulary (HELLO, TC, MPR, MPR
selector, willingness, ANSN) in which the rest of the project is expressed.
""")

md("""
<a id="step-3" name="step-3"></a>
## Step 3 — Literature review and feature harvesting
**Dates:** 2025-11-17, 2025-11-26

### Goal
Read widely on black-hole defenses in OLSR and, from each paper, extract **candidate
features** — both *active* features (evidence that specific messages are circulating)
and *passive* features (e.g. elevated routing overhead).

### Papers summarised
| Paper | Mechanism | Feature contribution |
|---|---|---|
| von Mulert, Welch, Seah (2012) — *Security Threats and Solutions in MANETs: AODV & SAODV* | Survey of AODV attack surface and cryptographic defenses | Taxonomy of attack classes; crypto-detection features (packet size, signature behaviour) |
| *An Effective Intrusion Detection Approach for OLSR* | Semantic consistency checks on HELLO/TC | The four consistency rules that later inform DCFM's contradiction rules |
| *Misbehavior Nodes Detection and Isolation for MANETs OLSR* | `PVM` probe + `AFM` control messages | Detection of probe/response traffic patterns |
| *An Improved Security OLSR against Black Hole based on FANET* | Per-neighbour reliability table (−1…1) biasing MPR selection | MPR-vs-connectivity mismatch; route-change frequency |
| **Baiad et al. (2014)** — *Cooperative Cross-Layer Detection … VANET-OLSR* | Cross-layer watchdog (network + MAC RTS/CTS) | **Selected → Defense 1 (Watchdog)** |
| **Tan et al. (2015)** — *Trust Based Routing … OLSR-based MANET* | Fuzzy Petri Net trust, propagated via TC | **Selected → Defense 2 (FPNT)** |
| **Schweitzer et al. (2025)** — *Achieving MANET Protection without Superfluous Fictitious Nodes* | GCOP/GCOHP graph colouring + contradiction rules | **Selected → Defense 3 (DCFM/GCOP)** |
| *New MPR Computation … Against Single Black Hole* | `ttl=2` HELLO + `ACK_HELLO` + K score | Surveyed; **not implemented** |
| **Srinivasan (2025)** — *Innovative Cross-Layer Defense Mechanisms … Black Hole and Wormhole Attacks* (Scientific Reports) | Cross-layer "Enhanced SVM" | **Shortlisted, attempted, abandoned** — the published algorithm cannot work; see [Step 6](#step-6) |

### A negative result worth recording
One of the shortlisted papers — **Srinivasan (2025)**, in *Scientific Reports* — was found,
on close reading and attempted implementation, to contain **substantive errors**: the
mechanism as specified cannot perform the classification the paper claims for it. The
project went on to submit a formal comment to the journal's editor. Because that is a dated
piece of work in its own right, it is recorded at its own date — see [Step 6](#step-6),
which sets out the three specific defects.

### Feature funnel — first pass
Roughly **150** candidate features were harvested across the surveyed papers. After
discarding features requiring an observer position unavailable to a passive adversary,
approximately **100** remained. This set was later formalised as the **Core-95** schema
(Step 27).

### Outcome
Three defense families selected for implementation — **Trust-based**, **Cryptographic**,
and **Statistical-anomaly** — with one paper chosen from each on the basis of its
publication venue and content.
""")

md(f"""
<a id="step-4" name="step-4"></a>
## Step 4 — Implementing the black-hole attack
**Date:** 2025-12-03 (weekly goal); implementation and refinement through May 2026

### Design
The attack is implemented **inside the OLSR routing protocol itself** rather than as a
separate module, and is toggled per node by two attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `IsMalicious` | `bool` | Enables all four attack mechanisms on this node |
| `SpoofedLinksCount` | `uint32` | Number of phantom neighbours advertised |

### The four mechanisms
The attacker does not merely drop packets — dropping alone attracts no traffic. It first
**manufactures topological centrality**, then exploits it:

| # | Mechanism | Layer | Effect |
|---|---|---|---|
| 1 | **Willingness manipulation** | Control | Sets `WILL_ALWAYS` in HELLO → neighbours preferentially elect it as MPR |
| 2 | **ANSN poisoning** | Control | Adds **+200** to the Advertised Neighbor Sequence Number → its TC messages always look "fresher" than legitimate ones and supersede them |
| 3 | **Link spoofing** | Control | Advertises symmetric links to non-existent neighbours at `200.0.0.x` → inflates apparent 2-hop coverage, which is exactly the quantity MPR election maximises |
| 4 | **Silent packet drop** | Data | `RouteInput` returns `true` without invoking the unicast callback — the packet vanishes with no ICMP error |

Mechanisms 1–3 attack the **control plane** to win the MPR election; mechanism 4 attacks
the **data plane** once won. ANSN poisoning turned out to have consequences well beyond
its intended role — it pollutes the topology set of *every* legitimate node, which is
what later forced Contradiction Rule 2 to be disabled (Steps 8, 15).

### Source
- Attack implementation: {ref(ATTACK)}, {ref(ATTACK_H)}

### Note on a later correction
As originally written, the attack advertised links only to **fictitious** nodes. It was
corrected in May 2026 to also claim links to **real nodes from the network** — see
[Step 18](#step-18).
""")

md(f"""
<a id="step-5" name="step-5"></a>
## Step 5 — The defense-strategy interface
**Date:** implemented alongside the attack (late 2025), stable thereafter

### Rationale
Four defenses had to be comparable, swappable at configuration time, and implementable
without touching the OLSR core. The **Strategy pattern** was adopted:
`RoutingProtocol` holds a `Ptr<OlsrDefenseStrategy> m_defenseStrategy`, set via the
`DefenseStrategy` attribute. Every node is an instance of `RoutingProtocol`, so **each
node can carry a different defense — or none**.

Three considerations drove the choice:

1. **Comparability** — defenses must be exchanged without perturbing the protocol.
2. **Isolation** — the OLSR core stays a clean RFC 3626 reference implementation with a
   small set of instrumentation points; all security logic lives in strategy classes.
3. **Testability** — a Null-Object default (`OlsrDefenseNull`, all hooks empty) lets the
   module run with no defense at all, which is exactly what the *baseline* measurement
   window requires.

### The hooks
The interface ({ref(IFACE)}) exposes **19 pure-virtual hooks** in seven categories, every
one of which `OlsrDefenseNull` implements as an empty body:

| Category | Hooks |
|---|---|
| Lifecycle | `Setup`, `DoDispose`, `PeriodicCheck` |
| Identification | `IsMalicious`, `GetBlacklist` |
| Control plane | `OnRecvHello`, `OnRecvTc`, `OnTcGenerated` |
| Data plane | `OnDataPacketReceived`, `OnDataPacketForwarded`, `OnDataPacketDropped` |
| Promiscuous sniffer | `OnNeighborForwardedPacket`, `OnRtsReceived`, `OnCtsReceived` |
| Cross-layer metrics | `OnQueueStatusReport`, `OnEnergyStateUpdate`, `OnMacTxFailure`, `OnSelfReliabilityReport` |
| Fictitious-node decision | `RequiresFictitiousNode` |

### Enforcement sites in the OLSR core
The strategy is consulted at nine call sites — the defense does not need to reimplement
routing, it only needs to be asked:

- `DoInitialize()` → `Setup()`
- `RecvOlsr()` → `OnRecvHello()` / `OnRecvTc()`, **before** duplicate- and
  malicious-filtering, so the defense sees every incoming control message
- `RecvOlsr()` → drops the message if `IsMalicious(originator) || IsMalicious(sender)`
- `SendHello()` → `RequiresFictitiousNode()`; when true, appends a fictitious link at
  the deterministic address `m_mainAddress + 65536`
- `SendTc()` → `OnTcGenerated()` (ground-truth mirror, before jitter)
- `MprComputation()` → filters blacklisted nodes from the N and N2 sets
- `RoutingTableComputation()` → filters blacklisted neighbours / topology / HNA tuples
- `RouteInput()` → the **IMP** mechanism: drops packets whose next hop is blacklisted
- 1 Hz `HandleDefenseTimer` → `PeriodicCheck()` plus queue / energy / self-reliability reports

Additionally, `SetupPromiscuousMonitor` / `MonitorSnifferRx` feed neighbour-forwarding
and RTS/CTS observations, and `DoInitialize` forces `RtsCtsThreshold = 0` on every WiFi
device so that **every** unicast — not only large frames — is preceded by RTS/CTS. That
last decision is what makes the Watchdog's MAC-layer signal observable at all, and it
became a question in its own right: does forcing RTS/CTS constitute a give-away that the
ML model could cheat on? (See [Step 18](#step-18).)

### Source
- Interface: {ref(IFACE)}, {ref(IFACE_CC)}
""")

md("""
<a id="step-6" name="step-6"></a>
## Step 6 — A published defense that cannot work, and the comment to *Scientific Reports*
**Dates:** flaw found while attempting implementation (Nov – Dec 2025); comment letter submitted **2026-02-18**

### Context
The survey of [Step 3](#step-3) shortlisted candidate defenses to build. One of them was
**Jagadeesan Srinivasan, *"Innovative cross-layer defense mechanisms for black hole and
wormhole attacks in wireless ad-hoc networks"*, Scientific Reports (April 2025)** —
attractive because it was recent, cross-layer in the same sense as Baiad et al., and aimed
at exactly our attack. Implementation was attempted. It failed, and the failure was not in
our code.

### What the attempt revealed
Three findings, in ascending order of severity.

**1. The classifier is mathematically null.**
Detection rests on an *"Enhanced Support Vector Machine"* (E-SVM). Any SVM decision
function has the form `f(x) = ⟨w, x⟩ + b`. In the *Detection and Isolation Phase* section
the bias term is derived as *"the difference of this value from one"* — that is, writing
the computed weight factor as `V`, the paper defines `b = 1 − V`. Substituting:

```
f(x) = ⟨w, x⟩ + b = V + (1 − V) = 1
```

The decision function collapses to the **constant 1, independent of the input `x`**. Every
input yields the same output, so no class separation is possible: the mechanism is a **null
classifier**. It follows that the performance reported in *Results and discussion* —
including an 89.28% packet delivery ratio — cannot have been produced by the algorithm the
paper describes.

**2. The method is not an SVM.**
The manuscript presents the standard SVM apparatus — Lagrangian multipliers, hyperplane
optimisation — but the mechanism actually described in *Proposed Work* classifies each point
by a **mean ± standard-deviation threshold per input dimension**. There is no quadratic
optimisation, no identification of support vectors, no margin maximisation and no kernel.
Functionally this is **Z-score anomaly detection**. "Enhanced SVM" names a different
algorithm from the one implemented.

**3. The pseudo-code is not executable.**
Algorithm 1 (the "E-SVM Algorithm", p. 7) branches on `If blackhole condition` and
`Elif wormhole condition`. Neither is defined anywhere — not as a formal criterion, not as
a mathematical expression, not in the surrounding prose. There is nothing to implement.

### The comment letter
A formal comment setting out these three points was submitted to the editor of *Scientific
Reports* on **2026-02-18**, asking the editorial board to contact the author for a
clarification to readers. It is signed by five authors — **Nadav Schweitzer, Oded Ofek,
Hananel Kadron, Dror Mughaz and Ariel Stulman**: the project's two students, both
supervisors, and a co-author of the DCFM paper this project later implemented
(Schweitzer et al., 2025; see [References](#references)).

### Why a negative result about someone else's paper belongs in this report
Three reasons, and the third generalises beyond this project:

1. **It consumed real project time.** Without it, a reader reconstructing the timeline is
   left wondering where the months between the survey and the first working defense went.
2. **Negative findings about published work are rarely reported**, and are therefore rarely
   available to the next group that picks the same paper off the same search results.
3. **The flaw is invisible to reading and visible only to implementation.** Points 1 and 3
   in particular do not surface from skimming a methodology section; they surface the moment
   someone tries to turn the equations into code and finds there is nothing to write.
   **Attempted implementation is a form of peer review that publication does not currently
   include** — and this project performed it by accident, simply by trying to build what it
   had read.

There is also a symmetry worth naming, because it runs through the whole report. Everything
that follows is an extended argument that **a reported number is only as good as the
mechanism that could have produced it**: the FlowMonitor artefact of [Step 9](#step-9), the
three separate bugs of [Steps 9, 13, 15](#step-9) that each made the metrics look *better*
while the system was more broken, and the entire leakage analysis of [Part V](#step-27).
Point 1 above is that same argument turned outward: an 89.28% cannot come from a constant
function.

### Status
**[VERIFIED]** — the letter exists, is dated 2026-02-18, and carries the five signatures
named above; the three technical points are taken from its text. The **editorial outcome is
not recorded in this report**, and nothing here should be read as a resolved dispute.

### Sources
- Project document: *"Comment on 'Innovative cross-layer defense mechanisms for blackhole
  and wormhole attacks in wireless ad-hoc networks'"* — letter to the editor, *Nature
  Scientific Reports*, 2026-02-18 (held with the project papers; not in either code
  repository)
- The paper commented on: Srinivasan (2025), listed under [References](#references)
- The survey that shortlisted it: [Step 3](#step-3)
""")

# ==========================================================================
# PART II
# ==========================================================================
md("""
---
# Part II — Defense Implementation and Validation
*February – May 2026*
---

The four defenses were not built in the order they are numbered, and each required a
different kind of work. The Watchdog needed **adaptation** (the paper's per-packet
algorithm is not expressible in ns-3). DCFM/GCOP needed **correction** (the
implementation diverged from the paper in ways that mattered). Both needed a shared
evaluation harness before any comparison between them meant anything.

A convention used throughout: the **four-phase measurement design**. Every validation
run measures four windows back to back within a single simulation — `baseline`,
`attack_only`, `defense_only`, `defense_vs_attack` — separated by stabilisation
intervals. Running all four in one process keeps topology, RNG stream, and channel
identical across phases, so differences between windows are attributable to attack and
defense state rather than to simulation variance. This design later became the source of
the dataset's four vectors per run.
""")

md(f"""
<a id="step-7" name="step-7"></a>
## Step 7 — Two working defenses: Watchdog and FPNT
**Date:** 2026-02-18

### Status at this point
Two defenses implement the `OlsrDefenseStrategy` interface and work. Both are, at
bottom, forms of watchdog.

### Defense 1 — Watchdog (Baiad et al., 2014)
Each node acts as a watchdog over its neighbours. When node *A* forwards a packet
through *B* toward *C*, *A* listens promiscuously for *B* to relay it. If *B* does not
relay within a timeout, the cross-layer logic disambiguates:

| Observation | Interpretation |
|---|---|
| *B* sent no RTS at all | **Malicious** — it never even tried |
| *B* sent RTS, *C* answered CTS, no data followed | **Malicious** — it was cleared to send and didn't |
| *B* sent RTS, no CTS observed | **Collision** — exculpated, not accused |

**Deviation from the paper.** The paper specifies a *cooperative* decision across nodes
but does not specify the aggregation protocol. In our implementation **each node decides
independently**. This is not merely a shortcut — see the deliberate justification in
[Step 13](#step-13).

**Two implementation additions beyond the paper:**
1. A sophisticated attacker might send RTS but never transmit, while *A* is simply too
   far from *C* to hear the CTS. To close this, *A* also flags *B* if it observes *B*
   emitting **excessive RTS**. This heuristic later proved to be the source of a serious
   false-positive bug — see [Step 11](#step-11).
2. A **reputation system with decay** replaced binary accusation: forwarding raises the
   score, each missed forward lowers it, and only crossing a threshold blacklists. This
   was introduced specifically to avoid punishing a node that drops packets because it
   is *congested* rather than malicious.

**An early failure that shaped the design.** In the first validation topology (left
column, right column, two bridge nodes in the middle), PDR never exceeded 76%. The
defense correctly identified the attacker and rerouted through the honest backup — and
then, seconds later, began flagging **the backup too**. The backup was not malicious; it
was drowning in the traffic that had just been diverted onto it. *Congestion is
indistinguishable from malice to a naive watchdog.* This observation drove the
reputation system, the congestion-disambiguation logic of [Step 10](#step-10), and the
three-guard commit policy of [Step 13](#step-13).

### Defense 2 — FPNT (Tan et al., 2015)
Each node watchdogs its neighbours and feeds the observations into a **Fuzzy Petri Net**
— a graph/automaton that maps the accumulated evidence about a neighbour to a trust
score. Unlike Defense 1, nodes **share** these scores by embedding them in TC messages,
so every node holds trust values for the whole network, not only its neighbours.

Routing is then computed by **Dijkstra weighted by trust**, using a max-min criterion:
for each candidate path take its *lowest*-trust node, and choose the path whose lowest-
trust node is the *highest*. The best path is the one whose weakest link is strongest.

**Consequence for classification:** FPNT is an **active** defense — it adds data to TC
messages. That extra payload is exactly what the ML model later latched onto, and it is
the origin of the FPNT leakage story that dominates Part III.

### Weekly goal set
Begin extracting features from the two working defenses; find a **passive** black-hole
defense for OLSR to serve as Defense 3.

### Sources
- Watchdog: {ref(D_WATCHDOG)}, {ref(D_WATCH_H)} *(an earlier variant named `olsr-defense-cooperative.{{cc,h}}` was later consolidated into this file and is not tracked in the current repository)*
- Interface: {ref(IFACE)}
""")

md(f"""
<a id="step-8" name="step-8"></a>
## Step 8 — DCFM/GCOP: algorithm correctness
**Date:** 2026-04-16

### The defense
DCFM/GCOP implements Schweitzer et al. (2025). It is a **control-plane** defense: it
never watches packet forwarding. Instead each node fact-checks the topology claims in
incoming HELLO messages against its own view, and — crucially — **injects a fictitious
node into its own HELLO** when doing so would force a lying neighbour into a detectable
contradiction. The paper's contribution is deciding *when* that injection is necessary,
since injecting always is wasteful:

- **GCOP** (Algorithm 1) — depth-limited BFS with node colouring (green = 1-hop,
  blue = 2-hop, yellow = 3-hop). Decides whether a fictitious node must be advertised.
- **GCOHP** (Algorithm 2) — detects the 6-node "hexagon" topology in which GCOP returns
  a false negative.

The paper is explicit about its own scope (§3.4, §7): *"gcop and gcohp are limited to
node-isolation and gray-hole attacks."* Black-hole is **not** a claimed capability. This
bound framed every evaluation that followed.

### Issues found and fixed
| # | Issue | Fix |
|---|---|---|
| A | `RunGcohpAlgorithm()` implemented only **Case 1** (yellow closing node); **Case 2** (blue closing node, Fig. 5(b)) was missing → some hexagons undetected | Added the Case-2 loop with the `(g′,b) ∉ E₃ ∧ (g″,b) ∉ E₃` no-chord conditions |
| B | Contradiction rules evaluated **before network convergence** (TC = 5 s, HELLO = 2 s) → false positives on legitimate nodes | Added `m_startTime`; suppress evaluation for the first 45 s |
| C | **Rules 2 and 3 are corrupted by ANSN poisoning** — both read the topology set, which the attacker has polluted | Added `HasKnownMaliciousNeighbor()`; suppress Rules 2/3 once an attacker is known |
| D | When false positives are widespread, **IMP blocks every route** — with 18 of 19 legitimate nodes blacklisted, every next hop is flagged and all packets drop | Consequence of A–C; addressed by fixing them |

Issue C is the important one. The attack's ANSN poisoning does not merely mislead
routing — it **weaponises the defense against the network**, turning honest nodes into
suspects. The defense's own rules become the denial-of-service.

Also added: `ReactivateDefenseStrategy()` in the routing protocol, because
`DoInitialize()` runs only once at `Simulator::Run()`, and the four-phase harness
installs the defense **mid-simulation** at t = 200 s.

### Results — small network (20-node bridge, static, `SpoofedLinksCount=5`)
| Phase | Tx | Rx | PDR | Suspects | Attacker blacklisted |
|---|---:|---:|---:|---:|:---:|
| baseline | 18 | 18 | 100.0% | 0 | — |
| attack_only | 18 | 0 | **0.0%** | 0 | — |
| defense_only | 18 | 18 | **100.0%** | 0 | NO |
| defense_vs_attack | 7 | 0 | **0.0%** | 18 | **YES** |

**The attacker is detected and the PDR is still zero.** Detection is necessary but not
sufficient: ANSN poisoning false-positives 18 legitimate nodes, IMP blocks both the
attacker *and* the backup, and no path remains.

### Results — large network (50 nodes, 750×1000 m, `SpoofedLinksCount=40`, Rule 2 disabled)
| Phase | PDR | Suspects | Attacker blacklisted |
|---|---:|---:|:---:|
| baseline | 100.0% | 0 | — |
| attack_only | **100.0%** | 0 | — |
| defense_only | 100.0% | 0 | NO |
| defense_vs_attack | **46.0%** | 14 | **YES** |

`attack_only = 100%` because in a 50-node random network with ~6 neighbours per node,
senders simply route around the attacker. **Rule 3 effectiveness scales with network
size** — it fires reliably at n ≥ 50 and fails at n < 20, confirming a theoretical
prediction.

### Verdict
GCOP provides **partial detection** of black-hole but **cannot prevent** it — consistent
with the paper's own stated scope. Its role in the project is complementary: Watchdog
monitors the **data plane**, DCFM the **control plane**.

### Sources
- Defense: {ref(D_GCOP)}, {ref(D_GCOP_H)}
""")

md(f"""
<a id="step-9" name="step-9"></a>
## Step 9 — DCFM/GCOP: the MAC-saturation artefact and penalty tuning
**Date:** 2026-04-17

### The artefact — a measurement lying to us
The first simulation (22 nodes, ten concurrent 20 kbps OnOff flows = 200 kbps) reported:

| Phase | PDR |
|---|---:|
| baseline | 60.0% |
| **attack_only** | **100.0%** |
| defense_only | 89.5% |
| defense_vs_attack | 32.6% |

**The attack appeared to be a defense.** The cause is a subtle property of FlowMonitor:
its `tx` counter is read at the IP layer immediately before MAC hand-off. When the MAC
saturates and the transmit queue overflows, packets are dropped from the transmit path
**and never counted as `tx` at all**. The reported PDR is therefore
`surviving_rx / artificially_deflated_tx` — a ratio over a denominator that shrank.

The lesson generalises well beyond this project: *a congestion-bound topology cannot
measure a routing attack*, because relieving contention (which a black hole does, by
absorbing traffic) improves the metric you are using to detect it.

**Fix:** redesign around a single `UdpClient`/`UdpServer` flow, 18 packets × 512 B at
2 s intervals — a rate the channel sustains. `attack_only` then correctly dropped to 0%.

### Contradiction rules — final form
| Rule | Test | Paper-faithful? |
|---|---|---|
| **1a — Bait** | Sender claims a link to *our* fictitious address (`MainIP + 65536`) → only a liar could know it | Yes |
| **1b — Spoofed range** | Sender claims links in `200.0.0.0/8` | **No — specific to our attacker** |
| **1c — Asymmetry** | Sender claims a symmetric link to our 1-hop neighbour *z*, but *z* does not confirm | Yes (paper's Rule 1) |
| **3 — Over-coverage** | Sender claims ≥ 70% of all known nodes as neighbours (guarded at > 5 known nodes) | Heuristic weakening |
| **2 — MPR-missing** | *Disabled* — reads the ANSN-poisoned topology set | — |

**Rule 1b is an honest weakness and is documented as such.** It works only because our
attacker uses a recognisable address range; a smarter attacker choosing plausible IPs
bypasses it entirely, at which point Rule 1c is the paper-faithful fallback.

### Penalty-duration tuning
The window during which a flagged node stays blacklisted without renewal:

| Penalty | `defense_only` PDR | `defense_vs_attack` PDR | Verdict |
|---|---:|---:|---|
| **5 s** | **~100%** | **35–46%** | **Adopted** |
| 10 s | 55.6% | 70.0% | State leaks across phases |
| 30 s | 77.8% | 50.0% (Tx collapsed to 2/18) | Catastrophic leakage |

**Why 5 s is the right answer, not just the best-scoring one:** `HELLO_INTERVAL = 2 s`,
so an *active* attacker's next HELLO always re-arms the flag within the 5 s window —
suspicion is continuous. But once the attacker is switched off, the flag expires cleanly
before the next measurement phase begins. The penalty duration is tuned to the HELLO
cadence, not to the score.

### Seed sweep and the Helper node
Across five seeds, the attacker was frequently **off the routing path** (the Victim was
equidistant to attacker and relay, so HELLO arrival order — a function of the seed —
decided the route). Moving the Relay out of range fixed that but left the Victim with
*no* route once the attacker was blacklisted. The resolution was a **Helper node** at
(200, 150): a 1-hop neighbour of both Victim and Relay but **not** of the Sender, giving
a 3-hop bypass without displacing the 2-hop baseline route. Result: 4 of 5 seeds
"HIGHLY EFFECTIVE"; attacker flagged in all 5.

### Sources
- Defense: {ref(D_GCOP)}
- Simulation: {ref("scratch/olsr-gcop-simulator.cc")}
""")

md(f"""
<a id="step-10" name="step-10"></a>
## Step 10 — Watchdog: hardening and the four-phase harness
**Date:** 2026-04-19

Six defects were found by auditing the implementation against both the paper and ns-3's
MAC/OLSR semantics. Each would independently have broken detection, false-positive
suppression, or route recovery.

| # | Defect | Fix |
|---|---|---|
| 1 | `PeriodicCheck` scanned the **entire** `m_macObservations` map — any neighbour's RTS/CTS could be misattributed to the suspect | Added `GetMacForIp()` (ARP-cache walk); look up `m_macObservations[suspectMac]` directly |
| 2 | `OnCtsReceived` set `receivedByNeighbor = true` on **every** pending packet — any CTS on the medium advanced the state machine for everything | Cache local MAC (`m_myMac`); add `m_lastRtsTarget` map to infer the CTS transmitter (ns-3's CTS carries only `Addr1`); update only the matching pending packet |
| 3 | `m_macObservations.clear()` every cycle — with `PeriodicCheck` at 1 s and `m_watchdogTimeout` at 0.5 s, observations were wiped **before** their packet's evaluation window closed | Added `firstSeen` / `lastUpdated` timestamps; erase selectively (> 2 × timeout) |
| 4 | The shared 1 s `HandleDefenseTimer` meant a pending packet could wait 1.5 s for a 0.5 s timeout; changing the shared timer would perturb GCOP's cross-layer reports | Gave the Watchdog its **own** `m_watchdogTimer`; left the shared cadence at 1 s |
| 5 | Blacklisting had **no teeth**: rejected in `RouteInput` but not `RouteOutput`; OLSR's neighbour/link/2-hop/MPR/topology sets untouched, so the blacklisted node was re-selected on the next recomputation; HELLO/TC from blacklisted senders still processed, re-establishing the adjacency every cycle | Added a `RouteOutput` check; added `EvictNeighbor()` (purges all OLSR state, drops matching routes, re-runs `MprComputation` + `RoutingTableComputation`); added blacklist early-returns in `ProcessHello` / `ProcessTc` |
| 6 | **Congestion vs. black hole** — an overloaded honest neighbour is behaviourally identical to a silent attacker | `m_neighborActivity[transmitter]` counts overheard forwards; if `activity ≥ ACTIVITY_THRESHOLD` the miss is reclassified as congestion, not malice |

Defect 5 is the instructive one: a defense that *detects* perfectly but does not evict
from protocol state achieves nothing — OLSR simply re-adds the attacker on the next
HELLO. **Detection and enforcement are separate problems.**

### The four-phase harness
| Phase | Interval | Configuration |
|---|---|---|
| — | 0–60 s | stabilisation |
| 1 | 60–100 s | `baseline` |
| — | 100–160 s | stabilise under attack |
| 2 | 160–200 s | `attack_only` |
| — | 200–260 s | stabilise under defense |
| 3 | 260–300 s | `defense_only` |
| — | 300–360 s | stabilise under both |
| 4 | 360–400 s | `defense_vs_attack` |

**Topology:** 22 static nodes — ten "left" at x = 300, ten "right" at x = 700, attacker
at (500, 500), honest backup at (500, 400); 250 m range. Left and right are 400 m apart,
so all traffic *must* cross the middle. The attacker is placed more centrally than the
backup so that OLSR finds it attractive once its willingness manipulation takes effect.

**Measurement methodology:** FlowMonitor installed once and **never reset** — per-flow
counters are snapshotted at window start and subtracted at window end, avoiding
`ResetAllStats()`, which leaves residual state. Flows are filtered to UDP port 9 only,
excluding OLSR control traffic on port 698.

State is **not** reset between phases: blacklist accumulated in phase 3 persists into
phase 4. This is the more realistic scenario and it deliberately stresses the
false-positive dimension.

### Sources
- Defense: {ref(D_WATCHDOG)}, {ref(D_WATCH_H)} *(this work was done on the then-named `olsr-defense-cooperative` file, since consolidated into `olsr-watchdog-defense`)*
- Harness: {ref("scratch/olsr-watchdog-validation.cc")}
""")

md(f"""
<a id="step-11" name="step-11"></a>
## Step 11 — Watchdog: two false-positive bugs
**Date:** 2026-04-22

### Bug #1 — the RTS-Spam heuristic accuses everyone
**Symptom.** `defense_only` PDR *below* `baseline`, and `defense_vs_attack` showing no
recovery — the signature of legitimate nodes being blacklisted with no attack running.

**Root cause.** The heuristic added in [Step 7](#step-7):

```cpp
else if (obs.rtsCount > 0 && !obs.hasClearance) {{
    if (obs.rtsCount <= 7) {{ isMalicious = false; }}   // "congestion"
    else                   {{ isMalicious = true;  }}   // "RTS Spam"
}}
```

This presupposes that watchdog *A* **can hear** the CTS from *C*. In this topology the
source (x = 300) is 400 m from the right column (x = 700) — beyond the 250 m
propagation limit. `obs.hasClearance` is therefore **structurally `false` for every
observation**, regardless of behaviour. Meanwhile the attacker legitimately forwards
~10 packets/s per destination, so within the 1 s window `rtsCount ≈ 10 > 7`. The
"RTS Spam" branch fires **unavoidably**, and honest neighbours are evicted.

The heuristic is not merely mistuned — it is **unsound whenever the watchdog is out of
range of the next-next-hop**, which is a topological property, not a parameter.

**Fix (diagnostic).** Raised `7` → `100` to disable the branch without removing it,
isolating the effect.

**Result:**
| Phase | PDR | AtkDet | FPs |
|---|---:|---:|---:|
| baseline | 60.1% | 0 | 0 |
| attack_only | 70.0% | 0 | 0 |
| defense_only | 100.2%* | 0 | 0 |
| defense_vs_attack | 71.9% | **0** | 0 |

\\* > 100% is a FlowMonitor windowing artefact: packets sent late in one window are
received in the next.

False positives eliminated — but **the attacker is not detected either**. Recovery is
1.9 pp. This exposed Bug #2.

### Bug #2 — the self-reliability filter suppresses all suspicion
**Evidence.** The per-second diagnostic:

```
>>> [EVAL] node=10.1.1.1 t=366.50 expired=5 step1(noCts)=0 step2(forwarded)=0
           step3(noise)=5 step4(noMac)=0 step4(noObs)=0 step5(congest)=0 SUSPECT=0
```

Five packets expired unforwarded — `step2(forwarded)=0` is the **correct signature of a
black hole**. All five were then discarded by Step 3 (`step3(noise)=5`). Steps 4 and 5
never ran.

**Root cause.** Step 3 implements Algorithm B (self-reliability): a watchdog whose own
PHY is dropping packets must not accuse its neighbours, since the missing evidence may
be local. The threshold `m_noiseThreshold` is **5**. With ten concurrent UDP flows on
802.11b plus OLSR control traffic and RTS/CTS, PHY drops routinely exceed five per
second **even with no attack**. The guard fires continuously; the defense's
discriminating logic never executes.

**Recommendation:** `m_noiseThreshold = 200` — a value reflecting genuine channel
pathology rather than the normal collision rate of a saturated ad-hoc network.
Verification deferred to the next session.

### Why `attack_only` (70.0%) exceeds `baseline` (60.1%)
Not a defect. The baseline is **congestion-bound**: ten flows on a shared 802.11b
medium, dominated by collisions. When the attacker silently absorbs traffic, contention
falls and the surviving flows do *better*. Compare [Step 9](#step-9) — the same
pathology, a different mechanism.

Further, `[PENDING-SNAPSHOT]` showed only **three of ten** sources routing through the
attacker; the other seven use the honest backup, because OLSR's MPR selection admits
both. The attacker receives a minority of traffic, bounding any achievable defense
effect.

### Sources
- Defense: {ref(D_WATCHDOG)}
""")

md(f"""
<a id="step-12" name="step-12"></a>
## Step 12 — Watchdog: risk analysis and algorithm design
**Date:** 2026-04-23 · **Author:** Oded Ofek

### Codebase audit
Full review of the modified OLSR (≈ 4,700 LoC). **Verdict: the routing protocol is
fully instrumented for Strategy-pattern defenses — no core modifications are needed to
plug in a concrete strategy.**

### Five integration risks
| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **`OnDataPacketForwarded` has dual semantics** — fires from `RouteOutput` (locally originated) *and* `RouteInput` (forwarded). Arming a timer identically in both is wrong: if `nextHop == dst` the next hop **consumes** and never relays → timer expires unjustly; locally-originated packets are armed before the MAC has even tried | **Critical** | Guard inside the strategy: `if (nextHop == dst) return; if (nextHop.IsBroadcast()) return;` |
| 2 | **Unconditional `RtsCtsThreshold = 0`** for all devices regardless of strategy — required by Baiad's Algorithm 1, but pure airtime overhead for strategies that don't need it | Low | Make it a capability query (`RequiresRtsCts()`), analogous to `RequiresFictitiousNode()`. Deferred — optimisation, not correctness |
| 3 | **Aggressive control-plane filtering** — `RecvOlsr` drops a message if *either* originator *or* sender is blacklisted. A false positive on *B* therefore costs the topology of **everything reachable through *B*** | Important | Cannot be fixed locally; motivates conservative thresholds |
| 4 | **`m_isMalicious` supersedes the strategy** in `RouteInput` — a node that is both malicious and defended still drops | By design | Respect it experimentally: attacker and monitor sets must be **disjoint** |
| 5 | **`ucb` may fail after the watchdog is armed** — the strategy is notified *before* the unicast callback; if the callback fails, the watchdog still expects to overhear the forward | Important | `OnMacTxFailure` must cancel the pending observation |

Risk 3 deserves emphasis: it means **false positives are strictly more expensive here
than in a pure data-plane defense** — accusing one neighbour blinds you to an entire
region of the network. Every threshold decision downstream is conservative because of it.

### The core adaptation: ns-3 breaks the paper's algorithm
Two adaptations were forced by the simulator's semantics:

1. **Per-packet identity is not preserved across hops.** ns-3's `Packet::GetUid()`
   changes when the IP layer re-encapsulates during forwarding, so the paper's
   buffer-and-match scheme **cannot be implemented literally**. Adopted instead: a
   **statistical windowed** variant comparing, over a 2 s window, packets *asked* to be
   forwarded vs. packets *heard* forwarded.
2. **MAC ↔ IP translation.** Sniffer callbacks yield `Mac48Address`; the strategy reasons
   in `Ipv4Address`. Adopted: on-demand ARP lookup with in-strategy memoisation.

### The implemented algorithm
```
For each neighbor B, per 2 s window:
  F(B) = # packets I asked B to forward        (network layer)
  H(B) = # data frames I heard B transmit      (promiscuous sniffer)
  RTS(B), CTS_to(B)                            (MAC counters)

  if F(B) < MinForwardsToEvaluate:  skip
  elif H(B)/F(B) >= ForwardRatioThreshold:  no action        # behaving
  else:
      if RTS(B) > 0 and CTS_to(B) == 0 and RTS(B) < RtsBurstThreshold:
          skip                                 # collision at B's next hop
      else:
          RecordViolation(B)                   # weighted by self-reliability

  DecaySuspicion()      # exponential, half-life 30 s
  UpdateBlacklist()     # hysteresis: enter at 3.0, exit at 1.5
```

### Design decisions register
| # | Decision | Rationale |
|---|---|---|
| D-1 | Statistical windowed rather than per-packet | `GetUid()` not preserved across re-encapsulation |
| D-2 | 2 s evaluation window | Smooths transient noise; still responsive |
| D-3 | Exponential decay, half-life 30 s | Lets falsely-accused nodes recover |
| D-4 | Blacklist hysteresis (enter 3.0, exit 1.5) | Prevents oscillation of borderline neighbours |
| D-5 | Weight violations by `m_selfReliability` (Algorithm B) | Neutralises accusations from a noisy monitor |
| D-6 | ARP-based MAC↔IP with memoisation | Sniffer yields MAC; strategy needs IP |
| D-7 | Exculpation via `OnMacTxFailure` | Directly addresses Risk 5 |

### Acceptance targets (from the paper, at 25% attacker density)
| Metric | Paper | Target |
|---|---|---|
| Detection rate | 89.28% | ≥ 85% |
| False-alarm rate | 1.28% | ≤ 5% |

The paper's own numbers degrade sharply with attacker density (35.33% detection at 50%
attackers), establishing 89.28% as a **practical upper bound** for this detector family
rather than an aspiration.

### Sources
- Header: {ref(D_WATCH_H)}
""")

md(f"""
<a id="step-13" name="step-13"></a>
## Step 13 — Watchdog: multi-topology evaluation
**Date:** 2026-04-25 (dev timeline 2026-04-10 → 2026-04-25)

### A design decision with consequences for the whole project
The Watchdog is **decentralised with no peer-to-peer sharing** — each node decides from
its own observations only. Four reasons, and the second is the project's thesis:

1. The reference paper does not specify a concrete sharing protocol.
2. **Sharing would make the defense *active*, defeating the classification challenge.**
   An ML classifier detects an active defense trivially, via its extra control traffic.
   Keeping Defense 1 passive is what makes it *interesting* to study.
3. A sharing protocol is itself an attack surface (false accusations, Sybil).
4. When several nodes independently identify an attacker, the aggregate approximates
   cooperation anyway.

This is the clearest instance of the project's inverted framing shaping an
implementation choice: the defense was deliberately built to be **hard to detect**,
because detectability is what is under study.

### Iterative fixes
| Date | Fix | Detail |
|---|---|---|
| 04-18 | **MAC↔IP learned from broadcast frames only** | For a *unicast* forward, WiFi `Addr2` = MAC of the **forwarder** but the IP source in the payload = the **original sender**. The defense stored reversed mappings, so every legitimate forward looked like a missed forward → honest bridges blacklisted, traffic collapsed to Tx = 0. Broadcasts (HELLO/TC) are never relayed at L2, so their `Addr2`↔IP pairing is sound |
| 04-19 | **Warmup window** (`WarmupDuration = 15 s`) | When enabled mid-simulation the MAC↔IP map is empty for 1–2 s; in-flight packets time out and fabricate evidence |
| 04-19 | **Tolerant defaults** | `ForwardTimeout` 100 ms → **500 ms**; `BlacklistThreshold` 3 → **10** |
| 04-20 | **Longer stabilisation** | 60 s is not enough for OLSR to converge in a 22-node mesh — the baseline was measured on an unconverged network, yielding a misleadingly low ~60%. Extended to 160 s; added a `baseline_late` diagnostic phase at 560–600 s to confirm the network returns to a converged state |
| 04-21 | **Stronger attack** | One attacker + one backup dropped PDR by only 10 pp. Changed to **two attackers** (y = 490, 470) plus a backup at y = 430 — close enough to be overheard, far enough to be OLSR's second choice |
| 04-22 | **Three-guard cautious commit** | Guard 1: ≥ 3 DATA frames overheard (a silent neighbour is more likely link-broken than malicious). Guard 2: if > 40% of unicasts to it fail at MAC, the *link* is unhealthy and cannot support accusation. Guard 3: 10 s probation on first threshold crossing — commit only if evidence keeps accumulating, else halve it and give another chance |

The 04-20 fix is a reminder worth recording: **an unconverged baseline invalidates every
comparison drawn against it**, and the failure is silent.

### Validation results — single seed
| Phase | PDR | Attacker detected by | FPs |
|---|---:|---|---:|
| baseline | 99.8% | 0 / 21 | 0 |
| attack_only | **10.0%** | 0 / 21 | 0 |
| defense_only | 100.0% | 0 / 21 | 0 |
| **defense_vs_attack** | **100.1%** | **6 / 21** | **0** |
| baseline_late | 100.0% | 0 / 21 | 0 |

Attack −89.8 pp; defense **+90.1 pp — full recovery**, zero false positives.

### Multi-seed variance (10 seeds)
| Phase | PDR mean | PDR stdev |
|---|---:|---:|
| baseline | 95.89% | 8.38 |
| attack_only | 25.74% | **30.91** |
| defense_only | 99.95% | 0.11 |
| **defense_vs_attack** | **100.04%** | **0.05** |

The high variance in `attack_only` is **topology sensitivity, not defense instability** —
whether the attacker lands on a critical route is a property of the seed. The
`defense_vs_attack` stdev of **0.05%** is the striking number: the defense is
deterministic.

### Extended evaluation — where it breaks
**Random square (1000×1000 m, 50 nodes, 2 attackers):**

| Phase | PDR mean | Tx mean |
|---|---:|---:|
| Baseline | 94.71% | 3740 |
| Attack only | 93.31% | 3750 |
| Defense only | 81.18% | 3162 |
| **Attack + Defense** | **57.14%** | **767** |

**Tx collapse in 3 / 7 seeds.** And the seeds reporting "100% PDR" transmitted only
**1/3 of baseline Tx** — four of six flows never sent a packet, because their sources
were **structurally isolated** after the defense blacklisted a critical bridge. *A PDR
of 100% over almost no traffic is not success.* This is the clearest case in the project
of a headline metric concealing a failure — a theme that returns in Part III.

This is a known limitation of hard-blacklist defenses, and the FPNT paper says so:
> *"It is unreasonable to isolate a node completely through blacklist in the routing protocol."* — Tan et al. (2015)

**Highway (1000×200 m, 300 m range — matched to Baiad et al.'s original scenario):**

| Phase | PDR mean | PDR stdev |
|---|---:|---:|
| Baseline | 96.49% | 3.89 |
| Attack only | 81.80% | 14.03 |
| Defense only | 97.45% | 2.41 |
| **Attack + Defense** | **95.56%** | **2.99** |

**Tx collapse: 0 / 7 seeds.** Recovery +13.8 pp; no degradation in `defense_only`.

**The defense was not broken — it was being evaluated outside the topology it was
designed for.** Baiad et al. evaluated on a highway; our random square deviated
substantially, and the deviation, not the algorithm, produced the collapse.

### Two behavioural signatures — a benefit, not a problem
| Scenario | `defense_vs_attack` | Signature |
|---|---|---|
| Highway (paper-matched) | ~95%, low variance | Successful mitigation |
| Random square | Bimodal: ~100% or 0% | Bridge-isolation collapse |

For the classification objective these two distinct modes give the model **more**
distinguishing structure per defense, not less.

### Sources
- Defense: {ref(D_WATCHDOG)}, {ref(D_WATCH_H)}
- Harnesses: {ref("scratch/olsr-watchdog-validation.cc")}, {ref("scratch/olsr-watchdog-eval.cc")}, {ref("scratch/olsr-watchdog-eval-highway.cc")}
""")

md(f"""
<a id="step-14" name="step-14"></a>
## Step 14 — The propagation-loss root-cause bug
**Date:** 2026-04-29 · **Environment:** ns-3.45, WSL / Ubuntu 24

> This is the most consequential infrastructure finding of the project. It had been
> silently corrupting **every** large-topology result up to this point.

### Symptom
`avgRoutes ≈ 7–33` out of a possible 49 across all seeds; `minRoutes = 0` universally.
The network was fragmenting into "convergence islands". Critically, a **minimal
`pureOlsrTest.cc` with no project modifications loaded at all reproduced the failure**
(`avgRoutes ≈ 13`) — which is what proved the bug was not ours.

### Diagnostic
Instrumentation computed, per node, the count of nodes within 190 m Euclidean distance
versus the number of routing-table entries. **The two were largely uncorrelated** —
nodes with 10+ physical neighbours held only 2–3 routes. Physical proximity was not
producing connectivity.

### Root cause
```cpp
YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();   // <-- the bug
wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel",
                               "MaxRange", DoubleValue(190.0));
```
In **ns-3.40+**, `YansWifiChannelHelper::Default()` already installs a
`LogDistancePropagationLossModel`. Adding `RangePropagationLossModel` afterwards does not
*replace* it — **both models operate in series**. Log-distance attenuates signal with
distance, so many frames well inside the 190 m cutoff still fall below receiver
sensitivity. The result is stochastic, distance-dependent frame loss.

The reference simulation (`iolsr-tests-mitigation.cc`) ran on an **older ns-3** whose
`Default()` did *not* include log-distance. **The two simulations diverged silently
despite byte-identical user-facing code.** This is the failure mode that makes the bug
worth reporting: nothing in the API surface indicates that `Default()` has an opinion.

### Fix
```cpp
YansWifiChannelHelper wifiChannel;                    // build from scratch
wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel",
                               "MaxRange", DoubleValue(190.0));
```

### Verification
`pureOlsrTest.cc` → `avgRoutes = 49, minRoutes = 49` on seeds 1, 2, 3 — **full
connectivity**. Convergence rate rose from **~15% to essentially 100%** of topologies.

### A second infrastructure bug fixed the same day
Every run aborted at t = 1.0 s with `"The size of this queue disc is not limited"`.
`HandleDefenseTimer` queries `QueueDisc::GetMaxSize()` every second; in ns-3.45 the
default disc (`MqQueueDisc` + `FqCoDelQueueDisc` children) is **unlimited**, and the
query triggers `NS_FATAL_ERROR`. Two fixes failed first — `FifoQueueDisc` passed the
assertion but silently dropped all control traffic (single-queue is incompatible with
the WiFi MAC's four access categories), and `MqQueueDisc` has no `MaxSize` attribute at
all (it is a pure multiplexer). **Resolution:** `PfifoFastQueueDisc` with an explicit
`MaxSize` of 1000 packets.

### Sources
- Simulation: {ref("scratch/gcopBaseSimulation.cc")} *(the diagnostic `pureOlsrTest.cc` used to isolate the bug was a throwaway and is not tracked in the repository)*
""")

md(f"""
<a id="step-15" name="step-15"></a>
## Step 15 — DCFM/GCOP: milestone and deviations from the paper
**Date:** 2026-04-29

### Final bridge-topology results (8 nodes, 280 m range, seed 1)
| Phase | Tx | Rx | PDR | Attacker flagged at Victim | Nodes with non-empty blacklist | Victim next hop |
|---|---:|---:|---:|:---:|:---:|---|
| baseline | 18 | 18 | 100.0% | No | 0 / 7 | Attacker |
| attack_only | 18 | 0 | **0.0%** | No | 0 / 7 | Attacker |
| defense_only | 18 | 18 | 100.0% | No | 0 / 7 | Attacker (no anomaly exists) |
| **defense_vs_attack** | 18 | 13 | **72.2%** | **Yes** | **6 / 7** | **Relay** |

**+72.2 pp recovery.** All six legitimate nodes on the attacker's side independently
flagged it; the Victim's next hop migrated from `[ATTACKER]` to `[RELAY-safe]` within
the window.

### Interpreting the residual 27.8% loss
Five of eighteen packets lost — **not a defense defect**. The attack starts at t = 300 s;
the defense flags within a few HELLO cycles; `MprComputation` and
`RoutingTableComputation` re-run on every control message — but populating the IP
routing table with the new next hop takes several more seconds. Packets sent during that
transient still go to the attacker and are dropped. **This is intrinsic to proactive
link-state routing**, not addressable by a better detection algorithm. It can be
mitigated only by shorter HELLO/TC intervals (at an overhead cost) or by
application-layer retransmission.

### Cycle 7 — the blacklist-renewal race
The subtlest bug of the session. `OnRecvHello` contained an early-return "optimisation":

```cpp
void OlsrDefenseGcop::OnRecvHello(Ipv4Address senderAddress, ...) {{
    if (IsMalicious(senderAddress)) return;   // <-- intended to preserve the penalty
    EvaluateContradictionRules(senderAddress);
}}
```

The intent was to preserve the existing penalty window. The effect was to **prevent its
renewal**:

| Time | Event | Blacklisted? |
|---|---|:---:|
| t | HELLO evaluated → blacklisted, expiry t+5 | Yes |
| t+2 | HELLO → early return, **no renewal** | Yes |
| t+4 | HELLO → early return, **no renewal** | Yes |
| **t+5** | **Penalty expires via GC** | **No** |
| t+6 | HELLO → evaluated afresh → blacklisted, expiry t+11 | Yes |

The **one-second window of non-suspicion at each cycle** was enough for
`RoutingTableComputation` — which runs on every HELLO/TC arrival — to reinsert a route
through the attacker. By the time the penalty renewed, the routing table was already
polluted. `RouteOutput` then oscillated between success and failure, and packets were
dropped at the sender's IP layer — **invisible to FlowMonitor's `tx` counter, producing
an inflated PDR that masked the instability**. Removing the early return fixed it.

Three separate bugs in this project (Steps 9, 13, 15) produced *better-looking* metrics
while the system was *more* broken. That pattern is the reason Part III insists on
operational metrics.

### Documented deviations from the paper
| # | Deviation | Justification |
|---|---|---|
| 8.1 | **Rule 2 disabled** | ANSN poisoning pollutes the topology set that Rule 2 reads → false positives on legitimate nodes. Retained in-source as commented documentation. Re-enable only alongside an ANSN-jump detector (threshold ≈ 50) |
| 8.2 | **Rule 3 uses a 70% heuristic** | The paper requires the sender to claim *all* of `V \\ ADJ(v)`; our topology view is necessarily partial, so a percentage with a minimum-population guard is used |
| 8.3 | **Rule 1b is attacker-specific** | The `200.x.x.x` check is calibrated to our attacker's address convention. A smarter attacker bypasses it; Rule 1c is the paper-faithful fallback |
| 8.4 | **`OnRecvHello` receives the interface address, not the main address** | Coincide in single-interface nodes; would break on multi-interface. One-line fix noted (`msg.GetOriginatorAddress()`) |
| 8.5 | **Validation on an 8-node bridge** | Paper validates on 50–100 random nodes. Appropriate for proof-of-mechanism; not directly comparable to the paper's PDR curves |
| 8.6 | **Overhead not measured** | The paper reports % of nodes advertised per TC; our harness does not yet trace this |

### Sources
- Defense: {ref(D_GCOP)}, {ref(D_GCOP_H)}
- Simulation: {ref("scratch/olsr-gcop-simulator.cc")}
""")

md(f"""
<a id="step-16" name="step-16"></a>
## Step 16 — Fixing the evaluation methodology: F1–F5
**Date:** 2026-05-01

> This session fixed the **experimental design**, not the defense. It is the point at
> which the pipeline became fit to generate a dataset.

### Fixed parameters — locked for all subsequent work
| Parameter | Value |
|---|---|
| Area | 750 × 1000 m |
| Nodes | 50 |
| Attackers | 1 |
| Mobility | static (a mobile campaign follows) |
| Traffic | UDP Node 1 → Node 0, 18 packets per 40 s window |
| Runtime | 402 s, four measurement windows |
| Radio range | 190 m |

### The five fixes
| ID | Problem | Action | Impact |
|---|---|---|---|
| **F1** | `spoofedLinks = 0` by default → the attacker emits **no** spoofed neighbours → **none** of the contradiction rules can fire. The defense was silent *by construction* | Run with `--spoofedLinks=3` | Without this the entire GCOP validation would have produced null-effect results |
| **F2** | The 45 s convergence wait was keyed to `m_startTime` (set in `Setup()`), but the harness enables the defense mid-simulation at t = 200 s → the wait ran 200→245 s, **eating the entire 40 s `defense_only` window** | Absolute check: `if (Simulator::Now().GetSeconds() < 45.0) return;` | Recovered both full measurement windows |
| **F3** | Hook parameter **names** in the GCOP override didn't match the interface (`nextHop` declared as `source`). `override` checks **types, not names**, so it compiled cleanly — but `if (IsMalicious(nextHop))` actually tested the *destination*, which is never malicious, silently killing a debug channel | Corrected the names | No behavioural change; restored the diagnostic used later in the session |
| **F4** | The attacker was pinned at the geometric centre; with 47 random nodes and a 190 m range, OLSR routinely found paths that bypassed the centre → `attack_pdr ≈ 100%` in ~60% of runs, i.e. **the run carried no information about the defense** | Place all 50 nodes randomly; at t = 60 s select the attacker **dynamically** as the first symmetric neighbour of Node 0 with ID ≥ 2. Added `NEIGHBOR_ABORT` (Node 1 is itself a 1-hop neighbour of Node 0 → no multi-hop route to attack) and `NO_ATTACKER` rejections | Attack effectiveness rose sharply |
| **F5** | `defense_pdr` fell **below 100%** with no attacker present — as low as 27.8%. Legitimate nodes were flagging each other. Cause: **Rule 1c (Asymmetry)** reads the `twoHopNeighbors` set; at 50 nodes with 2 s HELLOs, MAC collisions drop occasional HELLOs, producing a transient asymmetric view of a perfectly legitimate link | **Confirmation policy**: `m_violationCounter` per neighbour; blacklist only after **two consecutive** violations; any clean evaluation resets the counter | **`defense_pdr` returned to 100% on every seed** |

**Why F5's logic is sound, not just effective:** a real attacker exhibits the same
violation on *every* HELLO, so a genuine attack is confirmed within 4 s. A transient
false positive does not repeat on the next HELLO, so the counter resets and no penalty
is ever applied. The policy separates *persistent* from *momentary* evidence.

### Campaign C — after all five fixes
| Metric | Value |
|---|---|
| Baseline | 100.0% (all seeds) |
| Attack | 70.0% mean (effective in ~30% of runs; 0% when effective) |
| **Defense-only** | **100.0% — all 10 seeds** |
| Defense + Attack | 68.9% mean |
| **Recovery in effective-attack runs** | **+44.5% mean** (+27.8%, +77.8%, +27.8%) |

### Methodological notes recorded for the thesis
- **Attacker placement and bias.** Dynamic selection places the attacker where it *can*
  affect traffic without altering any physical parameter — the standard controlled-
  experiment design in the referenced literature (Schweitzer et al., 2024; the reference
  `iolsr` implementation). **The limitation is real and must be documented:** the model
  observes only 1-hop-attacker scenarios.
- **Connectivity rejection rate.** ~65–70% of seeds are rejected for incomplete
  connectivity. This is a topological property of 50 random nodes in 750 × 1000 m at
  190 m range, **not** a bug. Reaching 10,000 successful runs per defense requires
  attempting roughly **30,000–40,000 seeds**.
- **Off-path attackers.** Even with dynamic 1-hop selection, ~70% of successful runs
  still show `attack_pdr = 100%`. Node 0 typically has several 1-hop neighbours and OLSR
  may route via a different one. **This is a critical concern for the dataset**: `attack`
  vectors from off-path seeds are indistinguishable from `baseline` vectors, injecting a
  ~70% noise rate into training data. The recommended fix — select the attacker as the
  **actual next hop** in Node 1's routing table toward Node 0 — is recorded in
  [Open Questions](#open-questions).

### Sources
- Defense: {ref(D_GCOP)}
- Simulation: {ref("scratch/gcopBaseSimulation.cc")}
- Runner: {ref("run_gcop_base_multi_seeds.py")}
""")

md(f"""
<a id="step-17" name="step-17"></a>
## Step 17 — Watchdog: harness parity and calibration
**Date:** 2026-05-03

### Harness parity — the precondition for comparison
The Watchdog harness was produced by cloning the GCOP harness. The two are
**byte-for-byte identical** apart from three lines:

- `#include "ns3/olsr-watchdog-defense.h"` instead of `"ns3/olsr-defense-gcop.h"`
- `CreateObject<OlsrWatchdogDefense>()` instead of `CreateObject<OlsrDefenseGcop>()`
- the log strings naming the defense

Every parameter is preserved: `nNodes=50`, `750×1000 m`, `402 s`, `512 B`, `18` packets,
`190 m`, `TxGain 12.4`, `spoofedLinks=3`, identical phase boundaries. **Any difference
in results is therefore attributable to the defense and nothing else.** This is what
makes cross-defense comparison meaningful.

### A structural finding worth recording
**Seven interface hooks are declared but never invoked** from anywhere in
`olsr-routing-protocol.cc`: `OnNeighborForwardedPacket`, `OnRtsReceived`,
`OnCtsReceived`, `OnQueueStatusReport`, `OnEnergyStateUpdate`, `OnSelfReliabilityReport`,
`PeriodicCheck`. `HandleDefenseTimer()` is scheduled correctly but its **body is empty**.

**Any defense relying on these hooks would silently fail.** The Watchdog survives only
because it is self-sufficient: `Setup()` attaches directly to the WiFi PHY
`MonitorSnifferRx` and `PhyRxDrop` traces and schedules its own timer. This must be
documented for any future defense that lacks that self-sufficiency.

### Bug fix — `MacTxDrop` next-hop attribution
`MacTxDrop` peeked the IPv4 header of the dropped frame and passed
`ipHeader.GetDestination()` — the **final** destination — as the failed neighbour. For
any multi-hop packet that is the wrong node entirely.

**Effect:** `NeighborStats::macTxFailures` increments against the wrong address, so
Guard 2 in `MaybeBlacklist()` (suppress accusation when MAC failure rate > 0.4) becomes
**ineffective in multi-hop scenarios** — the correct neighbour's counter stays at zero.

**Fix:** look up the routing table — `Lookup(finalDest, entry1)` → `FindSendEntry(...)`
→ `nextAddr` — and pass that. Broadcast, multicast, and self-address results are
rejected; runs with no current route are skipped rather than guessed.

**Measured impact on PDR: none.** In the black-hole regime the attacker ACKs at Layer 2
and drops at Layer 3, so `MacTxDrop` rarely fires. **The fix was retained anyway**,
because `macTxFailures` is intended to become a **training feature** and would otherwise
carry mis-attributed noise across every multi-hop path in the 10,000-seed campaign. A
bug that does not affect today's metric can still poison tomorrow's dataset.

### A rejected modification — recorded because the failure is instructive
An EtherType filter was added to `OnNeighborForwardedPacket` on the hypothesis that
non-IPv4 frames were polluting the counter. Diagnostics showed that at that point the
packet **no longer carries LLC/SNAP encapsulation**: `llc.GetType()` returned `0xFFFF`
on **100.00% of calls** (55,000 samples). The filter therefore discarded **every** frame,
collapsing Recovery from 3.33% to 0.00%. **Reverted.** The correct location is inside
`SnifferRxCallback`, after the `WifiMacHeader` is parsed.

### Calibration — matching the detection budget to the window
**The concern, stated quantitatively before the runs:** with `BlacklistThreshold = 10`
and `MinSelfReliability = 0.3`, a self-reliability collapse raises the effective
threshold to 10 / 0.3 ≈ **33 evidence units** — but a 40 s window carries only **18
packets**, i.e. ~18 units maximum. *The defense was asked to gather more evidence than
the experiment could supply.*

| Attribute | Default | Final | Rationale |
|---|---:|---:|---|
| `BlacklistThreshold` | 10 | **3** | Detection completes on the ~4th missed forward (~t+8 s) instead of the ~10th |
| `ProbationDuration` | 10 s | **2 s** | No false positive appeared across ten seeds |
| `MinSelfReliability` | 0.3 | **0.6** | Caps the effective threshold at 3 / 0.6 = 5 units even under full collapse |
| `MinDataObservations` | 3 | **2** | Avoids stalling when the attacker has just entered the observation set |

### Results
| Configuration | Def+Atk PDR (avg) | Recovery | Effective seeds (21, 25, 30) |
|---|---:|---:|---|
| Defaults | 73.33% | 3.33% | 11.11% each |
| Round 1 (thr 4, prob 4 s) | 82.78% | 12.78% | 33.3%, 27.8%, 66.7% |
| **Round 2 — final (thr 3, prob 2 s)** | **87.78%** | **17.78%** | **50.0%, 50.0%, 77.8%** |

**The calibration is aggressive — so how do we know it isn't just trigger-happy?** The
seven seeds where the attacker is off-path form a **natural control experiment**: the
attacker is present and malicious, but no legitimate traffic reaches it. If the
parameters caused spurious blacklisting, those seeds would show Recovery < 100%. They do
not. The parameters are better *matched to the observation budget*, not merely looser.

**Confirmatory test.** Extending the window from 40 s (18 packets) to 72 s (36 packets)
raised Def+Atk PDR to **93.89%**, proving the residual gap is bounded by **OLSR
convergence time, not by the defense**. The change was reverted to preserve comparability
with GCOP.

### Watchdog vs. GCOP on identical seeds
| Seed | GCOP Def+Atk | Watchdog Def+Atk |
|---|---:|---:|
| 21 | 27.78% | **50.00%** |
| 25 | **77.78%** | 50.00% |
| 30 | 27.78% | **77.78%** |
| **Seven attack-inert seeds (avg)** | **≈ 82%** | **100%** |

On effective-attack seeds the two defenses **trade wins** — expected, given different
detection modalities. The decisive difference is the last row: on seeds where **no
threat exists**, GCOP degrades PDR to ~82% — its topological rules accuse innocent nodes
— while the Watchdog does **no** damage. That is a genuine architectural advantage of a
data-plane defense over a control-plane one in this topology.

### Sources
- Attack/protocol: {ref(ATTACK)}
- Defense: {ref(D_WATCHDOG)}
- Simulation: {ref("scratch/watchdogBaseSimulation.cc")}
- Runner: {ref("run_watchdog_base_multi_seeds.py")}
""")

md(f"""
<a id="step-18" name="step-18"></a>
## Step 18 — Correcting the attack; supervisor review
**Dates:** 2026-05-06, 2026-05-13, 2026-05-18

### 2026-05-06 — the attack was not attacking properly
Review of 30-seed runs across the defenses. Two defenses look good; **DCFM's results are
strange**. Diagnosis: the black-hole attack needed correction.

**The problem.** The attack claimed links only to **fictitious** nodes. A defense
checking consistency can dismiss those trivially — and, more importantly, the attack was
weaker than the literature's, because a spoofed link to a *real* node is far harder to
refute than one to an address nobody has ever seen.

**Decisions taken:**
- Fix the attack to advertise links to **real nodes from across the network**.
- **Feature filtering principle established:** keep features describing messages that
  **circulate through the network** — even if they don't necessarily reach the attacker,
  sensors or several attackers could be distributed to capture them. **Drop all features
  tied to HELLO messages**, because HELLOs do not circulate; they reach only immediate
  neighbours and are therefore not observable by a realistic adversary.
- Strengthen the attack (consider additional attackers).
- Simulation constraints: attacker within **1 hop** of the victim; **≥ 3 hops** between
  sender and receiver; only the victim sends packets.

### 2026-05-13 — supervisor meeting
- DCFM's results remain very strange → **compare our implementation against the paper's
  GitHub repository** and apply the necessary changes.
- **Feature funnel:** started at ~150 candidates; after filtering out *observer*
  features (those requiring a vantage point a real adversary lacks), ~**100** remain.
- The attack fix is in: it now lies about **real neighbours**, not fictitious ones.
- **Work split:** Hananel takes Defense 3 (DCFM); Oded continues on the vector-generating
  simulation.

### 2026-05-18 — DCFM works, and a phenomenon is explained
After the code change, **DCFM works and shows excellent results across a range of
`spoofedLinks` values.**

**The `both` column finally explained.** Why does `defense+attack` often show a PDR drop
while `defense_only` and `attack_only` are both high?

| Condition | Behaviour | Why |
|---|---|---|
| **Attack only** | PDR stays high | As long as the attacker isn't on the packets' path, nothing happens. No defense is running, so nothing suspects anyone → **no false positives** |
| **Defense only** | PDR stays high | The topology is **not poisoned** — there is no attacker — so no node trips the contradiction rules → **no false positives** |
| **Both** | **PDR drops** | The attacker **poisons the topology**; the defense then begins suspecting **innocent** nodes; some are blocked and packets never arrive |

The drop in `both` is **not** a failure to stop the attacker. It is the defense
**mis-firing on honest nodes**, and it can only occur when both are active — the
attacker supplies the poison, and the defense supplies the reaction. This is the same
mechanism as Step 8's Issue C, now understood as a general property.

### Open question raised at this meeting
The harness must account for **structural differences between defenses**. Specifically:
two defenses force **RTS/CTS even on small packets** (see [Step 5](#step-5)). **Is that
a cheat for the ML model?** — i.e., does it hand the classifier an artefact of our
configuration rather than a behaviour of the defense? This question anticipates the
entire leakage analysis of Part III, and it is recorded in
[Open Questions](#open-questions).

### Sources
- Attack: {ref(ATTACK)}
- Defense: {ref(D_GCOP)}
""")

md(f"""
<a id="step-19" name="step-19"></a>
## Step 19 — Adding the fourth defense: TRUST2

### Rationale
Three defenses were judged insufficient for a meaningful multi-class classification
problem, so a fourth — **TRUST** (labelled **TRUST2** in the ML pipeline's condition
names) — was implemented and evaluated through the same harness.

### What it is (from the source)
Reading `olsr-trust-defense.h` settles a question earlier documentation left open: TRUST
is based on **Adnane, Bidan & de Sousa, *Computer Communications* 36 (2013),
"Trust-based security for the OLSR routing protocol"** — a formula-driven trust framework,
**not** a second Tan-style scheme. It is the only **modular** defense: `OlsrTrustDefense`
owns five toggleable sub-modules — a forward monitor (the black-hole core, paper
Formula 10), a trust-state verdict log (Formula 15), consistency rules (Formulas 6/7/8/9/12),
a provable-identity module (Formula 13, off by default), and an alert distributor. It also
**separates detection from response**: with `ResponseEnabled=false` it records mistrust
without acting on it, so detection can be measured in isolation. Being a trust defense that
piggybacks on control traffic, it is a *deep* swap ([§ swap sets](#guide-swap)). See the
[annotated file guide](#guide-model) for the full breakdown.

### Note
TRUST2 turns out to be the **most scientifically interesting** defense in the study,
though not for a flattering reason: it is the defense the classifier can barely detect.
Across all 95 Core features, **no single feature exceeds a univariate AUC of 0.64** for
TRUST2 (see [Step 31](#step-31)). Its best clean result — TRUST2/mobile at
TPR@1%FPR = 0.047 — is **operationally unusable**, only 4.7× better than chance.

Precisely because no shortcut was ever available for it, **TRUST2's numbers are the most
trustworthy in the entire study.** The defenses that scored worst are the ones we can
believe.

### Sources
- Defense: {ref(D_TRUST)}, {ref(D_TRUST_H)}
""")

# ==========================================================================
# PART III — MACHINE LEARNING, CAMPAIGN 1 (defense_ml)
# ==========================================================================
md("""
---
# Part III — Machine Learning, Campaign 1: the `defense_ml` pipeline
*June – early July 2026*
---

With four validated defenses and a parity harness, the project reached its actual
question: **from passively observable OLSR control traffic alone, can a model tell whether
a black-hole defense is running — and, if so, which one?**

The machine-learning work happened in **two campaigns**, and this report presents them in
the order they occurred. **Campaign 1** (this part) is the earlier, publication-oriented
effort, built as the `defense_ml` Python package and run through **June–early July** on the
defenses *as they then stood* — including the pre-realignment DCFM that still injected
phantom nodes. It produced the project's central scientific narrative: the observability
ladder, the FPNT-artifact-vs-DCFM-signature distinction, the Trust 1.0-vs-2.0 study, and the
**detectability↔efficacy tradeoff thesis**. Its findings then *motivated* the DCFM
realignment and feature normalisation ([Part IV](#step-26)), after which a second, leaner
pipeline — `defense_detection_v4` — was run on the normalised data ([Part V](#step-27)).

> **Repository.** Campaign 1 lives in the **private** ML repo `hananelk26/ML-for-NS3`
> (branch `main`), package `defense_ml/defense_ml_project/`. Files below are given as
> repo-relative paths in code font, not links (the repo is private). Its datasets and
> `results/` tree are git-ignored local artifacts.
""")

md(f"""
<a id="step-20" name="step-20"></a>
## Step 20 — The first ML campaign: dataset, task, and a leak-free pipeline
**Dates:** 2026-06-14 → 2026-07-02

### The dataset
Per defense, **24,000 simulations** were generated (the full campaign, not the pilot):

| Window order | Mobility | Simulations |
|---|---|---|
| canonical | static | 10,000 |
| mixed (shuffled windows) | static | 2,000 |
| canonical | mobile | 10,000 |
| mixed | mobile | 2,000 |

Each run produces **4 windows** (one per scenario: baseline / attack-only / defense-only /
defense+attack), 40 s each, starting at t = 120 s after convergence. Every window is
described by the project's own **95-feature schema** (groups A–K). **canonical** is the
training/evaluation data; **mixed** is reserved for *bias validation only* — a control that
asks whether detection rides the fixed window order rather than the defense's behaviour.

### The task
The headline task is **`presence_active`** (mandated by the supervisor): a binary classifier
of *defense ON vs OFF* trained over **all four scenarios** (label = `defense_enabled`,
independent of attacker state), built **separately per mobility** (static and mobile are
different models — never pooled by default). A 4-class **`family_presence`** (which defense)
is the secondary task. Defined in {refml(DML + "/defense_ml/data_loading.py", "data_loading.py")}
(`build_presence_active`) and orchestrated by {refml(DML + "/defense_ml/main.py", "main.py")}.

### The pipeline — and why its numbers can be trusted
The package enforces leakage safety at three levels, which is what makes a headline
ROC-AUC here meaningful rather than an artefact of the protocol:

1. **Grouped repeated cross-validation.** {refml(DML + "/defense_ml/cv_harness.py", "cv_harness.py")}
   uses repeated `StratifiedGroupKFold` keyed by `run_id`, so all **four windows of a run
   always stay together** in train *or* test — and a **hard runtime guard raises an error**
   if any run ever spans the split.
2. **In-fold preprocessing.** Constant-feature removal and scaling live *inside* the sklearn
   pipeline ({refml(DML + "/defense_ml/feature_prep.py", "feature_prep.py")}), fit on training
   folds only — no selection leakage.
3. **Grouped permutation null.** Significance ({refml(DML + "/defense_ml/stats.py", "stats.py")})
   permutes labels **within each run-group**, preserving the dependency structure; a `Dummy`
   classifier pins the chance floor at AUC 0.500 in every run.

The model set ({refml(DML + "/defense_ml/model_zoo.py", "model_zoo.py")}) is a 13-model zoo
(logistic/ridge, kNN, SVM-RBF, RF/ExtraTrees, HistGB — the reference model — XGBoost,
LightGBM, CatBoost, MLP, stacking, plus Dummy), all `class_weight='balanced'`. Reporting
({refml(DML + "/defense_ml/metrics.py", "metrics.py")}) is deliberately multi-metric —
ROC-AUC, balanced accuracy, MCC, and the metric a security-detection paper is actually
judged on: the **operating-point `TPR@1%FPR`** — the *true-positive rate at a fixed 1%
false-positive-rate budget*, i.e. of all **defense-ON** windows, the fraction the model
correctly flags while raising a false alarm on **at most 1%** of defense-OFF windows
(chance ≈ 0.01; `TPR@5%FPR` is the same at a 5% budget). It is far more revealing than
ROC-AUC near the ceiling — with Nadeau-Bengio corrected CIs and
Friedman/Nemenyi model comparison ({refml(DML + "/defense_ml/stats.py", "stats.py")}).

### Sources
- Schema authority: {refml(DML + "/defense_ml/config.py", "config.py")}
- Front-door results: {refml(DML + "/RESULTS.md", "RESULTS.md")} · narrative: {refml(DML + "/RESEARCH_SUMMARY.md", "RESEARCH_SUMMARY.md")}
- Outputs: master tables `{DML}/results/00_summary/` (`SUMMARY.md`, `ML_results_master.xlsx`, `tables/*.csv`)
""")

md(f"""
<a id="step-21" name="step-21"></a>
## Step 21 — Baseline detection, the red flag, and FPNT as a single artifact

### The result was too good
On the full 95-feature schema, **presence detection reached ROC-AUC ≈ 1.00 for all three
original defenses** (FPNT, DCFM, Watchdog); the 4-class *which-defense* task reached ≈ 83%.
Perfect binary detection is a warning sign, not a triumph — it usually means leakage. A
data audit found none in the *protocol* (the grouped CV is clean), so the investigation
turned to *which features* were doing the work.

### FPNT — a disguised "IF rule" **[VERIFIED]**
Permutation importance attributed essentially **all** of FPNT's signal to a single feature,
`TcMessageSizeMean` (the mean byte size of TC messages). Mechanistically this is honest —
FPNT pads TC messages with trust data — but the model had not learned a rich behavioural
signature; it had learned **one threshold**, with every other feature at importance ≈ 0.

### The ablation **[VERIFIED]**
Removing `TcMessageSizeMean` **and every feature that measures the same thing** — the seven
**TC-byte-size** features (`TcMessageSizeMean/Std/P95/Max`, `TcBytesPerSecond`,
`PerNodeTcBytesStd`, `PerNodeTcBytesGini`; their cross-defense Cliff's δ ≈ 1.0) — collapsed
FPNT detection to **≈ 60%**. Conclusion: FPNT's perfect detection was almost entirely a
**defense-dependent implementation artifact of TC byte size**, not a broad structural
signature. Crucially, the advertised-*link-count* features (`AdvertisedLinksPerTc*`) were
**kept** — they are record counts, and FPNT inflates bytes, not link counts.

This distinction — *why does a feature's value change: because the network behaves
differently, or because the defense writes bytes differently?* — is the question that
organises the rest of Campaign 1.

### Sources
- Feature audit + the 7 TC-size features: {refml(DML + "/defense_ml/config.py", "config.py")} (`TC_SIZE_FEATURES`)
- Signatures / importance: {refml(DML + "/defense_ml/defense_signatures.py", "defense_signatures.py")}, {refml(DML + "/defense_ml/interpret.py", "interpret.py")}
- Outputs: `results/10_core_detection/contrast_95_vs_58/` (the `*_all` vs `*_passive` before/after)
""")

md(f"""
<a id="step-22" name="step-22"></a>
## Step 22 — DCFM: a broad, "holographic" phantom signature
**Dates:** 2026-06-23 → 2026-06-25

DCFM's detection was *also* ≈ 1.00 — so the obvious question was whether it, too, was a
single artifact like FPNT. **It is not**, and establishing that is one of Campaign 1's key
results.

### Depth of redundancy — greedy ablation **[VERIFIED]**
Iteratively removing the single most important feature and re-running (until best AUC < 0.90)
showed DCFM detection holding at ≈ 1.00 down to **8 features (static) / 29 (mobile)** — after
~60 / ~53 removals. **No feature family survived intact**; entire families (control-volume,
timing, entropy, defense-breadth) were wiped while detection held. The phantom signature is
imprinted **redundantly across the whole control plane** — it cannot be removed by feature
selection. (`tools/greedy_ablation.sh`.)

### It is causal in phantom injection — the control experiment **[VERIFIED]**
A dataset with **phantom injection disabled** was generated as a control. Detection
**collapsed** to ROC-AUC ≈ **0.80 (static) / 0.62 (mobile)**; the only residual signal came
from `NumberOfMprChurnEvents`, and **only under attack**. Directly measured, the phantoms
inflate the control plane wholesale: advertised distinct addresses **50 → 87**, degree-one
graph nodes **≈ 6 → 37**, TC packet rate **120 → 453 s⁻¹ (×3.8)**.

### Why the injection was kept anyway
The phantom injection is not a bug to remove — it is **core to the DCFM mechanism** (it is
how the defense forces contradictions; see [Step 8](#step-8)). So DCFM's detectability is a
*genuine, mechanism-driven* signature, unlike FPNT's byte-padding artifact. The redundancy is
**deep** under the 95-feature control-plane schema but **shallow** under a performance-only
33-feature schema (where only ~6 control features carry it, and detection collapses after
~6 removals) — a contrast that itself is informative.

### Sources
- Dated log: {refml(DML + "/CLAUDE.md", "CLAUDE.md")} · narrative {refml(DML + "/RESEARCH_SUMMARY.md", "RESEARCH_SUMMARY.md")} §11
- Tools: {refml(DML + "/tools/greedy_ablation.sh", "greedy_ablation.sh")}, {refml(DML + "/tools/run_dcfm33.py", "run_dcfm33.py")}
- Outputs: the greedy-ablation + no-phantom DCFM runs live on the collaborator's machine (folded into `results/00_summary/`, `source=documented/xlsx`); local 33-schema runs under `results/30_schema33/`
""")

md(f"""
<a id="step-23" name="step-23"></a>
## Step 23 — The observability ladder 95 → 67 → 18, and the external observer

### Threat models as feature sets
A detector is only as realistic as the vantage point it assumes. Campaign 1 makes the
threat model explicit by defining nested feature sets in {refml(DML + "/defense_ml/config.py", "config.py")}
(each with an enforced count assertion):

| Set | Count | Threat model |
|---|---:|---|
| `all` | **95** | everything, incl. god's-eye topology and byte-size artifacts |
| `observable` | **67** | a single passive OLSR **node** (control plane it overhears) |
| `passive` | **58** | the 67 minus TC-byte-size + phantom + an addressing artifact — the **clean** set (the trustworthy number) |
| `external` | **18** | an observer **outside the MANET**, control-plane only (also drops the whole topology-graph group J) |
| `generic` | **12** | defense-agnostic churn, for the unseen-defense (LODO) test |

The 95 partition exactly: **67 observable + 25 non-observable + 3 MAC-local**.

### The ladder — which defenses survive **[VERIFIED]**
Detection ROC-AUC as the observer weakens (static, at-rest):

| Defense | 95 (all) | 67 (observable) | 18 (external) | Reading |
|---|---:|---:|---:|---|
| **FPNT** | 1.00 | 1.00 | **0.59** | **collapses** — all its signal was the TC-size artifact, present in 95/67 and removed only at 18 |
| **Watchdog** | 0.99 | 0.91 | **0.89** | **survives** — a real, distributed behavioural signal |
| **DCFM** | 1.00 | 1.00 | **1.00** | **survives fully** — the phantom signature reaches even the 18 external features |

The 95→18 gap for FPNT is *exactly* the TC-size set — which is why FPNT, and only FPNT,
falls to chance at level 18. On the **clean 58-feature** set the honest per-mobility
`presence_active` numbers are FPNT 0.647/0.623, **Watchdog 0.941**/0.640, Trust2 0.665 — and
for **all three** the top feature is `NumberOfMprChurnEvents` (the rerouting churn that is the
genuine passive signature). Family (which-defense) degrades along the ladder too: balanced
accuracy 0.979 → 0.971 → **0.730** (static), because telling defenses *apart* needs the
topology-graph group that only an internal observer has.

### A caveat kept on the record **[stated in the summary]**
The dataset was collected from an **omniscient** vantage; reducing to 18 columns is a
**necessary but not sufficient** proxy for a true external observer (the retained values were
still measured under full coverage). The external-18 run is a **feasibility check**; a
publishable external-observer claim needs re-collection from a real external vantage plus an
explicit no-encryption assumption.

### Sources
- Feature sets: {refml(DML + "/defense_ml/config.py", "config.py")} · campaign runner {refml(DML + "/tools/run_external_campaign.py", "run_external_campaign.py")}
- {refml(DML + "/RESULTS.md", "RESULTS.md")} §1 · {refml(DML + "/RESEARCH_SUMMARY.md", "RESEARCH_SUMMARY.md")} §6
- Outputs: `results/20_observability_ladder/` (`external/`, `ladder/obs67_canonical/`, `ladder/all95_canonical/`)
""")

md(f"""
<a id="step-24" name="step-24"></a>
## Step 24 — Two planes, and the central tradeoff thesis
**Dates:** Trust study + contrasts, late June – early July 2026

### DCFM and Watchdog leak through *different planes* **[VERIFIED]**
Run on an identical performance-oriented 33-feature schema, the two defenses are detected via
opposite feature families: **DCFM** via **control-plane** features (`TcMessageRate`; AUC
1.00), **Watchdog** via **data-plane / performance** features (`FlowLossRateStd`, `AvgFlowDelay`;
AUC 0.965 static / 0.738 mobile). DCFM **"shouts"** in the control plane (phantom-induced TC
flooding); Watchdog **"whispers"** in the data plane (rerouting around suspects changes flow
delay/loss). Their overhead ratios confirm it: DCFM ≈ 31, FPNT ≈ 13, **Watchdog ≈ 10** (≈
baseline). And `path_hops` stays **≈ constant at 5.0** across every scenario and defense — *no
defense visibly lengthens routes*, so what leaks is the control-mechanism footprint, not
rerouting.

### Trust-OLSR, two versions — the cleanest demonstration
The fourth defense was studied in **two calibrations** (the NS-3 attributes
`min_forward_failures` 1→3, `mistrust_permanent` 1→0):

| | efficacy (defense-only PDR) | under attack (ΔPDR) | passive detectability (AUC, 67) |
|---|---|---|---|
| **Trust 1.0** (aggressive) | **57%** — breaks the clean network | median **−33%** (harmful) | ≈ **1.00** — "shouts" |
| **Trust 2.0** (calibrated) | **98.9%** — no false positives | **+3.5%** (Wilcoxon p ≈ 1.6·10⁻³¹) | **0.67** (perm-null p = 0.005) — "whispers" |

Trust 1.0 is aggressive, so it churns the network violently and is trivially detectable;
Trust 2.0 barely touches the network, so it is nearly invisible (a *real but weak* signal —
significant at p = 0.005, ≈ 21 σ from the null, but AUC only 0.67).

### The central thesis of the project
> **Passive detectability is orthogonal to — even inverse to — defense efficacy.** A better,
> gentler defense leaves fewer traces and is *harder* to detect passively.

| Defense | PDR recovery (static) | detectability (clean set) | character |
|---|---:|---:|---|
| FPNT | **64%** (most effective) | 0.647 | effective **and** evasive |
| DCFM | 55% | 1.00 (phantom "shout") | effective but loud |
| Watchdog | 42% | 0.941 (static) | moderate and visible |
| Trust 1.0 | −30% (harmful) | ≈ 1.00 | failed and loud |
| Trust 2.0 | 10% | 0.665 | gentle and near-silent |

Trust 1.0 and 2.0 are two measured endpoints of one curve — the strongest evidence for the
thesis, because they are the *same* defense at two operating points.

### Sources
- Efficacy from the oracle: {refml(DML + "/defense_ml/efficacy.py", "efficacy.py")}
- {refml(DML + "/RESULTS.md", "RESULTS.md")} §2–3 · {refml(DML + "/RESEARCH_SUMMARY.md", "RESEARCH_SUMMARY.md")} §5, §11.7
- Outputs: `results/40_trust_defense/` (`binary_obs67/`, `permutation_null/`); `results/00_summary/tables/trust_tradeoff.csv`, `efficacy_pdr_master.csv`
""")

md(f"""
<a id="step-25" name="step-25"></a>
## Step 25 — Advanced experiments: generalisation, novelty, and defending the method

Campaign 1's publication layer adds three experiment families, all on the *existing* data
(no new simulation), that go beyond in-domain detection.

### Transfer — does the signal generalise? {refml(DML + "/defense_ml/transfer.py", "transfer.py")}
Train-on-A / test-on-B, group-disjoint by construction (asserted):
- **cross-order** (canonical → mixed): if a canonical-trained model still scores on shuffled
  windows, the signal is **not** an artefact of window order or the slot-0 transient. It does
  — the external-18 numbers are stable canonical↔mixed — an important validity result.
- **cross-mobility** (static ↔ mobile): the one environmental axis in the data.
- **defense → defense**: train a "defense present" detector on defense A, test on B. **Low
  transfer is expected — and is itself a result:** each defense leaves a distinct
  control-plane signature, so a detector tuned to one should not recognise another.

### Open-set — can it flag an *unseen* defense? {refml(DML + "/defense_ml/openset.py", "openset.py")}
Leave-One-Defense-Out: known = {{none}} + two defenses; the held-out defense's windows are
the "unknown" class. Three novelty scores (max-softmax, Mahalanobis-to-nearest-centroid,
IsolationForest) are scored by AUROC(known vs unknown) and the **OSCR** curve. This is the
operational question — meeting a defense never trained on and flagging it as "other" rather
than mislabelling it.

### Selection audit — is the accepted sample biased? {refml(DML + "/defense_ml/selection_audit.py", "selection_audit.py")}
Five checks (S1 accounting; S3 accepted-vs-rejected topology covariates + a "selection
classifier" AUC; S4 cross-directory comparability). The centrepiece is **S5, a covariate-only
placebo**: a classifier given *only* the topology covariates (never the 95 features) must score
**≈ chance** on `presence_active` — proving detection cannot come from the *selected topology*.
It does score at chance, because within a run both scenarios share one topology.

### SHAP — direction, not just ranking {refml(DML + "/defense_ml/interpret.py", "interpret.py")}
Adds per-class attribution over the reference tree model (what makes a window look like
Watchdog rather than none), on a group-disjoint split.

Together these make Campaign 1 defensible as a paper. **But** its central finding — that
DCFM's perfect detection is a real, broad signature while FPNT's is a byte artifact — combined
with the recognition that the DCFM implementation still diverged from its paper, is exactly
what triggered the next phase: realign DCFM, normalise the features, and re-run. That is
[Part IV](#step-26).

### Sources
- {refml(DML + "/defense_ml/transfer.py", "transfer.py")}, {refml(DML + "/defense_ml/openset.py", "openset.py")}, {refml(DML + "/defense_ml/selection_audit.py", "selection_audit.py")}, {refml(DML + "/defense_ml/stats.py", "stats.py")}
- Outputs: `results/70_publication/` (`core/`, `mixed/`, `publication/`, `campaign_summary.txt`); `results/50_generic_features/lodo_audit/`
""")

# ==========================================================================
# PART IV — THE TRANSITION
# ==========================================================================
md("""
---
# Part IV — The Transition
*Realignment, upgrade, and normalisation — late June – mid July 2026*
---

Campaign 1 left one thing unresolved: DCFM's implementation still diverged from its paper,
and its perfect detectability — though a *real* signature — was partly a fingerprint of that
divergence. The response was to realign DCFM to the paper, upgrade the learning code, and
**normalise** the scale-dependent features, then regenerate the dataset. That regenerated,
normalised dataset is what the second campaign ([Part V](#step-27)) measures.
""")

md(f"""
<a id="step-26" name="step-26"></a>
## Step 26 — DCFM realignment to the paper; the 24 June feature run
**Date:** 2026-06-24

> **Provenance.** This step bridges a gap in the dated session logs, which run to
> 2026-05-03 and resume at 2026-07-13. It is reconstructed from the project journal
> rather than from a contemporaneous session log, and is dated to the day of the feature
> run it describes.

### The 24 June run — and why it mattered
A **new 33-feature set** was obtained directly from **Nadav Schweitzer** — first author of
the DCFM paper and project supervisor. The sample was regenerated on these features and
the learning pipeline re-run.

**The result was the now-familiar warning sign: DCFM classification accuracy was still
100%.** Supplying a defense's designer's own hand-picked features did not remove the
leakage — which was the clue that the leakage was not in the *feature selection* but in
the *implementation being measured*. The DCFM code, as it then stood, still diverged from
the paper in ways that produced an implementation-specific fingerprint the model could
read.

### The realignment
The response was to rewrite the DCFM implementation to follow the paper far more
closely. This was done in **two stages**, and the distinction is worth recording:

1. **First, in a different code structure** (git: `code from Nadav`, **2026-06-24**) — a
   standalone rework, separate from the project's own module layout, supplied by the
   supervisor to get the algorithm demonstrably right. On this reworked version, learning
   on the 33-feature set (`V2Only`) finally produced **reasonable accuracy** for DCFM: the
   100% collapsed to sensible numbers, per defense, in both static and dynamic regimes.
2. **Then, ported into the project's code structure** (git: *"replaced the files with the
   DCFM defense … aligned closely with the paper"*, **2026-07-10**). The realigned version
   is materially different from the earlier one documented in [Steps 8, 9, 15](#step-8):
   it **removes the blacklist and the time-based penalties entirely**, **retains the
   suspicious-MPR handling**, uses the **updated GCOHP**, and applies **a small adjustment
   to the contradiction rules**. In other words, the version measured in the July campaign
   is a leaner, more paper-faithful DCFM than the penalty-based one validated in April–May.

### Normalisation introduced here
Around the same time (git: *"Normalize scale-dependent features in schema v5 for Core and
V2 CSV outputs"*, **2026-07-09**), **many features that require normalisation were
normalised** — for example, dividing by the number of nodes in the network — so that a
model trained on the simulation's topology could generalise to networks of a **different
topology or size** than the one the vectors were generated on. This is recorded in the
feature schema as **"schema v5, normalised"** ({ref(FEATURES)}).

> **This is the origin of the normalisation that [Step 32](#step-32) later identifies as
> the probable root cause of the reintroduced leakage.** The normalisation was introduced
> here for a sound reason (cross-topology generalisation) that the single-configuration
> pilot dataset ([Step 28](#step-28)) cannot actually exercise — which is exactly why the
> hypothesis in Step 32 is plausible. The decision that helped generalisation in
> principle may have injected a configuration-dependent leak in practice. That tension is
> not yet resolved.

### Why both 24 June and 13 July are correct dates
The 33-feature set was run **twice**, and the two runs answer different questions:

| Run | Date | DCFM implementation | DCFM accuracy | Meaning |
|---|---|---|---|---|
| First | **2026-06-24** | pre-realignment | **still 100%** | Leakage persists even on the supervisor's own features → the problem is the *implementation*, not the feature list |
| Second | **2026-07-13 → 07-15** | realigned + normalised, inside the 128-feature dataset | reasonable (then leakage re-analysed in depth) | The full pilot campaign and the leakage study of [Part V](#step-27) |

### After the realignment
The learning code was then upgraded to a higher level, and the sample was regenerated for
**all** defenses over the full **128** features (95 Core + 33 V2), with every feature that
needed normalisation normalised beforehand. That regenerated dataset is the subject of
[Part V](#step-27).

### Sources
- Defense: {ref(D_GCOP)}, {ref(D_GCOP_H)}
- Feature schema: {ref(FEATURES)}
- Normalisation spec: `OLSR_Feature_Normalization_Table.docx`
""")

# ==========================================================================
# PART III
# ==========================================================================
md("""
---
# Part V — Machine Learning, Campaign 2: the `defense_detection_v4` pipeline
*July 2026*
---

After the realignment and normalisation of [Part IV](#step-26), the learning was re-run —
this time with a **second, leaner pipeline**, `defense_detection_v4`, on the normalised
128-feature dataset. This is the project's **most recent** ML work (git 2026-07-19). It
consolidates the instructor's earlier `defense_detection_v2` with the rigour of the
`defense_ml` package, and it re-derives — on the *normalised* data — the same leakage story
Campaign 1 established, then surfaces one further concern (the **normalisation hypothesis**,
[Step 32](#step-32)) that sits on top of Campaign 1's cleaner feature audit.

> The short version is unchanged from Campaign 1: the model was not learning network
> behaviour, it was reading configuration constants off the wire. What Campaign 2 adds is a
> controlled ablation ladder (32 → 29 → 26 → 76 features) on the normalised dataset, and the
> observation that **normalisation can re-introduce a leak that feature-removal alone cannot
> close**.

> **Provenance of the pipeline — who wrote what.** The single-file structure, the control
> flow, the engineered feature families (CDR, TDR, QoS, delay/hop ratios and the
> log/sqrt/x²/x³ transforms), the multi-criterion feature selection, the model pool including
> Stacking, and the probability-calibration and threshold-tuning machinery all come from the
> supervisor's earlier `defense_detection_v2`. This project's additions are concentrated in
> the **statistical-validation layer**: the grouped repeated cross-validation and its runtime
> guard, moving calibration and thresholding inside the fold, the `Dummy` chance floor, the
> `TPR@FPR` operating points, the grouped permutation test, LaTeX export, and the CLI
> feature-subset selection without which the ablation ladder of [Steps 29–34](#step-29) could
> not have been run at all. The division is worth stating plainly: **the pipeline is largely
> inherited; what this project contributed is the apparatus that decides whether its numbers
> are believable** — which, given that the central finding of both campaigns is that early
> numbers were *not* believable, turned out to be the part that mattered.
""")

md(f"""
<a id="step-27" name="step-27"></a>
## Step 27 — The 128-feature schema

### Definition
The schema is defined in {ref(FEATURES)} via the `FeatureMode` enum **[VERIFIED by
direct column count from source]**:

| Mode | Groups | Columns |
|---|---|---:|
| `Core` | A–K | **95** |
| `V2Only` | L (`strict_observable_v2` parity) | **33** |
| `CoreAndV2` | A–L | **128** |

- The **Core-95** set is the project's own, distilled from ~150 candidates via the
  filtering of Steps 3 and 18.
- The **V2-33** set was supplied by **Nadav Schweitzer**, first author of the DCFM paper
  and project supervisor.
- The dataset is emitted in `CoreAndV2` mode — hence `dataset_128_all_defenses`.

### The `metrics32` set **[VERIFIED]**
The pipeline's default `--feature-set metrics32` is **exactly the 33-feature V2 group
minus `RoutingOverheadBytesRatio`**. Verified by extracting `METRICS` from the pipeline
source and cross-checking every name against the dataset columns: all 32 map to real
columns, no silent fill.

Two members — `MidMessageRate` and `HnaMessageRate` — are **constant** across all rows
and are auto-removed by the constant-feature filter. **The effective base is therefore
30, not 32.**

### Slicing, not re-simulating — a key methodological property
Because the dataset is emitted over all 128 features, **any subset can be trained on by
slicing the vectors** — no re-simulation is required. Every experiment in Steps 29–31
draws on the *same* dataset. This is what made the ablation study cheap enough to run at
all, and it means the comparisons are exact: identical runs, identical windows, only the
feature columns differ.

### ⚠️ Discrepancy on the record
The **"67-feature group" referenced in project discussion does not exist** in either
{ref(FEATURES)} or {refml(ML_PIPE)} — **the string `67` appears in neither file**. Only the
95 / 33 / 128 groupings are formally defined; the "32" group is derived in Python, not in
the header. This report follows the source. See [Open Questions](#open-questions).

### Sources
- Schema: {ref(FEATURES)}
- Pipeline: {refml(ML_PIPE)}
- Normalisation spec: `OLSR_Feature_Normalization_Table.docx` — **central to [Step 32](#step-32)**
""")

md(f"""
<a id="step-28" name="step-28"></a>
## Step 28 — Dataset generation
**Date:** 2026-07-13

### Design
| Property | Value |
|---|---|
| Defenses | FPNT, TRUST2, DCFM, WATCHDOG |
| Mobility regimes | static, mobile |
| **Conditions** | **8** (4 × 2) |
| Simulations per condition | **2,000** |
| Windows per simulation | **4** (baseline / attack-only / defense-only / defense+attack), 40 s each |
| Label | `defense_enabled ∈ {{0, 1}}` |

Generation parameters (WATCHDOG run):
```
--nNodes=50 --nMaxGridX=750 --nMaxGridY=1000 --bHighRange=false
--maliciousNodes=2 --spoofCount=5 --attackerJitter=25
```

### Verified integrity — FPNT/static **[VERIFIED]**
| Property | Value |
|---|---:|
| Feature rows | 8,021 (8,020 data + header) |
| Label rows | 8,021 (consistent) |
| Distinct `run_id` (CV groups) | **2,005** |
| Windows per run | **4** (exactly) |
| `defense_enabled = 0` | 4,010 |
| `defense_enabled = 1` | 4,010 |

**Perfectly balanced** (2 ON / 2 OFF per run); 2,005 × 4 = 8,020 confirms integrity.
Only FPNT/static was audited at this depth; the other seven conditions were confirmed to
contain both required CSVs.

### Layout
| Condition | Path |
|---|---|
| FPNT / static | `fpnt_static/` |
| FPNT / mobile | `fpnt_dynamic/` |
| TRUST2 / static | `trust_static/` |
| TRUST2 / mobile | `trust_dynamic/` |
| DCFM / static | `Dcfm_All_128_features/static/` |
| DCFM / mobile | `Dcfm_All_128_features/mobile/` |
| WATCHDOG / static | `Watchdog_All_128_features/static/` |
| WATCHDOG / mobile | `Watchdog_All_128_features/mobile/` |

Naming is **not uniform**; each run passes an explicit `--data-root`. Directories named
`*_dynamic` are the regime the pipeline labels `mobile`.

Each directory holds `windows_features.csv` (133 columns = 5 identifiers + 95 Core + 33
V2) and `windows_labels.csv`.

### ⚠️ A design consequence that matters enormously later
**All simulations use a single network configuration** — N = 50, fixed grid, fixed window
length. Therefore **feature normalisation intended to support generalisation across
network sizes provides no benefit in this experiment** — while, as [Step 32](#step-32)
argues, potentially causing severe harm. Normalisation was introduced for a benefit this
dataset cannot realise.

### Sources
- Schema: {ref(FEATURES)}
""")

md(f"""
<a id="step-29" name="step-29"></a>
## Step 29 — Experiment 1: baseline on 32 features; leakage discovered
**Runtime:** 80 min (8 runs) · **Results:** `scripts_for_all_128/step28_exp1_baseline_32/results_run1/`

### Pipeline
{refml(ML_PIPE)} implements:

| Stage | Description |
|---|---|
| [1] Loading | Merge `windows_features.csv` + `windows_labels.csv` on `run_id`, `scenario` |
| [2] Feature engineering | Mechanism-based composites — **CDR**, **TDR**, **QoS**, and delay/hop ratios — plus monotone transforms (log1p, sqrt, squared, cubed) applied **dynamically to every selected feature** rather than to a fixed hand-written list |
| [3] Selection | **Refit per fold**: constant removal, correlation pruning (\\|r\\| > 0.95), and multi-criterion rank aggregation over **mutual information, ANOVA F, RandomForest and ExtraTrees** importances |
| [4] Model pool | Dummy, LogisticRegression, Ridge, SVM-RBF, MLP, RandomForest, ExtraTrees, HistGB, AdaBoost, LightGBM, XGBoost, CatBoost, Stacking — **13 models** |
| [5] CV | Repeated StratifiedGroupKFold (2×5), **grouped by `run_id`**, with a guard that raises if a run spans train/test |
| [6–7] Reporting | Every metric as **mean ± sd across folds with a Nadeau-Bengio corrected confidence interval** (the correction for the dependence between overlapping training sets in repeated CV); grouped permutation test; LaTeX table; figures |

**Why the protocol can be trusted.** Four guards, and the third is the one most easily
overlooked:

1. **Grouping by `run_id`** prevents windows of the same simulation from appearing in both
   train and test, enforced by a runtime guard that raises if a run ever spans the split.
2. **Selection is refit inside each fold** — constant removal, correlation pruning and rank
   aggregation all see training data only — so there is no selection leakage.
3. **Probability calibration and decision thresholds are fitted inside the fold too**, on an
   inner **80/20 calibration split** of the training portion. This matters more than it
   looks: `TPR@1%FPR` and MCC are threshold-dependent, so an operating point chosen on data
   the model is then scored against would inflate precisely the operational metrics that
   [Step 30](#step-30) goes on to rely upon. **No number in this report was measured on data
   that took part in any stage of its own model's fitting.**
4. **`Dummy` returns exactly AUC = 0.5000** in every run, pinning the chance floor and
   confirming the evaluation is unbiased.

*The leakage found below is in the data, not in the protocol.*

Feature expansion: 32 base → **182** engineered columns — the multiplier is stage [2]
applying the four transforms to every surviving feature, which is also the mechanism that
[Step 31](#step-31) later finds capable of *reconstructing* a removed leak from features
screened as clean.

### A documented omission — no synthetic oversampling
`defense_detection_v2` applied **SMOTE**; v4 does not. Two independent reasons, either of
which would be sufficient:

1. **The class balance is a property of the experimental design, not of luck.** Every
   simulation emits two defense-OFF and two defense-ON windows, verified exactly in
   [Step 28](#step-28) (4,010 / 4,010 for FPNT/static). There is no minority class to
   oversample.
2. **Synthetic interpolation contradicts the grouped split.** SMOTE manufactures points
   between existing samples; interpolating between windows of *different* runs would
   fabricate rows belonging to no `run_id` at all — dissolving exactly the group structure
   that guard 1 above exists to enforce. A resampler that cannot honour the grouping has no
   place inside a grouped protocol.

Recorded because it is a **removal**, and removals leave no trace in the code they are
absent from.

### Results **[VERIFIED]**
| Defense | Mobility | ROC-AUC | MCC | TPR@1%FPR |
|---|---|---:|---:|---:|
| FPNT | static | **1.0000** | 0.998 | **1.0000** |
| FPNT | mobile | 0.9989 | 0.996 | 0.9985 |
| TRUST2 | static | 0.8517 | 0.530 | 0.1497 |
| TRUST2 | mobile | 0.6946 | 0.294 | 0.0469 |
| DCFM | static | 0.9874 | 0.913 | 0.9193 |
| DCFM | mobile | 0.9983 | 0.975 | 0.9841 |
| WATCHDOG | static | 0.9680 | 0.807 | 0.6476 |
| WATCHDOG | mobile | 0.7533 | 0.371 | 0.0658 |

**AUC = 1.0000 is not a result. It is a symptom.**

### Diagnosis — permutation importance **[VERIFIED]**
| Defense | Mobility | Top feature | Share of total importance | Features holding 90% |
|---|---|---|---:|---:|
| FPNT | static | `AvgTxPacketSize` | **100.0%** | **1** |
| FPNT | mobile | `AvgTxPacketSize` | 63.9% | 2 |
| TRUST2 | static | `AverageEndToEndDelay` | 16.2% | 16 |
| TRUST2 | mobile | `AvgTxPacketSize` | 9.9% | 31 |
| DCFM | static | `FlowCount` | **94.8%** | **1** |
| DCFM | mobile | `FlowCount` | **95.9%** | **1** |
| WATCHDOG | static | `AverageEndToEndDelay` | 18.4% | 20 |
| WATCHDOG | mobile | `AverageMprCount` | 22.4% | 13 |

**FPNT and DCFM are each decided by a single feature.** TRUST2 and WATCHDOG distribute
importance across 13–31 features with behaviourally plausible leaders — the expected
profile of genuine learning.

> **The inversion at the heart of this project:** the two "worst" conditions
> (TRUST2 at 0.85, WATCHDOG/mobile at 0.75) are the **most trustworthy** results in the
> table — precisely because no shortcut was available to them.

### A genuine, reportable finding **[VERIFIED]**
In FPNT/mobile, **LogisticRegression scored 0.5165** while tree ensembles scored 0.99+.
The same pattern appears in DCFM/mobile (LogReg 0.9208 vs 0.998). **The defense signature
under mobility is non-linear** and is not recoverable by linear decision boundaries.

### Sources
- Pipeline: {refml(ML_PIPE)}
- Scripts: `scripts_for_all_128/step28_exp1_baseline_32/run_all_defenses.sh`, `rank_importance.py`, `per_defense_tables.py`, `verify_features.py`
- Outputs (`scripts_for_all_128/step28_exp1_baseline_32/results_run1/`): `per_defense_all_models.csv`, `importance_overview.csv`, `univariate_auc_matrix.csv`; and per condition `summary.csv`, `folds.csv`, `importance.csv`, `run_config.json`, `summary.tex`, `final_model.pkl`, `figures/`
""")

md("""
<a id="step-30" name="step-30"></a>
## Step 30 — Experiment 2: univariate screen and ablation to 26 features
**Runtime:** ~160 min (16 runs) · **Results:** `scripts_for_all_128/step29_exp2_ablation_26/results_run2_behavioral/`

### The screen **[VERIFIED]**
**Method.** For each feature *f* and condition *c*, compute `AUC(f, defense_enabled)`
using **that feature alone**, direction-agnostic: `max(auc, 1−auc)`. A feature
approaching 1.0 by itself is, by construction, **a label proxy rather than a behavioural
signal**.

| Feature | FPNT/s | FPNT/m | TRUST2/s | TRUST2/m | DCFM/s | DCFM/m | WD/s | WD/m | **MAX** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `AvgTxPacketSize` | **1.000** | **0.996** | 0.506 | 0.555 | 0.663 | 0.616 | 0.527 | 0.544 | **1.000** |
| `FlowCount` | 0.500 | 0.508 | 0.500 | 0.503 | **0.942** | **0.954** | 0.501 | 0.506 | 0.954 |
| `RoutingOverheadRatio` | 0.505 | 0.512 | 0.508 | 0.528 | **0.813** | **0.925** | 0.530 | 0.521 | 0.925 |
| `AverageMprCount` | 0.510 | 0.508 | 0.515 | 0.500 | 0.679 | **0.885** | 0.603 | 0.533 | 0.885 |
| `AverageAdvertisedLinksPerTCMessage` | 0.501 | 0.523 | 0.502 | 0.547 | 0.520 | **0.842** | 0.517 | 0.537 | 0.842 |
| `NormalizedRoutingLoad` | 0.543 | 0.620 | 0.501 | 0.511 | 0.727 | **0.813** | 0.523 | 0.523 | 0.813 |

**Key structural observation: leakage is defense-specific.** `AvgTxPacketSize` leaks only
in FPNT; `FlowCount` and its cluster leak only in DCFM. **In every other condition these
same features sit at ≈ 0.5 — worthless.** The leak is not a property of the feature; it
is a property of the (feature, defense) pair.

### Design decision — uniform, not per-defense, removal
A **single uniform removal list** was adopted for all defenses. Two reasons:

1. **Comparability** — defenses trained on different feature sets cannot be compared.
2. **Avoiding data snooping** — removing a feature *because it performed well* is
   post-hoc selection and is **not defensible**. Removal must follow a principle fixed
   **in advance**: *"configuration-dependent magnitudes are inadmissible"* — a principle
   the project's own normalisation table already encodes independently.

Two thresholds test sensitivity to the cut-off:
- **drop3** (MAX ≥ 0.90): `AvgTxPacketSize`, `FlowCount`, `RoutingOverheadRatio` → **29 features**
- **drop6** (MAX ≥ 0.80): + `AverageMprCount`, `AverageAdvertisedLinksPerTCMessage`, `NormalizedRoutingLoad` → **26 features**

### Ablation results **[VERIFIED]**
| Defense | Mob | AUC(32) | AUC(29) | AUC(26) | Δ(26−32) |
|---|---|---:|---:|---:|---:|
| FPNT | static | 1.0000 | 0.9569 | 0.9561 | −0.0439 |
| FPNT | mobile | 0.9989 | 0.8423 | 0.8369 | **−0.1620** |
| TRUST2 | static | 0.8517 | 0.8506 | 0.8519 | **+0.0002** |
| TRUST2 | mobile | 0.6946 | 0.6915 | 0.6805 | −0.0141 |
| DCFM | static | 0.9874 | 0.9721 | 0.8603 | −0.1271 |
| DCFM | mobile | 0.9983 | 0.9894 | 0.8773 | −0.1210 |
| WATCHDOG | static | 0.9680 | 0.9679 | 0.9663 | **−0.0016** |
| WATCHDOG | mobile | 0.7533 | 0.7485 | 0.7335 | −0.0198 |

### Why the 0.80 threshold is empirically justified
The a-priori worry was that a 0.80 cut would remove features that are **legitimate** for
TRUST2 and WATCHDOG (where they score ≈ 0.5) merely because they leak in DCFM. **The
data refuted the worry:** TRUST2/static moved **+0.0002**, WATCHDOG/static **−0.0016**.
The removals cost the clean defenses essentially nothing.

**More telling: drop3 barely dented DCFM** (0.9874 → 0.9721) *despite removing
`FlowCount`, which held 94.8% of its importance*. When the primary shortcut vanished the
model simply **migrated to the next one** — the moderate DCFM cluster. Only removing the
whole cluster (drop6) collapsed DCFM to 0.86. **That substitution behaviour is stronger
evidence that the cluster carries redundant leakage than any single univariate score
could be.**

**Adopted: the 26-feature set (drop6) as the clean baseline.**

### A material correction — metric selection **[VERIFIED]**
ROC-AUC alone is **misleading** for this task:

| Condition | AUC(32) → AUC(26) | TPR@1%FPR (32) → (26) |
|---|---|---|
| FPNT / mobile | 0.9989 → 0.8369 (−0.16) | **0.9985 → 0.1778 (−5.6×)** |
| DCFM / mobile | 0.9983 → 0.8773 (−0.12) | **0.9841 → 0.3164 (−3.1×)** |
| DCFM / static | 0.9874 → 0.8603 (−0.13) | **0.9193 → 0.2534 (−3.6×)** |

**ROC-AUC compressed the magnitude of the leakage** because it is insensitive near the
ceiling. A −0.16 AUC change reads as a minor regression; the same event is a **5.6-fold
collapse** in operational terms.

**Recommendation: report three metrics.**

| Metric | Question answered |
|---|---|
| **ROC-AUC** | Does the model *rank* correctly? (threshold-free; field standard; comparable to literature) |
| **MCC** | How correct is it once it must *decide*? (0 = chance, 1 = perfect) |
| **TPR@1%FPR** | Operational: what fraction of defended windows are caught at a 1% false-alarm budget? (chance = 0.01) |

**Documented assumption:** all "best model" selections use ROC-AUC as the criterion. A
different criterion could select a different model.

### Clean baseline — 26 features **[VERIFIED]**
| Defense | Mob | ROC-AUC | MCC | TPR@1%FPR | Reading |
|---|---|---:|---:|---:|---|
| FPNT | static | 0.9561 | 0.782 | **0.619** | usable |
| FPNT | mobile | 0.8369 | 0.524 | 0.178 | weak |
| TRUST2 | static | 0.8519 | 0.529 | 0.150 | weak |
| TRUST2 | mobile | 0.6805 | 0.263 | **0.047** | ≈ unusable (4.7× chance) |
| DCFM | static | 0.8603 | 0.552 | 0.253 | weak |
| DCFM | mobile | 0.8773 | 0.588 | 0.316 | weak |
| WATCHDOG | static | 0.9663 | 0.799 | **0.648** | usable |
| WATCHDOG | mobile | 0.7335 | 0.334 | **0.066** | ≈ unusable |

**Reportable findings:**
1. **Only static conditions yield operationally meaningful detection** (FPNT 0.62,
   WATCHDOG 0.65). All mobile conditions fall below TPR@1%FPR = 0.32.
2. **Mobility degrades passive detectability across every defense.** Physically coherent:
   motion adds variance to routing metrics and, for WATCHDOG specifically, degrades the
   reliability of neighbour overhearing.
3. **An AUC of 0.68–0.73 corresponds to TPR@1%FPR of just 0.05–0.07** — AUC
   substantially overstates practical detectability.

### Sources
- Scripts: `scripts_for_all_128/step29_exp2_ablation_26/diagnose_leakage.py`, `run_behavioral.sh`, `compare_runs.py`, `three_metrics.py`
- Outputs (`scripts_for_all_128/step29_exp2_ablation_26/results_run2_behavioral/`): `comparison_32_vs_29_vs_26.csv`, `three_metrics.csv`, `core95_scan.csv`; and `drop3/` + `drop6/` (8 condition dirs each)
""")

md("""
<a id="step-31" name="step-31"></a>
## Step 31 — Experiment 3: expansion to 76 features; leakage returns
**Runtime:** 134 min (8 runs) · **Results:** `scripts_for_all_128/step30_exp3_expansion_76/results_run3_expanded/`

### Motivation
The 26-feature baseline is clean but weak. Could the **Core-95** set supply legitimate
behavioural signal to strengthen it — screened by the same univariate criterion?

### Core-95 screen **[VERIFIED]**
The Core-95 list was **derived from the data** (all columns − 5 identifiers − 33 V2)
rather than hard-coded; the script asserts the count is 95.

| Bucket | Criterion | Count |
|---|---|---:|
| Leaky | MAX ≥ 0.90 | 23 |
| Suspect | 0.80 ≤ MAX < 0.90 | 16 |
| **Safe** | **MAX < 0.80** | **50** |
| Constant (NaN) | no variance | 6 |

**Notable findings from the scan:**
- `TcMessageSizeMeanPerNode` = **1.000** in FPNT (both regimes) — a **second, independent
  perfect leak** of the same underlying quantity as `AvgTxPacketSize`. **FPNT's TC
  padding is detectable through any TC-size feature.**
- Leakage is heavily concentrated in **DCFM/mobile**: dozens of features sit at ≈ 0.5
  everywhere and jump above 0.9 only there — `EphemeralAddressFraction` 0.979 (and
  **constant/NaN in DCFM/static**), `FracDegreeOneNodes` 0.961,
  `ConnectedComponentsPerNode` 0.947.
- WATCHDOG has a distinct MPR signature: `DistinctMprSetsPerSender` 0.914,
  `MprChurnEventsPerSenderPerSecond` 0.907 (static only).
- **TRUST2 never exceeds 0.64 across all 95 Core features.** No feature in the Core
  schema detects TRUST2 better than what is already available.
- **Exact duplicate pairs exist** — `AdvertisedAverageDegreeNorm` ≡
  `AdvertisedGraphDensity`; `AdvertisedLinksPerTcMeanNorm` ≡
  `MprSelectorCountPerTcMeanNorm`; `FracDistinctTcSenderAddresses` ≡
  `FracNodesOriginatingTc`; `DataBytesPerSecondPerFlow` ≡ `PacketsSentPerFlowPerSecond` ≡
  `DataPacketRatePerFlow`. The correlation pruning (|r| > 0.95) removes these
  automatically.

### The 76-feature set
```
clean 26 (from metrics32)   : 26
Core-95 SAFE  (MAX < 0.8)   : 50
Core-95 suspect (0.80-0.90) : 16  (excluded)
Core-95 leaky   (>= 0.90)   : 23  (excluded)
Core-95 constant/NaN        : 6   (excluded)
TOTAL -> 76 features
```
Generated **from the scan CSV**, not typed by hand; the script asserts no dropped feature
reappears and no duplicates exist. Expansion: 76 base → **390** engineered columns.

### Result: leakage returned **[VERIFIED]**
| Defense | Mob | AUC 26 → 76 | MCC 26 → 76 | **TPR@1%FPR 26 → 76** | Best model (76) |
|---|---|---|---|---|---|
| FPNT | static | 0.9561 → **0.9997** | 0.782 → 0.996 | 0.619 → **0.9998** | Ridge |
| FPNT | mobile | 0.8369 → **1.0000** | 0.524 → 0.999 | 0.178 → **1.0000** | Ridge |
| TRUST2 | static | 0.8519 → 0.9037 | 0.529 → 0.625 | 0.150 → 0.512 | CatBoost |
| TRUST2 | mobile | 0.6805 → 0.7117 | 0.263 → 0.299 | 0.047 → 0.099 | Stacking |
| DCFM | static | 0.8603 → **0.9903** | 0.552 → 0.924 | 0.253 → **0.9313** | CatBoost |
| DCFM | mobile | 0.8773 → **0.9982** | 0.588 → 0.975 | 0.316 → **0.9885** | CatBoost |
| WATCHDOG | static | 0.9663 → 0.9863 | 0.799 → 0.881 | 0.648 → 0.839 | XGBoost |
| WATCHDOG | mobile | 0.7335 → 0.7717 | 0.334 → 0.405 | 0.066 → 0.095 | Stacking |

> **This is not improved learning. This is leakage reintroduced.** Fifty features that
> are each individually weak (**all MAX < 0.80**) cannot legitimately produce perfect
> detection. FPNT/mobile TPR@1%FPR went 0.178 → **1.0000**.

### Mechanism — multivariate interaction **[VERIFIED]**
Permutation importance for FPNT/static:

| Feature | Importance | **Its univariate AUC in FPNT/static** |
|---|---:|---:|
| `TcMessageSizeMaxToMeanRatio` | 0.478 | 0.716 |
| `AdvertisedLinksPerTcMaxToMeanRatio` | 0.312 | **0.500** |
| `AverageJitter` | 0.142 | — |

The same two dominate FPNT/mobile (0.505, 0.273).

> **The second feature is univariately worthless — exactly chance (0.500) — yet carries
> 31% of the model's importance.** Together the two account for ≈ 79% and produce
> near-perfect separation.

**Mechanistic reading [HYPOTHESIS]:** `TcMessageSizeMaxToMeanRatio` measures dispersion
of TC message size; `AdvertisedLinksPerTcMaxToMeanRatio` measures dispersion of
advertised link count. Their **ratio approximates bytes per advertised link** — precisely
FPNT's padding signature (TC messages grow without a corresponding growth in advertised
links). Consistent with this: the winning model is **Ridge**, a *linear* model — the
pipeline had already pre-computed the interaction as an engineered composite, so no
non-linearity was needed to exploit it.

### The central methodological lesson
> **Univariate leakage screening is insufficient when the pipeline engineers ratios and
> products.** Two features that are individually uninformative can **jointly reconstruct
> a removed leak**. A feature at exactly AUC = 0.500 alone was the second most important
> feature in a model achieving AUC = 1.0000.

This is a **transferable** result — it applies to any ML pipeline that performs automated
feature engineering on top of a screened feature set — and should be stated as such in
the thesis.

### The whack-a-mole pattern
| Step | Feature removed | The model's next shortcut |
|---|---|---|
| 1 | `AvgTxPacketSize` (AUC 1.000) | `TcMessageSizeMeanPerNode` (AUC 1.000) |
| 2 | `TcMessageSizeMeanPerNode` | `TcMessageSizeMaxToMeanRatio` (+ interaction) |

**Every feature touching TC message size reveals FPNT.** This raises a question that
**feature removal cannot answer** — see [Step 32](#step-32).

### Secondary observation — importance diagnostics fail under redundancy **[VERIFIED]**
For DCFM/mobile, the **maximum** importance is **0.023**, all values diffuse. This is
**not** evidence of absence of leakage; it is a **known failure mode of permutation
importance under high feature redundancy** — with 390 correlated columns, permuting any
single column is compensated by the others. **The DCFM/mobile diagnosis is inconclusive
by this method** and requires grouped permutation or leave-one-group-out.

### Sources
- Scripts: `scripts_for_all_128/step30_exp3_expansion_76/scan_core95.py`, `build_features76.py`, `run_expanded.sh`, `compare_26_vs_76.py`
- Outputs: `scripts_for_all_128/step30_exp3_expansion_76/results_run3_expanded/compare_26_vs_76.csv` (+ 8 condition dirs); generated feature list `scripts_for_all_128/step30_exp3_expansion_76/features_clean.txt`
- Generated list: `scripts_for_all_128/step30_exp3_expansion_76/features_clean.txt`
""")

md("""
<a id="step-32" name="step-32"></a>
## Step 32 — Root-cause analysis: the normalisation hypothesis

> ## ⚠️ [HYPOTHESIS — NOT YET TESTED]
> **This section is a hypothesis.** It is consistent with all observations and with the
> project's own normalisation documentation, but **no test has been run**. It must be
> verified before being reported as a finding. **No result in [Step 30](#step-30) may be
> reported as final until this is tested.**

### The proposed mechanism
The normalisation table (`OLSR_Feature_Normalization_Table.docx`) documents, for
`AvgTxPacketSize`:

> *"In your simulations this is always 512 because it is a configuration constant
> (UDP_PACKET_SIZE)."*

> **Correction ([Step 37](#step-37)).** The table says 512; the value actually emitted on the
> wire is **540** — 512 B of UDP payload plus 28 B of UDP/IP headers. Measured on the
> un-normalised dataset: mean 540.0, std 0.0, `nunique = 1` across all 8012 static windows.
> The argument below is unaffected — the numerator is constant either way — but the constant
> is 540.
> *"Normalisation: divided by the mean size of all packets observed in the window
> (**including control messages**)."*

Therefore:

```
AvgTxPacketSize_normalised = 540 / mean_size_of_all_packets_including_control
                             ^^^                    ^^^^^^^^^^^^^^^^^^^^^^^^
                          constant                the only varying term
```

**The numerator is constant. All variance in the normalised feature originates in the
denominator — i.e. in control-message size. And FPNT pads TC messages.** The normalised
feature therefore **does not measure data-packet size at all**; it measures
control-message size, which *is* FPNT's signature. This would explain AUC = 1.000
exactly.

**This also explains an earlier project observation** that an *un-normalised* run
produced "more reasonable" results: un-normalised, the feature is **constant** and is
auto-removed as worthless. **Normalisation converted a useless constant into a perfect
FPNT detector.**

### The static/mobile asymmetry — same mechanism
Three features are normalised by **(nObs − 1)**, the number of observed nodes
(`AverageMprCount`, `AverageAdvertisedLinksPerTCMessage`, and related Core features).

| Regime | `nObs` behaviour | Predicted effect | Observed |
|---|---|---|---:|
| **Static** | all 50 nodes always observed → `nObs` **constant** | division by a constant preserves ranking → AUC unchanged | **0.520** |
| **Mobile** | nodes enter/leave range → `nObs` **varies** | if `nObs` correlates with defense state, the normaliser **injects that leak into every feature divided by it** | **0.842** |

This predicts **exactly** the asymmetry observed for
`AverageAdvertisedLinksPerTCMessage` (0.520 static vs 0.842 mobile) and, more strikingly,
`EphemeralAddressFraction` (**constant/NaN** in DCFM/static vs **0.979** in DCFM/mobile).

### If confirmed, the implication is structural
**The leak is not in any individual feature but in the normalisation denominator, and it
is inherited by every feature sharing that denominator.** No amount of feature removal
can eliminate it — **which is precisely the whack-a-mole pattern of [Step 31](#step-31)**.

### A competing explanation that must be excluded
An alternative account of the DCFM/mobile anomaly is that **the DCFM/mobile data were
generated differently** (different parameters, code version, or attack configuration).
Note the anomaly's **direction**: for FPNT, TRUST2 and WATCHDOG, mobility makes detection
**harder**; for DCFM, mobility makes it **easier**. **This inversion is anomalous** and
warrants a data-provenance audit of the DCFM cells.

### The required test (not yet performed)
Compute, for each condition:
1. `AUC(nObs, defense_enabled)` — **does the node-count normaliser itself predict the label?**
2. `AUC(mean_packet_size, defense_enabled)` — **does the packet-size normaliser predict the label?**
3. Per-group descriptive statistics (mean, std, range) of these denominators, split by
   static/mobile × ON/OFF.

**If the denominators leak, the remediations are:**
- **(a)** re-emit features **without** normalisation — valid here, since the
  single-configuration design gains nothing from it ([Step 28](#step-28)); or
- **(b)** normalise by a **configuration constant** (the known N = 50) rather than by an
  observed, defense-sensitive quantity; or
- **(c)** disable composite/transform engineering (`--no-engineered --no-transforms`) to
  prevent ratio-based reconstruction.
""")

# ==========================================================================
# PART IV
# ==========================================================================
md(f"""
<a id="step-33" name="step-33"></a>
## Step 33 — Experiment 2b: single- and pair-feature ablation of the DCFM cluster
**Date:** 2026-07-24 – 2026-07-26

### Motivation
[Step 30](#step-30) could only compare two points: removing the two highest-MAX cluster
features (drop3 -> 29, DCFM barely moved) and removing all three cluster members
(drop6 -> 26, DCFM collapsed to ~0.86). That leaves the middle unexplored. Is any **single**
cluster member responsible for the DCFM leak? Is the cluster **redundant** (each member
individually removable)? Or is there a **hierarchy** — some members stronger than others?
This step fills the gap with two families of runs on the same 8 conditions, same
`--drop-features` mechanism, leakage guard, and Dummy=0.5000 baseline as [Step 30](#step-30):
- **single ablation** — drop3 + one cluster feature -> **28 features** (3 runs)
- **pair ablation** — drop3 + two cluster features -> **27 features** (3 runs)

The three cluster features are `AverageMprCount` (Mpr), `AverageAdvertisedLinksPerTCMessage`
(Adv), and `NormalizedRoutingLoad` (NRL). Only DCFM is affected — each is a DCFM-only signal
(univariate AUC ~0.5 elsewhere) — so the tables below report the two DCFM conditions; the
other six conditions stayed within noise of their [Step 30](#step-30) values.

### Single-feature ablation (28 features) — DCFM
Reference: drop3 (29) is DCFM static/mobile = 0.9721 / 0.9894; drop6 (26) ~0.86 / ~0.88.

| Removed | Cond | AUC | MCC | TPR@1% | Feature the model moved to | share |
|---|---|---:|---:|---:|---|---:|
| Mpr | static | 0.9714 | 0.8388 | 0.7980 | `sqrt_NormalizedRoutingLoad` | 0.15 |
| Mpr | mobile | 0.9870 | 0.8930 | 0.8277 | `AverageAdvertisedLinksPerTCMessage` | 0.28 |
| Adv | static | 0.9739 | 0.8553 | 0.7897 | `NormalizedRoutingLoad` | 0.49 |
| Adv | mobile | 0.9781 | 0.8726 | 0.8501 | `AverageMprCount` | 0.44 |
| NRL | static | 0.9688 | 0.8286 | 0.7641 | `AverageMprCount` | 0.63 |
| NRL | mobile | 0.9789 | 0.8591 | 0.7876 | `AverageMprCount` | 0.66 |

Every single removal left DCFM at ~0.97–0.99, and the "moved-to" column shows the model
migrating to whichever cluster members remain — the whack-a-mole of [Step 31](#step-31) made
explicit.

### Pair ablation (27 features) — DCFM
| Removed (remaining) | Cond | AUC | MCC | TPR@1% | Dominant feature | share |
|---|---|---:|---:|---:|---|---:|
| Mpr+Adv (**NRL left**) | static | 0.9636 | 0.8266 | 0.7944 | `sqrt_NormalizedRoutingLoad` | 0.15 |
| Mpr+Adv (**NRL left**) | mobile | 0.9742 | 0.8523 | 0.7402 | `NormalizedRoutingLoad` | 0.76 |
| Adv+NRL (**Mpr left**) | static | 0.9657 | 0.8349 | 0.7287 | `AverageMprCount` | 0.78 |
| Adv+NRL (**Mpr left**) | mobile | 0.9655 | 0.8250 | 0.7749 | `AverageMprCount` | 0.80 |
| Mpr+NRL (**Adv left**) | static | **0.8703** | **0.5789** | **0.2926** | `interact_Throughput_AverageEndToEndDelay` | 0.22 |
| Mpr+NRL (**Adv left**) | mobile | **0.9364** | **0.7225** | **0.5172** | `AverageAdvertisedLinksPerTCMessage` | 0.71 |

### Comparison — DCFM AUC across the whole ladder
| Features | Set | static | mobile |
|---|---|---:|---:|
| 32 | full | 0.9874 | 0.9983 |
| 29 | drop3 | 0.9721 | 0.9894 |
| 28 | drop3 − Mpr | 0.9714 | 0.9870 |
| 28 | drop3 − Adv | 0.9739 | 0.9781 |
| 28 | drop3 − NRL | 0.9688 | 0.9789 |
| 27 | + only NRL left | 0.9636 | 0.9742 |
| 27 | + only Mpr left | 0.9657 | 0.9655 |
| 27 | + only Adv left | **0.8703** | **0.9364** |

> **Footnote added by [Step 37](#step-37).** The 27-set is nominally 27 columns but
> **effectively 23**. Three of them carry one quantity — the emitter writes `L_pdr` to both
> `PacketDeliveryRatio` and `RxTxPacketRatio`, and `1 - L_pdr` to `PacketLossRatio` — and
> `MidMessageRate` and `HnaMessageRate` are exactly zero in all 16 028 windows of both the
> normalised and un-normalised datasets. `FeatureSelector`'s zero-variance filter and its
> |r| > 0.95 correlation pruning remove all five in-fold, so no result here is invalidated;
> the count is simply larger than the effective dimensionality. [VERIFIED]
| 26 | drop6 (none left) | ~0.86 | ~0.88 |

### Findings
**[VERIFIED]** No single cluster feature is necessary. Removing any one (28 features) leaves
DCFM at ~0.97–0.99; the model migrates to the members that remain.

**[VERIFIED]** The cluster is **not** uniformly redundant — there is a hierarchy. When only one
member is left (27 features): `NormalizedRoutingLoad` alone holds DCFM at ~0.96–0.97, and
`AverageMprCount` alone holds it at ~0.966; but `AverageAdvertisedLinksPerTCMessage` alone is
**not** sufficient — DCFM/static falls to 0.870 (TPR@1% 0.29) and the model starts reaching for
non-cluster interaction terms. So `Mpr` and `NRL` are the strong members; `Adv` is weak.

**[VERIFIED]** The effective core of the leak is the pair `{{Mpr, NRL}}`: as long as either is
present DCFM stays ~0.97; only removing both (leaving Adv) begins the collapse, and removing all
three (drop6 -> 26) completes it at ~0.86. This refines [Step 30](#step-30)'s drop6 result,
which was stricter than necessary, and is consistent with [Step 32](#step-32): the members are
interchangeable carriers of the same normalisation-denominator signal, not independent features.

> **Naming note:** the single-run sub-folders are named `Run_With_29_Features_Without_<X>`, but
> each drops a 4th feature and is therefore a **28-feature** run; the "29" refers to the drop3
> starting point. The pair-run folders `Run_Without_<X>_AND_<Y>` are **27-feature** runs.

### Sources
- Runner {refml("scripts_for_all_128/Step_32_Experiment_Between_26_and_29_Features/run_drop4.sh", "run_drop4.sh")}
  (takes the feature(s) to remove as arguments); comparison
  {refml("scripts_for_all_128/Step_32_Experiment_Between_26_and_29_Features/compare_drop4.py", "compare_drop4.py")}
- Pipeline {refml(ML_PIPE)}
- Single-ablation outputs (28): {refml("scripts_for_all_128/Step_32_Experiment_Between_26_and_29_Features/Run_With_29_Features_Without_AverageMprCount/results_run4_drop4/")},
  and the `_Without_AverageAdvertisedLinksPerTCMessage` / `_Without_NormalizedRoutingLoad` siblings
- Pair-ablation outputs (27): {refml("scripts_for_all_128/Step_32_Experiment_Between_26_and_29_Features/Run_Without_MprCount_AND_Advertised/results_run4_drop4/")},
  `Run_Without_Adv_AND_NRL`, and `Run_Without_MprCount_AND_NRL`
- Combined DCFM summary: {refml("scripts_for_all_128/Step_32_Experiment_Between_26_and_29_Features/step32_dcfm_summary.csv", "step32_dcfm_summary.csv")}
""")

md(f"""
<a id="step-34" name="step-34"></a>
## Step 34 — Defense-independent features: from normalisation leakage to an observable, implementation-general set
**Date:** 2026-07-27

### Motivation
[Step 32](#step-32) left the normalisation hypothesis **untested and blocking**: no
[Step 30](#step-30) result was final until we knew whether the near-perfect scores came from
defense *behaviour* or from a normalisation *denominator* that moves with the defense. With the
feature schema {ref(FEATURES)} and the two defense implementations {ref(D_FPNT)} and
{ref(D_GCOP)} in hand, we could finally settle this from source and then build a final feature
set justified on first principles rather than on measured performance.

---

### A. Why normalisation created the leak — [VERIFIED from source]
Most Core/V2 features are *normalised* (divided by something) so they generalise across network
sizes. The hidden assumption is that the **denominator is a property of the network, not of the
defense**. Reading {ref(FEATURES)} shows that assumption breaks for three denominators, and the
break is exactly aligned with what each defense does on the wire:

- **`AvgTxPacketSize`** = a constant (the data packet size) ÷ **mean size of all packets
  *including control messages*.** FPNT piggybacks a 3-byte evaluation vector per advertised
  neighbour onto every TC message ({ref(D_FPNT)}). When FPNT is on, control messages grow, the
  denominator grows, and — because the numerator is frozen — *all* the variance is FPNT's TC
  padding. The feature is labelled "data packet size" but measures control-message size.
- **`nObs`** = the count of distinct observed addresses, the denominator of the whole
  MPR / advertised-links family. DCFM/GCOP advertises a fictitious address (`main+65536`) inside
  the node's own TC ({ref(D_GCOP)}); every injected address inflates `nObs`, so every feature
  divided by it inherits a DCFM signal regardless of its numerator.
- **`graphN`** = the advertised-graph node count. The same fictitious addresses appear in the
  advertised graph, so graph-structure features divided by `graphN` shift too — in **both**
  mobility regimes.

Because many features share one denominator, the leak is **collective**: removing one feature
just moves it to the next feature that divides by the same quantity. This is precisely the
[Step 31](#step-31) whack-a-mole and the interchangeable `Mpr`/`NRL` carriers of
[Step 33](#step-33) — they were never independent features, only different numerators over the
same contaminated denominator.

**The fix (this step):** keep only features whose normalisation *cannot* carry a defense signal
— those that are **scale-free by construction** (ratios, coefficients of variation, standardised
moments, Gini, relative entropy) or divided by a **genuine constant** (the observer's own window
length). Everything divided by `nObs`, `graphN`, or the packet-size mean is dropped. **59
features** survive this audit; the ~69 excluded features are removed *in aggregate* for one
reason — a defense-sensitive or non-neutral observed denominator — not case by case.

**The 59 defense-independent ("green") features**, by family:
- *Per-node TC volume / shape:* `PerNodeTcRateCv`, `PerNodeTcBytesCv`, `PerNodeTcBytesGini`
- *TC size & advertised-link shape:* `TcMessageSizeCv`, `TcMessageSizeP95ToMeanRatio`,
  `AdvertisedLinksPerTcCv`, `AdvertisedLinksPerTcP95ToMeanRatio`
- *TC sequence / content / validity:* `TcAnsnIncrementMean`, `TcAnsnSkipRatePerTc`,
  `TcMessageContentEntropyNorm`, `TcVtimeMean`, `TcVtimeStd`
- *MPR:* `MprSelectorCountPerTcCv`
- *TC timing & origination rate:* `TcInterArrivalMean`, `TcInterArrivalStd`, `TcInterArrivalP95`,
  `TcBurstinessHurst`, `ControlMessageInterArrivalSkew`, `ControlMessageInterArrivalKurtosis`,
  `TcOriginationRateMin`, `TcOriginationRateMean`, `TcOriginationRateStd`, `TcOriginationRateMax`
- *Address entropy:* `TcSenderAddressEntropyNorm`, `TcAdvertisedAddressEntropyNorm`
- *Packet-size distribution:* `PacketSizeDistributionSkew`, `PacketSizeDistributionKurtosis`
- *Advertised-graph structure:* `AdvertisedDegreeCv`, `AdvertisedDegreeSkew`,
  `AdvertisedDegreeKurtosis`, `AdvertisedClusteringCoefficient`, `AdvertisedRadiusToDiameterRatio`,
  `AdvertisedGraphDensity`, `EdgePersistenceFraction`
- *MAC-local (own radio):* `Layer2RetransmissionRate`, `ChannelBusyTimeFraction`
- *Paths / hops (data-plane):* `ObservedHopCountCv`, `PathChangesPerFlowPerSecond`
- *Latency / delivery (data-plane):* `EndToEndLatencyCv`, `JitterToLatencyRatio`,
  `DeliveredToSentBytesRatio`, `AverageJitter`, `PacketDeliveryRatio`, `PacketLossRatio`,
  `Throughput`, `RxTxPacketRatio`
- *Flow-level (data-plane):* `AvgFlowDuration`, `FlowDurationStd`, `AvgFlowJitter`,
  `AvgFlowLossRate`, `FlowThroughputStd`, `FlowDelayStd`, `FlowJitterStd`, `FlowLossRateStd`,
  `AvgTxBytesPerFlow`, `AvgRxBytesPerFlow`, `AvgTxPacketsPerFlow`, `AvgRxPacketsPerFlow`,
  `AvgRxPacketSize`

Trained with the pipeline ({refml(ML_PIPE)}) on all 8 conditions, the 59-feature set preserves
FPNT/DCFM detection with the denominator leak gone (see the Results table below).

---

### B. Transfer test → the 27 that generalise — [VERIFIED]
A clean feature can still score perfectly *within* a single frozen topology (N=50) for the wrong
reason. Since **leakage does not survive a change of conditions but a genuine signal does**, we
trained on one mobility regime and tested on the other (static ↔ mobile), per defense, using
{refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/transfer_test.py", "transfer_test.py")}.
Selecting per-feature survivors (target ROC-AUC ≥ 0.75 on the held-out regime for at least one
defense) with
{refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/build_final_set.py", "build_final_set.py")}
gives **27 features**:

| # | Feature | Carrier defense | Transfer AUC |
|---|---|---|---:|
| 1 | `TcAdvertisedAddressEntropyNorm` | dcfm | 0.956 |
| 2 | `TcMessageSizeCv` | fpnt | 0.946 |
| 3 | `AdvertisedGraphDensity` | dcfm | 0.942 |
| 4 | `PathChangesPerFlowPerSecond` | fpnt | 0.938 |
| 5 | `ChannelBusyTimeFraction` | dcfm | 0.925 |
| 6 | `AdvertisedDegreeCv` | dcfm | 0.915 |
| 7 | `PerNodeTcBytesGini` | fpnt | 0.914 |
| 8 | `ControlMessageInterArrivalKurtosis` | dcfm | 0.905 |
| 9 | `ControlMessageInterArrivalSkew` | dcfm | 0.904 |
| 10 | `PacketSizeDistributionKurtosis` | dcfm | 0.902 |
| 11 | `PerNodeTcBytesCv` | fpnt | 0.882 |
| 12 | `PacketSizeDistributionSkew` | dcfm | 0.878 |
| 13 | `EdgePersistenceFraction` | watchdog | 0.854 |
| 14 | `TcMessageContentEntropyNorm` | watchdog | 0.850 |
| 15 | `TcInterArrivalMean` | dcfm | 0.847 |
| 16 | `TcMessageSizeP95ToMeanRatio` | fpnt | 0.846 |
| 17 | `Layer2RetransmissionRate` | dcfm | 0.841 |
| 18 | `TcInterArrivalP95` | dcfm | 0.812 |
| 19 | `AdvertisedDegreeKurtosis` | dcfm | 0.806 |
| 20 | `ObservedHopCountCv` | dcfm | 0.794 |
| 21 | `TcInterArrivalStd` | dcfm | 0.781 |
| 22 | `TcOriginationRateMean` | watchdog | 0.774 |
| 23 | `TcOriginationRateStd` | watchdog | 0.772 |
| 24 | `TcOriginationRateMin` | watchdog | 0.771 |
| 25 | `AverageJitter` | fpnt | 0.756 |
| 26 | `JitterToLatencyRatio` | fpnt | 0.756 |
| 27 | `AvgFlowJitter` | fpnt | 0.755 |

At the **whole-set** level the split between defenses is stark: FPNT transfers at ROC-AUC 1.000
in both directions and DCFM at 0.907 / 0.649, but **TRUST and WATCHDOG collapse to ~0.50
(random)** — their in-condition scores were single-topology inflation, not generalising signal.

---

### C. Which of the 27 are observable by a single node — the 22 — [reasoned, not re-run]
The passive attacker is a single node in promiscuous mode (the Watchdog observer model).
{refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/classify_observability.py", "classify_observability.py")}
classifies each of the 27 by *how it is computed* in {ref(FEATURES)}. Because TC is flooded
network-wide, TC-derived and reconstructed-advertised-graph features are observable; but five
features are computed from multi-hop **data-plane** traffic (the extractor is fed a latency value
and a last-forwarder TTL that require seeing both ends of a flow) and are therefore **not**
observable by one node:

> **Removed as GLOBAL (5):** `PathChangesPerFlowPerSecond`, `ObservedHopCountCv`, `AverageJitter`,
> `JitterToLatencyRatio`, `AvgFlowJitter`.

The remaining **22** are single-node observable. **Note:** the learning pipeline was **not**
re-run on this 22-feature set; it is a classification result used to reach the final set in §E.

---

### D. Which of the 27 are implementation-general — the 8 — [reasoned, not re-run]
A signature is only useful if *any* faithful implementation of the defense would produce it, not
just our specific code.
{refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/classify_implementation.py", "classify_implementation.py")}
applies this using the mechanisms in {ref(D_FPNT)} (FPNT enlarges every TC) and {ref(D_GCOP)}
(DCFM advertises **one** fictitious address inside existing TC — it adds **no** channel traffic
and does **not** change TC timing; the code's own comments warn that spurious hexagon detections
distort feature vectors). By that test, channel/MAC/timing signatures for DCFM are artefacts, and
the **behaviour-general core is 8 features**:

| Feature | Defense mechanism (any faithful implementation) |
|---|---|
| `TcMessageSizeCv` | FPNT enlarges every TC message |
| `TcMessageSizeP95ToMeanRatio` | FPNT enlarges TC (size-distribution shift) |
| `PerNodeTcBytesGini` | FPNT adds per-neighbour bytes to TC |
| `PerNodeTcBytesCv` | FPNT adds per-neighbour bytes to TC |
| `TcAdvertisedAddressEntropyNorm` | DCFM injects fictitious advertised addresses |
| `AdvertisedGraphDensity` | DCFM adds nodes/edges to the advertised graph |
| `AdvertisedDegreeCv` | DCFM perturbs advertised-graph degrees |
| `AdvertisedDegreeKurtosis` | DCFM perturbs advertised-graph degrees |

These map exactly onto the two loud defenses' defining behaviours (FPNT enlarges TC; DCFM injects
addresses/structure), and cover **no** TRUST or WATCHDOG signature — those defenses have no
control-plane mechanism a general feature could catch. **Note:** the pipeline was **not** re-run
on this 8-feature core either; like the 22, it is a classification result, reported to show how
tight the behavioural core is.

---

### E. The final set — 21 features (observable AND implementation-general) — [VERIFIED]
Rather than chain §B→§C→§D (which lets the empirical transfer filter pre-decide the space), the
final set applies the two **a-priori** criteria — single-node observability and
implementation-generality (lenient: kept if a *reasonable* FPNT/DCFM implementation would move
it) — **directly to all 59 green features, without looking at any performance number.**
{refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/classify_apriori.py", "classify_apriori.py")}
yields **21 features**:

| # | Feature | Mechanism | Observability |
|---|---|---|---|
| 1 | `PerNodeTcBytesCv` | FPNT | local |
| 2 | `PerNodeTcBytesGini` | FPNT | local |
| 3 | `TcMessageSizeCv` | FPNT | local |
| 4 | `TcMessageSizeP95ToMeanRatio` | FPNT | local |
| 5 | `AdvertisedLinksPerTcCv` | DCFM | local |
| 6 | `AdvertisedLinksPerTcP95ToMeanRatio` | DCFM | local |
| 7 | `TcAnsnIncrementMean` | DCFM | local |
| 8 | `TcAnsnSkipRatePerTc` | DCFM | local |
| 9 | `TcMessageContentEntropyNorm` | FPNT+DCFM | local |
| 10 | `MprSelectorCountPerTcCv` | DCFM | local |
| 11 | `ChannelBusyTimeFraction` | FPNT | local |
| 12 | `TcAdvertisedAddressEntropyNorm` | DCFM | local |
| 13 | `PacketSizeDistributionSkew` | FPNT | local |
| 14 | `PacketSizeDistributionKurtosis` | FPNT | local |
| 15 | `AdvertisedDegreeCv` | DCFM | tc-graph |
| 16 | `AdvertisedDegreeSkew` | DCFM | tc-graph |
| 17 | `AdvertisedDegreeKurtosis` | DCFM | tc-graph |
| 18 | `AdvertisedClusteringCoefficient` | DCFM | tc-graph |
| 19 | `AdvertisedRadiusToDiameterRatio` | DCFM | tc-graph |
| 20 | `AdvertisedGraphDensity` | DCFM | tc-graph |
| 21 | `EdgePersistenceFraction` | DCFM | tc-graph |

This 21-feature set **was** trained and evaluated on all 8 conditions
({refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/run_apriori21.sh", "run_apriori21.sh")},
{refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/compare_apriori.py", "compare_apriori.py")}).

---

### Results — [VERIFIED]
ROC-AUC (and, for the 21-set, TPR@1%FPR) across all 8 conditions, comparing the final 21-feature
set with the 59-feature green set and the earlier drop6 (26) baseline:

| Condition | 26 AUC | 59 AUC | 21 AUC | 21 TPR@1% |
|---|---|---|---|---|
| fpnt_static | 0.956 | 1.000 | 1.000 | 1.000 |
| fpnt_mobile | 0.837 | 1.000 | 1.000 | 1.000 |
| dcfm_static | 0.860 | 0.988 | 0.983 | 0.888 |
| dcfm_mobile | 0.877 | 0.999 | 0.999 | 0.992 |
| trust_static | 0.852 | 0.882 | 0.698 | 0.102 |
| trust_mobile | 0.681 | 0.691 | 0.622 | 0.034 |
| watchdog_static | 0.966 | 0.984 | 0.925 | 0.588 |
| watchdog_mobile | 0.734 | 0.750 | 0.643 | 0.040 |

**What it means.** Dropping **38 of 59** features on principle costs almost nothing for FPNT/DCFM
(both far above the 26-baseline on TPR@1%), while TRUST/WATCHDOG fall — the intended outcome, not
a loss. Re-running the transfer test on the 21-set **improves** DCFM generalisation
(mobile→static rises from 0.649 on the 59-set to **0.942** on the 21-set): the a-priori-removed
features were injecting topology-dependent noise. Crucially, the a-priori selection (from code)
and the transfer-survival ranking (from performance) — two independent routes — converge on the
same core: TC size / bytes for FPNT, advertised-address entropy and graph structure for DCFM.

**Central result — [VERIFIED]:** generalising passive detection works for the *loud* defenses
(FPNT perfectly, DCFM strongly) and essentially **fails** for the *gentle* ones (TRUST,
WATCHDOG), with every surviving feature justified on three independent a-priori grounds — clean
denominator, single-node observability, implementation-general mechanism. This is the sharpest
statement of the detectability↔efficacy tradeoff in the project.

**Limitations — [HYPOTHESIS].** Implementation-generality is argued from a *single*
implementation of each defense; surviving an alternative faithful implementation is untested and
is the natural next campaign. All features were also extracted at a single network size (N=50),
so size-generalisation remains untested. And because the 21-set was selected using all 8
conditions, its numbers are the **selected set's** performance (optimistic); a clean held-out
estimate needs the full multi-size campaign — see [Open Questions](#open-questions).

### Sources
- NS-3: feature schema {ref(FEATURES)}; FPNT {ref(D_FPNT)}; DCFM/GCOP {ref(D_GCOP)}
- Pipeline {refml(ML_PIPE)}
- Classifiers: {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/classify_apriori.py", "classify_apriori.py")},
  {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/classify_observability.py", "classify_observability.py")},
  {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/classify_implementation.py", "classify_implementation.py")}
- Transfer + selection: {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/transfer_test.py", "transfer_test.py")},
  {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/build_final_set.py", "build_final_set.py")}
- Runner {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/run_apriori21.sh", "run_apriori21.sh")};
  comparison {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/compare_apriori.py", "compare_apriori.py")}
- Feature lists: {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/features_clean_step33.txt", "features_clean_step33.txt")} (59),
  {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/features_apriori_lenient.txt", "features_apriori_lenient.txt")} (21)
- Outputs: {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/results_run9_apriori21/")} and the transfer CSVs under `.../transfer_test/`
""")

md(f"""
<a id="step-35" name="step-35"></a>
## Step 35 — Generalisation as a pipeline capability: mobility transfer, the cross-defense matrix, and Leave-One-Defense-Out
**Date:** 2026-07-29

### Motivation — three open threads, one missing instrument
[Step 34](#step-34) settled the final feature set on a principle it stated explicitly —
**leakage does not survive a change of conditions, but a genuine signal does** — and used
that principle to decide which features are admissible. It exercised the principle through
{refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/transfer_test.py", "transfer_test.py")}:
a standalone script, written for one selection decision, over one axis (static ↔ mobile).
A test that decides *which features are real* deserves to be a standing capability of the
pipeline rather than a script that outlives its experiment by accident.

The [planned full-scale campaign](#full-campaign) names **cross-defense generalisation** as
its third research question and calls it *"the strongest available test that the learned
features are behavioural rather than implementation-specific — and the one that matters
most."* Campaign 2 had no instrument for it at all.

[Campaign 1](#step-25) did build transfer and open-set modules
({refml(DML + "/defense_ml/transfer.py", "transfer.py")},
{refml(DML + "/defense_ml/openset.py", "openset.py")}) — but they live in the `defense_ml`
package and were run on the **pre-realignment** DCFM over **un-normalised** data. Nothing
carried them onto the normalised 128-feature dataset that [Steps 27–34](#step-27) analyse.

This step folds all three capabilities into {refml(ML_PIPE)} itself.

### What was added
Three experiments, each **off by default** and enabled by a dedicated flag. They compose —
one, two, or all three in a single invocation.

| # | Experiment | Flag | Question it answers |
|---|---|---|---|
| 1 | **Mobility transfer** | `--transfer-mobility` | Does a model trained on a static network still detect the defense on a mobile one (and the reverse), per defense? |
| 2 | **Cross-defense matrix** | `--transfer-defense` | N×N: trained on defense A, how well is each other defense recognised? |
| 3 | **Leave-One-Defense-Out** | `--lodo` | Trained on three defenses, is the fourth — never seen — still detected? |

> **Operational note.** When any transfer flag is given, the transfer experiments run
> **instead of** that invocation's ordinary cross-validation, not alongside it.

### The methodological core — the frozen source model
In all three experiments "transfer" means one thing: a **frozen source model**. The fitted
pipeline, the isotonic calibration, and the selected decision threshold are all determined
on source data alone, then applied to the target with no re-fitting and no information from
the target at any stage. Four guards make the resulting number mean what it claims:

| # | Guard | Why it is load-bearing |
|---|---|---|
| 1 | **Nothing is re-fitted on the target** — pipeline, calibrator **and threshold** all frozen at source | Re-fitting *anything* on the target stops measuring transfer and starts measuring how well a target-tuned model can do — in-domain performance in a transfer experiment's clothes. **The threshold is the subtle one:** MCC and `TPR@1%FPR` are threshold-dependent, so quietly re-choosing the operating point on the target restores much of what the domain shift took away |
| 2 | **Group namespacing** — `run_id` is prefixed to `<defense>/<run_id>` before any union | Two defenses number their runs from the same origin, so the same `run_id` exists in both and refers to unrelated simulations. Unioned unprefixed, `GroupShuffleSplit` would fuse their windows into one group — conservative in direction, but a silent corruption of the grouping the entire protocol rests on |
| 3 | **Common-feature intersection** — the model is fitted and applied on the columns shared by every domain involved; the count is written to `n_features` in each output row and to the log | The monotone transforms (`log1p`, `sqrt`) are applied conditionally on the data, so two domains need not yield identical engineered column sets. Fitting on one and scoring on the other otherwise risks a hard failure or — worse — a silent fill |
| 4 | **In-domain reference cells** — matrix diagonals use a grouped 70/30 holdout; LODO additionally reports `known_holdout_roc_auc` over the three seen defenses | A transfer number alone says nothing. ROC-AUC 0.78 on the target is strong if the same model scores 0.80 in-domain and weak if it scores 0.99. **The drop is the measurement, not the level** — so every unseen cell has its own control beside it, and even the "local" number is measured on runs the model never trained on |

### The new surface
| Flag | Default | Meaning |
|---|---|---|
| `--transfer-mobility` | off | Enable experiment 1 |
| `--transfer-defense` | off | Enable experiment 2 |
| `--lodo` | off | Enable experiment 3 |
| `--transfer-model` | `auto` | Model used for the transfer experiments. `auto` resolves LightGBM → XGBoost → HistGB (first available); any name from the v4 model zoo is also accepted |

One consequential side change: the default of `--defense` moved from `fpnt` to **`None`**.
An ordinary run still selects `fpnt`, but `--transfer-mobility` without an explicit
`--defense` now sweeps **all** defenses — the sensible default for an experiment whose
whole point is comparison across defenses.

```bash
# static <-> mobile transfer, all four defenses
python defense_detection_v4.py --transfer-mobility

# defense-to-defense matrix, both mobility regimes
python defense_detection_v4.py --transfer-defense --mobility both

# all three experiments, both regimes, with an explicit model
python defense_detection_v4.py --transfer-mobility --transfer-defense --lodo \\
    --mobility both --transfer-model HistGB
```

The new code is confined to one section, **`[9] Transfer experiments`**, placed ahead of
the CLI:

| Function | Role |
|---|---|
| `pick_transfer_model` | Resolves `--transfer-model auto` to the strongest available booster |
| `load_engineered_condition` | Loads one (defense, mobility) condition and applies **exactly** the CV path's feature engineering, with the defense-prefixed groups |
| `common_columns` | The feature-column intersection across the domains involved |
| `fit_frozen_model` | Fits on one domain under the standard calibration and threshold-selection protocol; returns calibrated model + threshold |
| `score_frozen_model` | Evaluates a frozen model on a target domain **at the source's threshold** |
| `in_domain_reference` | The grouped 70/30 holdout that fills the diagonal |
| `save_transfer_heatmap` | Renders a transfer matrix |
| `run_transfer_mobility` / `run_transfer_defense` / `run_lodo` | The three experiment drivers |
| `run_transfer_experiments` | Entry point when any transfer flag is given; also writes `transfer_config.json` |

### Containment — why no earlier result moves
The core CV and statistics path is **untouched**: `make_splits`, `eval_fold`, `aggregate`
and the permutation test are unchanged, and the new section reuses `load_dataset`,
`engineer_features`, `make_pipeline_for`, `_calibrate_prefit` and `tpr_at_fpr` rather than
reimplementing them. Combined with the flags being off by default, this means **no number
reported in [Steps 29–34](#step-29) is affected by this change** — worth stating outright,
because a modification to the pipeline that produced the report's results would otherwise
put every one of those results back in question.

### Outputs
Written to `{DML}/results/30_schema33/paper_v4/transfer/` (the v4 default results root, or
whatever `--results-dir` selects):

| File | Contents |
|---|---|
| `transfer_mobility.csv` | Experiment 1 — every cell of every defense |
| `transfer_defense_<mobility>.csv` | Experiment 2 — one matrix per mobility regime |
| `lodo_<mobility>.csv` | Experiment 3, including the `known_holdout_*` reference columns |
| `transfer_config.json` | Reproduction manifest — model, seed, feature set, which experiments ran, timestamp |
| `figures/transfer_mobility_<defense>.png` | 2×2 heatmap per defense |
| `figures/transfer_defense_<mobility>.png` | N×N heatmap |
| `figures/lodo_<mobility>.png` | Known-defenses vs. unseen-defense bar chart |

Every CSV row carries the full metric set — `roc_auc`, `pr_auc`, `balanced_accuracy`, `f1`,
`mcc`, `tpr_at_fpr01`, `tpr_at_fpr05`, `threshold` — alongside `n_features`, `n_train`,
`n_test` and the source/target identifiers. Reporting `TPR@1%FPR` beside ROC-AUC is not
decoration: [Step 30](#step-30) showed AUC compressing a 5.6-fold operational collapse into
a −0.16 change, and a transfer table read on AUC alone would hide the same thing.

### Status — [VERIFIED]
| Check | Result |
|---|---|
| `python -m py_compile defense_detection_v4.py` | passes |
| `--help` | lists all four new flags |
| Smoke test (`--limit-runs 30`), all three experiments | completes; every CSV and figure written |
| Full run — three experiments × four defenses × two mobility regimes | completes, **exit code 0**; all outputs written |
| Behaviour with no transfer flag given | unchanged from before this step |

### What is **not** claimed here
> **This step documents an instrument, not a result.** The full run completed and the tables
> exist, but **no cell of any transfer matrix has been analysed or reported in this report.**
> Nothing in this step may be cited as a finding about generalisation. The result tables are
> the *input* to the next step, not its conclusion.

### Pre-registered predictions — [HYPOTHESIS]
Recorded **before** the tables were read, for the same reason [Step 30](#step-30) fixed its
removal criterion in advance: a threshold chosen after seeing the scores is not defensible.

| Experiment | Prediction | Grounding |
|---|---|---|
| 1 — mobility | FPNT ≈ 1.00 in both directions; DCFM strong (≈ 0.94 mobile → static on the 21-set); **TRUST and WATCHDOG ≈ 0.50** | [Step 34](#step-34) §B measured exactly this with the standalone script |
| 2 — defense → defense | Off-diagonal near chance throughout. Specifically **FPNT ↮ DCFM**, because the 21-set's FPNT features (TC size and per-node TC bytes) and DCFM features (advertised-address entropy, advertised-graph structure) are **disjoint mechanism families**; and TRUST/WATCHDOG rows *and* columns near chance, because the 21-set encodes no TRUST or WATCHDOG signature at all | [Step 34](#step-34) §D; [Step 25](#step-25)'s *"low transfer is expected — and is itself a result"* |
| 3 — LODO | A large gap between `known_holdout_roc_auc` (high) and the unseen defense (low), for the same reason: the surviving features are mechanism-specific, so a defense whose mechanism was never in training has nothing for the model to recognise | [Step 34](#step-34) §D |

> **Experiment 1 overlaps [Step 34](#step-34) §B by construction — deliberately.** It
> re-derives, inside the pipeline, a result already measured by an independent script. If
> the two disagree, one of the two implementations is wrong. The first experiment is
> therefore also the instrument's own control.

### A distinction worth preserving
The LODO implemented here is a **detection-transfer** test: a presence classifier trained on
three defenses, evaluated on the fourth. It is **not** the open-set novelty scoring of
[Step 25](#step-25) ({refml(DML + "/defense_ml/openset.py", "openset.py")}: max-softmax,
Mahalanobis-to-nearest-centroid, IsolationForest, scored by OSCR), which asks whether an
unseen defense can be flagged as *"other"* rather than mislabelled. The two answer different
operational questions — *"would I still notice it?"* versus *"would I know I hadn't seen it
before?"* — and only the first exists in v4 today.

### Sources
- Pipeline: {refml(ML_PIPE)} — new section `[9] Transfer experiments`
- Outputs: `{DML}/results/30_schema33/paper_v4/transfer/` (`transfer_mobility.csv`,
  `transfer_defense_<mobility>.csv`, `lodo_<mobility>.csv`, `transfer_config.json`, `figures/`)
- Antecedents: {refml(DML + "/defense_ml/transfer.py", "transfer.py")} and
  {refml(DML + "/defense_ml/openset.py", "openset.py")} (Campaign 1);
  {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/transfer_test.py", "transfer_test.py")} (Step 34)
- Feature set under test: {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/features_apriori_lenient.txt", "features_apriori_lenient.txt")}
""")

md(f"""
<a id="step-36" name="step-36"></a>
## Step 36 — Cross-defense intersection: detecting an unseen defense, and the Step-35 prediction overturned
**Date:** 2026-08-04

### Motivation — turning the instrument on the question, with a stricter selection protocol
[Step 35](#step-35) built the Leave-One-Defense-Out **instrument** and, before reading any
table, pre-registered a [HYPOTHESIS]: a defense whose mechanism was never in training would
sit near chance (TRUST and WATCHDOG ≈ 0.50), because the surviving features are
mechanism-specific. This step runs the experiment in earnest — with a **different, stricter
feature-selection protocol supplied by the supervisor** — and that prediction does not
survive.

The work has its own self-contained script tree, `{S34}/`, kept apart from the v4 pipeline
so that a change to v4 cannot silently alter it.

### The selection protocol (the methodological core)
Model-based importance is replaced by a **per-defense univariate ranking, aggregated by the
worst case** — the supervisor's protocol:

1. a univariate separation score (**mutual information**) for each feature, computed
   **per development defense** (not on a pooled set, which the strongest defense would dominate);
2. turned into a rank matrix (feature × defense);
3. aggregated by the **worst rank**, R(f) = max over the development defenses — so a feature
   survives only if it ranks well for *every* development defense. This is exactly the
   **intersection of the per-defense top-k lists**: a feature is in the intersection at cut k
   iff its worst rank ≤ k;
4. the cut **k is chosen by a nested LODO** inside the three development defenses (the k with
   the best mean inner-AUROC), not fixed by hand.

Everything runs inside the outer LODO: a run-level 60/20/20 split stratified by defense; the
held-out defense removed from Train **and** Val and evaluated only on its Test runs; the
scaler and the ranking fitted on development defenses at every layer. Two classifiers are
run — **Logistic Regression** (the conservative a-priori choice) and **Random Forest** (a
non-linear sensitivity model).

### The instrument
| File | Role |
|---|---|
| {refml(S34 + "/step34_lodo.py", "step34_lodo.py")} | Orchestrator: outer LODO, nested-LODO k, worst-rank top-k, final model, evaluation, attack-split, stability; `--permute-labels` null test |
| {refml(S34 + "/step34_common.py", "step34_common.py")} | Self-contained loaders, run-level split, MI rank matrix, worst-rank selection, metrics — **does not import v4** |
| {refml(S34 + "/summarize_step34.py", "summarize_step34.py")} | The 8-row table (regime × held-out) + static↔mobile core overlap |
| {refml(S34 + "/dominance_check.py", "dominance_check.py")} | Reproduces each fold's final model and reads its importances — the "who is dominant" audit |
| {refml(S34 + "/run_step34_on_27.sh", "run_step34_on_27.sh")} | One-shot runner for the 27-feature experiment (both models) |

### The metrics, and why each is present
- **ROC-AUC** — primary, threshold-independent: the probability a defense-on window outscores
  a defense-off one (1.0 = perfect separation, 0.5 = chance). It hides operating-point
  collapse, so it never travels alone — see [Step 30](#step-30).
- **TPR@FPR=5%** and **FPR@TPR=95%** — two a-priori operating points (a conservative and an
  aggressive attacker).
- **Attack-split** — paired AUROC with vs without an attacker present. The **no-attack** side
  isolates detection of the defense's *mere presence* (no attacker to react to).
- **Permutation null** — shuffle the label within each run and re-run; a pipeline free of
  structural leakage must collapse to ≈ 0.50.

### The two feature sets under test
Each is the survivor of an earlier leakage decision.

**Set A — 99 features** ({refml(S34 + "/features/features_99_surviving.txt", "features_99_surviving.txt")}):
the survivors of a **message-structure** filter over the 128-feature schema (features whose
value is fixed by byte size, message content/format, or single-message composition were
pruned; behavioural, timing, routing-dynamics and topology features were kept).

**Set B — 27 features** ({refml(S34 + "/features/features_27_step32.txt", "features_27_step32.txt")}):
*(nominally 27; effectively 23 — see the footnote at [Step 33](#step-33) and
[Step 37](#step-37).)*
the [Step 33](#step-33) set — the canonical 32 minus the five leak / normalisation-denominator
carriers below. It is the focus of this step.

*The five removed to form the 27:*
| Removed from the 32 | Why |
|---|---|
| `AvgTxPacketSize`, `FlowCount`, `RoutingOverheadRatio` | the drop3 leak carriers (univariate AUC ≥ 0.90) — [Step 29](#step-29) |
| `AverageMprCount`, `NormalizedRoutingLoad` | the DCFM normalisation-denominator cluster — [Step 32](#step-32), [Step 33](#step-33) |

*The 27 in full, by family:*
| Family | Features |
|---|---|
| OLSR control-plane | `TcMessageRate`, `MidMessageRate`, `HnaMessageRate`, `AverageAdvertisedLinksPerTCMessage` |
| Delivery / loss | `PacketDeliveryRatio`, `PacketLossRatio`, `AvgFlowLossRate`, `FlowLossRateStd`, `RxTxPacketRatio` |
| Delay / jitter / latency | `AverageEndToEndDelay`, `AverageJitter`, `AvgFlowDelay`, `AvgFlowJitter`, `FlowDelayStd`, `FlowJitterStd` |
| Throughput / rate | `Throughput`, `DataPacketRate`, `AvgFlowThroughput`, `FlowThroughputStd` |
| Flow structure / size | `AverageHopCount`, `AvgFlowDuration`, `FlowDurationStd`, `AvgTxBytesPerFlow`, `AvgRxBytesPerFlow`, `AvgTxPacketsPerFlow`, `AvgRxPacketsPerFlow`, `AvgRxPacketSize` |

> `DataPacketRate` is stored as `MacDataPacketRate` in the extended-schema datasets; the
> runner detects which name is present and substitutes automatically.

### Results — Set A (99 features): AUROC on the held-out defense
| Regime · held-out | LogReg | RF | RF − 7 suspects |
|---|---|---|---|
| static · fpnt | 0.977 | 0.999 | 0.999 |
| static · trust | 0.699 | 0.938 | 0.954 |
| static · dcfm | **0.269** | 0.995 | 0.995 |
| static · watchdog | 0.944 | 1.000 | 0.999 |
| mobile · fpnt | 0.769 | 0.999 | 0.999 |
| mobile · trust | 0.661 | 0.982 | 0.982 |
| mobile · dcfm | 0.911 | 1.000 | 1.000 |
| mobile · watchdog | 0.726 | 0.999 | 0.999 |

Two findings already visible here. **(1)** Under LogReg, DCFM-static *inverts* to 0.27 (below
chance); under RF it is 0.995 — the signal is present but **non-linear**. **(2)** Removing the
seven normalisation-denominator suspects (`FlowCount`, `NormalizedRoutingLoad`,
`RoutingOverheadRatio`, `EphemeralAddressFraction`, `FracDegreeOneNodes`,
`ConnectedComponentsPerNode`, `AverageMprCount`) changes nothing (DCFM-static 0.9946 → 0.9946):
the result does **not** rest on them. A Spearman check on the rank matrix explained the
LogReg inversion: fpnt↔trust ranks correlate at **0.72**, but DCFM↔anything is **≈ 0.00** —
each defense ranks a *different* feature family (routing-churn, latency, graph-topology), so a
single linear boundary cannot fit all of them at once.

### Results — Set B (27 features): AUROC on the held-out defense
| Regime · held-out | LogReg | RF |
|---|---|---|
| static · fpnt | 0.811 | 0.995 |
| static · trust | 0.675 | 0.981 |
| static · dcfm | 0.670 | 0.984 |
| static · watchdog | 0.877 | 1.000 |
| mobile · fpnt | 0.705 | 0.998 |
| mobile · trust | 0.582 | 0.982 |
| mobile · dcfm | 0.717 | 0.999 |
| mobile · watchdog | 0.654 | 0.998 |

**How many features the intersection kept (k), per condition:**
| Regime · held-out | LogReg | RF |
|---|---|---|
| static · fpnt | 27 | 27 |
| static · trust | 27 | 15 |
| static · dcfm | 27 | 27 |
| static · watchdog | 27 | 27 |
| mobile · fpnt | 20 | 20 |
| mobile · trust | 20 | 20 |
| mobile · dcfm | 20 | 20 |
| mobile · watchdog | 20 | 15 |

A point worth stating plainly: in several folds the intersection kept **all 27 features** — no
filtering at all — and RF still reached ≈ 1.0. So the strong result is not an artefact of a
lucky small subset.

### Dominance — is one feature carrying it? (the supervisor's question)
**No.** Reproducing each fold's RF and reading its importances
({refml(S34 + "/dominance_check.py", "dominance_check.py")}):

| Set / regime | top-1 share | #features for 80% | leakage-suspect in top-12? |
|---|---|---|---|
| 27 / static | 13–15 % | 8–10 | none |
| 27 / mobile | 9–14 % | 10–12 | none |
| 99 / static | 6–12 % | 19–43 | none |
| 99 / mobile | 3–6 % | 55–60 | (two, at low weight) |

The importance is **diffuse**, and the leaders are **generic and behavioural**, recurring in
the top-5 of all four folds:

| Regime (27-set) | Generic leaders (top-5 in 4/4 folds) |
|---|---|
| static | `AvgFlowDelay` (0.13), `AvgFlowJitter` (0.13), `AverageJitter` (0.12), `AverageEndToEndDelay` (0.12) |
| mobile | `AverageAdvertisedLinksPerTCMessage` (0.12), `AverageHopCount` (0.10), `AverageEndToEndDelay` (0.08) |

The signature also **re-shapes** between regimes (delay/jitter in static; control-plane and
hop-count in mobile) — consistent with a genuine physical signature that manifests
differently under mobility, not a fixed leaking column.

### The permutation null — [VERIFIED]
Shuffling `defense_enabled` within each run and re-running (`--permute-labels`) collapsed
**all eight** conditions to ≈ 0.50 (range 0.486–0.535), attack-split included. Had any
structural shortcut remained — a leaking column, a per-defense source fingerprint — it would
have kept some fold above chance under a shuffled label. None did. The real 0.98–1.00
therefore comes only from the true feature→defense relationship.

### What we concluded
1. **A generic, cross-defense "a defense is running" signature exists and generalises to an
   unseen defense** — with RF, every condition reaches 0.94–1.00, held-out defense included.
   This **overturns the [Step 35](#step-35) [HYPOTHESIS]** that TRUST/WATCHDOG would sit at
   chance.
2. **The signature is non-linear.** LogReg is weak and uneven (0.58–0.88); RF is strong. The
   same lesson recorded for mobility in [Step 29](#step-29) — a linear model is the wrong
   instrument for this signal, not evidence of its absence.
3. **It is not leakage.** It survives dropping the seven denominator suspects; no single
   feature dominates; no suspect ranks in the top-12; and the permutation null is ≈ 0.50.
4. **LODO beats in-condition.** TRUST-mobile scored ≈ 0.70 when trained on TRUST alone
   ([Step 29](#step-29) baseline) but ≈ 0.98 when trained on the other three and tested on
   TRUST. Pooling three defenses learns the shared fingerprint better than one weak defense in
   isolation — the inversion is the finding, not a bug.
5. **What is detected is *presence*, not efficacy.** The no-attack side of the attack-split is
   the more separable one — the model sees the mechanism *running and perturbing the traffic*,
   not "the attacker was defeated." This is the honest scope of the claim.

### Status and the standing caveat
[VERIFIED]: all runs completed; the permutation null is ≈ 0.50; the 7-suspect removal leaves
every cell unchanged. Caveat [HYPOTHESIS]: DCFM shares the pipeline and simulator with the
others (confirmed with the owner), which weakens — but does not fully close — the possibility
that a share of its near-perfect scores reflects distribution shift rather than defence
signature; the cross-generalisation among FPNT/TRUST/WATCHDOG stands independently of it. See
[Open Questions](#open-questions).

### Sources
- Instrument: {refml(S34 + "/step34_lodo.py", "step34_lodo.py")},
  {refml(S34 + "/step34_common.py", "step34_common.py")},
  {refml(S34 + "/summarize_step34.py", "summarize_step34.py")},
  {refml(S34 + "/dominance_check.py", "dominance_check.py")},
  {refml(S34 + "/run_step34_on_27.sh", "run_step34_on_27.sh")}
- Feature sets: {refml(S34 + "/features/features_99_surviving.txt", "features_99_surviving.txt")},
  {refml(S34 + "/features/features_27_step32.txt", "features_27_step32.txt")}
- Configs: {refml(S34 + "/step34_config.json", "step34_config.json")} (+ `step34_config_27.json`,
  `step34_config_27_rf.json`, generated by the runner)
- Outputs (each with `summary/step34_results.csv`): `{S34}/results_rf/` (99·RF),
  `results_rf_noleak/` (99·RF−7), `results_27/` (27·LogReg), `results_27_rf/` (27·RF),
  `results_27_rf_PERMUTED/` (permutation null)
- Antecedents: [Step 35](#step-35) (the LODO instrument and its pre-registered predictions);
  [Step 33](#step-33) (origin of the 27-set)
""")

md(f"""
<a id="step-37" name="step-37"></a>
## Step 37 — The normalisation hypothesis measured: DCFM re-run on un-normalised features
**Date:** 2026-08-05

### Motivation — the one test that had only ever been argued, never measured
[Step 32](#step-32) proposed that DCFM's apparent detectability was an artefact of the
**denominators** used to normalise the feature schema rather than of the defense's behaviour,
and it named the experiment that would settle it: measure whether the denominator itself
predicts the label. [Step 34](#step-34) confirmed the mechanism **from source** — reading the
emitter and identifying `nObs`, `graphN` and the all-packet size mean as the contaminated
divisors — and marked the finding `[VERIFIED from source]`. What never existed was the
counterfactual: the same simulations, emitted **without** normalisation.

It exists now. On 2026-08-04 the DCFM dataset was regenerated with an un-normalised emitter
under the same `runner.config` — identical `START_SEED=1`, identical `EXTRA_ARGS`, same ns-3
tree, same scratch program; only `OUT_DIR` and `DATE_STARTED` differ. This step runs the two
feature sets whose normalised results are already published — the 27-set of
[Step 33](#step-33) and the 32-set of [Step 29](#step-29) — on the raw data, changing nothing
else.

### The two schema headers, diffed
`olsr_window_features.h` (raw) and its schema-v5 normalised counterpart (the `NORM-001`
block) were compared column by column. Of the 95 Core columns **77 are renamed** and 18 keep
their names — and those 18 are exactly the ones that were already scale-free by construction
(`PerNodeTcBytesGini`, `TcBurstinessHurst`, `AdvertisedGraphDensity`, the Skew/Kurtosis and
`TcInterArrival*` families). The 33-column V2 group keeps **every** name and changes only
values; the header says so explicitly and warns never to mix pre- and post-v5 rows.

**This is why only this experiment is clean.** `--feature-set all` is not comparable across
the two datasets, because 77 of its columns changed identity. The 32-metric block is
comparable, because its names are stable. [VERIFIED from source]

Two mechanisms fall straight out of the emission code:

- `FlowCount_normalised = m_dataSentByFlow.size() / nObs`. The raw numerator takes **three
  values** (mean 2.995 of a maximum 3 — almost always exactly 3), so the normalised feature is
  very nearly **3 / nObs**: a direct readout of the observed address count. Arithmetic check:
  3/54 = 0.0556 against a measured normalised mean of 0.0554. `FlowCount` carried **94.8% of
  the importance in DCFM/static** in [Step 29](#step-29). [VERIFIED from source]
- **Eight of the 27** still divide by a defense-sensitive observed quantity: `TcMessageRate`,
  `MidMessageRate` and `HnaMessageRate` by `nObs`; `AverageAdvertisedLinksPerTCMessage` by
  `nObs-1`; `AverageEndToEndDelay` and `AvgFlowDelay` by the observed mean hop count;
  `AverageHopCount` by the observed diameter; `DataPacketRate` by the observed flow count.
  Dropping `AverageMprCount` and `NormalizedRoutingLoad` in [Step 33](#step-33) removed two
  carriers and left these — which is why the 27-set sat at 0.87 rather than at chance.
  [VERIFIED from source]

### Validity gate — four checks, all run before any learning
A pre-flight script established that the comparison isolates normalisation:

| Check | Result |
|---|---|
| **Control group** — four of the 27 divide by the window duration, a genuine constant, so their univariate AUC *cannot* change | max abs delta = **0.0000**, both regimes |
| **Pairing** — do the two datasets hold the same simulations? | `run_id` overlap **2003 / 2004** static, **2004 / 2004** mobile |
| **Baseline provenance** — do the located baselines reproduce the published numbers? | `baseline_drift = 0.0` on all four |
| **Pipeline parity** — seed, folds, repeats, composites, transforms | every knob matches |

The control group is the strongest of the four: it is a built-in falsification test. Had those
four features moved at all, every other delta would have been confounded by re-simulation.
They did not move. This is precisely the assurance that [Step 36](#step-36)'s standing
provenance caveat could not provide for its own comparison. [VERIFIED]

### Result 1 — the denominator alone identifies the defense
Because the datasets are paired, the per-window ratio raw ÷ normalised **recovers the
denominator exactly**. Two independent routes:

| Route | static | mobile |
|---|---|---|
| Direct: the raw `NumDistinctAddressesInTcAdvertisements` counter | **0.9469** | **0.9863** |
| Recovered: per-window raw ÷ normalised ratio | 0.9468 | 0.9871 |

They agree to three decimals. The recovered denominator is **53.8** (static) and **57.8**
(mobile) — **not 50**. That excess of roughly four to eight addresses is the `main+65536`
injection, measured rather than inferred.

**`AUC(denominator, defense_enabled)` ≈ 0.95–0.99, with no model at all.** This answers
question 1 of [Step 32](#step-32). Any feature divided by this quantity inherits an almost
complete defense signal regardless of what its numerator measures. [VERIFIED]

### Result 2 — `FlowCount`, the project's largest single leak, dissolved
Univariate AUC of the individual carriers, raw versus normalised (static / mobile):

| Feature | raw | normalised |
|---|---|---|
| `FlowCount` | **0.501 / 0.517** | 0.942 / 0.954 |
| `AvgTxPacketSize` | constant 540.0, `nunique=1` / 0.500 | 0.663 / 0.616 |

`FlowCount` raw is **at chance**. It contains no behavioural information whatever. Normalised,
it reaches 0.942 — and that figure is almost exactly the denominator's own 0.947, because the
normalised feature *is* the denominator. The classifier that assigned it 94.8% of its
importance in [Step 29](#step-29) was counting DCFM's fictitious addresses. [VERIFIED]

### Result 3 — normalisation also destroyed real signal
This direction was not anticipated:

| Feature | regime | raw | normalised |
|---|---|---|---|
| `TcMessageRate` | mobile | **0.910** | 0.720 |
| `TcMessageRate` | static | **0.812** | 0.680 |
| `AverageAdvertisedLinksPerTCMessage` | static | **0.777** | 0.520 |
| `AverageHopCount` | mobile | **0.763** | 0.699 |
| `AverageEndToEndDelay` | mobile | **0.593** | 0.501 |

DCFM floods TC because it injects nodes — a real behavioural signature. But `nObs` rises for
exactly the same reason, so dividing one by the other **cancels the signal**. Normalisation
was harmful in both directions at once: it promoted content-free features into detectors *and*
suppressed the features that carry the mechanism. [VERIFIED]

### Result 4 — the two sets move in opposite directions, and that is the point
**`HistGB` fixed on both sides**, so the model is not a second variable. Every pipeline knob
pinned to the published baseline (seed 42, 2×5 grouped CV, composites on, transforms on, no
`--n-features` cap):

| set | regime | normalised | raw | delta AUC | TPR@1%FPR |
|---|---|---|---|---|---|
| **27** | static | 0.8660 | **0.9415** | **+0.076** | 0.258 → **0.691** |
| **27** | mobile | 0.9341 | **0.9711** | **+0.037** | 0.514 → **0.789** |
| **32** | static | 0.9864 | 0.9591 | −0.027 | 0.921 → 0.750 |
| **32** | mobile | 0.9983 | 0.9760 | −0.022 | 0.984 → 0.814 |

The opposite signs are structural, not contradictory: **the 32-set contains the leak and the
27-set does not.** Stripping the representation costs the 32-set the leak it was partly running
on, while the 27-set has no leak to lose and instead recovers the behavioural signal that
division by `nObs` had been cancelling. Each set experiences only one of the two effects.

Note the operating point on the 27-set: at 1% false-alarm rate, detection goes from **a quarter
of windows to two thirds** — a change far larger than the 0.076 of AUC suggests. This is
[Step 29](#step-29)'s central lesson running in reverse: the ranking metric masks, the
operating point exposes.

### Result 5 — the ablation ladder was largely measuring normalisation

| | 27 | 32 | gap |
|---|---|---|---|
| normalised | 0.866 | 0.986 | **0.120** |
| raw | 0.942 | 0.959 | **0.018** |

In the normalised representation the five removed features appear to carry 0.12 of AUC. Raw,
they carry 0.018 — a seventh as much. **The single-feature and pair ablations of
[Step 30](#step-30) through [Step 33](#step-33) were, to a large extent, mapping the
normalisation scheme rather than a hierarchy of behavioural features.** This does not
invalidate that work; it explains why it felt like whack-a-mole, exactly as
[Step 34](#step-34) argued — now with a measurement instead of an inference. [VERIFIED]

### Result 6 — what the model learns instead
`TDR` — the composite `TcMessageRate × AverageAdvertisedLinksPerTCMessage`, the rate of
topology dissemination — is the **top-ranked feature in all four raw runs**. The full leading
groups:

| run | top features |
|---|---|
| 27 · static | `TDR`, `AverageAdvertisedLinksPerTCMessage`, `Delay_Per_Hop` |
| 27 · mobile | `TDR`, `AverageAdvertisedLinksPerTCMessage`, `DataPacketRate` |
| 32 · static | `TDR`, `AverageMprCount`, `AverageAdvertisedLinksPerTCMessage` |
| 32 · mobile | `TDR`, `AverageAdvertisedLinksPerTCMessage`, `AverageMprCount` |

That is DCFM's mechanism read directly: fictitious-node injection floods the control plane
with TC traffic. The model now measures the flooding rather than counting addresses through a
denominator — which corroborates, in a clean representation, what [Step 22](#step-22) claimed
for Campaign 1. Importance is also less concentrated than in [Step 29](#step-29): `TDR` takes
0.176 on 27·static against `FlowCount`'s 94.8% there, with the remainder spread thinly
(`AverageMprCount` 0.049, `AverageAdvertisedLinksPerTCMessage` 0.044).

### Controls
- **Permutation null.** 100 grouped permutations on 27·static, labels shuffled within each run
  so the 2-ON/2-OFF balance is preserved: observed **0.9466**, null **0.4989 ± 0.0072**,
  p = 0.0099 (the floor for 100 permutations). The observed value sits 62 null standard
  deviations away. Because this step *reverses* a pre-registered expectation, the null matters
  more than usual: the pipeline finds nothing once the labels are destroyed. It also matches
  [Step 36](#step-36)'s null band (0.486–0.535). [VERIFIED]
- **Anchor stability.** The comparison was repeated with **nine** fixed models (HistGB,
  LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, Ridge, Stacking, AdaBoost) — 36
  comparisons. **The sign agrees in 36 of 36**: +0.075…+0.101 and +0.036…+0.042 for the 27-set,
  −0.021…−0.043 and −0.021…−0.036 for the 32-set, and likewise for all 36 TPR@1% deltas. The
  conclusion does not depend on the choice of anchor. [VERIFIED]
- **Linearity.** `Ridge` shows the **largest gain anywhere in the anchor table** (+0.1009 on
  27·static, 0.8140 → 0.9148). A *linear* model improves most, so normalisation had also made
  the problem less linearly separable — dividing by a quantity that moves with the label bends
  a boundary that was close to flat. Conversely `ExtraTrees` and `RandomForest` lose most on
  the 32-set (−0.043, −0.038) against `LightGBM`'s −0.021: the non-boosted forests leaned
  hardest on the single dominant leak feature and suffer most when it evaporates.
  [HYPOTHESIS — consistent with the importance tables, not tested directly]
- **Non-linearity, again.** `SVM_rbf` reaches only 0.5567 and `LogisticRegression`
  0.8065 ± 0.131 on 27·static, against 0.9415 for `HistGB`. [Step 29](#step-29)'s finding
  holds in the raw representation too.

### Three schema findings recorded along the way
1. **`AvgTxPacketSize` is 540, not 512** — 512 B of UDP payload plus 28 B of UDP/IP headers.
   The normalisation table quotes 512. Corrected at [Step 32](#step-32).
2. **`NormalizedRoutingLoad` was never normalised**, despite its name. It is byte-identical in
   both datasets (177.5476 vs 177.5827 static; 315.7069 vs 315.7069 mobile), as is
   `RoutingOverheadRatio` (scale-free by construction). Its signal is therefore behavioural
   (univariate AUC 0.728 static / 0.813 mobile). **Only two of the five features removed to
   build the 27-set were normalisation artefacts** — the other three may have carried real
   signal that was discarded. Recorded in *Open Questions*.
3. **A schema collision in both headers.** The emitter writes `L_pdr` to both
   `PacketDeliveryRatio` and `RxTxPacketRatio`, and `1 - L_pdr` to `PacketLossRatio` — three
   columns, one quantity. Separately, `MidMessageRate` and `HnaMessageRate` are exactly zero in
   all 16 028 windows of both datasets. `FeatureSelector` removes all five in-fold, so nothing
   is invalidated, but **the 27-set is effectively 23 features**. Footnoted at
   [Step 33](#step-33) and [Step 36](#step-36).

### Conclusion
DCFM is genuinely detectable, but the published figures were measured through the wrong lens.
Normalising by an observed quantity that the defense itself perturbs is **doubly harmful**: it
promotes content-free features into detectors, and it cancels the signal in features that do
carry the mechanism. In the raw representation DCFM is detected at **0.94–0.98 on the basis of
control-plane flooding**, and **27 leak-free behavioural features suffice** — the clean set
sought since [Step 30](#step-30) was present all along, hidden by the representation.

The generalisable caution, well beyond OLSR: **before normalising a feature, ask whether its
denominator is influenced by the phenomenon being detected.** A denominator that is is not a
scaling choice but a label leak.

### What this does not establish
This is **DCFM only**. No un-normalised dataset exists for FPNT, TRUST or WATCHDOG, so no
general claim about normalisation is licensed yet. FPNT is the informative next case, because
its contaminated divisor is the mean all-packet size rather than `nObs`. And the result says
nothing about [Step 36](#step-36)'s cross-defense generalisation, which ran entirely on
normalised data — whether a *generic* defense signature survives de-normalisation is now the
sharpest open question in the project. Both are carried into *Open Questions*.

### Sources
- Schema headers (NS-3): {ref("scratch/olsr_window_features.h")} (raw) and its schema-v5
  `NORM-001` counterpart
- Pipeline: {refml("defense_detection_v4.py")} — **unmodified**; this step adds no v4 change
- Pre-flight and denominator diagnostics:
  {refml(S35 + "/preflight.py", "preflight.py")}
- Runner (activates the `defense` conda env itself, and aborts if LightGBM / XGBoost /
  CatBoost are missing, so a 9-model zoo can never be compared against a 12-model baseline):
  {refml(S35 + "/run_dcfm_nonorm.sh", "run_dcfm_nonorm.sh")}
- Comparison (best-model and fixed-anchor tables, plus a config-parity check):
  {refml(S35 + "/compare_normalized_vs_raw.py", "compare_normalized_vs_raw.py")}
- Feature sets, in `METRICS` order so that `--features-file` reproduces the Step-33
  `--drop-features` selection exactly — order matters, because `FeatureSelector`'s correlation
  pruning is a deterministic greedy keep-first over column order:
  {refml(S35 + "/features_27.txt", "features_27.txt")},
  {refml(S35 + "/features_32.txt", "features_32.txt")}
- Source-derived name and denominator map, all 128 columns:
  {refml(S35 + "/feature_name_map.csv", "feature_name_map.csv")}
- Dataset: `Dcfm_All_128_features_no_normalization/` — `static/` and `mobile/`, 2004 runs and
  8016 windows each, generated 2026-08-04 (2.17 h and 2.76 h wall clock, yields 0.348 and 0.241)
- Outputs: `{S35}/results_run_nonorm_27/` and `{S35}/results_run_nonorm_32/` (each with
  `dcfm_static/` and `dcfm_mobile/`, holding `summary.csv`, `folds.csv`, `importance.csv`,
  `run_config.json`, `summary.tex`, `permutation_test.json` for 27·static, `final_model.pkl`,
  `figures/`); `{S35}/preflight_report/` (`scale_comparison_*.csv`,
  `univariate_by_class_*.csv`, `denominator_direct_*.csv`, `denominator_recovered_*.csv`,
  `pairing_*.json`); `{S35}/comparison/` (`comparison_best_model.csv`,
  `comparison_fixed_HistGB.csv`, `config_parity.csv`)
- Antecedents: [Step 32](#step-32) (the hypothesis and the test it specified);
  [Step 34](#step-34) (the source-level confirmation); [Step 33](#step-33) (the 27-set and its
  0.8703 / 0.9364 baseline); [Step 29](#step-29) (the 32-set and its 0.9874 / 0.9983 baseline)

> **Folder numbering.** The script folder is `step_35_dcfm_non_normalized`, following the
> repository's own counter, which runs two behind the notebook — the same offset that makes
> `Step_34_Cross_Defense_Intersection` the home of notebook [Step 36](#step-36).
""")

md(f"""
<a id="step-38" name="step-38"></a>
## Step 38 — Traffic load as a variable: DCFM un-normalised with one CBR flow
**Date:** 2026-08-09

### Motivation — the representation was fixed, so the simulation could be varied
[Step 37](#step-37) removed normalisation as a confound and showed DCFM is detected at
0.94–0.98 from control-plane flooding alone. It left one question sharply open: **how much of
that signal depends on the data plane?** Every dataset in the project had carried **three CBR
flows of 18 messages**. If DCFM's signature really is fictitious-node injection flooding TC,
then reducing the data traffic should leave it largely untouched — and should erode any
feature that measures the data plane.

A new dataset was generated for exactly that test: un-normalised, identical schema, **one CBR
flow of 18 messages** instead of three. This step runs three feature sets on it — the full 33
V2 group, the 32-metric schema of [Step 29](#step-29), and the 27-set of
[Step 33](#step-33) — against the three-flow results of [Step 37](#step-37).

**Read the comparison with care.** Unlike [Step 37](#step-37), the two datasets differ in the
*simulation*, not only in the emitter, so there is **no control group of the Step-37 kind**
here: no quantity is constant by construction across both. Every 1-flow-vs-3-flow delta is
therefore confounded with traffic load. Findings *within* the new dataset are `[VERIFIED]`;
causal statements of the form "removing two flows caused X" are `[HYPOTHESIS]`.

### The three feature sets
All three are declared through `--features-file` in `METRICS` order — never through the
preset — so the three runs differ **only** in the feature set:

| Set | Definition |
|---|---|
| **33** | The complete V2 / `strict_observable_v2` parity group, i.e. `V2FeatureCsvHeader()` |
| **32** | The 33 minus `RoutingOverheadBytesRatio` — identical to v4's `METRICS` and to the `metrics32` preset |
| **27** | The 32 minus five carriers: `AvgTxPacketSize`, `FlowCount`, `RoutingOverheadRatio` (the drop3 leak carriers of [Step 30](#step-30)) and `AverageMprCount`, `NormalizedRoutingLoad` (the DCFM denominator cluster of [Step 32](#step-32) / [Step 33](#step-33)) |

The `DataPacketRate` → `MacDataPacketRate` alias trap of [Step 37](#step-37) does **not** apply
here: `MacDataPacketRate` appears in neither schema header, so the explicit name resolves
directly. A different collision does: the raw `CoreAndV2` header emits **`DataPacketRate`
twice** — once in Core group A, once in V2 — both from the same variable
(`m_dataPackets / dur`). `pandas` mangles the second to `DataPacketRate.1`; the two columns
were verified numerically identical. The raw CSV therefore has **133 columns, not 128**.
[VERIFIED]

### Pre-flight — seven checks before any learning
| Check | Result |
|---|---|
| **Schema parity** — the 33 V2 names present and in identical order in all four conditions | pass; `DataPacketRate.1` byte-identical to its Core twin |
| **Integrity** — 4 windows per run, single window length, four scenarios | pass; 8016 windows / 2004 runs (static), 8012 / 2003 (mobile), all 40 s |
| **Class balance** | exactly 4008 / 4008 and 4006 / 4006 |
| **Emitter identities** — `PacketDeliveryRatio` ≡ `RxTxPacketRatio`, PDR + PLR ≡ 1 | hold in all four conditions |
| **Baseline provenance** — do the [Step 37](#step-37) trees reproduce their published figures? | yes, max drift **0.000038** across all four |
| **Pipeline parity** — seed 42, 2×5 folds, `cal_fraction` 0.2, composites and transforms on, no `--n-features` cap | every knob matches |
| **`run_id` pairing** | overlap 2004/2004 static, 1997/2003 mobile — **provenance only, not a counterfactual** |

The baseline check is what makes the comparison quantitative rather than quoted: the
[Step 37](#step-37) numbers were **re-measured on this machine**, not copied from the report.
[VERIFIED]

### Result 1 — the data plane collapses, and takes nine features with it
`FlowCount` is `m_dataSentByFlow.size()`. With one flow it falls from a mean of 2.995 to
**0.998**, and the five per-flow `Std` features become `Std()` over a single-element vector —
identically zero:

| | 3 flows | 1 flow |
|---|---|---|
| `FlowCount` mean | 2.995 | **0.998** |
| Constant features among the 33 | 2 (static: 3) | **7** |
| `AvgTxPacketSize` | constant 540.0 in static | `nunique=2`, \\|r\\| = 1.000 with `FlowCount` |

`AvgTxPacketSize` deserves note: in [Step 37](#step-37) it was constant in static and therefore
harmless. Here it takes two values and is perfectly collinear with `FlowCount` — both are
indicators of *whether any data was sent at all* in the window (13 empty windows static, 235
mobile). **[Step 37](#step-37)'s "constant, therefore auto-removed" argument does not carry
over.** [VERIFIED]

Effective dimension, measured two ways — on the base features, and on the engineered matrix
using v4's own `engineer_features()` and `FeatureSelector`:

| Set | base: 3f static / mobile | base: 1f static / mobile | engineered: 3f static / mobile | engineered: 1f static / mobile |
|---|---|---|---|---|
| 33 | 21 / 25 | **12 / 14** | 58 / 75 | **32 / 46** |
| 32 | 21 / 25 | **12 / 14** | 58 / 75 | **32 / 46** |
| 27 | 17 / 20 | **8 / 10** | 50 / 66 | **26 / 37** |

The engineered figures are the ones that matter — squares, cubes and ratios of a pruned base
feature can survive selection — and `run_config.json` does **not** report them (it logs
`n_total_features`, i.e. before selection). Roughly **half** the usable dimension is gone.
[VERIFIED]

### Result 2 — sets 33 and 32 are bit-identical, in every model and every metric
`RoutingOverheadBytesRatio` is pruned against `RoutingOverheadRatio` at \\|r\\| = 0.966–0.987 in
**all four** (dataset × regime) cells. The consequence is total: across 13 models × 2 regimes ×
6 metrics, sets 33 and 32 agree **to four decimals in every cell**. The single exception is
`LightGBM`/mobile, which differs by 4.1 × 10⁻⁶ in ROC-AUC — a thousandth of the fold standard
deviation, and invisible in all five threshold-dependent metrics, so the matrices reaching the
models are identical and the difference is floating-point summation order.

**This makes the 33-run a control rather than a third experiment.** It confirms in practice
the reasoning by which v4 excluded `RoutingOverheadBytesRatio` from `METRICS` a priori. Had the
two disagreed materially, it would have signalled a bug — not a finding. [VERIFIED]

Worth recording separately: `RoutingOverheadBytesRatio` is the **strongest univariate feature
in three of the four conditions** (0.844 / 0.903 one-flow, 0.846 / 0.930 three-flow), ahead of
`TcMessageRate` in static. It is discarded only because of its correlation with
`RoutingOverheadRatio`. Carried to *Open Questions*.

### Result 3 — all 13 models, one flow, static
`balanced_accuracy` at the in-fold threshold; `T1%` / `T5%` = TPR at 1% / 5% FPR. Sets 33 and
32 are one column because they are identical.

| Model | AUC 33/32 | Acc 33/32 | MCC 33/32 | T1% 33/32 | AUC 27 | Acc 27 | MCC 27 | T1% 27 |
|---|---|---|---|---|---|---|---|---|
| LightGBM | **0.9818** | **0.9400** | **0.8804** | 0.8468 | 0.9689 | 0.9104 | 0.8221 | **0.7226** |
| Stacking | 0.9799 | 0.9389 | 0.8785 | 0.8226 | 0.9693 | 0.9154 | 0.8317 | 0.6901 |
| CatBoost | 0.9786 | 0.9343 | 0.8689 | 0.8296 | **0.9697** | **0.9169** | **0.8344** | 0.6961 |
| HistGB | 0.9780 | 0.9329 | 0.8672 | 0.8228 | 0.9677 | 0.9140 | 0.8292 | 0.7094 |
| XGBoost | 0.9779 | 0.9316 | 0.8639 | **0.8326** | 0.9670 | 0.9110 | 0.8234 | 0.6881 |
| AdaBoost | 0.9746 | 0.9238 | 0.8488 | 0.7621 | 0.9651 | 0.9111 | 0.8227 | 0.6553 |
| RandomForest | 0.9709 | 0.9165 | 0.8336 | 0.7345 | 0.9678 | 0.9128 | 0.8270 | 0.6862 |
| ExtraTrees | 0.9660 | 0.9031 | 0.8081 | 0.6785 | 0.9688 | 0.9164 | 0.8335 | 0.6957 |
| MLP | 0.9510 | 0.8941 | 0.7907 | 0.7008 | 0.9472 | 0.8938 | 0.7904 | 0.5770 |
| Ridge | 0.9177 | 0.8633 | 0.7395 | 0.6962 | 0.8799 | 0.8241 | 0.6613 | 0.5659 |
| LogisticRegression | 0.8569 | 0.8119 | 0.6324 | 0.5826 | 0.9029 | 0.8392 | 0.6842 | 0.6014 |
| SVM_rbf | 0.6234 | 0.5912 | 0.1832 | 0.0587 | 0.6520 | 0.6162 | 0.2345 | 0.1404 |
| Dummy | 0.5000 | 0.5000 | 0.0000 | 0.0100 | 0.5000 | 0.5000 | 0.0000 | 0.0100 |

### Result 4 — all 13 models, one flow, mobile

| Model | AUC 33/32 | Acc 33/32 | MCC 33/32 | T1% 33/32 | AUC 27 | Acc 27 | MCC 27 | T1% 27 |
|---|---|---|---|---|---|---|---|---|
| CatBoost | **0.9705** | **0.9179** | **0.8375** | **0.8082** | **0.9642** | **0.9024** | **0.8086** | **0.7739** |
| XGBoost | 0.9686 | 0.9153 | 0.8337 | 0.8050 | 0.9624 | 0.8995 | 0.8028 | 0.7690 |
| Stacking | 0.9686 | 0.9160 | 0.8335 | 0.7775 | 0.9629 | 0.8983 | 0.7990 | 0.7639 |
| HistGB | 0.9681 | 0.9165 | 0.8353 | 0.7969 | 0.9617 | 0.9005 | 0.8036 | 0.7726 |
| LightGBM | 0.9675 | 0.9128 | 0.8276 | 0.7946 | 0.9606 | 0.9003 | 0.8044 | 0.7654 |
| AdaBoost | 0.9652 | 0.9091 | 0.8206 | 0.7457 | 0.9613 | 0.8990 | 0.7999 | 0.7494 |
| RandomForest | 0.9616 | 0.8973 | 0.7956 | 0.7454 | 0.9610 | 0.8939 | 0.7906 | 0.7489 |
| ExtraTrees | 0.9517 | 0.8821 | 0.7665 | 0.7136 | 0.9546 | 0.8865 | 0.7756 | 0.7226 |
| Ridge | 0.9413 | 0.8864 | 0.7759 | 0.6967 | 0.9407 | 0.8883 | 0.7786 | 0.6903 |
| MLP | 0.8326 | 0.8094 | 0.6306 | **0.1981** | 0.8273 | 0.8074 | 0.6205 | **0.0752** |
| LogisticRegression | 0.6702 | 0.6317 | 0.2892 | 0.0428 | 0.6578 | 0.6318 | 0.2849 | 0.0324 |
| SVM_rbf | 0.6187 | 0.5869 | 0.1762 | 0.0271 | 0.6137 | 0.5841 | 0.1717 | 0.0141 |
| Dummy | 0.5000 | 0.5000 | 0.0000 | 0.0100 | 0.5000 | 0.5000 | 0.0000 | 0.0100 |

Two readings of these tables:

- **The eight tree models are separated by 0.005 (static) to 0.010 (mobile) of AUC** — inside
  the fold standard deviation. The "winner" is not a meaningful ranking, which is why the
  cross-step comparison below is anchored on a fixed model.
- **`MLP`/mobile is the cautionary case.** AUC 0.833 looks serviceable; `T1%` is **0.198** on
  the 32-set and **0.075** on the 27-set. A model that misses more than nine in ten attacks at
  a 1% false-alarm budget. Its ranking is fair, its probabilities are not calibrated in the
  tail. [VERIFIED]

### Result 5 — the set gap lives in the tail, not in AUC
Fixed `HistGB` anchor, one flow:

| Regime | metric | 33/32 | 27 | gap |
|---|---|---|---|---|
| static | ROC-AUC | 0.9780 | 0.9677 | 0.010 |
| static | balanced acc | 0.9329 | 0.9140 | 0.019 |
| static | MCC | 0.8672 | 0.8292 | 0.038 |
| static | **TPR@1%FPR** | **0.8228** | **0.7094** | **0.114** |
| mobile | ROC-AUC | 0.9681 | 0.9617 | 0.006 |
| mobile | **TPR@1%FPR** | **0.7969** | **0.7726** | **0.024** |

One point of AUC, eleven of TPR — in static. In mobile the tail gap is only 0.024. So the five
features removed to build the 27-set buy something real, but **only in static and only at a low
false-alarm budget**. This is [Step 29](#step-29)'s lesson and [Step 37](#step-37)'s Result 4
recurring a third time: the ranking metric masks, the operating point exposes. [VERIFIED]

### Result 6 — against three flows: static improves, mobile regresses
Fixed `HistGB` anchor, three flows ([Step 37](#step-37)) versus one flow:

| set | regime | ROC-AUC 3f → 1f | MCC 3f → 1f | fold std 3f → 1f |
|---|---|---|---|---|
| **27** | static | 0.9415 → **0.9677** (**+0.026**) | 0.7402 → 0.8292 (**+0.089**) | 0.0094 → **0.0040** |
| **27** | mobile | 0.9711 → 0.9617 (−0.009) | 0.8313 → 0.8036 (−0.028) | 0.0035 → 0.0030 |
| **32** | static | 0.9591 → **0.9780** (**+0.019**) | 0.7971 → 0.8672 (**+0.070**) | 0.0080 → **0.0028** |
| **32** | mobile | 0.9760 → 0.9681 (−0.008) | 0.8462 → 0.8353 (−0.011) | 0.0027 → 0.0041 |

In balanced accuracy the same pattern is larger, and it holds across the **whole** model zoo:

| set | regime | mean delta, 13 models | range |
|---|---|---|---|
| **27** | static | **+0.037** | −0.026 … +0.082 |
| **32** | static | **+0.028** | −0.020 … +0.073 |
| **27** | mobile | −0.016 | −0.024 … 0.000 |
| **32** | mobile | −0.016 | −0.063 … 0.000 |

**In mobile all twelve non-`Dummy` models decline, without exception.** In static every model
except `Ridge` improves. Per-model accuracy, three flows → one flow:

| Model | 27 static | 27 mobile | 32 static | 32 mobile |
|---|---|---|---|---|
| CatBoost | 0.8720 → 0.9169 | 0.9178 → 0.9024 | 0.9027 → 0.9343 | 0.9258 → 0.9179 |
| HistGB | 0.8688 → 0.9140 | 0.9148 → 0.9005 | 0.8965 → 0.9329 | 0.9227 → 0.9165 |
| LightGBM | 0.8706 → 0.9104 | 0.9142 → 0.9003 | 0.9046 → 0.9400 | 0.9232 → 0.9128 |
| XGBoost | 0.8694 → 0.9110 | 0.9141 → 0.8995 | 0.8973 → 0.9316 | 0.9261 → 0.9153 |
| Stacking | 0.8745 → 0.9154 | 0.9168 → 0.8983 | 0.9018 → 0.9389 | 0.9235 → 0.9160 |
| AdaBoost | 0.8685 → 0.9111 | 0.9107 → 0.8990 | 0.8865 → 0.9238 | 0.9192 → 0.9091 |
| RandomForest | 0.8744 → 0.9128 | 0.9114 → 0.8939 | 0.8791 → 0.9165 | 0.9139 → 0.8973 |
| ExtraTrees | 0.8729 → 0.9164 | 0.9097 → 0.8865 | 0.8744 → 0.9031 | 0.9087 → 0.8821 |
| MLP | 0.8628 → 0.8938 | 0.8186 → 0.8074 | 0.8915 → 0.8941 | 0.8187 → 0.8094 |
| Ridge | 0.8497 → **0.8241** | 0.9089 → 0.8883 | 0.8835 → **0.8633** | 0.9088 → 0.8864 |
| LogisticRegression | 0.7576 → 0.8392 | 0.6563 → 0.6318 | 0.7384 → 0.8119 | 0.6944 → 0.6317 |
| SVM_rbf | 0.5574 → 0.6162 | 0.6063 → 0.5841 | 0.5574 → 0.5912 | 0.6046 → 0.5869 |

### Result 7 — why the two regimes move in opposite directions
The pre-flight univariate table explains both signs with **one** mechanism. The five per-flow
`Std` features that went to zero were **noise in static and signal in mobile**:

| Feature | 3f static | 3f mobile | 1f (both) |
|---|---|---|---|
| `FlowJitterStd` | 0.5047 | **0.5889** | 0.5000 |
| `FlowDelayStd` | 0.5005 | **0.5583** | 0.5000 |
| `FlowLossRateStd` | 0.5065 | **0.5783** | 0.5000 |
| `FlowThroughputStd` | 0.5010 | 0.5079 | 0.5000 |
| `FlowDurationStd` | 0.5006 | 0.5012 | 0.5000 |

Under mobility, variance *between* concurrent flows is itself informative — link breakage hits
flows unequally. In a static topology it is close to pure noise. Removing it therefore cleaned
the static problem and impoverished the mobile one. The **collapse of fold variance in static**
(0.0094 → 0.0040 and 0.0080 → 0.0028, a factor of 2–3) is independent corroboration: fewer
noise dimensions, a more stable model. [VERIFIED as observation]
[HYPOTHESIS as causal claim — traffic load is confounded with the change]

`Ridge` is the one model that declines in static as well (−0.026 and −0.020). Consistent with
this reading: L2 shrinks all coefficients rather than selecting, so a linear model benefits from
extra weak dimensions and is hurt when they vanish — the same behaviour that makes it *stronger*
on the 32-set than the 27-set in static (0.8633 vs 0.8241). [HYPOTHESIS]

### Result 8 — the control-plane signature is invariant to data-plane load
This is the answer to the question the step was built to ask. Univariate AUC, three flows
versus one:

| Feature | 3f static | 1f static | 3f mobile | 1f mobile |
|---|---|---|---|---|
| `TcMessageRate` | 0.8116 | 0.8103 | 0.9095 | 0.9098 |
| `AverageAdvertisedLinksPerTCMessage` | 0.7771 | 0.7765 | 0.7835 | 0.7836 |
| `RoutingOverheadRatio` | 0.8127 | 0.8100 | 0.9252 | 0.8964 |
| `AverageHopCount` | 0.6256 | **0.5628** | 0.7626 | **0.6633** |
| `PacketDeliveryRatio` | 0.5356 | 0.5199 | 0.6014 | 0.5821 |

The two features that carry DCFM's signature are **unchanged to the third decimal** while the
data plane thins to a third of its traffic. The data-plane features weaken exactly as expected
from having a third of the samples. [Step 37](#step-37)'s Result 6 — that the model measures
control-plane flooding — is confirmed under a manipulation it was never fitted to. [VERIFIED]

Consistently, `TDR` (`TcMessageRate × AverageAdvertisedLinksPerTCMessage`) remains the
top-ranked feature in **five of the six** runs:

| run | top features (permutation importance share) |
|---|---|
| 33/32 · static | `TDR` 0.459, `Delay_Per_Hop` 0.281, `AverageMprCount` 0.124 |
| 33/32 · mobile | `TDR` 0.447, `AverageAdvertisedLinksPerTCMessage` 0.173, `Delay_Per_Hop` 0.168 |
| 27 · mobile | `TDR` 0.533, `Delay_Per_Hop` 0.179, `AverageAdvertisedLinksPerTCMessage` 0.111 |
| **27 · static** | **`Delay_Per_Hop` 0.550**, `TDR` 0.288, `AverageAdvertisedLinksPerTCMessage` 0.077 |

The exception is instructive rather than contrary. `Delay_Per_Hop` =
`AverageEndToEndDelay / AverageHopCount`; with a single flow it becomes a clean per-hop
measurement of the queueing load that TC flooding imposes, no longer averaged across three
competing flows. The mechanism being read is the same one. [HYPOTHESIS]

### Controls
- **Permutation null.** 100 grouped permutations on 27·static, labels shuffled within run:
  observed **0.9710**, null **0.5002 ± 0.0080**, p = 0.0099 — the floor for 100 permutations,
  so no permutation came close. The observed value is 59 null standard deviations out, and the
  null band matches [Step 37](#step-37)'s 0.4989 ± 0.0072 and [Step 36](#step-36)'s
  0.486–0.535. Note that `permutation_test` uses a **different protocol** from the main CV — 3
  folds, no calibration, no threshold — which is why its observed 0.9710 differs from the
  summary table's 0.9697; it measures both sides under the same protocol, so the comparison is
  internally consistent. [VERIFIED]
- **Anchor stability.** The static-improves / mobile-regresses pattern holds for **12 of 12**
  non-`Dummy` models in mobile and **11 of 12** in static (`Ridge` excepted). The conclusion
  does not depend on the anchor.
- **Non-linearity, a fourth time.** `SVM_rbf` reaches 0.61–0.65 and `LogisticRegression`
  0.66–0.90 against 0.96–0.98 for the boosted trees. Consistent with
  [Steps 29](#step-29), [36](#step-36) and [37](#step-37). `LogisticRegression` on 32·static
  should not be quoted without its spread: 0.8569 with a **fold std of 0.131**.

### A reporting defect found in `run_config.json`
Every one of the ten runs records `feature_set: "metrics32"`, including those driven by
`--features-file` with 33 or 27 names. `cfg.feature_set` is simply never updated by
`resolve_features()`. The companion field `n_base_features_used` **is** correct (33 / 32 / 27),
so no run is mis-specified and no result is affected — but a reader auditing the JSON alone
would mis-identify every run. Recorded here rather than patched, since changing v4 mid-campaign
would break parity with [Step 37](#step-37). [VERIFIED]

### Conclusion
DCFM's detectability rests on the **control plane** and is largely indifferent to how much data
traffic the network carries. Cutting the CBR flows from three to one left `TcMessageRate` and
`AverageAdvertisedLinksPerTCMessage` unchanged to the third decimal, cost the data-plane
features roughly a tenth of AUC each, and — because those features were noise in a static
topology — actually **improved** static detection to 0.968–0.978 while costing mobile 0.008.

Two consequences beyond DCFM. First, the static↔mobile inversion that
[Step 32](#step-32) and *Open Questions* item 3 have carried since Campaign 2 began is **not a
fixed property of the defense**: it reverses when data-plane variance is removed, which points
at per-flow variance under mobility as its source rather than anything about DCFM. Second, a
feature set's worth cannot be read from AUC — the 32-set beats the 27-set by 0.010 of AUC and
by **0.114 of TPR at a 1% false-alarm budget** in static.

### What this does not establish
Still **DCFM only**, and now with a weaker experimental design than [Step 37](#step-37): the
two datasets differ in the simulation, so no delta here is a clean counterfactual. The
mechanism proposed in Result 7 — per-flow variance as the source of the mobility advantage —
is a `[HYPOTHESIS]` supported by univariate AUCs and by the fold-variance collapse, not a
measurement. The decisive test would be a **three-flow dataset with the five per-flow `Std`
features explicitly dropped**: if static then rises and mobile falls, the mechanism is
confirmed with traffic load held constant. Carried into *Open Questions*.

### Sources
- Pipeline: {refml("defense_detection_v4.py")} — **unmodified**; this step adds no v4 change
- Dataset probe (schema, degeneracy, univariate AUC, effective dimension), run on both
  datasets: {refml(S36 + "/probe_1ch.py", "probe_1ch.py")}
- Pre-flight gate — seven checks, exits non-zero on failure; imports v4 and calls its own
  `engineer_features()` and `FeatureSelector` so the effective dimension is measured with the
  pipeline's code rather than a reimplementation:
  {refml(S36 + "/preflight.py", "preflight.py")}
- Runner (activates the `defense` conda env, aborts if LightGBM / XGBoost / CatBoost are
  missing, caps `MAX_JOBS` at 8 for 10 physical cores, and diffs each feature list against
  `head -1` of the CSV before running — v4 warns and continues on a missing name):
  {refml(S36 + "/run_all.sh", "run_all.sh")}
- Comparisons: {refml(S36 + "/compare_1ch_vs_3ch.py", "compare_1ch_vs_3ch.py")} (best-model,
  fixed-anchor, config parity, top importances) and
  {refml(S36 + "/compare_accuracy_prev_vs_cur.py", "compare_accuracy_prev_vs_cur.py")}
  (per-model metric deltas against [Step 37](#step-37); matches rows **by model name**, since
  the summary tables are sorted by AUC and the order differs between steps)
- Feature sets, in `METRICS` order:
  {refml(S36 + "/33_features/features_33.txt", "features_33.txt")},
  {refml(S36 + "/32_features/features_32.txt", "features_32.txt")},
  {refml(S36 + "/27_features/features_27.txt", "features_27.txt")}
- Pre-registered predictions, written before any learning run of this step:
  {refml(S36 + "/PREDICTIONS.md", "PREDICTIONS.md")}
- Dataset: `Dcfm_All_128_features_no_norm_1ch_18msgs/` — `static/` and `mobile/`, 2004 / 2003
  runs and 8016 / 8012 windows, generated 2026-08-09
- Outputs: `{S36}/{{33,32,27}}_features/results_{{static,mobile}}/` (each with `summary.csv`,
  `folds.csv`, `importance.csv`, `run_config.json`, `summary.tex`, `final_model.pkl`,
  `figures/`, plus `permutation_test.json` on 27·static); `{S36}/preflight_report/`
  (`integrity.csv`, `control_group.csv`, `pairing.csv`, `degeneracy_univariate.csv`,
  `effective_dimension.csv`, `baseline_provenance.csv`, `probe_1ch.log`, `probe_3ch.log`);
  `{S36}/comparison/` (`comparison_best_model.csv`, `comparison_fixed_HistGB.csv`,
  `comparison_matrix.csv`, `config_parity.csv`, `importance_top.csv`,
  `acc_prev_vs_cur.csv`, `acc_prev_vs_cur_tables.txt`, `whatsapp_tables.txt`);
  `{S36}/logs/`
- Antecedents: [Step 37](#step-37) (the three-flow baseline, re-measured here);
  [Step 33](#step-33) (the 27-set); [Step 29](#step-29) (the 32-set)

### Runtime
Six runs in **39 minutes** (2026-08-09 20:56–21:36 UTC), plus 11 min 38 s for the
27·static re-run carrying the permutation null. 20 logical / **10 physical** cores, 31 GB RAM.

> **Folder numbering.** The script folder is `step_36_dcfm_non_normalized_1ch_18msgs`,
> following the repository's own counter, which runs two behind the notebook — the same offset
> that makes `step_35_dcfm_non_normalized` the home of notebook [Step 37](#step-37).
""")

md(f"""
<a id="step-39" name="step-39"></a>
## Step 39 — Realigning the Watchdog defense to its source papers, and a measurement bug that invalidated every earlier blacklist reading
**Date:** 2026-08-19 – 2026-08-23

### Motivation — a citation the implementation could not support
The Watchdog defense has been cited throughout this project as an implementation of
Baiad et al. (2014). Preparing a publication makes that citation load-bearing: if the
implementation diverges materially from the paper it claims to implement, it is a new
defense, and the write-up becomes a defense-design paper rather than a
defense-*detection* paper. The supervisor's framing was exact — where the source leaves
something undefined we may choose, but the burden is on us to show the choice is
reasonable and does not contradict the paper.

Auditing the implementation against the paper required first assembling the sources.
Baiad et al. (2014) is a six-page conference paper that specifies little and defers to
its references. Two further papers were located:

- **Marti et al. (MobiCom 2000)** — the origin of the watchdog technique, cited as [3]
  by Baiad but never explained there.
- **Baiad, Alhussein, Otrok, Muhaidat (Vehicular Communications 5, 2016)** — a journal
  extension of the 2014 paper by the same first author, which the project had not been
  aware of. It restates the 2014 algorithms verbatim and adds a third, physical-layer
  scheme.

### What the sources actually specify — and what they do not
Reading all three changed the audit substantially.

**Marti retroactively justifies code that looked invented.** The 2014 paper never
describes how a watchdog works, deferring to [3]. Marti does: a buffer of recently
forwarded packets, a timeout after which an unmatched packet increments a failure
tally, and a threshold on that tally. He also states explicitly that a watchdog should
*not* accuse on first evidence but watch over a period. The implementation's
`ForwardTimeout`, `BlacklistThreshold` and probation window are therefore not
heuristics we added — they are the mechanism of the reference the paper cites.
[VERIFIED — Marti et al. §3.1]

**The 2016 paper closes what 2014 left open.** It supplies the detection-percentage
formula in a form consistent across four equations, and gives the only definition of
false-alarm rate in the corpus: *the percentage of normal nodes detected as attackers
due to collisions*. It also settles an apparent contradiction — 2014's
`CF(i) = (1/Rd)(1/N)V` minimised and 2016's `QoS(i) = Rd·N·(1/V)` maximised are exact
reciprocals, so the two describe the same cluster-head election.

**One variable is undefined in both.** `MAC_s`, the monitor status in Algorithm 4
Part B, is stated to be 0 or 1 with no criterion for which. Two papers, identical
pseudo-code, same silence.

**The reporting mechanism is absent by construction.** Both papers specify the
aggregation *rule* — equal weight, filtered against MAC-monitor reports — but neither
specifies how a report reaches an aggregator: no message format, no recipient, no
schedule. Both evaluate in MATLAB over Mobisim trace files, so no messages are
exchanged and none is needed. This is the single most consequential gap for an ns-3
implementation.

### Correction 1 — the cross-layer test was running in the opposite direction
This is the substantive correction: it is the mechanism that makes the design
cross-layer at all, and it was operating against its own purpose.

Both papers are unambiguous. 2014 §IV-B: *by monitoring the number of RTS sent and CTS
received, at the MAC layer level, as well as data received and forwarded, at the network
layer, we can determine the existence of attackers* — and the worked example that follows
is entirely about a node wrongly accused because a collision, not a drop, prevented
delivery. Algorithm 4 Part A of 2016 makes the consequence explicit: where a MAC report
names a node, the watchdog report against it is set to zero. **MAC evidence exonerates.**

The implementation did the reverse. In `EvaluateMissingForward`, a neighbour that had
issued many RTS frames but little DATA received **+2 evidence** — the reasoning being
that a blackhole would appear to be trying to forward. Plausible in isolation, but it
inverts the premise of the entire cross-layer design, whose reason for existing is to
suppress collision-induced false positives rather than manufacture them.

The heuristic was removed and replaced by `CollisionSuspectedFor()`: if a monitored node
issued more RTS frames than it was granted CTS over the observation window, the medium
was contended, and the report is voided without scoring. Two implementation points
carried decisions of their own:

- **Window length.** Counts are kept per aggregation round and the test sums the current
  and the immediately preceding round. A single round is too short: with a 500 ms forward
  timeout and a 1 s period, part of a pending packet's life precedes the current round,
  so a one-round test can miss the very contention that killed it. Two rounds cover a
  packet's lifetime while bounding how long one discrepancy can exonerate a node —
  verified to expire after exactly two rotations.
- **Unsigned underflow.** CTS may legitimately exceed RTS when a node's RTS was
  transmitted outside our hearing but the CTS answering it was not. Without an explicit
  `cts >= rts` guard the subtraction wraps in `uint32_t` and every node reads as
  permanently contended. The guard treats the case as evidence of a clear medium, which
  it is.

Verification exposed a property of the published mechanism worth recording: **the
exculpatory test is gameable.** A node that emits RTS frames it never completes maintains
`rts > cts` in every round and is exonerated indefinitely, however much it drops — five
uncompleted RTS per round over twenty rounds accumulates zero evidence. Neither paper
considers a detector-aware attacker; both model the blackhole as a passive dropper. The
implementation is faithful precisely in exhibiting the weakness. `[VERIFIED]`

### Correction 2 — CTS was never counted, so the comparison never happened
`OnCtsReceived()` was an empty stub, commented *reserved for future refinement*. The
`rtsFromThisNode` counter existed and was fed, but nothing on the other side of the
comparison did. The central quantity of §3.2 of the 2016 paper — RTS sent against CTS
received — was therefore never computed, in any run, since the defense was written.

Implementing it required one 802.11 detail. A CTS frame carries a single address field
(RA / Addr1) holding the address of the station whose RTS is being cleared — not the
station sending the CTS. Overhearing a CTS therefore tells the watchdog that **that**
station won the medium, and the frame is credited to it. This is what makes the count
comparable to the RTS count for the same node.

A self-filter was added on both handlers: a node's own RTS and the CTS clearing it say
nothing about a monitored neighbour.

### Correction 3 — `MAC_s` was graded where the papers make it binary
Algorithm 4 Part B computes `new_weight_a(i) = wd(i) × MAC_s(j)`, with `MAC_s` stated to
be *either 1 or 0*, then `new_weight = count(new_weight_a)`. The 2014 text states the
intent directly: *if the watchdog has problems while listening, it can be eliminated from
being watchdog.* A monitor that was itself colliding contributes **nothing** to that
round.

The implementation maintained `m_selfReliabilityScore`, an exponentially-weighted average
over local PHY drops, in the range [0.6, 1.0], used to scale the blacklist threshold. Two
departures followed. The score was graded rather than binary. More importantly, **the
floor of 0.6 meant no watchdog was ever eliminated** — a permanently deaf monitor still
accused, merely later. That is the opposite of the mechanism the papers rely on for their
reported improvement, and it is the reason the change matters rather than being cosmetic.

`LocalMacStatus()` now returns `MAC_s` as a boolean, checked before any per-neighbour
reasoning — the disqualification is a property of the monitor, not of the monitored node.
When false, the watchdog scores nothing that round. With the scaling gone, the blacklist
threshold is a fixed count again, as in Marti.

Behavioural testing found a defect introduced by the change itself. `SelfDropsThreshold`
had been calibrated against a **single** observation window; comparing a two-round sum
against a one-round threshold silently halves the configured tolerance, and a watchdog
sustaining exactly the permitted number of drops per round was muted permanently. The
comparand is now `2 × SelfDropsThreshold`, preserving the attribute's meaning as
*tolerated drops per round*. No reading of the papers supports the halved behaviour; it
was an arithmetic consequence of the window choice, invisible to inspection. `[VERIFIED]`

### Correction 4 — a relay to nowhere counted as a relay
Marti §3.1 states the limitation plainly: *for the watchdog to work properly, it must
know where a packet should be in two hops.* Under DSR the next hop is carried in the
packet; under a hop-by-hop protocol it is not, and *a malicious or broken node could
broadcast the packet to a non-existant node and the watchdog would have no way of
knowing.* He concludes that the watchdog *works best on top of a source routing
protocol.* OLSR is hop-by-hop, so this implementation sits squarely in the case he warns
about, and the Baiad papers inherit the watchdog without revisiting the point.

`OnNeighborForwardedPacket` discarded its `receiver` argument entirely and matched on
packet UID alone: any retransmission of the packet cleared the pending entry, including a
broadcast addressed to nothing.

`IsPlausibleOnwardHop()` now gates the credit. A retransmission counts as a forward only
if it is not broadcast or multicast (a unicast data relay never is — this is Marti's
literal attack), not addressed back to this node, not addressed to the forwarder itself,
and — where the address resolves — present somewhere in this node's link-state view:
neighbour set, two-hop neighbour set, or topology set. A fabricated address appears in
none of the three.

The original plan was stronger: verify that the relay was addressed toward the
destination, by computing what the neighbour's own routing table would say. **That is not
cleanly implementable.** Determining a neighbour's next hop requires the neighbour's
routing table; this node has only its own, and reconstructing one from the topology set
yields an approximation that would produce false positives. What was implemented covers
the failure Marti actually names, without the approximation.

One decision was required. When the receiver's MAC does not resolve to an IP — the
mapping is learned from overheard broadcasts and is necessarily incomplete — the forward
is **credited**, not rejected. Rejecting on non-resolution would manufacture false
positives for legitimate relays to nodes not yet mapped, which is exactly the failure
mode this work is trying to remove. The rule adopted, and stated as such, is that the
watchdog does not accuse on absence of information.

Note the dependency this creates: a node whose link-state view is still empty rejects
every resolved receiver, because nothing is yet known. The existing 15 s warmup covers
it. **If `WarmupDuration` is ever shortened, this check is the first thing to
re-examine.** `[VERIFIED]`

### Correction 5 — a verdict that never expires, adopted on Marti's authority
This one is not a contradiction of the papers but a gap they leave open in opposite
directions, and the reasoning is worth recording because it decided the change.

Neither Baiad paper acts on a detection at all. Their outputs are a detection percentage
and a false-alarm rate; the words *isolate*, *exclude* and *avoid* appear in neither, and
neither reports any network-performance metric — which is what one would measure if nodes
were being removed. They are detection systems and stop at detection.

The response therefore derives from Marti, where the pathrater assigns a suspected node a
rating of −100 and the source routes around it. Marti §3.2 then adds a recommendation:
a node marked misbehaving *due to a temporary malfunction or incorrect accusation* should
not be excluded permanently, and its rating should be restored *after a long timeout* —
followed by the admission that this *is not implemented in our simulations*.

It was not implemented here either, and the measurements above quantify what that cost:
9 mobile runs in 121 where an innocent node was blocked, and 10 where a correct block
still cost more than 20 points of PDR. `ReleaseExpiredBlacklist()` now lapses a verdict
after `BlacklistDuration`, clearing evidence, probation state and timestamp on release —
leaving the count in place would re-convict on the next timed-out packet, which is
permanence under another name. Verified: a single drop immediately after release does not
re-convict, while sustained dropping re-convicts within a few rounds.

Two departures from Marti remain and are declared: exclusion here is absolute rather than
a path-rating penalty, and it is enforced during MPR selection and routing-table
construction rather than by the source. Enforcement is required by this project — without
it there is no PDR effect to measure — but it is an addition to the published detector.

### Parameters after the realignment
| Attribute | Value | Provenance |
|---|---|---|
| `ForwardTimeout` | 500 ms | Marti §3.1 (mechanism; no value given) |
| `BlacklistThreshold` | 3 | Marti §3.1 (a count, not the *bandwidth* he specifies — see below) |
| `ProbationDuration` | 2 s | Marti §3.1 — *do not immediately accuse … continue to watch over a period* |
| `WarmupDuration` | 15 s | ns-3 necessity; no counterpart in any source |
| `SelfDropsThreshold` | 5 per round | Criterion for `MAC_s`; the papers define the range {{0,1}} but no criterion |
| `RtsCtsDiscrepancyThreshold` | 1 | **New.** The papers say only *a difference*; 1 is the literal reading |
| `VerifyOnwardHop` | true | **New.** Marti §3.1; false reproduces the unverified behaviour |
| `BlacklistDuration` | 30 s | **New.** Marti says only *a long timeout*; 0 restores permanent exclusion |
| `RtsToDataRatioThreshold` | — | **Inert.** Drove the inverted heuristic of Correction 1 |
| `MinRtsForHeuristic` | — | **Inert.** Same |
| `MinSelfReliability` | — | **Inert.** Floor of the graded score replaced in Correction 3 |

The three inert attributes are still registered, so existing harness invocations that set
them continue to run; `defense_params.txt` now marks them `INERT` and records the three
new ones, having previously advertised superseded parameters as live.

### The audit as a whole — three categories of difference
Every difference from the sources was classified, on the supervisor's framing that a
choice is permissible where the source is silent but must be justified:

- **Corrected (7)** — the source specifies it and the code did not comply: the five
  above, plus the fixed threshold restored in Correction 3 and the probation window
  retained on Marti's authority.
- **Chosen (6)** — the source leaves it open: the `MAC_s` criterion, the observation
  window, the discrepancy threshold, the expiry duration, the treatment of an unresolved
  relay target, and the count-versus-rate threshold reading.
- **Structural (5)** — a difference between an offline computation and a network
  simulation, not resolvable by implementation: the meaning of *detection*, decision
  aggregation, the MAC-monitor role, the response to a detection, and the clustering
  layer.

One entry in the *chosen* group deserves its own note. Marti defines the threshold as a
**bandwidth**, and the choice is load-bearing: his sixth weakness is that a node can
evade detection by dropping below the configured rate, *but is thereby forced to forward
at the threshold rate* — the threshold is what makes the watchdog enforce a minimum
service level rather than merely detect total failure. This implementation uses an
absolute count. Under a full blackhole the two order neighbours identically, so nothing
observable differs and the change could not even be validated here; it would matter for a
grayhole, which is out of scope. Recorded rather than corrected.

### Measuring what the papers measure
The harness recorded network-performance metrics (PDR, throughput, delay) but neither
of the two quantities the papers report. Seven columns were added to
`windows_oracle.csv` (`HEADER_VERSION` 4 → 5): `true_detections`,
`false_detections`, `watchdogs_total`, `watchdogs_in_range`, `detection_percent`,
`detection_percent_in_range`, `false_alarm_rate`.

No new instrumentation was needed. A loop already walked every non-attacker node,
fetched its blacklist and tested whether the attacker was in it — then collapsed the
result into an aggregate. The change counts what that loop already computed.

**`windows_features.csv` and `windows_labels.csv` are byte-identical to before**, so
the ML pipeline is unaffected; the oracle file is already excluded from features by the
harness's own instructions. [VERIFIED — header diff]

### ⚠️ The measurement bug: `blacklist_max_size` is always zero
Analyses of defense efficacy in this project have used the oracle column
`blacklist_max_size`. **That column is zero in every window of every run, for every
defense, whether or not the defense works.**

The harness samples it at `windowStart − 2 s`, and the UDP flows run **only inside** the
measurement window. At the sampling instant no data has been exchanged, so no watchdog
has observed anything and every blacklist is empty — by construction, always. The same
timing applies to `min_attacker_trust`, whose companion fields are named `*PrevWindow`
precisely because this offset is deliberate for the path-related quantities it was
designed for.

The consequence is that **any comparison of defenses resting on `blacklist_max_size` is
uninformative** — a zero reading says nothing about whether a defense detected anything.
The contradiction was visible and went unnoticed: PDR recovered by 17 points while the
column read zero across all 484 windows. If no node was ever blocked, the routing had no
reason to improve. [VERIFIED — scheduling chain traced from `installWindow` to `EndSlot`]

The same trap was very nearly repeated. The detection counters were first added inside
that same start-of-window function; had they shipped there, a defense-enabled window
would have reported the *previous* slot's verdicts. `SampleDetectionAccuracy` therefore
runs at `windowEnd − 1 ms`, as a separate pass so that the existing columns keep their
established timing. **Eighteen unit tests of the arithmetic passed against the wrong
sampling point** — the formulas were correct and were being evaluated at the wrong
instant, which no unit test could see.

> **Rule for future work.** Before relying on an oracle column, locate its
> `Simulator::Schedule` and confirm when it is sampled; and treat a column that is
> constant across every row as a measurement defect until proven otherwise, not as a
> finding.

### The finding: the published detection metric measures opportunity, not detection
With detection instrumented, the 2016 metric could be computed — and does not mean what
its name suggests.

Algorithm 2 of the 2016 paper computes network-layer detection as a **neighbour
relation**: for each attacker and each watchdog, `if N(attacker, wd) = 1 then
detections++`. There is no packet comparison, no buffer, no timeout. A monitor within
transmission range counts as having detected. [VERIFIED — Alg. 2 as printed]

In an implementation that tracks packets, only the node that *handed a packet to* the
attacker can determine that it was not forwarded. Every other neighbour hears traffic it
has no reference for. The numerator is therefore bounded by the number of relaying
predecessors — typically one per flow — not by the number of nodes in range.

Measured over 300 runs per configuration: a mean of **7.3 watchdogs within range** of the
attacker, of which typically **one** detects it. Applying the published formula yields
3.4%; the same events against an opportunity-normalised denominator yield 7.1%. The
papers report 87–97%.

**The gap is definitional, not a matter of detector quality.** A validation run in the
papers' own topology (1000 × 100 m corridor, 30 nodes, extended radio range) was
performed to test the alternative explanation — that the gap is a consequence of our
sparser grid. It is not: `detection_percent_in_range` remained ≈ 7% there as well. That
run also showed the corridor is far less connected than assumed — `watchdogs_in_range`
was 14–19 of 29, matching the geometric prediction that a 250 m radio covers half of a
1000 m corridor — and its acceptance yield fell from 28% to 11%, so it was stopped at
three accepted runs once it had answered the question. `[VERIFIED]` for the measured
values; `[HYPOTHESIS]` that neighbour-counting is the whole of what the MATLAB
implementation does, since that code is not published.

Both denominators are emitted, together with the raw counts, so either can be
recomputed without re-running.

### Validation: the defense still works
Regression over **300 runs per configuration**, paired within run (attack-only window
versus defense+attack window on the same topology):

| | PDR recovery | 95% CI | caught attacker | false accusations (dva / defense-only) |
|---|---|---|---|---|
| **static** | **+17.23 pts** | [+13.73, +20.72] | 33% of windows | **0 / 0** |
| **mobile** | **+12.05 pts** | [+7.61, +16.48] | 32% of windows | 62 / 79 |

Neither interval crosses zero. Zero false accusations across 604 static windows locates
the false-positive problem in mobility-induced link breakage — a link breaking mid-transfer
means the watchdog never hears the relay — which is precisely the *ambiguous collision*
case Marti names and the cross-layer correlation of Baiad exists to suppress.

Decomposing the mobile runs that lost more than 20 points of PDR gave three roughly equal
groups: a third had a false accusation, a third had a **correct** detection and lost
delivery anyway, and a third had no detection at all and are attributable to mobility
variance rather than to the defense. The middle group is a genuine cost of the mechanism:
excluding an attacker that is the only link holding part of the network together is worse
than the packets it was discarding. Neither Baiad paper can observe this, as neither
measures packet delivery. [VERIFIED]

Adopting Marti's release recommendation carried a measurable cost of its own: releasing a
node permits it to be misjudged again, so a single persistent misjudgement becomes several
counted events (0.21 → 0.26 false accusations per defense-only mobile window, different
samples). The recommendation was adopted because the source specifies it, not because it
improved a number; the cost is reported because Marti never implemented the
recommendation and so never observed it. `[HYPOTHESIS]` on the magnitude — the two
samples are different runs, not the same seeds.

### What was deliberately not implemented
| Item | Reason |
|---|---|
| Decision aggregation between watchdogs | The aggregation *rule* is specified; the reporting mechanism is not, in either paper. Implementing it means designing a protocol absent from the source — message format, recipient, timing — which would be a contribution of ours presented as theirs |
| MAC-monitor nodes as a separate population | The papers select them "in the same range of the watchdogs", i.e. by proximity. Since each node reads its own PHY directly, the roles were merged: direct measurement replaces inference from proximity, and no reporting protocol is required |
| Algorithm 1 of the 2014 paper (the hidden-terminal geometric test) | **Deleted by the authors themselves** in the 2016 version |
| The VANET clustering layer (CH election, QoS, direction-based voting) | This project targets general OLSR, not the vehicular variant; cluster-head election does not bear on the detection mechanism |
| The 2016 physical-layer scheme (signature keys, maximum-likelihood test) | A third detection scheme added in the journal version. The 2016 paper reports its Net+MAC scheme separately, so citing it does not require implementing the PHY layer |
| Threshold as a *rate* rather than a count | Marti defines a threshold **bandwidth**, whose purpose is to enforce a minimum service level. Under a full blackhole the two are equivalent and the difference is unobservable here; it would matter for a grayhole, which is out of scope |
| MPR-targeted attacker selection, and the papers' simulation parameters | Both require changing attack code or project infrastructure, which was out of scope for this work |

### Defects found during this work
Three, all caught before shipping, each of a kind the others could not have caught:

1. **Threshold halved silently.** `SelfDropsThreshold` was calibrated against a single
   observation window; summing two windows without adjusting the comparand muted any
   watchdog sitting exactly at the configured tolerance. Found by a behavioural test.
2. **Sampling point.** Described above. Found by tracing the scheduling chain; all
   arithmetic tests passed against it.
3. **Duplicate implementation.** Blacklist release was implemented twice, and
   `BlacklistDuration` registered as an attribute twice — which aborts ns-3 during
   TypeId construction, so the defense would not have loaded at all. Found by a
   structural duplicate scan; every behavioural test passed against the duplicated file,
   because each copy was individually correct. A duplicate-symbol check over member
   declarations, method declarations, struct fields, function definitions and attribute
   names is now part of the QA pass.

### Sources
- Defense: {ref(D_WATCHDOG)}, {ref(D_WATCH_H, "olsr-watchdog-defense.h")}
- Harness: {ref("scratch/olsr-watchdog-eval-mitigation.cc")}
- New attributes: `RtsCtsDiscrepancyThreshold`, `VerifyOnwardHop`, `BlacklistDuration`;
  superseded and now inert: `RtsToDataRatioThreshold`, `MinRtsForHeuristic`,
  `MinSelfReliability` (retained so existing harness invocations still run, and marked
  `INERT` in `defense_params.txt`)
- Outputs: `windows_oracle.csv` gains seven columns (`HEADER_VERSION` 5);
  `Release_Check/{{static,mobile}}/` (300 runs each); `Temporary_Experiment_1/paper_topology/`
  (the papers'-topology validation run, 3 accepted)
- Papers: Baiad et al., IWCMC 2014; Baiad, Alhussein, Otrok, Muhaidat,
  *Vehicular Communications* 5 (2016) 9–17; Marti, Giuli, Lai, Baker, MobiCom 2000
- Antecedents: [Step 7](#step-7) (the Watchdog first built); [Steps 10](#step-10)–[12](#step-12)
  (hardening, the two false-positive bugs, and the risk analysis whose parameters this step
  re-derives from the sources); [Step 18](#step-18) (the `RtsCtsThreshold = 0` question,
  which this step touches but does not resolve)

### Consequence for the ML campaign
`HEADER_VERSION` moved 4 → 5. The features and labels files are unchanged, so existing
models and results stand. But a comparison across defenses requires all defenses to be
generated by the same harness build, and the Watchdog defense itself has changed
behaviour — the full-scale campaign should regenerate Watchdog rather than reuse
pre-2026-08-19 data.
""")

md("""
---
# Part VI — Synthesis
---
""")

md("""
<a id="open-questions" name="open-questions"></a>
## Open Questions

### A. Requiring domain input on the defense implementations
These cannot be resolved by analysis alone.

| # | Question | Why it decides something |
|---|---|---|
| 1 | **Is FPNT's TC padding inherent to the method, or an implementation choice?** | If **inherent**, detection via TC size is a **legitimate finding** — a real weakness of the defense, observable by any passive attacker — and the whack-a-mole should **stop**. If an **artefact**, removal is correct. The entire status of the FPNT result turns on this. **Addressed ([Step 34](#step-34)):** FPNT's TC enlargement survives the static<->mobile transfer test (ROC-AUC 1.000 both directions), so detection via TC size is a legitimate, generalising finding — though confirmed against a *single* implementation only. |
| 2 | **Does DCFM's mechanism inherently alter MPR structure and TC advertisement?** | Determines whether the DCFM cluster (`AverageMprCount`, `AdvertisedLinks`, `NormalizedRoutingLoad`) is a legitimate signature or an artefact. |
| 3 | **Why is DCFM/mobile easier to detect than DCFM/static**, inverting the pattern of every other defense? | Either a genuine mechanistic property (DCFM acts on topology dynamics, so mobility "activates" it) or a data-generation artefact ([Step 32](#step-32)). **[Step 37](#step-37) narrows this:** the data-artefact branch is now excluded for the normalisation channel — the two datasets are the same simulations and the inversion survives de-normalisation (0.9591 static vs 0.9760 mobile raw). The recovered `nObs` is also larger under mobility (57.8 vs 53.8), consistent with the mechanistic reading. **[Step 38](#step-38) then reverses the inversion:** with one CBR flow instead of three, static overtakes mobile in every one of the six runs (e.g. 0.9780 vs 0.9681 on the 32-set). So the advantage is **not a fixed property of the defense**. The candidate source is per-flow variance under mobility — the five `Flow*Std` features carry 0.558–0.589 univariate AUC in three-flow mobile against 0.500–0.507 in static, and go identically zero with one flow. `[HYPOTHESIS]`; the decisive test is a three-flow run with those five features explicitly dropped. |
| 5 | **Is `RoutingOverheadBytesRatio` a discarded real signal?** | It is the strongest univariate feature in three of [Step 38](#step-38)'s four conditions (0.844 / 0.903 one-flow, 0.846 / 0.930 three-flow), ahead of `TcMessageRate` in static — yet it is excluded from `METRICS` a priori as an FPNT byte-padding artefact, and pruned in every fold at \\|r\\| = 0.966–0.987 against `RoutingOverheadRatio`. For DCFM it measures the byte share of TC flooding, which is the mechanism itself. Whether the a-priori exclusion is right for DCFM is untested. |
| 4 | **Does forcing `RtsCtsThreshold = 0` on small packets constitute a cheat for the model?** | Raised at the 2026-05-18 meeting ([Step 18](#step-18)). Two defenses force RTS/CTS; if the classifier reads that, it is reading our configuration, not the defense. **This question anticipated the whole leakage analysis and is still open.** |

> **The framing that resolves Q1–Q2** is *not* whether a feature name sounds behavioural,
> but **why its value changes**. A **behavioural** feature changes because the network
> behaves differently. An **artefact** changes because the defense *writes* it differently.

### B. Reconciling the record
Documented because contemporaneous logs and later recollection diverge. **This report
follows the logs.**

| # | Discrepancy | Log says | Recollection said |
|---|---|---|---|
| 5 | **The "67-feature group"** | The string `67` appears in **neither** `olsr_window_features.h` **nor** `defense_detection_v4.py`. Only 95 / 33 / 128 are formally defined; "32" is derived in Python | A 67-feature group was used for several experiments |
| 6 | ~~**Date of the 33-feature (V2) set**~~ **RESOLVED** ([Step 26](#step-26)) | The 128-feature dataset and analysis are dated **2026-07-13 → 07-15** | The 33 features were first run **2026-06-24** — **both are correct**: an early run on 24 June (pre-DCFM-realignment, still 100%) *and* the July campaign after realignment |
| 7 | **The 2026-05-01 DCFM fix** | The fix that restored `defense_pdr` to 100% is **F5 — the two-consecutive-violations confirmation policy**. Rule 2 had already been disabled on 2026-04-16/17, for ANSN poisoning | "I removed Rule 2 from the contradiction rules" |
| 8 | **Defense numbering** | `session_report_2026-05-01` calls GCOP "**Defense #1**"; other logs call it "**Defense 3**" | — |

### C. Methodological items still outstanding
| # | Item | Status |
|---|---|---|
| 9 | ~~**Test the normalisation hypothesis** ([Step 32](#step-32))~~ **RESOLVED** ([Step 34](#step-34) from source; **[Step 37](#step-37) by measurement**) | Confirmed from source: `AvgTxPacketSize` divides by mean all-packet size (FPNT padding), the MPR/advertised family divides by `nObs` (inflated by DCFM's `main+65536` injection). [Step 37](#step-37) then measured it on a paired un-normalised DCFM dataset: `AUC(nObs, defense_enabled)` = **0.947 static / 0.987 mobile**, and `FlowCount` — 94.8% of DCFM/static importance — collapses to **0.501** raw. Two corrections fall out: the constant is 540 not 512, and **`NormalizedRoutingLoad` was never normalised at all** (byte-identical in both datasets), so only *two* of the five features removed to build the 27-set were normalisation artifacts |
| 10 | **Dynamic-attacker selection is too loose** | ~70% of successful runs have an **off-path** attacker → their `attack` vectors are indistinguishable from `baseline`, injecting ~70% noise. Fix: select the attacker as the **actual next hop** in Node 1's routing table toward Node 0 ([Step 16](#step-16)) |
| 11 | **Rule 1c residual false positives** | Options: (A) disable Rule 1c (Rule 1b alone gave 100% detection); (B) require **three** consecutive violations for 1c; (C) exclude blacklisted addresses from 1c verification. **Decision required before feature extraction** |
| 12 | **Data provenance** | Relationship between `~/dataset_paper/` and `dataset_128_all_defenses` was never independently audited. **Partly closed by [Step 37](#step-37)** for DCFM: `runner.config` differs between the normalised and un-normalised trees only in `OUT_DIR` and `DATE_STARTED`, and the `run_id` overlap is 2003/2004 (static) and 2004/2004 (mobile) — the same simulations. Note also that v4's `DATASETS` registry loads DCFM from `~/Downloads/DCFM_33_size/…/v2only_static` while the other three come from `dataset_final/…_canonical`; that asymmetry is still unaudited |
| 13 | **Does de-normalisation generalise beyond DCFM?** ([Step 37](#step-37)) | [Step 37](#step-37) is DCFM only. FPNT is the informative case: its contaminated divisor is the mean all-packet size, not `nObs`, and [Step 21](#step-21) showed its entire signal was TC padding. Requires regenerating the FPNT dataset un-normalised (~3 h) |
| 14 | **Does the cross-defense signature survive de-normalisation?** ([Step 36](#step-36) × [Step 37](#step-37)) | [Step 36](#step-36) reached 0.98–1.00 on an unseen defense, but ran entirely on normalised data — and [Step 37](#step-37) shows normalisation injects a strong non-behavioural component. If LODO holds up raw, the generic "a defense is running" signature is real and proven clean; if it collapses, [Step 36](#step-36)'s result was largely shared normalisation. **This is the sharpest open question in the project.** Requires question 13 first |
| 13 | **Seed-set breadth** | Ten seeds sufficed for calibration; the 87.78% Watchdog figure is **not statistically robust**. 50–100 seeds warranted |
| 14 | **Training distribution is 1-hop-attacker only** | A model expected to generalise to *n*-hop attackers needs the generator to sample attacker hop-distance ([Step 16](#step-16)) |
""")

md("""
<a id="full-campaign" name="full-campaign"></a>
## Planned Full-Scale Campaign

The 2,000-simulation dataset is a **pilot** whose purpose was to identify a
leakage-free, behavioural feature set. That objective is **not yet met** — [Step 32](#step-32)
must be resolved first. The full campaign, once the feature set is settled:

### Per defense
| Regime | Simulations |
|---|---:|
| Dynamic network, ordered windows | 10,000 |
| Static network, ordered windows | 10,000 |
| Dynamic network, **randomised window order** | 2,000 |
| Static network, **randomised window order** | 2,000 |
| **Per defense** | **24,000** |

### Totals
| Quantity | Value |
|---|---:|
| Defenses | 4 |
| Simulations | 4 × 24,000 = **96,000** |
| Vectors (4 windows each) | **384,000** |

**Randomised window order** exists to test whether the classifier has learned the
*phase sequence* rather than the *defense behaviour* — a leakage control at the
experiment-design level, and a direct response to the lessons of Part III.

> **This is already implemented, not merely planned.** The primary runner
> {ref("run_simulations.sh")} accepts `--random-window-order` (shuffles the four
> measurement windows, seeded by `--run`) and `--mixed-fraction F`, which orchestrates a
> canonical-order batch and a shuffled-order batch into separate `normal/` and `mixed/`
> output directories with disjoint seed ranges. Output directories for the ordered and
> shuffled runs of DCFM and Watchdog (`out_dcfm_ordered/`, `out_dcfm_shuffled/`,
> `out_watchdog_ordered/`, `out_watchdog_shuffled/`) are already present in the repository
> root — the randomised-order campaign is underway.

> **Cost warning from [Step 16](#step-16):** ~65–70% of seeds are rejected for incomplete
> connectivity. Reaching 10,000 **successful** runs requires attempting roughly
> **30,000–40,000 seeds** per cell.

### Research questions for the full campaign
1. **Presence:** is a black-hole defense active? (binary)
2. **Identity:** if so, which of the four? (multi-class)
3. **Cross-defense generalisation:** can a model trained on the vectors of **one**
   defense correctly classify vectors of **another**? This is the strongest available
   test that the learned features are behavioural rather than implementation-specific —
   and, given Part III, the one that matters most.

> **The instrument for question 3 now exists** ([Step 35](#step-35)). `defense_detection_v4.py`
> carries three flag-gated generalisation experiments — `--transfer-mobility` (static ↔ mobile),
> `--transfer-defense` (the N×N cross-defense matrix that question 3 asks for) and `--lodo`
> (train on three defenses, test on the unseen fourth) — all under a frozen-source-model
> protocol in which nothing, including the decision threshold, is re-fitted on the target.
> They have been run end to end on the pilot dataset; **their tables have not yet been
> analysed.** When the full campaign lands, question 3 is a re-run of an existing experiment
> on more data rather than a new piece of engineering.

### Training protocol
Static and dynamic are trained **separately**, per defense.
""")

# ==========================================================================
# PART V — ANNOTATED SOURCE-FILE GUIDE
# ==========================================================================
md(f"""
---
# Part VII — Annotated Source-File Guide
---

The chronological narrative above explains *why* each piece exists. This part explains
*what each source file is*, for a reader opening the repository
[`hananelk26/manet-olsr-project`]({REPO.rsplit("/blob/", 1)[0]}) (branch `master`). Every
file the user identified as relevant is documented: the modified OLSR core and the four
defenses in `src/olsr/model/`, the feature schema and simulations in `scratch/`, the
per-defense swap sets in `files for all defenses/`, and the batch scripts in the
repository root. Descriptions are grounded in the code as it stands on `master`.

> **One thing to know before reading the file list.** The repository is **mono-checkout**:
> only one defense's source is present in `src/olsr/model/` at a time (currently
> **DCFM**). Switching defenses means copying a different set of files over it — see
> [§ the swap sets](#guide-swap). So "the FPNT routing protocol" and "the DCFM routing
> protocol" are *different files that occupy the same path at different times*.
""")

md(f"""
<a id="guide-coldstart" name="guide-coldstart"></a>
## How the four windows are measured — the `Enabled` cold-start

Before the files, the mechanism that ties them together. Every dataset row comes from a
simulation that measures **four windows on one network** — `baseline`, `attack_only`,
`defense_only`, `defense_vs_attack` — so that differences between windows are caused by
attack/defense state and not by topology or RNG variance. Two problems had to be solved
for that to be valid, and both are visible throughout the defense code:

1. **A defense-OFF window must be indistinguishable from a network that has no defense at
   all.** Every defense therefore has an **`Enabled` ns-3 attribute** (default `false`).
   While disabled the strategy is *fully inert*: `IsMalicious` returns `false` for
   everyone, `GetBlacklist` returns empty, `RequiresFictitiousNode` returns `false` (so a
   disabled DCFM node does not even emit fictitious entries on the air), and every
   state-mutating hook early-returns. A defense-OFF window is thus byte-identical on the
   wire to a no-defense baseline.

2. **State must not leak across window boundaries.** `SetEnabled` performs a **symmetric
   cold start**: *every* real transition — enable→disable **and** disable→enable — wipes
   all accumulated detection state. The harness forces a reset at each slot boundary by
   toggling `Enabled` twice (`ForceDefenseColdStart()`), which always produces two real
   transitions and therefore a guaranteed full wipe, restoring the slot's intended value.
   A no-op guard makes a redundant set-to-same-value a strict no-op, so a stray
   `SetAttribute` mid-window cannot erase live state.

Each defense also exposes a read-only `GetDebugStateSizes()` returning the raw sizes of
its state containers (**not** gated by `Enabled`), surfaced by the harness's
`--debugDefenseState` flag: immediately after a cold start every size must read zero, and
any non-zero value is direct evidence of a cross-window leak. This is the on-air, in-code
counterpart to the *data*-level leakage analysis of [Part III](#step-21).
""")

md(f"""
<a id="guide-model" name="guide-model"></a>
## `src/olsr/model/` — protocol core, interface, defenses

### `olsr-routing-protocol.{{cc,h}}` — the modified OLSR core (+ the attack)
{ref(ATTACK_H)} · {ref(ATTACK)}

A fork of ns-3's stock OLSR (RFC 3626), whose header credits *"Modified by: Oded Ofek,
2025 — Implementation of Blackhole & Link Spoofing Attacks."* It is the foundation for
**both** the attack and every defense, and it is the file that differs most between
defenses (each defense ships its own variant — see [§ swap sets](#guide-swap)).

- **Attack surface.** Two ns-3 attributes, `IsMalicious` (bool) and `SpoofedLinksCount`
  (uint32), arm the four-mechanism black-hole attack ([Step 4](#step-4)). In the code:
  willingness is forced to `WILL_ALWAYS`; `SendTc` inflates ANSN by +200; `SendHello`
  injects phantom `200.0.0.x` links via `BuildSpoofTargets(m_spoofedLinksCount)`;
  `RouteInput` silently drops transit packets for a malicious node.
- **Defense scaffolding.** Holds `Ptr<OlsrDefenseStrategy> m_defenseStrategy` and calls
  its hooks at nine sites (control-message receipt, TC generation, MPR computation,
  routing-table computation, data forwarding, `RouteInput` IMP enforcement, the 1 Hz
  `HandleDefenseTimer`). Adds `SetupPromiscuousMonitor` / `MonitorSnifferRx` (the
  promiscuous overhearing the Watchdog needs), `EvictNeighbor` (purges a blacklisted node
  from every OLSR set and recomputes routes), and `ReactivateDefenseStrategy` (re-arms a
  defense installed mid-simulation). In the FPNT/TRUST variants it additionally carries
  `RunTrustDijkstra` (trust-weighted routing) and the machinery to piggyback trust on TC.
- **Header dependencies** (`olsr-header.h`, `olsr-state.h`, `olsr-repositories.h`) are
  stock for Watchdog/DCFM but modified for FPNT/TRUST, which is why those two defenses
  swap more than just this file.

### `olsr-defense-strategy.{{cc,h}}` — the pluggable-defense interface
{ref(IFACE)} · {ref(IFACE_CC)}

The Strategy-pattern contract every defense implements. In the **Watchdog/DCFM** variant
it declares **19 pure-virtual hooks** (lifecycle, blacklist queries, control-plane,
data-plane, promiscuous sniffer, cross-layer metrics, and `RequiresFictitiousNode`) plus
a `DropReason` enum, and provides `OlsrDefenseNull` — a no-op default so the protocol can
run with no defense (exactly what the baseline window needs). **Important:** the
**FPNT/TRUST** variant of this same file *extends* the interface with trust-propagation
methods (`GetEvaluationVectors`, `GetNodeTrust`, `OnRecvEvaluationVectors`,
`IsTrustRoutingEnabled`, and a richer `OnDataPacketForwarded` signature). That interface
difference is the reason FPNT and TRUST cannot share the Watchdog/DCFM core and must swap
their own `olsr-defense-strategy` too.

### `olsr-watchdog-defense.{{cc,h}}` — Defense 1: cross-layer Watchdog
{ref(D_WATCHDOG)} · {ref(D_WATCH_H, "olsr-watchdog-defense.h")}

Implements Baiad et al. (2014). Each node independently watchdogs its 1-hop neighbours:
it records packets it forwarded (`m_pendingByNeighbor`), listens promiscuously via
`SnifferRxCallback` for the neighbour to relay them, and on a timeout runs
`EvaluateMissingForward` — a decision tree that uses MAC-layer RTS/CTS evidence to tell an
intentional drop from a collision. Notable, all in the code:
- **MAC↔IP learned from broadcasts only** (`m_macToIp`), because a *unicast* frame's L2
  source is the forwarder while its IP source is the original sender — learning from
  unicast would reverse the mapping ([Step 13](#step-13)).
- **Self-reliability (Algorithm B):** `PhyRxDropCallback` counts local PHY drops;
  `UpdateSelfReliability` scales the effective threshold up when the node's own reception
  is noisy, so a bad listener does not accuse others.
- **Anti-false-positive guards:** a `WarmupDuration`, a three-guard cautious-commit
  (`MinDataObservations`, `MacFailureRateThreshold`, `ProbationDuration`), and evidence
  decay. The calibrated defaults live in `GetTypeId` (`BlacklistThreshold=3`,
  `ProbationDuration=2 s`, `MinSelfReliability=0.6`, `MinDataObservations=2` —
  [Step 17](#step-17)).
- Self-sufficient wiring: `AttachWifiTraces` / `DetachWifiTraces` connect the PHY traces
  directly (the routing core's sniffer hooks are not required), and `Enabled` triggers
  `ResetAccumulatedState`.

### `olsr-defense-fpnt.{{cc,h}}` — Defense 2: FPNT (Fuzzy-Petri-Net trust)
{ref(D_FPNT)} · {ref(D_FPNT_H, "olsr-defense-fpnt.h")}

A full implementation of Tan et al. (2015). Per neighbour it accumulates four behavioural
factors (`NodeBehaviorMetrics`: load, packet-forwarding-rate, average forwarding delay,
protocol-deviation), turns them into an evaluation vector through a **Fuzzy Petri Net**
(`RunFuzzyPetriNet`, the paper's seven rules → 15 propositions / 11 transitions, via the
matrix operators `MatrixOp_Threshold/Max/WeightedMax`), and fuses direct plus
recommendation evidence with an **L1 "slander" filter** (`AggregateEvaluations`, Eq. 1–3)
before Eq. 4/5 synthesis and temporal smoothing. Trust is **shared network-wide**:
evaluation vectors are piggybacked onto TC and consumed via `OnRecvEvaluationVectors`,
and routing is trust-weighted (`RoutingProtocol::RunTrustDijkstra`, a max-min bottleneck
criterion). This active, control-plane trust propagation is the very thing the ML model
found easiest to detect ([Part III](#step-21)). A node is malicious iff its aggregated
trust `T(V_j) < MaliciousThreshold`.

### `olsr-defense-gcop.{{cc,h}}` — Defense 3: DCFM (GCOP/GCOHP + contradiction rules)
{ref(D_GCOP)} · {ref(D_GCOP_H, "olsr-defense-gcop.h")}

Implements Schweitzer et al. (2025). Purely control-plane: on every received HELLO,
`EvaluateContradictionRules` runs three rules and flags the sender on any violation —
`CheckRule1` (asymmetry on a real neighbour, plus a "bait" sub-check that catches an
attacker swallowing our own fictitious node), `CheckRule2` (MPR-coverage contradiction
over the topology set), `CheckRule3` (over-coverage). Separately, `RequiresFictitiousNode`
decides whether to advertise a fictitious node to *provoke* a contradiction downstream,
using `RunGcohpAlgorithm` (the hexagon test). Code-level facts that matter:
- **"PROF PORT" (the 2026-07-10 realignment, [Step 26](#step-26)):** the file is now
  byte-faithful to the paper author's reference — **no warmup, no penalty window, no
  strike counter**. The only state container is `m_riskyNodes`, recomputed from scratch on
  every HELLO (a node clears the instant it sends a clean HELLO).
- **`RunGcopAlgorithm` (Algorithm 1, BFS depth-2) exists but is NOT on the live path:**
  matching the reference, `RequiresFictitiousNode` calls **GCOHP only**. GCOP is retained
  for A/B comparison (and carries a documented bug-fix folding topology-set edges into
  G₂). GCOHP is a faithful port of the reference `sixNodesArrangedInCircle()`, deliberately
  accepting only the *yellow*-closer hexagon case.
- Two ns-3 attributes: `Enabled` (cold-start switch) and **`UseFictitiousNodes`** (the
  2026-06-23 flag — `false` runs the three contradiction rules only, with no fictitious
  injection and no bait sub-check).

### `olsr-trust-defense.{{cc,h}}` (+ `defense/` submodules) — Defense 4: TRUST
{ref(D_TRUST)} · {ref(D_TRUST_H, "olsr-trust-defense.h")}

> **Correction surfaced by reading the code.** TRUST is **not** a second Tan-style trust
> scheme. Its header cites **Adnane, Bidan & de Sousa, *Computer Communications* 36
> (2013)** — a different, formula-driven trust framework. Earlier documentation left
> TRUST's paper unconfirmed; the source settles it.

Unlike the other three (each a single class), TRUST is **modular**: `OlsrTrustDefense`
owns and wires five independently toggleable sub-modules, referenced via `defense/`
includes and configured through one `OlsrTrustDefenseConfig` mirrored as ns-3 attributes:
- `OlsrForwardMonitor` — the black-hole core (paper **Formula 10**), driving
  `OnForwardFailure`;
- `OlsrTrustState` — the mistrust verdict / detection log (**Formula 15**, `MN_x`);
- `OlsrConsistencyRules` — consistency checks (**Formulas 6/7/8/9/12**);
- `OlsrProvableIdentity` — Section 6 / **Formula 13** (off by default);
- `OlsrAlertDistributor` — third-party-verifiable alert propagation.

A deliberate design point: **detection is separated from response.** Recording a mistrust
(measurable as precision/recall) is decoupled from `IsMalicious` driving the routing
countermeasure, so with `ResponseEnabled=false` the defense runs in a "detection-only"
mode. (Note: the `defense/` submodule files are referenced by the includes but are not
part of the current `master` checkout of `src/olsr/model/` — a repository-tracking gap
worth closing, since TRUST will not build without them.)
""")

md(f"""
<a id="guide-scratch" name="guide-scratch"></a>
## `scratch/` — feature schema and simulations

### `olsr_window_features.h` — the 128-feature schema (the heart of the dataset)
{ref(FEATURES)}

A single defense-agnostic collector (namespace `ns3::olsreval`; the `ns3::fpnt` name was a
historical artifact of the FPNT harness being written first). It observes one measurement
window and emits a feature row. The `enum class FeatureMode {{ Core, V2Only, CoreAndV2 }}`
selects the output: **Core** = groups A–K (**95** features, the project's own set),
**V2Only** = group L, the `strict_observable_v2` parity group (**33** features, supplied by
the DCFM paper's author), **CoreAndV2** = **128**. The emit path is marked *"schema v5,
normalised"* — scale-dependent features are divided by quantities like node count (the
normalisation whose leakage risk [Step 32](#step-32) analyses). Design principles baked in
and directly relevant to Part III: features an on-network passive adversary *cannot*
observe (HELLO-only quantities, 1-hop RTS/CTS/ACK) are excluded from the feature set and
diverted to the oracle; defense-internal state, attacker-on-path, and defense config are
kept out of features and written to the oracle/labels/runs tables instead.

### `olsr-{{dcfm,fpnt,watchdog,trust}}-eval-mitigation.cc` — the four dataset generators
{ref("scratch/olsr-dcfm-eval-mitigation.cc", "olsr-dcfm-eval-mitigation.cc")} ·
{ref("files for all defenses/FPNT/olsr-fpnt-eval-mitigation.cc", "olsr-fpnt-eval-mitigation.cc")} ·
{ref("scratch/olsr-watchdog-eval-mitigation.cc", "olsr-watchdog-eval-mitigation.cc")} ·
{ref("files for all defenses/Trust/olsr-trust-eval-mitigation.cc", "olsr-trust-eval-mitigation.cc")}

**These four harnesses produced the machine-learning dataset.** They are near-identical:
one canonical harness with the defense-layer include and casts swapped (the DCFM file's
own changelog states it is *"byte-identical to the Watchdog/FPNT harnesses"* apart from the
defense binding). What the shared harness does:
- **Timeline.** `t=[0,60)` initial stabilisation always runs *neutral* (attack OFF,
  defense OFF); then four slots, each `60 s` stabilisation + `40 s` measurement, for a
  `460 s` simulation. The `Enabled` cold-start ([§ above](#guide-coldstart)) fires at each
  slot boundary so a slot inherits nothing from the previous one.
- **Window order.** `--randomWindowOrder` shuffles the four windows using a *separate*
  `std::mt19937` seeded by `--run`, so topology is identical to the canonical-order run of
  the same seed (no ns-3 RNG draw is consumed). Records `window_order_perm` /
  `slot0_scenario` in `runs.csv`.
- **Traffic.** Each window carries `NUM_DATA_FLOWS = 3` simultaneous UDP flows (flow 0 is
  always the legacy node1→node0 pair whose acceptance gate is unchanged), so the
  network-level footprint of emergent isolation around the attacker becomes observable.
- **Output.** Five CSV tables per run, promoted atomically at the end: `windows_features`
  (the 128-feature vectors, ML input), `windows_labels` (`defense_enabled`, scenario),
  `windows_oracle` (ground truth kept *out* of features: PDR, blacklist size, on-path
  flags), plus `runs.csv` and probe data. The header carries `harness_version` /
  `HEADER_VERSION` for provenance.

### `gcopBaseSimulation.cc`, `watchdogBaseSimulation.cc` — 50-node base harnesses
{ref("scratch/gcopBaseSimulation.cc")} · {ref("scratch/watchdogBaseSimulation.cc")}

The 50-node, four-phase validation harnesses on the fixed parameters (750×1000 m, 190 m
range, dynamic 1-hop attacker) used to confirm each defense works before the full dataset
runs ([Steps 16–17](#step-16)). `watchdogBaseSimulation.cc` is a parity clone of
`gcopBaseSimulation.cc` with only the defense binding changed, so their results are
directly comparable. These pre-date, and fed into, the `*-eval-mitigation.cc` generators.

### `olsr-gcop-simulator.cc` — 8-node bridge validation
{ref("scratch/olsr-gcop-simulator.cc")}

The small, deterministic 8-node bridge proof-of-mechanism for DCFM ([Steps 9, 15](#step-9)):
Victim / Sender / Attacker / Relay / Helper + background, four phases, `--txRange` (use
280 m), CSV out. Ideal for isolating one variable at a time; not comparable to the paper's
50–100-node PDR curves.

### `olsr-watchdog-validation.cc`, `olsr-watchdog-eval.cc`, `olsr-watchdog-eval-highway.cc`
{ref("scratch/olsr-watchdog-validation.cc")} ·
{ref("scratch/olsr-watchdog-eval.cc")} ·
{ref("scratch/olsr-watchdog-eval-highway.cc")}

The Watchdog's own validation lineage ([Step 13](#step-13)): `-validation` is the 22-node
controlled scenario (five phases incl. `baseline_late`); `-eval` is the random-square
evaluation (where hard-blacklist bridge isolation can collapse Tx); `-eval-highway` is the
1000×200 m, 300 m-range highway matched to Baiad et al.'s original scenario (zero Tx
collapse). Together they establish the two behavioural signatures the classifier benefits
from.

### `bridge_attack.cc` — attack validation
{ref("scratch/bridge_attack.cc")}

A focused scenario that exercises the black-hole attack itself (independent of any
defense) — the earliest simulation, used to confirm the four attack mechanisms drive PDR
down before any defense was written.
""")

md(f"""
<a id="guide-swap" name="guide-swap"></a>
## `files for all defenses/` — the per-defense swap sets
{ref(f"{SWAP}", SWAP + "/")}

Because the repository keeps only one defense's core in `src/olsr/model/` at a time, this
folder stores, per defense, the exact files to copy over that directory when switching to
it. The **depth of each swap is itself informative**:

| Subdir | Files it swaps into `src/olsr/model/` | Why |
|---|---|---|
| {ref(f"{SWAP}/Watchdog", "Watchdog/")} | `olsr-routing-protocol.{{cc,h}}`, `olsr-defense-strategy.{{cc,h}}` | **shallow** — a data-plane observer needs only the core + interface |
| {ref(f"{SWAP}/DCFM", "DCFM/")} | same two pairs | **shallow** — control-plane observer, same reason |
| {ref(f"{SWAP}/FPNT", "FPNT/")} | the above **plus** `olsr-header.{{cc,h}}`, `olsr-state.{{cc,h}}`, `olsr-repositories.h`, `CMakeLists.txt` | **deep** — must change *what OLSR transmits* to carry trust in TC |
| {ref(f"{SWAP}/Trust", "Trust/")} | same deep set | **deep** — same reason |

> **The swap depth is a signature of passive vs. active defense.** Watchdog and DCFM only
> *read* the protocol (and act through the blacklist), so they leave OLSR's wire format
> untouched. FPNT and TRUST *piggyback trust onto TC* and reason over trust-weighted
> routing, so they must replace `olsr-header`/`olsr-state`/`olsr-repositories` and extend
> the defense interface. This is the same passive/active axis that governs how detectable
> each defense turned out to be in [Part III](#step-24) — the folder layout foreshadows the
> ML result.

The folder also holds {ref(f"{SWAP}/OLSR_Feature_Normalization_Table.docx", "OLSR_Feature_Normalization_Table.docx")},
the specification of which features are normalised and by what — the primary reference for
the normalisation hypothesis of [Step 32](#step-32).
""")

md(f"""
<a id="guide-scripts" name="guide-scripts"></a>
## Repository-root batch scripts

### `run_simulations.sh` — the primary dataset runner
{ref("run_simulations.sh")}

A resumable, parallel batch runner (a "post-audit" rewrite; needs bash ≥ 4.3). It repeats
a chosen `*-eval-mitigation` harness across seeds until *N accepted* runs are collected,
tolerating the ~65–70% connectivity rejections, and never touches the harness's CSVs
except for a header-version check and the final summary. Key controls:
`--defense {{fpnt,watchdog,dcfm}}` selects the harness; `-n` sets accepted-run target;
`-j` sets parallel workers; `--random-window-order` forwards `--randomWindowOrder=1`; and
`--mixed-fraction F` turns the script into an **orchestrator** that splits the target into
a canonical-order batch (`normal/`) and a shuffled-order batch (`mixed/`) with disjoint
seed ranges. State lives in `<out>/.runstate/` (`accepted`/`rejected`/`errors`, seed
ledger); the PDR summary reads `windows_oracle.csv`.

### `run_gcop_base_multi_seeds.py`, `run_watchdog_base_multi_seeds.py`
{ref("run_gcop_base_multi_seeds.py")} · {ref("run_watchdog_base_multi_seeds.py")}

The earlier per-defense seed-sweep drivers for the *base* simulations
(`gcopBaseSimulation.cc` / `watchdogBaseSimulation.cc`): sweep seeds until *N* successful
runs, discard connectivity/attacker-selection rejects, print a per-run PDR table and
optional CSV. Superseded for dataset generation by `run_simulations.sh`, but still the
tools behind the validation numbers in [Steps 16–17](#step-16).

### `multi-seed-eval.sh`, `multi-seed-eval-highway.sh`, `variance-test.sh`
{ref("multi-seed-eval.sh")} · {ref("multi-seed-eval-highway.sh")} · {ref("variance-test.sh")}

Multi-seed sweep wrappers used in the Watchdog evaluation lineage ([Step 13](#step-13)):
random-square and highway evaluations, and the 10-seed variance study whose exceptionally
low `defense_vs_attack` stdev demonstrated the Watchdog is deterministic.
""")

md(f"""
<a id="guide-repro" name="guide-repro"></a>
## Reproducing the dataset — the exact commands

The pilot 128-feature dataset (2,000 accepted runs per defense × mobility) was generated
with the commands below, run from the ns-3 tree on a WSL/Ubuntu workstation. Watchdog and
DCFM are shown; FPNT and TRUST follow the same pattern with `--defense fpnt|trust`.

```bash
# Build (optimized), then run inside tmux to survive disconnection.
./ns3 configure -d optimized && ./ns3 build

E="--nNodes=50 --nMaxGridX=750 --nMaxGridY=1000 --bHighRange=false \\
   --maliciousNodes=2 --spoofCount=5 --attackerJitter=25"

# Static (2000 accepted, 5 parallel workers)
bash run_simulations.sh --defense watchdog -n 2000 -j 5 \\
  -o ~/dataset_paper/watchdog/All_128_features/static \\
  --extra "$E --bMobility=false"

# Mobile
bash run_simulations.sh --defense watchdog -n 2000 -j 5 \\
  -o ~/dataset_paper/watchdog/All_128_features/mobile \\
  --extra "$E --bMobility=true"
```

Each output directory holds `.runstate/{{accepted,rejected,errors}}`, `runner.config`,
and `windows_{{features,labels,oracle}}.csv`. The integrity invariants checked after a run
are worth recording as the dataset's acceptance criteria:

| Invariant | Expected |
|---|---|
| feature rows == label rows == oracle rows | equal |
| feature rows == `accepted × 4` | 4 windows per run |
| NUL bytes in the CSVs | 0 |
| feature-CSV columns | **133** (5 identifiers + 95 Core + 33 V2) |
| windows per `run_id` | exactly 4 |
| identical header across static/mobile | matching `md5sum` |

The oracle (`windows_oracle.csv`, columns include `scenario`, `blacklist_max_size`,
`pdr_percent`) is what makes leakage auditing possible: e.g. *"`defense_only` windows with
a non-empty blacklist"* counts false positives, and the attack/`defense_vs_attack` PDR gap
measures recovery — all **outside** the ML feature set. The finished datasets were copied
from `~/dataset_paper/<def>/All_128_features/` to
`/mnt/d/Hananel/{{Watchdog,Dcfm}}_All_128_features/`, which is the provenance the ML report
([Part V](#step-27)) analyses — resolving the caveat it flagged about the
`~/dataset_paper/` path.
""")

md("""
<a id="references" name="references"></a>
## References

1. **Clausen, T. & Jacquet, P.** (2003). *RFC 3626 — Optimized Link State Routing Protocol (OLSR)*. IETF. https://www.rfc-editor.org/rfc/rfc3626
2. **Perkins, C., Belding-Royer, E. & Das, S.** (2003). *RFC 3561 — Ad hoc On-Demand Distance Vector (AODV) Routing*. IETF.
3. **Baiad, R., Otrok, H., Muhaidat, S. & Bentahar, J.** (2014). *Cooperative Cross Layer Detection for Blackhole Attack in VANET-OLSR*. IEEE IWCMC, 863–868. — **Defense 1 (Watchdog)**
3b. **Baiad, R., Alhussein, O., Otrok, H. & Muhaidat, S.** (2016). *Novel cross layer detection schemes to detect blackhole attack against QoS-OLSR protocol in VANET*. Vehicular Communications, 5, 9–17. https://doi.org/10.1016/j.vehcom.2016.09.001 — journal extension of [3]; located in [Step 39](#step-39) and used there to resolve the detection-percentage formula and the false-alarm definition
4. **Tan, S., Li, X. & Dong, Q.** (2015). *Trust Based Routing Mechanism for Securing OLSR-Based MANET*. Ad Hoc Networks, 30. — **Defense 2 (FPNT)**
5. **Schweitzer, N., Cohen, L., Hirst, T., Dvir, A. & Stulman, A.** (2025). *Achieving MANET Protection without the Use of Superfluous Fictitious Nodes*. Computer Communications, 229, 107978. https://doi.org/10.1016/j.comcom.2024.107978 — **Defense 3 (DCFM/GCOP)**
6. **Adnane, A., Bidan, C. & de Sousa Júnior, R. T.** (2013). *Trust-based security for the OLSR routing protocol*. Computer Communications, 36(10–11), 1159–1171. — **Defense 4 (TRUST)** *(confirmed from the `olsr-trust-defense.h` source header)*
7. **Schweitzer, N., Stulman, A., Shabtai, A. & Margalit, R. D.** (2016). *Mitigating Denial of Service Attacks in OLSR Protocol Using Fictitious Nodes*. IEEE Transactions on Mobile Computing, 15(1), 163–172.
7. **Schweitzer, N., Stulman, A., Margalit, R. D. & Shabtai, A.** (2017). *Contradiction Based Gray-Hole Attack Minimization for Ad-Hoc Networks*. IEEE Transactions on Mobile Computing, 16(8), 2174–2183.
8. **von Mulert, J., Welch, I. & Seah, W. K. G.** (2012). *Security Threats and Solutions in MANETs: A Case Study Using AODV and SAODV*. Journal of Network and Computer Applications.
9. **Marti, S., Giuli, T. J., Lai, K. & Baker, M.** (2000). *Mitigating Routing Misbehavior in Mobile Ad Hoc Networks*. MobiCom, 255–265. — the origin of the watchdog technique, cited but not explained by [3]; supplies the buffer/timeout/tally mechanism, the six weaknesses, and the blacklist-release recommendation adopted in [Step 39](#step-39)
10. **Hayajneh, T., Krishnamurthy, P., Tipper, D. & Kim, T.** (2009). *Detecting Malicious Packet Dropping in the Presence of Collisions and Channel Errors in Wireless Ad Hoc Networks*. IEEE ICC, 1–6.
11. **Bianchi, G.** (2000). *Performance Analysis of the IEEE 802.11 Distributed Coordination Function*. IEEE JSAC, 18(3), 535–547.
12. *Countering Data and Control Plane Attack on OLSR Using Passive Neighbor Policing and Inconsistency Identification*. https://dl.acm.org/doi/10.1145/3345837.3355955
13. *Misbehavior Nodes Detection and Isolation for MANETs OLSR Protocol*. https://www.sciencedirect.com/science/article/pii/S1877050910003959
14. *An Improved Security OLSR Protocol against Black Hole Attack Based on FANET*. IEEE. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9828257
15. *Solutions to Black Hole Attacks in MANETs*. IEEE. https://ieeexplore.ieee.org/document/9249524
16. *New MPR Computation for Securing OLSR Routing Protocol Against Single Black Hole Attack*. — surveyed, not implemented
17. **Srinivasan, J.** (2025). *Innovative Cross-Layer Defense Mechanisms for Black Hole and Wormhole Attacks in Wireless Ad-Hoc Networks*. Scientific Reports (April 2025). — **shortlisted, attempted, abandoned**; the subject of this project's comment to the editor ([Step 6](#step-6))
18. **The ns-3 Consortium.** *ns-3 Network Simulator*. https://www.nsnam.org
""")

md(f"""
<a id="file-index" name="file-index"></a>
## File Index

> **NS-3 references are clickable links** to the public repository
> [`hananelk26/manet-olsr-project`]({REPO.rsplit("/blob/", 1)[0]}) (branch `master`).
> **ML references are repo-relative paths, not links** — `hananelk26/ML-for-NS3`
> (branch `main`) is private.

### NS-3 — core
| Component | File | Role |
|---|---|---|
| OLSR + attack | {ref(ATTACK)} | Modified OLSR (RFC 3626) with the four-mechanism black-hole attack and all defense hooks |
| OLSR header | {ref(ATTACK_H)} | Attributes `IsMalicious`, `SpoofedLinksCount`, `DefenseStrategy`; `EvictNeighbor`, `ReactivateDefenseStrategy` |
| Defense interface | {ref(IFACE)} | Abstract `OlsrDefenseStrategy` — **19 pure-virtual hooks** |
| Interface impl. | {ref(IFACE_CC)} | Type registration + `OlsrDefenseNull` no-op default |

### NS-3 — defenses (current implementations)
| # | Defense | File |
|---|---|---|
| 1 | **Watchdog** | {ref(D_WATCHDOG)}, {ref(D_WATCH_H, "olsr-watchdog-defense.h")} |
| 2 | **FPNT** | {ref(D_FPNT)}, {ref(D_FPNT_H, "olsr-defense-fpnt.h")} |
| 3 | **DCFM / GCOP** | {ref(D_GCOP)}, {ref(D_GCOP_H, "olsr-defense-gcop.h")} |
| 4 | **TRUST** | {ref(D_TRUST)}, {ref(D_TRUST_H, "olsr-trust-defense.h")} |

> **Repository layout changed 2026-08-04** (commits `07791e7`, `3746e1d`, `46f22d2`). FPNT and
> TRUST — both sources and their `*-eval-mitigation.cc` simulations — were moved out of
> `src/olsr/model/` and `scratch/` into a top-level `files for all defenses/` directory, and
> **removed from `src/olsr/CMakeLists.txt`**. The links above point at the new locations.
> `src/olsr/CMakeLists.txt` currently builds only `olsr-defense-strategy`, `olsr-defense-gcop`
> and `olsr-watchdog-defense`, so **FPNT and TRUST no longer compile as part of the OLSR
> module**; regenerating either dataset requires restoring them to the build first. DCFM and
> Watchdog are unaffected. [VERIFIED — GitHub API, master @ 2026-08-04T15:56Z]

### NS-3 — per-defense swap folder (`{SWAP}/`)
To run a given defense you overwrite the files in `src/olsr/model/` with that defense's
set. The folder holds one subdirectory per defense:

| Subdir | Swaps | Note |
|---|---|---|
| {ref(f"{SWAP}/Watchdog", "Watchdog/")} | `olsr-routing-protocol.{{cc,h}}` + `olsr-defense-strategy.{{cc,h}}` | shallow swap |
| {ref(f"{SWAP}/DCFM", "DCFM/")} | same two file pairs | shallow swap |
| {ref(f"{SWAP}/FPNT", "FPNT/")} | the above **plus** `olsr-header.{{cc,h}}`, `olsr-state.{{cc,h}}`, `olsr-repositories.h`, `CMakeLists.txt` | **deep** — modifies the OLSR core to carry trust in TC |
| {ref(f"{SWAP}/Trust", "Trust/")} | same deep set as FPNT | **deep** — same reason |

> **An architectural finding worth stating.** Watchdog and DCFM are *observers* — they read
> the protocol and act through the blacklist, so they need only the routing-protocol and
> strategy files. FPNT and TRUST *change what OLSR transmits* — they piggyback trust scores
> onto TC messages — so they must also replace `olsr-header`, `olsr-state`, and
> `olsr-repositories`. The depth of the swap is a direct signature of whether a defense is
> passive (data-plane observation) or active (control-plane modification) — the same
> distinction that governs how detectable each defense is in [Part III](#step-24).

The normalisation table {ref(f"{SWAP}/OLSR_Feature_Normalization_Table.docx", "OLSR_Feature_Normalization_Table.docx")} — central to [Step 32](#step-32) — also lives here.

### NS-3 — features
| Component | File |
|---|---|
| Feature schema (`FeatureMode`: Core 95 + V2 33 = 128; "schema v5, normalised") | {ref(FEATURES)} |

### NS-3 — simulations (`scratch/`)
| File | Role |
|---|---|
| {ref("scratch/olsr-dcfm-eval-mitigation.cc")} | **Dataset generator — DCFM** (four-window feature emission) |
| {ref("files for all defenses/FPNT/olsr-fpnt-eval-mitigation.cc")} | **Dataset generator — FPNT** |
| {ref("scratch/olsr-watchdog-eval-mitigation.cc")} | **Dataset generator — Watchdog** |
| {ref("files for all defenses/Trust/olsr-trust-eval-mitigation.cc")} | **Dataset generator — TRUST** |
| {ref("scratch/gcopBaseSimulation.cc")} | 50-node base harness, four phases, dynamic attacker |
| {ref("scratch/watchdogBaseSimulation.cc")} | Watchdog parity harness |
| {ref("scratch/olsr-gcop-simulator.cc")} | 8-node bridge validation |
| {ref("scratch/olsr-watchdog-validation.cc")} | 22-node validation |
| {ref("scratch/olsr-watchdog-eval.cc")} | Random-square evaluation |
| {ref("scratch/olsr-watchdog-eval-highway.cc")} | Highway evaluation (matched to Baiad et al.) |
| {ref("scratch/bridge_attack.cc")} | Attack validation |

### NS-3 — batch runners (repository root)
| File | Role |
|---|---|
| {ref("run_simulations.sh")} | **Primary runner** — resumable, parallel; `--defense {{fpnt,watchdog,dcfm}}`, `--random-window-order`, `--mixed-fraction` (orchestrates ordered + shuffled batches into `normal/` + `mixed/`) |
| {ref("run_gcop_base_multi_seeds.py")} | Seed-sweep runner (GCOP base) |
| {ref("run_watchdog_base_multi_seeds.py")} | Seed-sweep runner (Watchdog base) |
| {ref("multi-seed-eval.sh")}, {ref("multi-seed-eval-highway.sh")}, {ref("variance-test.sh")} | Multi-seed evaluation sweeps |

### ML — repository `hananelk26/ML-for-NS3` (branch `main`) *(private — paths shown, not linked)*

**Campaign 1 — the `defense_ml` package** ([Part III](#step-20)), at `{DML}/`:

| File | Role |
|---|---|
| {refml(DML + "/defense_ml/config.py", "config.py")} | **Schema authority** — the 95 features (groups A–K) and every feature set (67/58/18/12/33) with count assertions |
| {refml(DML + "/defense_ml/main.py", "main.py")} | Orchestrator + CLI (`presence_active`, `family_presence`, `--feature-set`, `--drop-features`) |
| {refml(DML + "/defense_ml/cv_harness.py", "cv_harness.py")} | Grouped repeated CV + **hard leakage guard** |
| {refml(DML + "/defense_ml/data_loading.py", "data_loading.py")} | Dataset loading, `build_presence_active`, canonical/mixed policy |
| {refml(DML + "/defense_ml/model_zoo.py", "model_zoo.py")}, {refml(DML + "/defense_ml/feature_prep.py", "feature_prep.py")}, {refml(DML + "/defense_ml/metrics.py", "metrics.py")}, {refml(DML + "/defense_ml/stats.py", "stats.py")} | 13-model zoo · in-fold preprocessing/composites · metrics incl. TPR@FPR · Wilcoxon/Nadeau-Bengio/Friedman-Nemenyi/permutation |
| {refml(DML + "/defense_ml/efficacy.py", "efficacy.py")} | PDR/overhead from the oracle + detectability↔efficacy tradeoff |
| {refml(DML + "/defense_ml/defense_signatures.py", "defense_signatures.py")} | Cliff's-δ ON-vs-OFF signatures, cross-defense heatmap, fingerprints |
| {refml(DML + "/defense_ml/transfer.py", "transfer.py")}, {refml(DML + "/defense_ml/openset.py", "openset.py")} | Train-A/test-B transfer (defense→defense) · Leave-One-Defense-Out novelty (OSCR) |
| {refml(DML + "/defense_ml/selection_audit.py", "selection_audit.py")}, {refml(DML + "/defense_ml/interpret.py", "interpret.py")} | Selection-bias audit S1–S5 (incl. covariate-only placebo) · SHAP |
| {refml(DML + "/RESULTS.md", "RESULTS.md")}, {refml(DML + "/RESEARCH_SUMMARY.md", "RESEARCH_SUMMARY.md")}, {refml(DML + "/RUNBOOK.md", "RUNBOOK.md")} | Front-door numbers · full narrative · operational guide |
| {refml(DML + "/tools/run_dcfm33.py", "tools/run_dcfm33.py")}, {refml(DML + "/tools/greedy_ablation.sh", "greedy_ablation.sh")}, {refml(DML + "/tools/run_external_campaign.py", "run_external_campaign.py")} | 33-schema loader · greedy ablation · external-observer campaign |

*Campaign 1 — result files.* The full `results/` tree is git-tracked under `{DML}/results/`,
reorganised (2026-07-02) into a numbered layer (`results/README.md` maps old → new names):

| Path (under `{DML}/results/`) | Contents → report step |
|---|---|
| `00_summary/` | **Master tables** — `SUMMARY.md`, `ML_results_master.xlsx`, and `tables/` (`detection_master.csv`, `detection_pivot.csv`, `dominant_features_master.csv`, `efficacy_pdr_master.csv`, `trust_tradeoff.csv`, `experiment_index.csv`) |
| `10_core_detection/` | Clean-58 detection + the 95-vs-58 contrast → [Step 21](#step-21), [Step 23](#step-23) |
| `20_observability_ladder/` | The 95→67→18 ladder + external-observer campaign (`external/`, `ladder/`) → [Step 23](#step-23) |
| `30_schema33/` | The 33-schema runs (+ the instructor's `lecturer_v2_verbatim`/`lecturer_v3`, and `paper_v4/` = the v4 default output dir) → [Step 22](#step-22), [Step 24](#step-24) |
| `40_trust_defense/` | Trust 1.0/2.0 binary, `permutation_null/`, `improvement_sweep/` → [Step 24](#step-24) |
| `50_generic_features/` | Generic set + Leave-One-Defense-Out (`lodo_audit/`) → [Step 25](#step-25) |
| `70_publication/` | Transfer / open-set / SHAP / LaTeX campaign (`core/`, `mixed/`, `publication/`) → [Step 25](#step-25) |
| `60_conference_poster/` | Historical poster runs |
| `90_archive/` | Smoke tests — **do not cite** |

Each per-run leaf dir holds the standard `main.py` outputs: `best_per_task.csv`,
`all_fold_metrics.csv`, `summary_<task>.csv`, `class_signatures.csv`, `importance_<task>.csv`,
`permutation_null.csv`, `average_ranks.csv`, `pairwise_vs_reference.csv`, `run_config.json`,
`report.md`/`report.html`, `figures/`, and `signatures/` (`signature_matrix_*.csv`,
`signature_uniqueness_*.csv`, `fingerprint_*.csv`, heatmap PNGs). **Note:** the DCFM and
Watchdog-33 raw run dirs live on the collaborator's machine; their numbers are folded into
`00_summary/` (marked `source=documented/xlsx`).

**Campaign 2 — the `defense_detection_v4` pipeline** ([Part V](#step-27)), at the repo root.

*The pipeline*

| File | Role |
|---|---|
| {refml("defense_detection_v4.py")} | **The pipeline** — consolidation of the instructor's v2 + `defense_ml` (structure and most components from v2; the additions are the statistical-validation layer). 13 models incl. Stacking; multi-criterion selection (MI + ANOVA F + RF + ET); grouped repeated CV with **in-fold** isotonic calibration and threshold tuning; Nadeau-Bengio corrected CIs; `Dummy` floor; TPR@FPR; grouped permutation test; **no SMOTE** ([Step 29](#step-29)). Section `[9] Transfer experiments` ([Step 35](#step-35)) adds three flag-gated generalisation experiments — `--transfer-mobility`, `--transfer-defense`, `--lodo`, plus `--transfer-model` — under a frozen-source-model protocol; off by default, and the CV/statistics core is untouched by them |

*Key scripts in `scripts_for_all_128/` (runs 1–3, Step 34, the Step-36 cross-defense tree, and the Step-37 un-normalised replication)*

| File | Role |
|---|---|
| {refml("scripts_for_all_128/step28_exp1_baseline_32/run_all_defenses.sh", "run_all_defenses.sh")} | Orchestrates **run 1** — 8 conditions × metrics32 → `scripts_for_all_128/step28_exp1_baseline_32/results_run1/` |
| {refml("scripts_for_all_128/step29_exp2_ablation_26/run_behavioral.sh", "run_behavioral.sh")} | Orchestrates **run 2** — ablation `--drop-features` (drop3 → 29, drop6 → 26) |
| {refml("scripts_for_all_128/step30_exp3_expansion_76/run_expanded.sh", "run_expanded.sh")} | Orchestrates **run 3** — 76-feature clean set (`--features-file`) |
| {refml("scripts_for_all_128/step30_exp3_expansion_76/features_clean.txt", "features_clean.txt")} | The generated 76-feature list consumed by run 3 |
| {refml("scripts_for_all_128/step28_exp1_baseline_32/verify_features.py", "verify_features.py")} | Confirms which base features a run actually used (schema sanity check) |
| {refml("scripts_for_all_128/step29_exp2_ablation_26/diagnose_leakage.py", "diagnose_leakage.py")} | Univariate-AUC leakage screen over `metrics32` |
| {refml("scripts_for_all_128/step30_exp3_expansion_76/scan_core95.py", "scan_core95.py")} | Univariate leakage screen over the Core-95 features |
| {refml("scripts_for_all_128/step30_exp3_expansion_76/build_features76.py", "build_features76.py")} | Builds the clean 76-feature list from the Core-95 scan |
| {refml("scripts_for_all_128/step28_exp1_baseline_32/rank_importance.py", "rank_importance.py")} | Importance-concentration diagnostic (flags single-feature dominance) |
| {refml("scripts_for_all_128/step28_exp1_baseline_32/per_defense_tables.py", "per_defense_tables.py")} | All-13-models table per defense, static vs mobile |
| {refml("scripts_for_all_128/step29_exp2_ablation_26/three_metrics.py", "three_metrics.py")} | AUC / MCC / **TPR@1%FPR** across the 32/29/26 runs |
| {refml("scripts_for_all_128/step29_exp2_ablation_26/compare_runs.py", "compare_runs.py")} | 32 vs 29 vs 26 aggregation |
| {refml("scripts_for_all_128/step30_exp3_expansion_76/compare_26_vs_76.py", "compare_26_vs_76.py")} | 26 vs 76 comparison + leakage re-check |
| {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/run_apriori21.sh", "run_apriori21.sh")} | Orchestrates **run 9** — the 21 a-priori features (`--features-file`) across 8 conditions |
| {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/classify_apriori.py", "classify_apriori.py")} | A-priori filter over the 59 green features: observability + implementation-generality → the 21-feature set |
| {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/transfer_test.py", "transfer_test.py")} | Train-one-mobility-regime / test-the-other transfer test (full-set and per-feature) |
| {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/compare_apriori.py", "compare_apriori.py")} | 21 vs 59 vs 26 comparison (AUC / MCC / TPR@1%FPR) |
| {refml("scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/features_apriori_lenient.txt", "features_apriori_lenient.txt")} | The final 21 defense-independent, observable, implementation-general features |
| {refml(S35 + "/preflight.py", "preflight.py")} | **[Step 37](#step-37)** pre-flight: schema check, zero-variance scan, constant-divisor control group, dataset pairing, and `AUC(denominator, defense_enabled)` — the test [Step 32](#step-32) specified |
| {refml(S35 + "/run_dcfm_nonorm.sh", "run_dcfm_nonorm.sh")} | Orchestrates the four un-normalised DCFM runs (27 and 32 features × static/mobile); activates the `defense` conda env and aborts if the boosting libraries are absent |
| {refml(S35 + "/compare_normalized_vs_raw.py", "compare_normalized_vs_raw.py")} | Normalised-vs-raw comparison: best-model table, fixed-anchor table (**HistGB** default), and a config-parity guard |
| {refml(S35 + "/features_27.txt", "features_27.txt")} | The 27-set in `METRICS` order, so `--features-file` reproduces the Step-33 `--drop-features` selection exactly |
| {refml(S35 + "/features_32.txt", "features_32.txt")} | The canonical 32-metric schema, declared explicitly rather than via the preset |
| {refml(S35 + "/feature_name_map.csv", "feature_name_map.csv")} | All 128 columns: raw name, normalised name, whether renamed, the `NORM-001` denominator, and whether that denominator is defense-sensitive |
| {refml(S36 + "/probe_1ch.py", "probe_1ch.py")} | **[Step 38](#step-38)** dataset probe: header/duplicate-column check, per-feature degeneracy and univariate AUC, and base-feature effective dimension — run on both the one-flow and three-flow datasets |
| {refml(S36 + "/preflight.py", "preflight.py")} | **[Step 38](#step-38)** validity gate (seven checks, non-zero exit on failure). Imports v4 and calls its own `engineer_features()` + `FeatureSelector`, so effective dimension is measured on the **engineered** matrix — the figure `run_config.json` omits |
| {refml(S36 + "/run_all.sh", "run_all.sh")} | Orchestrates the six one-flow runs (33/32/27 × static/mobile). Activates the `defense` env, aborts on a missing booster, caps `MAX_JOBS` at 8, and diffs each feature list against `head -1` of the CSV first |
| {refml(S36 + "/compare_1ch_vs_3ch.py", "compare_1ch_vs_3ch.py")} | One-flow vs three-flow vs normalised: best-model and fixed-anchor tables, delta matrix, config parity, top importances |
| {refml(S36 + "/compare_accuracy_prev_vs_cur.py", "compare_accuracy_prev_vs_cur.py")} | Per-model metric deltas against [Step 37](#step-37), any `summary.csv` column via `--metric`. Matches rows **by model name**, never by position |
| {refml(S36 + "/33_features/features_33.txt", "features_33.txt")}, {refml(S36 + "/32_features/features_32.txt", "features_32.txt")}, {refml(S36 + "/27_features/features_27.txt", "features_27.txt")} | The three sets under test, all in `METRICS` order |
| {refml(S36 + "/PREDICTIONS.md", "PREDICTIONS.md")} | Six pre-registered predictions for [Step 38](#step-38), written before any of its learning runs |
| {refml(S34 + "/step34_lodo.py", "step34_lodo.py")} | **Step 36** cross-defense LODO orchestrator: outer LODO + nested-k + worst-rank top-k intersection; `--permute-labels` null |
| {refml(S34 + "/step34_common.py", "step34_common.py")} | Self-contained loaders, run-level split, MI rank matrix, worst-rank selection, metrics (no v4 import) |
| {refml(S34 + "/summarize_step34.py", "summarize_step34.py")} | 8-row results table + static↔mobile core overlap |
| {refml(S34 + "/dominance_check.py", "dominance_check.py")} | Per-fold feature-importance / dominance audit |
| {refml(S34 + "/run_step34_on_27.sh", "run_step34_on_27.sh")} | One-shot runner for the 27-feature experiment (LogReg + RF) |
| {refml(S34 + "/features/features_99_surviving.txt", "features_99_surviving.txt")}, {refml(S34 + "/features/features_27_step32.txt", "features_27_step32.txt")} | The 99 (structure-filter survivors) and the 27 (Step-33 set) under test |

*Result files (git-tracked outputs of the three runs)*

| Directory | Files it contains |
|---|---|
| `scripts_for_all_128/step28_exp1_baseline_32/results_run1/` | `per_defense_all_models.csv`, `importance_overview.csv`, `univariate_auc_matrix.csv`, and 8 condition dirs `{{fpnt,trust,dcfm,watchdog}}_{{static,mobile}}/` each holding `summary.csv`, `folds.csv`, `importance.csv`, `run_config.json`, `summary.tex`, `final_model.pkl`, `figures/` |
| `scripts_for_all_128/step29_exp2_ablation_26/results_run2_behavioral/` | `comparison_32_vs_29_vs_26.csv`, `three_metrics.csv`, `core95_scan.csv`, and `drop3/` + `drop6/` (each with the same 8 condition dirs) |
| `scripts_for_all_128/step30_exp3_expansion_76/results_run3_expanded/` | `compare_26_vs_76.csv`, and 8 condition dirs (same per-condition file set as run 1) |
| `scripts_for_all_128/Step_33_Defense_Independent_Normalization_Features/results_run9_apriori21/` | `apriori_comparison.csv`, the three per-metric pivots, 8 condition dirs (same per-condition file set as run 1), and `transfer_test/` (`full_set_transfer.csv`, `per_feature_transfer.csv`) |
| `{S34}/` (**Step 36** cross-defense LODO) | Per regime (`static/`, `mobile/`) a `heldout_<defense>/` dir with `rank_matrix.csv`, `nested_k_selection.csv`, `selected_features.txt`, `metrics.json`, `roc_curve.csv`, plus `stability.json` and `summary/step34_results.csv`. Five run trees: `results_rf/` (99·RF), `results_rf_noleak/` (99·RF−7 suspects), `results_27/` (27·LogReg), `results_27_rf/` (27·RF), `results_27_rf_PERMUTED/` (permutation null) |
| `{S35}/` (**[Step 37](#step-37)** un-normalised DCFM) | `results_run_nonorm_27/` and `results_run_nonorm_32/`, each with `dcfm_static/` + `dcfm_mobile/` (`summary.csv`, `folds.csv`, `importance.csv`, `run_config.json`, `summary.tex`, `final_model.pkl`, `figures/`, and `permutation_test.json` on 27·static); `preflight_report/` (`scale_comparison_*.csv`, `univariate_by_class_*.csv`, `denominator_direct_*.csv`, `denominator_recovered_*.csv`, `pairing_*.json`); `comparison/` (`comparison_best_model.csv`, `comparison_fixed_HistGB.csv`, `config_parity.csv`) |
| `{DML}/results/30_schema33/paper_v4/transfer/` | **[Step 35](#step-35) generalisation experiments** (written under the v4 default results root, not under `scripts_for_all_128/`): `transfer_mobility.csv`, `transfer_defense_<mobility>.csv`, `lodo_<mobility>.csv`, `transfer_config.json` (reproduction manifest), and `figures/` (`transfer_mobility_<defense>.png`, `transfer_defense_<mobility>.png`, `lodo_<mobility>.png`) |
| `{S36}/` (**[Step 38](#step-38)** one-flow un-normalised DCFM) | `33_features/`, `32_features/`, `27_features/`, each with `results_static/` + `results_mobile/` (`summary.csv`, `folds.csv`, `importance.csv`, `run_config.json`, `summary.tex`, `final_model.pkl`, `figures/`, and `permutation_test.json` on 27·static); `preflight_report/` (`integrity.csv`, `control_group.csv`, `pairing.csv`, `degeneracy_univariate.csv`, `effective_dimension.csv`, `baseline_provenance.csv`, `probe_1ch.log`, `probe_3ch.log`); `comparison/` (`comparison_best_model.csv`, `comparison_fixed_HistGB.csv`, `comparison_matrix.csv`, `config_parity.csv`, `importance_top.csv`, `acc_prev_vs_cur.csv`, `acc_prev_vs_cur_tables.txt`, `whatsapp_tables.txt`); `logs/` |

### Reproduction environment
```bash
conda create -n defense python=3.10 -y
conda activate defense
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib \\
                            seaborn joblib lightgbm catboost -y
conda install -c conda-forge "xgboost>=2.1" -y     # <2.1 breaks calibration
```

**Critical reproduction notes:**
- **Pin `xgboost >= 2.1`.** Version 2.0.3 does not implement scikit-learn 1.7's
  estimator-tag protocol, so `is_classifier(XGBClassifier())` returns `False` and
  probability calibration raises `ValueError: FrozenEstimator should either be a
  classifier ... Got a regressor`.
- Scripts transferred from Windows need `sed -i 's/\\r$//' <file>` (CRLF → LF) under WSL.
- **Every run must print `leakage guard OK` in stage [2] and `Dummy AUC=0.5000` in stage
  [3].** Deviation indicates a protocol fault.
- **Confirm stage [1] reports the expected base-feature count** (32 / 29 / 26 / 76)
  before trusting any run.
- Long runs should execute inside `tmux` to survive disconnection.
""")

# --------------------------------------------------------------------------
# Emit notebook
# --------------------------------------------------------------------------

def to_cell(kind: str, src: str) -> dict:
    lines = src.split("\n")
    source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "name": "Detection and Classification of Black-Hole Defense Mechanisms in OLSR-Based MANETs",
            "toc_visible": True,
            "provenance": [],
        },
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": [to_cell(k, s) for k, s in CELLS],
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({len(CELLS)} cells)")
