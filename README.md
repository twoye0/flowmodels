# Flow Models

PyTorch implementations of flow-based generative models for toy data.

## Installation

```bash
git clone https://github.com/twoye0/flowmodels.git
cd flowmodels
pip install -e .
```

## Models

| Method | Paper | Imports & Examples |
|---|---|---|
| iSD | [Stabilizing Consistency Training: A Flow Map Analysis and Self-Distillation](https://arxiv.org/abs/2601.22679) | |
| FSF-DMD | [Distribution Matching Distillation without Fake Score Network](https://arxiv.org/abs/2605.19256) | |
| Beckmann | [Beckmann Transport Models: From Autonomous Flows to One-Step Maps](https://arxiv.org/abs/2608.01692) | |
| Flux Matching | [Generative Modeling with Flux Matching](https://arxiv.org/abs/2605.07319) | |
| W-Flow | [One-Step Generative Modeling via Wasserstein Gradient Flows](https://arxiv.org/abs/2605.11755) | |
| DMD2 | [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867) | `Generator`, `dmd_loss` · [`dmd.ipynb`](samples/dmd.ipynb) |
| Drifting | [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) | |
| FMM | [Flow Map Matching with Stochastic Interpolants: A Mathematical Framework for Consistency Models](https://arxiv.org/abs/2406.07507) | `FlowMap`, `flow_map_loss` · [`fmm.ipynb`](samples/fmm.ipynb) |
| Flow-map SD | [How to Build a Consistency Model: Learning Flow Maps via Self-Distillation](https://arxiv.org/abs/2505.18825) | |
| Score Matching | [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.org/papers/v6/hyvarinen05a.html) | `ScoreNetwork`, `score_matching_loss` · [`score.ipynb`](samples/score.ipynb) |
| Metric FM | [Metric Flow Matching for Smooth Interpolations on the Data Manifold](https://arxiv.org/abs/2405.14780) | |
| Platonic | [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) | |
| Unpaired Rosetta | [Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data](https://arxiv.org/abs/2610.09411) | |
| VCT | [VCT: Training Consistency Models with Variational Noise Coupling](https://proceedings.mlr.press/v267/silvestri25a.html) | |
| CAF | [Constant Acceleration Flow](https://arxiv.org/abs/2411.00322) | |
| LeJEPA | [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544) | |
| LeWAM | [LeWAM: A JEPA World Action Model with Diffusion-Steering-Based MPC](https://arxiv.org/abs/2610.12407) | |
