# Flow Models

PyTorch implementations of flow-based generative models for toy data.

## Installation

``` bash
git clone https://github.com/twoye0/flowmodels.git
cd flowmodels
pip install -e .
```

## Models

  --------------------------------------------------------------------------------------------------------------------------------------------
  Method                  Paper                                                             Imports & Examples
  ----------------------- ----------------------------------------------------------------- --------------------------------------------------
  iSD                     [Stabilizing Consistency Training: A Flow Map Analysis and        `ISDFlowMap`, `isd_loss` ·
                          Self-Distillation](https://arxiv.org/abs/2601.22679)              [`isd.ipynb`](samples/isd.ipynb)

  FSF-DMD                 [Distribution Matching Distillation without Fake Score            
                          Network](https://arxiv.org/abs/2605.19256)                        

  Beckmann                [Beckmann Transport Models: From Autonomous Flows to One-Step     
                          Maps](https://arxiv.org/abs/2608.01692)                           

  Flux Matching           [Generative Modeling with Flux                                    
                          Matching](https://arxiv.org/abs/2605.07319)                       

  W-Flow                  [One-Step Generative Modeling via Wasserstein Gradient            
                          Flows](https://arxiv.org/abs/2605.11755)                          

  DMD2                    [Improved Distribution Matching Distillation for Fast Image       `Generator`, `dmd_loss` ·
                          Synthesis](https://arxiv.org/abs/2405.14867)                      [`dmd.ipynb`](samples/dmd.ipynb)

  Drifting                [Generative Modeling via                                          
                          Drifting](https://arxiv.org/abs/2602.04770)                       

  FMM                     [Flow Map Matching with Stochastic Interpolants: A Mathematical   `FlowMap`, `flow_map_loss` ·
                          Framework for Consistency                                         [`fmm.ipynb`](samples/fmm.ipynb)
                          Models](https://arxiv.org/abs/2406.07507)                         

  Flow-map SD             [How to Build a Consistency Model: Learning Flow Maps via         `ConsistencyFlowMap`, `consistency_loss` ·
                          Self-Distillation](https://arxiv.org/abs/2505.18825)              [`consistency.ipynb`](samples/consistency.ipynb)

  Score Matching          [Estimation of Non-Normalized Statistical Models by Score         `ScoreNetwork`, `score_matching_loss` ·
                          Matching](https://jmlr.org/papers/v6/hyvarinen05a.html)           [`score.ipynb`](samples/score.ipynb)

  Metric FM               [Metric Flow Matching for Smooth Interpolations on the Data       
                          Manifold](https://arxiv.org/abs/2405.14780)                       

  Platonic                [The Platonic Representation                                      
                          Hypothesis](https://arxiv.org/abs/2405.07987)                     

  Unpaired Rosetta        [Shared Geometry As A Rosetta Stone: Cross-Modal Alignment        
                          Without Paired Data](https://arxiv.org/abs/2610.09411)            

  VCT                     [VCT: Training Consistency Models with Variational Noise          
                          Coupling](https://proceedings.mlr.press/v267/silvestri25a.html)   

  CAF                     [Constant Acceleration Flow](https://arxiv.org/abs/2411.00322)    

  LeJEPA                  [LeJEPA: Provable and Scalable Self-Supervised Learning Without   
                          the Heuristics](https://arxiv.org/abs/2511.08544)                 

  LeWAM                   [LeWAM: A JEPA World Action Model with Diffusion-Steering-Based   
                          MPC](https://arxiv.org/abs/2610.12407)                            
  --------------------------------------------------------------------------------------------------------------------------------------------

## iSD

A minimal 2D **iSD-style** toy implementation based on the flow-map and
self-distillation formulation in *Stabilizing Consistency Training: A
Flow Map Analysis and Self-Distillation*.

The implementation uses the time convention

\[ t = 0 ; `\text{(data)}`{=tex}, `\qquad `{=tex}t = 1 ;
`\text{(noise)}`{=tex}, \]

so generation runs backward from noise to data. `ISDFlowMap` predicts a
flow-map vector (F\_`\theta`{=tex}(x_t; t, s)), and the corresponding
mapped point is

\[ x_s = x_t + (s-t)F\_`\theta`{=tex}(x_t; t,s). \]

Training combines conditional flow matching (CFM) on the diagonal with
an SD-R-style self-distillation regression objective:

\[ `\mathcal{L}`{=tex} = `\mathcal{L}`{=tex}*{`\mathrm{CFM}`{=tex}} +
`\lambda`{=tex}*{`\mathrm{SD}`{=tex}}`\mathcal{L}`{=tex}\_{`\mathrm{SD-R}`{=tex}}.
\]

The SD-R target uses a stop-gradient target and a total derivative of
the flow map along the learned trajectory. The total derivative is
computed with `torch.func.jvp`.

The accompanying toy experiment also keeps a naive two-step
self-distillation baseline for comparison. On the shifted 2D Gaussian
task, the naive self-distillation objective becomes unstable, while the
CFM + SD-R-style objective trains stably and learns a direct one-step
map from noise at (t=1) to the target distribution at (t=0).

``` python
from flowmodels.isd import ISDFlowMap, isd_loss

model = ISDFlowMap()

loss, loss_cfm, loss_sd = isd_loss(
    model,
    batch_size=256,
    lambda_sd=1.0,
)
```

See [`samples/isd.ipynb`](samples/isd.ipynb) for the full toy
experiment, including the naive baselines, training curves, one-step
generation, and off-diagonal flow-map visualization.
