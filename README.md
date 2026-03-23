# Mini Mamba Agent 1.58b

This repository contains a PyTorch pipeline for training a BitMamba Small Language Model (SLM) optimized for reasoning, logic, and tool-use, all constrained to fit on a single 24GB consumer GPU (RTX 3090).

Rather than relying on quadratic Self-Attention, which limits context windows and crushes consumer hardware, this architecture merges the linear-time sequence modeling of the Mamba-2 architecture with the extreme parameter efficiency of the BitNet b1.58 paradigm.

**🧠 Architectural Highlights & Hardware Optimizations**

State Space Models are notoriously sensitive to numerical perturbations. A naive application of 1.58-bit ternary quantization directly to the state transition matrices causes the model to suffer from "amnesia". To solve this, our architecture utilizes a Hybrid Precision Allocation strategy:
 * Triton-Accelerated Ternary Projections: The massive, dense linear projection matrices (in_proj, out_proj, x_proj) dominate the layer's memory footprint. These are quantized to strictly \{-1, 0, 1\} using a custom Triton Kernel that fuses the Straight-Through Estimator (STE) directly into the forward pass, eliminating VRAM fragmentation.
 * High-Precision Recurrent Core: The highly sensitive state transition matrix (A) operates exponentially over time and must remain continuous. We strictly isolate the A matrix, the discrete step size (δ), and the input/output state mappings (B, C) in FP16/FP32 precision.
 * Linear Context Scaling (16k+): By using Tri Dao's mamba_inner_fn for the SSD core alongside our ternary projections, we scale the context window to 16,384 tokens with flat VRAM usage, allowing the agent to execute 50+ step tool-use trajectories.
 * State Bleed Prevention: Our 0% padding dataloader uses a custom seq_idx tensor passed directly to the Mamba core to flush the recurrent state at document boundaries, preventing logic contamination.

**🚀 The 3-Phase Training Engine**

*Phase 1: Pre-Training (The Logic Core)*

 * Isolated Multi-Optimizer Routing: 2D ternary weight matrices are routed to the Muon optimizer. The continuous State Space parameters (A, δ) are routed to a dedicated AdamW optimizer with a strictly lower, fixed learning rate (10x smaller) and zero weight decay to prevent the recurrent core from destabilizing during Quantization-Aware Training.
 * 5-Stage FG-WSD Curriculum: Dynamically expands the context window from 2k to 16k tokens only in the final phases of training to organically teach long-horizon linear recurrence.

*Phase 2: Supervised Fine-Tuning (The Reasoning Toggle)*

 * Dynamic System Prompts: Inspired by Nvidia's Nemotron, the dataloader injects "Reasoning ON" and "Reasoning OFF" system prompts, randomly stripping <think> blocks from 30% of the instruction data. This provides a "clutch pedal" to switch between deep Chain-of-Thought reasoning and rapid, direct responses.

*Phase 3: Cascaded Reinforcement Learning (Concise GRPO)*
 * VRAM-Free RL: Utilizes Group Relative Policy Optimization (GRPO) directly on the Actor, avoiding the need to load a separate Critic model into VRAM.
 * Conciseness Penalties: A custom reward function heavily penalizes the model if its thought-to-answer token ratio exceeds 10.0, forcing the discovery of elegant, highly efficient logical paths rather than aimless rambling.

## 📈 The Model Family Workflow
| Model Tier | Parameters | Layers | Dim | Steps | Purpose |
|---|---|---|---|---|---|
| Scout | ~150M | 24 | 512 | 100,000 | Fast validation of the BitMamba block, QAT stability, and pipeline integrity. |
| Parent | ~500M | 40 | 1024 | 1,000,000 | The primary pre-training marathon (~64 Billion tokens). |
| Upscaled | ~770M | 64 | 1024 | N/A | Generated via SOLAR-style layer duplication from the Parent. |

## 📂 Project & Data Structure

Place your downloaded datasets into the local_data folder exactly as mapped below:

```
local_data/
├── train/
│   ├── logic/ (SynLogic, FLDx2)
│   ├── code/ (tiny-codes, tiny-math-textbooks)
│   ├── web/ (fineweb-edu sample-10BT subset)
│   └── tools/ (toolformer-v0-postprocessed)
├── sft/
│   └── Agentic-Chain-of-Thought-Coding-SFT-Dataset
└── rl/
    └── natural_reasoning
```

## 🛠️ Getting Started

### (0. Optional Development Environment)
This repository includes devcontainer scripts for running the container inside Podman on VSCode.
To get Podman working with VSCode devcontainers you have to add this line to your User's settings.json :
```
"dev.container.dockerPath":"podman"
```

For Windows users who utilize WSL, to get GPU passtrough to work inside Podman running on WSL Ubuntu host image, this repository includes a helper setup script to get things set up.


### 1. Installation

Clone the repository and install the strict dependencies (including Triton and Mamba-SSM):
```
pip install torch transformers datasets wandb triton
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

### 2. Build the Custom Tokenizer

Generate the 64k mathematical vocabulary strictly from your local train/ data:

```bash
python train_tokenizer.py
```

### 3. Launch Phase 1 (Pre-Training)

Log in to Weights & Biases (wandb login). Open train.py and set the mode toggle at the top to MODE = "scout", then execute:

```bash
python train.py
```

### 4. Phase 2 & 3 (SFT and RL)

Once the Parent model is trained, execute the fine-tuning pipelines:

```bash
python sft_train.py
python rl_train.py
```

## 📝 License
This project's source code is licensed under the **Apache License 2.0**. See the `LICENSE` file for full details. 

*(Note: Pre-trained model weights, once released, may be subject to a separate commercial-use license).*

