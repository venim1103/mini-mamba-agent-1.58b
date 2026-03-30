# Mini Mamba Agent 1.58b

[![Coverage HTML Report](https://img.shields.io/badge/Coverage-HTML%20Report-0a7ea4?logo=github&logoColor=white)](https://venim1103.github.io/mini-mamba-agent-1.58b/)

This repository contains a PyTorch pipeline for training a BitMamba Small Language Model (SLM) optimized for reasoning, logic, and tool-use, all constrained to fit on a single 12GB–24GB consumer GPU (e.g., RTX 3060 to RTX 3090/4090).

Rather than relying on quadratic Self-Attention, which limits context windows and crushes consumer hardware, this architecture merges the linear-time sequence modeling of the Mamba-2 architecture with the extreme parameter efficiency of the BitNet b1.58 paradigm.

**🧠 Architectural Highlights & Hardware Optimizations**

State Space Models are notoriously sensitive to numerical perturbations. A naive application of 1.58-bit ternary quantization directly to the state transition matrices causes the model to suffer from "amnesia". To solve this, our architecture utilizes a Hybrid Precision Allocation strategy alongside extreme consumer-GPU memory optimizations:
 * Triton-Accelerated Ternary Projections: The massive, dense linear projection matrices (in_proj, out_proj, x_proj) dominate the layer's memory footprint. These are completely untied from the FP32 embeddings and quantized to strictly `{-1, 0, 1}` using a custom Triton Kernel that fuses the Straight-Through Estimator (STE) directly into the forward pass.
 * High-Precision Recurrent Core: The highly sensitive state transition matrix (A) operates exponentially over time and must remain continuous. We strictly isolate the A matrix, the discrete step size (δ), and the input/output state mappings (B, C) in FP16/FP32 precision.
 * Chunked Cross-Entropy & Dynamic Padding: Eliminates the massive `[BS, seq_len, vocab_size]` logits memory bomb. Gradients are computed in chunks with dynamic `valid_tokens` calculation to prevent SFT padding dilution, while custom collators ensure batches are only padded to the longest sequence in that specific batch.
 * Linear Context Scaling (16k+): By using Tri Dao's `mamba_chunk_scan_combined` for the native Mamba-2 SSD core alongside our ternary projections, we scale the context window to 16,384 tokens with flat VRAM usage.
 * Hybrid Mamba-Attention (Nemotron-H §2.1): ~8% of layers are lightweight Grouped-Query Attention blocks (GQA with 4 KV-heads) dispersed evenly among Mamba-2 blocks. This closes the retrieval gap for tool-name/parameter recall from system prompts while preserving the linear-context advantage.
 * Ampere/Ada Optimized: Fully integrates `torch.compile(mode="reduce-overhead")` and FP16 `GradScaler` to double throughput on RTX 3090 architectures.

**🚀 The 3-Phase Training Engine**

*Phase 1: Pre-Training (The Logic Core)*

 * Isolated Multi-Optimizer Routing: 2D ternary weight matrices are routed to the Muon optimizer. The continuous State Space parameters (A, δ) are routed to a dedicated AdamW optimizer with a strictly lower, fixed learning rate (10x smaller).
 * True 4-Phase FG-WSD Curriculum: Implements progressive data quality refinement (Nanbeige4-3B §2.2.2). The learning rate is held flat while the dataset mixture dynamically shifts from web-heavy (Phase 1) to distilled synthetic reasoning (Phase 4). Context is strictly fixed at 8k to stabilize training dynamics, expanding to 16k only in the final decay phase.

*Phase 2: Supervised Fine-Tuning (3-Stage Pipeline)*

 * **Stage 1 — Cold-start:** Exclusively high-quality reasoning data (math CoT, code, science). Multiple epochs at higher LR to establish a strong reasoning baseline and prevent degenerate outputs.
 * **Stage 2 — Mixed:** Introduces general chat alongside reasoning. Dynamic system prompts toggle between "Reasoning ON" and "Reasoning OFF" modes, randomly stripping `<think>` blocks from 30% of samples. This provides a "clutch pedal" to switch between deep CoT reasoning and rapid, direct responses.
 * **Stage 3 — Polish:** Tool-calling and function-use focused fine-tuning with structured output training.

*Phase 3: Cascaded Reinforcement Learning (Concise GRPO)*
 * VRAM-Free RL: Paging optimizer states to the CPU during autoregressive generation (`torch.cuda.empty_cache()`) frees up the exact VRAM needed to run Group Relative Policy Optimization (GRPO) directly on the Actor with `GROUP_SIZE=8`, avoiding a separate Critic model.
 * DAPO-Style PPO Clipping: Omits the heavy KL-divergence penalty in favor of PPO epsilon clipping (`clamp(ratio, 1-EPS, 1+EPS)`), preventing weight destruction on high-advantage batches.
 * On-Policy Difficulty Filtering: Before each training batch, a pre-pass estimates per-problem pass rates and retains only problems in the 10%–90% range, sorting them into an Easy→Hard curriculum.
 * Separated Rewards: Format rewards (tag structure), accuracy rewards, and conciseness penalties (thought-to-answer ratio > 10:1) are computed independently.

## 📈 The Model Family Workflow
| Model Tier | Parameters | Layers | Dim | Steps | Purpose |
|---|---|---|---|---|---|
| Scout | ~77M | 24 (2 attn) | 512 | 100,000 | Fast validation of the BitMamba block, QAT stability, and pipeline integrity. |
| Parent | ~360M | 40 (3 attn) | 1024 | 1,000,000 | The primary pre-training marathon. |
| Upscaled | ~550M | 64 (5 attn) | 1024 | 150,000+ (continued) | Generated via SOLAR-style layer duplication from the Parent and then continued pre-trained (required). |

## 📂 Project & Data Structure

Place your downloaded datasets into the `local_data` folder exactly as mapped below:

```
local_data/
├── train/                          # Pre-training (Phase 1)
│   ├── logic/                      # [HQ] NuminaMath-CoT, FLDx2
│   ├── code/                       # [HQ] tiny-codes
│   ├── web/                        # [MQ] fineweb-edu sample-10BT subset
│   └── tools/                      # [HQ] toolformer-v0-postprocessed
├── sft/                            # Supervised Fine-Tuning (Phase 2)
│   ├── reasoning/                  # Stage 1+2: OpenMathReasoning (CoT split parquets),
│   │                               #   Nemotron-Post-Training (SFT/code + SFT/science),
│   │                               #   OpenR1-Math-220k (default split)
│   ├── mixed/                      # Stage 2: Smol-SmolTalk
│   └── tool_calling/               # Stage 3: APIGen-Function-Calling,
│                                   #   xLAM-Irrelevance-7.5k
└── rl/                             # Reinforcement Learning (Phase 3)
    └── reasoning/                  # OpenMathReasoning (CoT split — has pass_rate_72b_tir)
```

### Recommended Datasets

Download each dataset from its HF Hub ID and place the files in the corresponding directory.

**Pre-training (Phase 1)**

| Directory | Dataset | HF Hub ID | Subset / Split | Size | License |
|---|---|---|---|---|---|
| train/logic | NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | train | 860K | Apache-2.0 |
| train/logic | FLDx2 | `hitachi-nlp/FLDx2` | train | 110K | CC-BY-4.0 (see FLD-corpus repo) |
| train/code | Tiny Codes | `nampdn-ai/tiny-codes` | train | 1.6M | MIT (gated access) |
| train/web | FineWeb-Edu (10BT) | `HuggingFaceFW/fineweb-edu` | sample-10BT | ~10B tokens | ODC-By |
| train/tools | Toolformer | `dmayhem93/toolformer-v0-postprocessed` | train | 2.2K | Not clearly declared on card |

**Supervised Fine-Tuning (Phase 2)**

| Directory | Dataset | HF Hub ID | Subset / Split | Size | License |
|---|---|---|---|---|---|
| sft/reasoning | OpenMathReasoning | `nvidia/OpenMathReasoning` | default / cot | 3.2M CoT solutions | CC-BY-4.0 |
| sft/reasoning | Nemotron Post-Training | `nvidia/Llama-Nemotron-Post-Training-Dataset` | SFT / code + science | 657K code, 709K science | CC-BY-4.0 * |
| sft/reasoning | OpenR1 Math | `open-r1/OpenR1-Math-220k` | default | 94K verified solutions | Apache-2.0 |
| sft/mixed | Smol-SmolTalk | `HuggingFaceTB/smol-smoltalk` | train | 460K | Apache-2.0 |
| sft/tool_calling | APIGen Function Calling | `argilla/apigen-function-calling` | train | 109K | CC-BY-4.0 |
| sft/tool_calling | xLAM Irrelevance | `MadeAgents/xlam-irrelevance-7.5k` | train | 7.5K | CC-BY-4.0 |

**Reinforcement Learning (Phase 3)**

| Directory | Dataset | HF Hub ID | Subset / Split | Size | License |
|---|---|---|---|---|---|
| rl/reasoning | OpenMathReasoning | `nvidia/OpenMathReasoning` | default / cot | 3.2M (has `pass_rate_72b_tir`) | CC-BY-4.0 |

> \* Nemotron Post-Training includes a small subset of responses generated by Llama models subject to the Llama Community License.

### Download Commands

```bash
# Optional: authenticate for higher rate limits
hf auth login

# Pre-training
hf download AI-MO/NuminaMath-CoT --repo-type dataset --local-dir local_data/train/logic/numinamath-cot
hf download hitachi-nlp/FLDx2 --repo-type dataset --local-dir local_data/train/logic/fldx2
hf download nampdn-ai/tiny-codes --repo-type dataset --local-dir local_data/train/code/tiny-codes
hf download HuggingFaceFW/fineweb-edu --repo-type dataset --local-dir local_data/train/web/fineweb-edu --include "sample/10BT/*"
hf download dmayhem93/toolformer-v0-postprocessed --repo-type dataset --local-dir local_data/train/tools/toolformer

# SFT
hf download nvidia/OpenMathReasoning --repo-type dataset --local-dir local_data/sft/reasoning/open-math-reasoning
hf download nvidia/Llama-Nemotron-Post-Training-Dataset --repo-type dataset --local-dir local_data/sft/reasoning/nemotron-post-training --include "SFT/code/*" --include "SFT/science/*"
hf download open-r1/OpenR1-Math-220k --repo-type dataset --local-dir local_data/sft/reasoning/openr1-math --include "default/*"
hf download HuggingFaceTB/smol-smoltalk --repo-type dataset --local-dir local_data/sft/mixed/smol-smoltalk
hf download argilla/apigen-function-calling --repo-type dataset --local-dir local_data/sft/tool_calling/apigen-fc
hf download MadeAgents/xlam-irrelevance-7.5k --repo-type dataset --local-dir local_data/sft/tool_calling/xlam-irrelevance

# RL (prefer symlink to SFT copy)
mkdir -p local_data/rl/reasoning
ln -sfn ../../sft/reasoning/open-math-reasoning local_data/rl/reasoning/open-math-reasoning

# RL (or download separately)
hf download nvidia/OpenMathReasoning --repo-type dataset --local-dir local_data/rl/reasoning/open-math-reasoning
```

## 🛠️ Getting Started

### (0. Optional Development Environment)
This repository includes devcontainer scripts for running the container inside Podman on VSCode.
To get Podman working with VSCode devcontainers you have to add this line to your User's `settings.json`:
```json
"dev.container.dockerPath":"podman"
```

For Windows users who utilize WSL, to get GPU passthrough to work inside Podman running on a WSL Ubuntu host image, this repository includes a helper setup script to get things set up.


### 1. Installation

Clone the repository and install the base dependencies for your environment:
```bash
pip install -r requirements-cpu.txt
```

For CUDA 12.8 environments, install the CUDA overlay instead:
```bash
pip install -r requirements-cuda.txt
```

If you need the full GPU training stack, install the CUDA-specific Mamba dependencies after the CUDA overlay:
```bash
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

### 1.1 Test Coverage (Public HTML on GitHub Pages)

This repo includes a GitHub Actions workflow at `.github/workflows/coverage-pages.yml` that:

- runs tests with coverage on each PR
- uploads the HTML report (`htmlcov/`) as an artifact on each push/PR run
- publishes to GitHub Pages only when manually triggered from the Actions tab on `main`

After enabling Pages once in your GitHub repo settings, anyone can open the latest public report at:

`https://<your-github-username>.github.io/<your-repo-name>/`

If you want to run the same report locally:

```bash
./run_tests.sh
```

Then open `htmlcov/index.html`.

### 2. Build the Custom Tokenizer

Generate the 64k mathematical vocabulary strictly from your local `train/` data:

```bash
python train_tokenizer.py
```

### 3. Launch Phase 1 (Pre-Training)

Log in to Weights & Biases (`wandb login`), then choose a mode via env var (`scout`, `parent`, `upscaled`):

```bash
MODE=scout python train.py
# MODE=parent python train.py
```

For post-upscale continued pre-training (required after running `upscale.py`):

```bash
MODE=upscaled python train.py
```

### 4. Phase 2 & 3 (SFT and RL)

Once the Parent model is trained, execute the fine-tuning pipelines:

```bash
python sft_train.py
python rl_train.py
```

### 5. (Optional) Synthetic Data Augmentation

Generate additional training data from your existing corpus using the trained model. This is strictly required before entering Phase 3/4 of the Fine-Grained WSD pre-training curriculum:

```bash
# Generate QA pairs from web data
python synth_data.py --strategy diverse_qa --input local_data/train/web --output local_data/synth/web_qa

# Distill code into concise examples
python synth_data.py --strategy distill --input local_data/train/code --output local_data/synth/code_distill

# Extract structured knowledge
python synth_data.py --strategy extract --input local_data/train/web --output local_data/synth/web_extract
```

Available strategies: `diverse_qa`, `distill`, `extract`, `knowledge`, `rephrase` (Nemotron-H §2.3).

## 📝 License
This project's source code is licensed under the **Apache License 2.0**. See the `LICENSE` file for full details. 

*(Note: Pre-trained model weights, once released, may be subject to a separate commercial-use license).*
