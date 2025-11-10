# Decision-Centric Fine-Tuning for Visual Language Models (VLMs) in GUI Automation

This repository implements an autonomous GUI Automation Agent that observes the computer screen, interprets UI context using a Vision-Language Model (VLM), and selects the next action using a decision-centric offline reinforcement learning pipeline.

Traditional GUI automation relies on brittle rule scripts or template-based coordinates. Even modern VLMs like GPT-4V can understand UI elements but cannot decide what to do next. This project enables VLMs to act, not just describe.

---

## Key Contributions

- **1.3B Parameter VLM Backbone** for multimodal UI reasoning.
- **Decision-Centric Fine-Tuning** that shifts the VLM from description to action.
- **Offline PPO with Advantage-Weighted Updates**, enabling safe learning without live interaction.
- **Stochastic-Aware Advantage Estimation** for stability under UI layout changes and noise.
- **vLLM + DeepSpeed Integration** for scalable inference and training.
- **53.8% Task Success Rate**, outperforming GPT-4V (8.3%) and CogAgent (38.5%).

---

## Why This Matters

Enterprise software, CRMs, SaaS dashboards, and vendor portals depend heavily on GUIs.  
However:

- Rule-based RPA breaks when UI layouts change.
- Supervised fine-tuning overfits to absolute screen coordinates.
- VLMs without reinforcement learning lack sequential decision reasoning.

This project reframes GUI control as a sequential decision-making problem.

---

## Architecture

Screen Capture → Vision-Language Model (vLLM + DeepSpeed)
↓
Action Distribution
↓
Offline PPO (Advantage-Weighted)
↓
UI Executor

---

## Training Pipeline

### 1) Supervised Warm-Up (BC)
The model learns to imitate expert demonstration trajectories.

### 2) Decision-Centric Offline PPO
The model improves by reweighting high-advantage actions:

π_new(a|s) ∝ π_old(a|s) * exp(A(s,a) / β)

This enables policy optimization **without interacting with live software**.

---

## Results

| Model / Agent | Task Success Rate |
|--------------|------------------|
| **This Work (1.3B VLM + Offline PPO)** | **53.8%** |
| CogAgent (Vision-Action) | 38.5% |
| GPT-4V (Prompted Reasoning) | 8.3% |
| Supervised Baseline Only | 24.3% |

---

## Repository Structure

data.py # Dataset structures and replay buffer
env_utils.py # Screen capture + environment wrappers
labeling.py # Action labeling and preprocessing
offline_rl.py # Advantage-weighted offline PPO
offline_rl_gui_2.py # Main training pipeline
offline_RL_GUIagent.py # Runtime inference agent
offline_gui_vllm_deepspeed.py # VLM acceleration wrapper
GUIAgent.md # System conceptual documentation

---

## Installation

```bash
conda create -n gui-agent python=3.10
conda activate gui-agent
pip install -r requirements.txt
python labeling.py --input trajectories/ --output demos.pt
python offline_rl_gui_2.py --stage bc
python offline_rl_gui_2.py --stage rl
python offline_RL_GUIagent.py --model checkpoint/
