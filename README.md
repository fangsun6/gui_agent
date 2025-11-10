
# Decision-Centric Fine-Tuning for Visual Language Models (VLMs) in GUI Automation

Modern GUIs are visually rich but *structurally inconsistent* environments. Elements shift, screens differ across devices, and there is no stable "API" for UI state. Traditional RPA relies on template matching or brittle pixel coordinates. Even advanced Vision-Language Models (VLMs) like GPT-4V can **describe** what they see but cannot **act** on the interface with reliability or intent.

This project addresses this gap by reframing GUI automation as a **sequential decision-making problem**, and enabling a **Vision-Language Model to become an agent**, not just a recognizer. Our system combines:

- A **1.3B parameter multimodal VLM** for UI perception
- A **decision-centric offline RL training pipeline** for action reasoning
- **Advantage-weighted policy updates** to learn *why* certain actions lead to success
- **vLLM + DeepSpeed inference/training optimization** for deployment scale

The result: a model that **interprets screen state, chooses the next step, and executes actions**, achieving robust end-to-end performance across real GUI workflows.

---

## 🌍 Core Problem & Motivation

### Why GUI Automation Is Hard
| Challenge | Why It Breaks Traditional Methods |
|---------|-----------------------------------|
| UI layouts shift dynamically | Hard-coded coordinates become invalid |
| Visual similarity across elements | Template matching confuses lookalikes |
| Mixed-modal semantics (icons, text, spatial cues) | Requires reasoning, not just detection |
| Long multi-step workflows | Needs planning, memory, and feedback |

Supervised fine-tuning on GUI demonstrations *only teaches imitation*.  
It cannot teach **why** one action is better than another, especially when:

- Multiple actions *appear reasonable*, but only one progresses the task
- Layout or element order changes between environments
- Observations are noisy (scroll offsets, dialog popups, shadows, etc.)

A **model that *understands what to do next*** must reason in terms of **action value**, not just visual description.

This is why reinforcement learning is necessary.

---

## 🎯 Key Insight

> Vision-Language Models already understand UI semantics.  
> What they *lack* is a decision-making prior.

Instead of training the model to **describe** the UI ("There is a blue Submit button"), we train it to **choose**:
```
Given the current screen,
what *action* most increases the probability of completing the task?
```

This shift is **decision-centric fine-tuning**.

---

## 🧠 System Overview

```
              (1) Screen Observation
                      ↓
          [ Vision-Language Model ]
               (Scene Understanding)
                      ↓
              Action Distribution π(a|s)
                      ↓
      [ Offline PPO with Advantage Weighting ]
         (Reinforce good decisions, suppress bad)
                      ↓
               Environment Executor
              (Click, Scroll, Drag)
```

- The VLM extracts *semantic scene embeddings* (text, icons, structure).
- The RL layer biases the model toward *state-progressing actions*.

---

## 🏋️ Training Pipeline — Full Thinking Flow

### Stage 0 — Collect Demonstration Trajectories
We gather ~700k UI interaction steps from expert or semi-expert operators.

These demonstrations **do not need to be perfect**.  
This is intentional — variability allows the advantage estimator to learn *quality differences* between actions.

### Stage 1 — Supervised Behavioral Cloning (Warm Start)
We first teach the model:
```
Given a screen, predict the human-chosen action.
```
This ensures:
- The VLM learns spatial grounding (what elements correspond to actions)
- The policy begins in a *reasonable region* of action space
- RL does not start from random behavior (avoiding collapse)

**But supervised BC alone overfits screen-position correlations**, not action logic.

### Stage 2 — Decision-Centric Offline RL (Offline PPO)
For each (state, action) we compute an **advantage score**:
```
A(s,a) = Q(s,a) - V(s)
```
This measures whether the chosen action was **better or worse than typical** in that same state.

We then update the policy:
```
π_new(a|s) ∝ π_old(a|s) * exp(A(s,a)/β)
```

Interpretation:
- **Good actions** (positive advantage) are strengthened exponentially
- **Bad actions** are naturally suppressed without forcing discontinuities
- β controls exploration vs. conservativeness (lower β = sharper updates)

This creates a stable **policy improvement cycle** **without any online interaction**.

---

## 📈 Results — What Changed After RL?

| Model Variant | Behavior | Failure Modes | Success Rate |
|--------------|----------|---------------|---------------|
| GPT-4V Prompting | Can describe UI but cannot act | No sequential planning | 8.3% |
| Supervised BC Only | Clicks correct elements sometimes | Overfits layout | 24.3% |
| CogAgent (Vision → Action) | Hard-coded action heuristics | Limited planning | 38.5% |
| **This Work** (BC → Offline PPO) | Understands *progress vs. no-progress* states | Robust across UI variation | **53.8%** |

---

## ⚙️ Repository Structure — Designed for Interpretability

```
data.py                       # Replay buffer + trajectory abstractions
env_utils.py                  # Screen capture + UI environment stubs
labeling.py                   # Demonstration preprocessing + tokenization
offline_rl.py                 # Core advantage-weighted policy update logic
offline_rl_gui_2.py           # Full pipeline (BC + RL training stages)
offline_RL_GUIagent.py        # Action loop for real GUI execution
offline_gui_vllm_deepspeed.py # Scalable inference (zero-copy KV caching)
GUIAgent.md                   # System diagrams & conceptual documentation
```

---

## 🚀 Installation & Execution

```bash
conda create -n gui-agent python=3.10
conda activate gui-agent
pip install -r requirements.txt

# Convert human demonstrations to training tensors
python labeling.py --input trajectories/ --output demos.pt

# Stage 1: Behavioral Cloning
python offline_rl_gui_2.py --stage bc

# Stage 2: Offline PPO (Decision-Centric Fine-Tuning)
python offline_rl_gui_2.py --stage rl

# Run agent live
python offline_RL_GUIagent.py --model checkpoint/
```

---

## 🧭 Future Work

- Incorporate **self-generated counterfactual rollouts**
- Add **hierarchical action abstraction** (macro-actions)
- Vision transformer distillation for **mobile on-device UI agents**

---

## 📩 Contact

This project is actively being extended.
For collaboration, research discussion, or benchmarking access:

```
Maintainer: Fang Sun
Role: Researcher — Decision-Centric VLM Control
```
