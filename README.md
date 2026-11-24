# HPMA: Image-Only Prototype-Memory Adaptation of SAM 3.1

2D image-only segmentation pipeline with hierarchical prototype memory. The default dataset is
ISLES-2022, represented as `[DWI, ADC, FLAIR]` image channels. EndoVis loaders remain available.

---

## Architecture Overview (Figure 2)

- **(a) Multi-scale Prototype-Memory Construction**:
  - Offline extraction of FPN features: $72 \times 72$ (Global), $144 \times 144$ (Structural), $288 \times 288$ (Local).
  - Masked Average Pooling (MAP) over ground-truth masks.
  - K-Means clustering ($K=4$) into a strictly frozen memory bank $P \in \mathbb{R}^{3 \times |C| \times 4 \times 256}$.
- **(b) Hierarchical Representation Level-to-Module Coupling**:
  - **Global Visual Level**: Image-derived tokens attend to all global prototypes before fusion.
  - **Structural Component Level**: Object queries use structural prototypes retrieved from image features.
  - **Local Detail Level**: High-resolution feature map $F_0$ ($288 \times 288$) aligned with projected $P_{local}$ via Cosine Distance loss during training (**Zero inference overhead**).
  - **SAM 3.1 Freezing Status**:
    - **Frozen ❄️**: Segmentation Head, Prototype Memory Bank, lower Vision/Fusion layers.
    - **Trainable 🔥**: Decoder, adapters, last 4 Vision Trunk blocks, last 2 Fusion layers.

Inference uses `model(images)`. Labels and masks are used only for memory construction, training losses, and metrics. The image-only path has no text encoder or text prompt.

The ISLES loader canonicalizes orientation, resamples ADC, FLAIR, and masks to the DWI grid as needed, then creates deterministic train/validation/test splits by case (80/10/10 by default). Slices from one case cannot cross splits. Training evaluates validation mDice each epoch and saves the best-validation checkpoint; use the test split only for final reporting after model selection.

---

## Directory Layout

```
├── configs/               # Hyperparameters and dataset configs
├── data/                  # ISLES 2D and EndoVis dataset loaders
├── memory/                # Multi-scale Prototype Memory Bank & K-Means builder
├── models/
│   ├── adapters/          # Global, Structural, and Local adapters (Figure 2c)
│   ├── sam3/              # SAM 3.1 model builder and mock architecture
│   └── hpma_model.py      # Core HPMA network
├── losses/                # SAM3 Focal/Dice loss and Local Alignment loss
├── utils/                 # Segmentation metrics, optimizer, and logger
├── tools/                 # Structural inspector and tensor shape verifiers
├── scripts/               # Bash execution scripts for pipeline
├── tests/                 # Unit test suite
├── build_memory.py        # Prototype bank construction entrypoint
├── train.py               # Training entrypoint
├── eval.py                # Evaluation entrypoint
└── docs/System_Design.md
```

---

## Quick Start

### 1. Install

```bash
python3 -m pip install -e ".[dev]"
```

Training and memory construction require the official `sam3` package and the checkpoint configured at
`model.sam3_checkpoint`. The mock is used only by shape checks; production entry points fail instead of silently training with random mock weights.

### 2. Verification and Tests
```bash
python3 tools/test_tensor_shapes.py
python3 -m unittest discover tests
```

### 3. Build Frozen Memory Bank
```bash
bash scripts/build_memory.sh
```

### 4. Train Model
```bash
bash scripts/train.sh
```

### 5. Evaluate Benchmarks
```bash
bash scripts/evaluate.sh
```
