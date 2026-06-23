# EgoSim Incremental Inference

Incremental egocentric world-model inference for multi-clip hand-object interaction video generation.

This repo is the continuous generation subfunction of EgoSim. It uses the same single-clip EgoSim inference code path, metadata format, and **EgoSim-14B** model directory as the main project, then reconstructs and carries an updatable 3D scene state between clips.

## What this repo contains

- **Incremental inference** - generate a sequence of clips from one source video while updating an updatable 3D scene state after each generated clip.
- **Per-clip EgoSim generation** - calls the same standard inference stack used by `egowm/inference/runner.py`.
- **Updatable 3D scene state backend** - runs the bundled modified `egosim_state` backend to reconstruct, align, fuse, and render scene state for the next clip.

## Installation

Requires the main EgoSim inference environment plus a separate scene environment for reconstruction.

```bash
cd continuous_simulation
```

Install the main EgoSim project first by following its `Installation` section. This provides the Python environment used for per-clip EgoSim generation.

Then create the scene reconstruction environment for incremental inference:

```bash
bash scripts/setup_envs.sh
```

The setup script creates `egosim-scene`, which runs EgoSim-state reconstruction, phrase prediction, and updatable 3D scene state rendering.

This subproject lives inside the `egosim-opensource/` checkout. The launcher imports standard EgoSim inference code from that repo root (`../` relative to `continuous_simulation/`).

## Model weights

Download weights from the **egosim-opensource** repo root:

```bash
cd egosim-opensource

# Scene reconstruction weights (required for recon_visualize / full)
bash continuous_simulation/scripts/download_scene_weights.sh

# EgoSim-14B (required for generation; same path as single-clip inference)
huggingface-cli download your-org/EgoSim-14B --local-dir ./EgoSim-14B
```

Expected `EgoSim-14B/` layout (at `egosim-opensource/EgoSim-14B/`):

```text
EgoSim-14B/
├── diffusion_pytorch_model.safetensors
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
└── google/umt5-xxl/
```

`continuous_simulation/configs/project.yaml` points at this directory via `models.model_root: ../EgoSim-14B` (relative to `continuous_simulation/`).

Optional flags: `--with-qwen`, `--with-alternate`, `--with-aot`, `--all`. Run `bash continuous_simulation/scripts/download_scene_weights.sh --help` for details.

Expected layout:

```text
egosim-opensource/
├── EgoSim-14B/
└── artifacts/models/
    ├── priorda/
    ├── sam3/
    ├── depth-anything-3/
    ├── groundingdino/
    ├── bert-base-uncased/
    ├── aot/
    ├── qwen-vl/                 # optional phrase prediction
    ├── unidepth-v2-vitl14/      # alternate scene pipelines only
    ├── video-depth-anything/    # alternate scene pipelines only
    └── moge-2-vitl-normal/      # alternate scene pipelines only
```

`configs/project.yaml` already points at the default paths. See `docs/models.md` for per-model details.

## Data preparation

The incremental metadata extends the main EgoSim CSV format.

Required columns are the same as single-clip inference:

```text
video,ego_prior_video,hand_keypoint_video,first_frame,prompt
```

Incremental reconstruction uses additional columns when available:

```text
task_name,part_idx,process_result_dir,hdf5_path,gt_process_result_dir
```

`video` should encode clip ranges in its filename, for example:

```text
test_3dmem/add_remove_lid/GT_14_0_60.mp4
test_3dmem/add_remove_lid/GT_14_60_120.mp4
```

Clips with the same task and source video id are grouped and processed in temporal order. See `../tests/samples/mini_sample/continuous_generation/metadata.csv` for an example.

The default config expects the continuous generation mini-sample **dataset root** at:

```text
../tests/samples/mini_sample/continuous_generation/process_result/
```

with `metadata.csv` next to it at `../tests/samples/mini_sample/continuous_generation/metadata.csv`. Paths in the CSV may omit the legacy `process_result/` prefix when `data.dataset_root` already points at that folder; older rows that still include `process_result/...` are also accepted.

## Inference

All commands below assume you are inside this repo:

```bash
cd continuous_simulation
```

This repository ships a default config at `configs/project.yaml`. With the standard layout, no manual config edit is needed.

Run a reconstruction visualization smoke test:

```bash
PYTHONPATH=src python -m egowm_incremental.cli run-incremental \
  --config configs/project.yaml \
  --mode recon_visualize
```

Run full incremental generation:

```bash
PYTHONPATH=src python -m egowm_incremental.cli run-incremental \
  --config configs/project.yaml \
  --mode full
```

The resolved backend command can be printed without running inference:

```bash
PYTHONPATH=src python -m egowm_incremental.cli print-command \
  --config configs/project.yaml \
  --mode full
```

Available modes:

- `recon_visualize` - generate the first clip, run scene reconstruction, and optionally open/write visualization outputs.
- `full` - run the multi-clip incremental pipeline with updatable 3D scene state updates.

## License

Apache 2.0
