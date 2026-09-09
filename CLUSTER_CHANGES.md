# Cluster-compatible research branch

This branch keeps the released HaloScope algorithm while adding the operational changes
needed for the CCDS Slurm environment:

- generation resumes at the first missing or invalid saved answer;
- threshold search checkpoints after every threshold and restores CPU/CUDA RNG state;
- detection no longer loads unused background-generation label files;
- Llama execution imports the required tracing hooks without importing Baukit's optional
  torchvision-dependent package initializer;
- unused TruthfulQA evaluation imports are disabled, avoiding an unused T5 dependency;
- `run_llama.sbatch` provides separate `generate`, `label`, and `detect` stages within the
  six-hour and 24 GiB cluster limits.

Large local state is intentionally excluded from Git: `.deps`, model links and weights,
logs, generated answers, embeddings, scores, and checkpoints.

## Running on Slurm

The launcher defaults to a sibling `LLM_Haloscope` checkout for Python and the shared
Hugging Face cache. Override those locations when necessary:

```bash
export HALOSCOPE_REIMPLEMENTATION_ROOT=/home/msai/siddhart022/LLM_Haloscope
export HALOSCOPE_PYTHON="$HALOSCOPE_REIMPLEMENTATION_ROOT/.venv/bin/python"

sbatch run_llama.sbatch generate
sbatch run_llama.sbatch label
sbatch run_llama.sbatch detect
```

Submitting a stopped `generate` or `detect` stage again resumes from its checkpoint.
