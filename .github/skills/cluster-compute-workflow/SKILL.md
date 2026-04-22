---
name: cluster-compute-workflow
description: 'Run nano-vllm-moe tasks on Slurm compute nodes from a login node. Use for srun --jobid entry, A100 fallback submission, conda activation, and full log capture. Trigger words: load this skill, cluster workflow, Slurm, A100, srun, sbatch, 载入skill, 集群计算节点, 登录节点进入计算节点, 保存日志.'
argument-hint: 'Load this skill and run/validate nano-vllm-moe on Slurm compute nodes with full logs'
user-invocable: true
---

# Cluster Compute Workflow

## When to Use

- You are on a login node and need a compute-node shell.
- You must enter an active Slurm job with srun --jobid.
- You need all outputs captured in log files.
- You need an A100 fallback plan when no active job exists.

## Decision Points

1. Is there a running user job for interactive entry?

   - Yes: use srun --jobid=<jobid> --pty bash.
   - No: submit A100 resources via sbatch first, then use srun with the new jobid.

2. Is pytest available in nano_moe?

   - Yes: run python -m pytest for target tests.
   - No: run the smoke test block in this skill.

## Proven Notes From Validation

- Verified in this repository on 2026-04-09.
- Working path: login node -> srun --jobid=15299 --pty bash -> conda activate nano_moe.
- Avoid set -u before conda activate in the compute shell, otherwise conda activation scripts can fail with unbound variables.
- In this environment, pytest is not installed in nano_moe, so use smoke tests unless pytest is added.
- A tcsetattr: Inappropriate ioctl for device line may appear after srun exits in non-interactive capture mode; if SRUN_STATUS=0, treat it as non-fatal.

## Procedure

1. Precheck cluster state.

   squeue -u "$USER"
   sinfo

2. If an interactive compute job exists, enter it with the target jobid.

   srun --jobid=15299 --pty bash

3. In compute node shell, activate runtime and run checks.

   source ~/.bashrc
   conda activate nano_moe
   cd /home/mumura/moe_spec/nano-vllm-moe
   python -c "import torch, nanovllm; print(torch.__version__)"

4. Save all outputs to logs when running from login node.

   mkdir -p /home/mumura/moe_spec/logs
   LOG=/home/mumura/moe_spec/logs/cluster_run_$(date +%Y%m%d_%H%M%S).log
   (your commands) 2>&1 | tee "$LOG"

5. If no active job exists, request A100 resources first (example batch flow).

   sbatch --partition=A100 --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --wrap "sleep infinity"
   squeue -u "$USER"
   srun --jobid=<new_jobid> --pty bash

6. Exit compute shell after all tasks complete.

   exit

## Completion Checks

- You are on a compute host (hostname differs from login host).
- CONDA_DEFAULT_ENV is nano_moe.
- torch and nanovllm import successfully.
- For A100 jobs, cuda_available is True and cuda_device_count is at least 1.
- Final SRUN_STATUS is 0.
- A log file path is printed and the file exists under /home/mumura/moe_spec/logs.

## One-Command Validated Flow

Use the helper script to execute precheck, compute entry, conda activation, and smoke test with full logging:

- Script: [validate_on_compute.sh](./scripts/validate_on_compute.sh)
- Usage:

  bash .github/skills/cluster-compute-workflow/scripts/validate_on_compute.sh
  bash .github/skills/cluster-compute-workflow/scripts/validate_on_compute.sh 15299

## Smoke Test That Worked

python - <<"PY"
import torch
import nanovllm
from nanovllm.sampling_params import SamplingParams

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
sp = SamplingParams(max_tokens=16, temperature=0.8, ignore_eos=False)
print("sampling_params_ok", isinstance(sp, SamplingParams), sp)
print("nanovllm_import_ok", nanovllm.__name__)
PY

## Cluster Documentation

- https://saids.hpc.gleamoe.com/
