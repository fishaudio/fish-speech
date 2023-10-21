import os
import subprocess as sp
import sys

SLURM_NTASKS = 6

processes = []
for i in range(SLURM_NTASKS):
    env = os.environ.copy()
    env["SLURM_PROCID"] = str(i)
    env["SLURM_NTASKS"] = str(SLURM_NTASKS)
    env["CUDA_VISIBLE_DEVICES"] = str(i % 8)

    processes.append(
        sp.Popen(
            f"python preparing_data/wenet_clean/clean_wenet_speech.py",
            shell=True,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    )


for p in processes:
    p.wait()
    print(p.communicate())
