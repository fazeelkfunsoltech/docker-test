import os, subprocess, sys, time, runpod

BURN = r"""
import os, time, torch
x = torch.randn(8192, 8192, device="cuda")
torch.cuda.synchronize(); t = time.time(); n = 0
while time.time() - t < SECONDS:
    x = x @ x; x = x / x.norm(); n += 1
torch.cuda.synchronize()
print(f"GPU {os.environ['CUDA_VISIBLE_DEVICES']} | {torch.cuda.get_device_name(0)} | {n} matmuls")
"""

def handler(job):
    seconds = int(job["input"].get("seconds", 30))
    gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()
    code = BURN.replace("SECONDS", str(seconds))
    start = time.time()
    procs = [subprocess.Popen([sys.executable, "-c", code],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for i in range(len(gpus))]
    results = []
    for p in procs:
        out, err = p.communicate()
        results.append(out.strip() or err.strip()[-300:])
    return {"gpus_seen": gpus, "parallel_results": results,
            "wall_time_s": round(time.time() - start, 1)}

runpod.serverless.start({"handler": handler})