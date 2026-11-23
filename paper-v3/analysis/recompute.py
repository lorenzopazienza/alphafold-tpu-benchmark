"""Recompute every number of the AF2 v1 paper from the raw files in the repository.

Reads only committed files (results/, configs/, src/, notebooks at commit faeaa4b via `git show`).
Runs no experiment and imports no project code. Standard library only.

Usage (from the worktree root):
    python3 paper-v3/analysis/recompute.py            # writes paper-v3/analysis/out/recomputed.json and .md
"""
import json
import math
import pathlib
import re
import statistics as st
import subprocess
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper-v3/analysis/out"
R = {}          # key -> value
SRC = {}        # key -> source string


def put(key, value, src):
    R[key] = value
    SRC[key] = src


def load(rel):
    return json.loads((ROOT / rel).read_text())


def git_show(rev, rel):
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{rel}"],
                          capture_output=True, text=True, check=True).stdout


def ts(line):
    """absl log timestamp (I0915 18:04:00.904643) -> seconds since midnight."""
    m = re.match(r"[IWE]\d{4} (\d\d):(\d\d):(\d\d\.\d+)", line.strip())
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def tail_after_predict(lines):
    """Seconds between AlphaFold's model.py:183 log line (predict() returned) and the script's
    'First predict() done' line. Only jax.block_until_ready and the profiler context exit run there."""
    end = next(i for i, l in enumerate(lines) if "First predict() done" in l)
    ret = max(i for i, l in enumerate(lines[:end]) if "model.py:183]" in l)
    return ts(lines[end]) - ts(lines[ret])


def second_tail(lines):
    """Same interval for the first steady-state call (no profiler): model.py:183 -> 'done'."""
    i_done = next(i for i, l in enumerate(lines) if re.search(r"(Second predict\(\) done|Steady-state predict\(\) 1/\d+ done)", l))
    ret = max(i for i, l in enumerate(lines[:i_done]) if "model.py:183]" in l)
    return ts(lines[i_done]) - ts(lines[ret])


def pct(x):
    return round(100 * x, 2)


# ---------------------------------------------------------------- baselines (August)
cpu = load("results/result_cpu-colab.json")
gpu = load("results/result_gpu-t4.json")
tpu = load("results/result_tpu-v5e-podslice.json")
for tag, d in [("cpu", cpu), ("gpu", gpu), ("tpu", tpu)]:
    f = {"cpu": "result_cpu-colab.json", "gpu": "result_gpu-t4.json", "tpu": "result_tpu-v5e-podslice.json"}[tag]
    put(f"B.{tag}.init", d["init_params_seconds"], f"results/{f} -> init_params_seconds")
    put(f"B.{tag}.first", d["first_predict_compile_and_run_seconds"], f"results/{f} -> first_predict_compile_and_run_seconds")
    put(f"B.{tag}.steady", d["second_predict_steady_state_seconds"], f"results/{f} -> second_predict_steady_state_seconds")
put("B.tpu.num_devices", tpu["num_devices"], "results/result_tpu-v5e-podslice.json -> num_devices")
put("B.tpu.has_host_processor_field", "host_processor" in tpu, "results/result_tpu-v5e-podslice.json keys")
c, g, t = R["B.cpu.steady"], R["B.gpu.steady"], R["B.tpu.steady"]
put("B10.gpu_over_cpu", round(c / g, 3), "B.cpu.steady / B.gpu.steady")
put("B11.tpu_over_gpu", round(g / t, 3), "B.gpu.steady / B.tpu.steady")
put("B12.tpu_over_cpu", round(c / t, 2), "B.cpu.steady / B.tpu.steady")
put("B13.cpu_first_over_steady", round(R["B.cpu.first"] / c, 3), "B.cpu.first / B.cpu.steady")
put("B14.gpu_first_over_steady", round(R["B.gpu.first"] / g, 3), "B.gpu.first / B.gpu.steady")
put("B15.tpu_first_over_steady", round(R["B.tpu.first"] / t, 2), "B.tpu.first / B.tpu.steady")
put("B.gpu_init_over_cpu_init", round(R["B.gpu.init"] / R["B.cpu.init"], 3), "B.gpu.init / B.cpu.init")

# ---------------------------------------------------------------- August notebooks: profiler tail and dates
for tag, nbname in [("cpu", "alphafold_cpu_benchmark"), ("gpu", "alphafold_gpu_benchmark")]:
    nb = json.loads(git_show("faeaa4b", f"notebooks/{nbname}.ipynb"))
    lines = []
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            if "text" in o:
                lines += "".join(o["text"]).splitlines()
    log = [l for l in lines if re.match(r"[IWE]\d{4} \d\d:", l.strip())]
    put(f"AUG.{tag}.profiler_tail_s", round(tail_after_predict(log), 3),
        f"git show faeaa4b:notebooks/{nbname}.ipynb, model.py:183 line -> 'First predict() done' line")
    put(f"AUG.{tag}.second_tail_s", round(second_tail(log), 4),
        f"git show faeaa4b:notebooks/{nbname}.ipynb, model.py:183 line -> 'Second predict() done' line")
    first = next(l for l in log)
    put(f"AUG.{tag}.first_log_line_clock", first.strip()[:21], f"git show faeaa4b:notebooks/{nbname}.ipynb, first log line (notebook clock, no time zone)")
    ep = next((re.search(r"I0000 00:00:(\d+\.\d+)", l) for l in lines if re.search(r"I0000 00:00:\d+\.\d+", l)), None)
    if ep:
        put(f"AUG.{tag}.first_epoch_utc", datetime.utcfromtimestamp(float(ep.group(1))).isoformat(timespec="seconds") + "Z",
            f"git show faeaa4b:notebooks/{nbname}.ipynb, first 'I0000 00:00:<epoch>' log line (Unix time, so UTC)")
    numpy_line = next((l for l in lines if "numpy 2." in l), "")
    m = re.search(r"you have numpy (\S+)", numpy_line)
    put(f"AUG.{tag}.numpy", m.group(1) if m else None, f"git show faeaa4b:notebooks/{nbname}.ipynb, pip output")
put("AUG.cpu.first_corrected", round(R["B.cpu.first"] - R["AUG.cpu.profiler_tail_s"], 2), "B.cpu.first - AUG.cpu.profiler_tail_s")
put("AUG.gpu.first_corrected", round(R["B.gpu.first"] - R["AUG.gpu.profiler_tail_s"], 2), "B.gpu.first - AUG.gpu.profiler_tail_s")
put("AUG.cpu.corrected_cold_over_steady", round(R["AUG.cpu.first_corrected"] / c, 3), "AUG.cpu.first_corrected / B.cpu.steady")
put("AUG.gpu.corrected_cold_over_steady", round(R["AUG.gpu.first_corrected"] / g, 3), "AUG.gpu.first_corrected / B.gpu.steady")
put("AUG.cpu.profiler_share_of_first", pct(R["AUG.cpu.profiler_tail_s"] / R["B.cpu.first"]), "tail / first")
put("AUG.gpu.profiler_share_of_first", pct(R["AUG.gpu.profiler_tail_s"] / R["B.gpu.first"]), "tail / first")

# ---------------------------------------------------------------- September reruns
SEPT = [("2026-09-15_cpu-colab", "cpu-colab-repro", "cpu", "0915"),
        ("2026-09-16_cpu-colab", "cpu-colab-repro", "cpu", "0916"),
        ("2026-09-17_cpu-colab", "cpu-colab-repro", "cpu", "0917"),
        ("2026-09-15_gpu-t4", "gpu-t4-repro", "gpu", "0915")]
for folder, tag, kind, day in SEPT:
    base = f"results/repro/{folder}"
    npf = load(f"{base}/result_{tag}_model_3_len118_recycle0_float32_noprofile.json")
    pf = load(f"{base}/result_{tag}_model_3_len118_recycle0_float32.json")
    runs = npf["steady_state_runs_seconds"]
    k = f"SEP.{kind}.{day}"
    put(f"{k}.profile_flag_noprofile_file", npf["profile_first_predict"], f"{base}/..._noprofile.json -> profile_first_predict")
    put(f"{k}.n", len(runs), f"{base}/..._noprofile.json -> steady_state_runs_seconds")
    put(f"{k}.mean", round(st.mean(runs), 4), "mean of steady_state_runs_seconds (recomputed)")
    put(f"{k}.sd", round(st.stdev(runs), 4), "sample sd of steady_state_runs_seconds (recomputed)")
    put(f"{k}.mean_in_file", npf["steady_state_mean_seconds"], f"{base}/..._noprofile.json -> steady_state_mean_seconds")
    put(f"{k}.sd_in_file", npf["steady_state_stdev_seconds"], f"{base}/..._noprofile.json -> steady_state_stdev_seconds")
    put(f"{k}.init_noprofile", npf["init_params_seconds"], f"{base}/..._noprofile.json -> init_params_seconds")
    put(f"{k}.first_noprofile", npf["first_predict_compile_and_run_seconds"], f"{base}/..._noprofile.json -> first_predict_compile_and_run_seconds")
    put(f"{k}.first_profile", pf["first_predict_compile_and_run_seconds"], f"{base}/result_..._float32.json -> first_predict_compile_and_run_seconds")
    put(f"{k}.steady_profile_run", pf["steady_state_mean_seconds"], f"{base}/result_..._float32.json -> steady_state_mean_seconds (n=1)")
    put(f"{k}.first_profile_minus_noprofile", round(pf["first_predict_compile_and_run_seconds"] - npf["first_predict_compile_and_run_seconds"], 2),
        "first_profile - first_noprofile (two processes, same session)")
    lp = (ROOT / f"{base}/log_{tag}_profile.txt").read_text().splitlines()
    ln = (ROOT / f"{base}/log_{tag}_noprofile.txt").read_text().splitlines()
    put(f"{k}.profiler_tail_s", round(tail_after_predict(lp), 3), f"{base}/log_{tag}_profile.txt, model.py:183 -> 'First predict() done'")
    put(f"{k}.noprofile_tail_s", round(tail_after_predict(ln), 4), f"{base}/log_{tag}_noprofile.txt, same interval")
    put(f"{k}.clean_cold_over_steady", round(npf["first_predict_compile_and_run_seconds"] / st.mean(runs), 3), "first_noprofile / mean steady")
    envf = ROOT / f"{base}/environment_{tag}.json"
    env = json.loads(envf.read_text())
    put(f"{k}.repo_commit", env.get("repo_commit"), f"{base}/environment_{tag}.json -> repo_commit")
    put(f"{k}.alphafold_commit", env.get("alphafold_commit"), f"{base}/environment_{tag}.json -> alphafold_commit")
    put(f"{k}.jax", env["package_versions"].get("jax"), f"{base}/environment_{tag}.json -> package_versions.jax")
    put(f"{k}.numpy", env["package_versions"].get("numpy"), f"{base}/environment_{tag}.json -> package_versions.numpy")
    put(f"{k}.cpu_model", env["hardware"].get("cpu_model"), f"{base}/environment_{tag}.json -> hardware.cpu_model")
    put(f"{k}.logical_cpus", env["hardware"].get("logical_cpus"), f"{base}/environment_{tag}.json -> hardware.logical_cpus")
    if kind == "gpu":
        put(f"{k}.nvidia_smi", str(env["hardware"].get("nvidia_smi"))[:200], f"{base}/environment_{tag}.json -> hardware.nvidia_smi")

cpu_means = [R[f"SEP.cpu.{d}.mean"] for d in ("0915", "0916", "0917")]
put("SEP.cpu.spread", round(max(cpu_means) / min(cpu_means), 4), "max/min of the three September CPU means")
put("SEP.cpu.over_aug_min", round(min(cpu_means) / c, 3), "min September CPU mean / B.cpu.steady")
put("SEP.cpu.over_aug_max", round(max(cpu_means) / c, 3), "max September CPU mean / B.cpu.steady")
put("SEP.gpu.aug_over_sep", round(g / R["SEP.gpu.0915.mean"], 3), "B.gpu.steady / SEP.gpu.0915.mean")
put("SEP.days_aug8_to_sep15", (datetime(2026, 9, 15) - datetime(2026, 8, 8)).days, "calendar")

# ---------------------------------------------------------------- TPU repeated runs
rr = load("results/sweep/repeated_runs.json")
for fld in ("init_params_seconds", "first_predict_seconds", "steady_state_seconds"):
    vals = [r[fld] for r in rr["runs"]]
    m, s = st.mean(vals), st.stdev(vals)
    put(f"RR.{fld}.values", vals, f"results/sweep/repeated_runs.json -> runs[].{fld}")
    put(f"RR.{fld}.mean", round(m, 4), "recomputed mean")
    put(f"RR.{fld}.sd", round(s, 4), "recomputed sample sd")
    put(f"RR.{fld}.cv_pct", round(100 * s / m, 3), "recomputed sd/mean")
tm = R["RR.steady_state_seconds.mean"]

# ---------------------------------------------------------------- platform ratios under each baseline choice
for d in ("0915", "0916", "0917"):
    put(f"RATIO.tpu047_over_sep_cpu_{d}", round(R[f"SEP.cpu.{d}.mean"] / t, 1), f"SEP.cpu.{d}.mean / 0.47")
    put(f"RATIO.tpuRR_over_sep_cpu_{d}", round(R[f"SEP.cpu.{d}.mean"] / tm, 1), f"SEP.cpu.{d}.mean / RR steady mean")
put("RATIO.tpu047_over_sep_gpu", round(R["SEP.gpu.0915.mean"] / t, 2), "SEP.gpu.0915.mean / 0.47")
put("RATIO.tpuRR_over_sep_gpu", round(R["SEP.gpu.0915.mean"] / tm, 2), "SEP.gpu.0915.mean / RR steady mean")
put("RATIO.tpuRR_over_aug_cpu", round(c / tm, 1), "B.cpu.steady / RR steady mean")
put("RATIO.tpuRR_over_aug_gpu", round(g / tm, 2), "B.gpu.steady / RR steady mean")
put("RATIO.sep_gpu_over_sep_cpu_0915", round(R["SEP.cpu.0915.mean"] / R["SEP.gpu.0915.mean"], 2), "same-day September CPU / GPU")
allcpu = [c] + cpu_means
allgpu = [g, R["SEP.gpu.0915.mean"]]
tpu_all = [t] + [r["steady_state_seconds"] for r in rr["runs"]]
put("RANGE.tpu_over_cpu_min", round(min(allcpu) / max(tpu_all), 0), "min CPU session / max TPU value")
put("RANGE.tpu_over_cpu_max", round(max(allcpu) / min(tpu_all), 0), "max CPU session / min TPU value")
put("RANGE.tpu_over_gpu_min", round(min(allgpu) / max(tpu_all), 1), "min GPU session / max TPU value")
put("RANGE.tpu_over_gpu_max", round(max(allgpu) / min(tpu_all), 1), "max GPU session / min TPU value")
# all TPU single-query toy-sequence steady states (same script, 118 res, recycle 0, float32, model_3)
cv = load("results/sweep/chip_visibility_sweep.json")
pr = load("results/sweep/precision_sweep.json")
mc = load("results/sweep/model_comparison.json")
rd = load("results/sweep/recycle_depth_sweep.json")
cc = load("results/sweep/compilation_cache.json")
tpu_steadies = {"baseline": t, "RR1": rr["runs"][0]["steady_state_seconds"], "RR2": rr["runs"][1]["steady_state_seconds"],
                "RR3": rr["runs"][2]["steady_state_seconds"], "CV1": cv[0]["steady_state_seconds"], "CV8": cv[1]["steady_state_seconds"],
                "PR_fp32": pr[0]["steady_state_seconds"], "MC_model3": mc[0]["steady_state_seconds"], "RD0": rd[0]["steady_state_seconds"],
                "CC_cold": cc["cold"]["steady_state_seconds"], "CC_warm": cc["warm"]["steady_state_seconds"]}
put("TPU.single_query_steady_all", tpu_steadies, "every model_3/118/recycle0/float32 single-query TPU steady state in results/")
put("TPU.single_query_steady_min_max", [min(tpu_steadies.values()), max(tpu_steadies.values())], "min, max of the above")
put("TPU.single_query_steady_n", len(tpu_steadies), "count")
firsts = {"baseline": tpu["first_predict_compile_and_run_seconds"], "RR1": rr["runs"][0]["first_predict_seconds"], "RR2": rr["runs"][1]["first_predict_seconds"],
          "RR3": rr["runs"][2]["first_predict_seconds"], "CV1": cv[0]["first_predict_seconds"], "CV8": cv[1]["first_predict_seconds"],
          "PR_fp32": pr[0]["first_predict_seconds"], "MC_model3": mc[0]["first_predict_seconds"], "RD0": rd[0]["first_predict_seconds"], "CC_cold": cc["cold"]["first_predict_seconds"]}
put("TPU.single_query_first_all", firsts, "every cold first predict (profiler on) of the same configuration")
put("TPU.single_query_first_min_max", [min(firsts.values()), max(firsts.values())], "min, max")

# ---------------------------------------------------------------- chip visibility, precision, model, recycle, length
put("CV.1.steady", cv[0]["steady_state_seconds"], "chip_visibility_sweep.json -> [0].steady_state_seconds")
put("CV.8.steady", cv[1]["steady_state_seconds"], "chip_visibility_sweep.json -> [1].steady_state_seconds")
put("CV.8.hbm_in_use_bytes", [m["bytes_in_use"] for m in cv[1]["memory_stats_per_device"]], "chip_visibility_sweep.json -> [1].memory_stats_per_device[].bytes_in_use")
put("CV.8.chips_nonzero", sum(1 for m in cv[1]["memory_stats_per_device"] if m["bytes_in_use"] > 0), "count of nonzero bytes_in_use")
put("CV.bytes_limit", cv[1]["memory_stats_per_device"][0]["bytes_limit"], "chip_visibility_sweep.json -> bytes_limit")
put("CV.all_bytes_multiple_of_1e6", all(m["bytes_in_use"] % 1_000_000 == 0 and m["peak_bytes_in_use"] % 1_000_000 == 0 for e in cv for m in e["memory_stats_per_device"]),
    "every memory value in chip_visibility_sweep.json is a whole number of MB (transcribed from the printed MB lines)")
put("PR.d_first_pct", pct(pr[1]["first_predict_seconds"] / pr[0]["first_predict_seconds"] - 1), "precision_sweep.json")
put("PR.d_steady_pct", pct(pr[1]["steady_state_seconds"] / pr[0]["steady_state_seconds"] - 1), "precision_sweep.json")
put("PR.d_hbm_pct", pct(pr[1]["hbm_bytes_in_use"] / pr[0]["hbm_bytes_in_use"] - 1), "precision_sweep.json")
put("PR.d_peak_pct", pct(pr[1]["hbm_peak_bytes_in_use"] / pr[0]["hbm_peak_bytes_in_use"] - 1), "precision_sweep.json")
put("MC.model5_vs_model3_pct", pct(mc[2]["steady_state_seconds"] / mc[0]["steady_state_seconds"] - 1), "model_comparison.json")
put("RD.factor_1", round(rd[1]["steady_state_seconds"] / rd[0]["steady_state_seconds"], 3), "recycle_depth_sweep.json")
put("RD.factor_3", round(rd[2]["steady_state_seconds"] / rd[0]["steady_state_seconds"], 3), "recycle_depth_sweep.json")
put("RD.per_recycle_s", round((rd[2]["steady_state_seconds"] - rd[0]["steady_state_seconds"]) / 3, 4), "(rec3 - rec0)/3")
put("RD.first", [r["first_predict_seconds"] for r in rd], "recycle_depth_sweep.json -> first_predict_seconds (profiler on)")
sl = load("results/sweep/sequence_length_sweep.json")
put("SL.steady", {r["num_residues"]: r["steady_state_seconds"] for r in sl}, "sequence_length_sweep.json")
put("SL.first", {r["num_residues"]: r["first_predict_seconds"] for r in sl}, "sequence_length_sweep.json (profiler on)")
xs = [math.log(r["num_residues"]) for r in sl]
ys = [math.log(r["steady_state_seconds"]) for r in sl]
mx, my = st.mean(xs), st.mean(ys)
k = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
put("SL.exponent_k", round(k, 3), "OLS slope of log(steady) on log(length), 4 points")
put("SL.time_factor_500_vs_60", round(sl[3]["steady_state_seconds"] / sl[0]["steady_state_seconds"], 2), "sequence_length_sweep.json")

# ---------------------------------------------------------------- compilation cache and trace
put("CC.speedup_init", round(cc["cold"]["init_params_seconds"] / cc["warm"]["init_params_seconds"], 3), "compilation_cache.json cold/warm")
put("CC.speedup_first", round(cc["cold"]["first_predict_seconds"] / cc["warm"]["first_predict_seconds"], 3), "compilation_cache.json cold/warm")
put("CC.combined_cold", round(cc["cold"]["init_params_seconds"] + cc["cold"]["first_predict_seconds"], 2), "init+first, cold")
put("CC.combined_warm", round(cc["warm"]["init_params_seconds"] + cc["warm"]["first_predict_seconds"], 2), "init+first, warm")
put("CC.combined_reduction_pct", pct(1 - R["CC.combined_warm"] / R["CC.combined_cold"]), "1 - warm/cold")
put("CC.first_reduction_pct", pct(1 - cc["warm"]["first_predict_seconds"] / cc["cold"]["first_predict_seconds"]), "1 - warm/cold")
put("TR.share_pct", pct(12.55 / 16.56), "profiling/trace_analysis.md:30-31 (reported, no raw trace)")

# ---------------------------------------------------------------- vmap
bt = load("results/sweep/batching_sweep.json")
for b in bt:
    put(f"VM.B{b['batch_size']}.thr_file", b["throughput_proteins_per_sec"], "batching_sweep.json")
    put(f"VM.B{b['batch_size']}.thr_recomputed", round(b["batch_size"] / b["steady_state_seconds"], 3), "batch/steady")
    put(f"VM.B{b['batch_size']}.rel_B1", round(b["throughput_proteins_per_sec"] / bt[0]["throughput_proteins_per_sec"], 4), "thr / thr(B=1)")
    put(f"VM.B{b['batch_size']}.hbm_tpu0", b["hbm_tpu0_mb"], "batching_sweep.json -> hbm_tpu0_mb")

# ---------------------------------------------------------------- pmap multi-query, auto-mesh, ensemble
sh = load("results/sweep/sharding.json")
pm, gs = sh["pmap_data_parallelism"], sh["mesh_auto_sharding"]
put("MQ.baseline_s", pm["single_protein_baseline_seconds"], "sharding.json -> pmap_data_parallelism.single_protein_baseline_seconds")
put("MQ.steady_s", pm["pmap_8chips_steady_state_seconds"], "sharding.json -> pmap_8chips_steady_state_seconds")
put("MQ.thr", round(8 / pm["pmap_8chips_steady_state_seconds"], 3), "8 / steady")
put("MQ.thr_file", pm["throughput_proteins_per_sec"], "sharding.json -> throughput_proteins_per_sec")
put("MQ.baseline_thr", round(1 / pm["single_protein_baseline_seconds"], 3), "1 / baseline")
put("MQ.speedup", round(R["MQ.thr"] / R["MQ.baseline_thr"], 3), "thr / baseline_thr")
put("MQ.speedup_file", pm["speedup_vs_single_protein_baseline"], "sharding.json -> speedup_vs_single_protein_baseline")
put("MQ.efficiency_pct", pct(R["MQ.speedup"] / 8), "speedup / 8")
put("MQ.mem", pm["memory_per_chip_mb"], "sharding.json -> memory_per_chip_mb")
put("MQ.speedup_vs_vmapB1", round(R["MQ.thr"] / bt[0]["throughput_proteins_per_sec"], 3), "pmap thr / vmap B=1 thr (both rotated alphabet, both time apply)")
put("GS.run1", gs["run_1"], "sharding.json -> mesh_auto_sharding.run_1")
put("GS.run2", gs["run_2"], "sharding.json -> mesh_auto_sharding.run_2")
put("GS.verdict_text", gs["verdict"], "sharding.json -> mesh_auto_sharding.verdict (hand-written)")
put("GS.description_text", gs["description"], "sharding.json -> mesh_auto_sharding.description (hand-written)")
en = load("results/sweep/ensemble_shard.json")
put("EN.first", en["first_compile_and_run_seconds"], "ensemble_shard.json")
put("EN.steady", en["second_steady_state_seconds"], "ensemble_shard.json")
put("EN.chips_nonzero", en["num_chips_with_nonzero_memory"], "ensemble_shard.json")
put("EN.pmean_check", en["pmean_reduction_consistent_across_chips"], "ensemble_shard.json")
put("EN.mem_in_use", [m["bytes_in_use_mb"] for m in en["memory_stats_per_device_mb"]], "ensemble_shard.json")
put("EN.mem_peak", [m["peak_mb"] for m in en["memory_stats_per_device_mb"]], "ensemble_shard.json")

# ---------------------------------------------------------------- scaling grid and fit
grid = load("results/sweep/scaling_law_data.json")
fit_file = grid["fitted_power_law"]
pts = grid["grid"]
X = [[1.0, math.log(p["chips"]), math.log(p["length"])] for p in pts]
Y = [math.log(p["throughput_per_sec"]) for p in pts]


def solve3(A, b):
    M = [row[:] + [v] for row, v in zip(A, b)]
    n = 3
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]


XtX = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
XtY = [sum(x[i] * y for x, y in zip(X, Y)) for i in range(3)]
b0, b1, b2 = solve3(XtX, XtY)
pred = [b0 + b1 * x[1] + b2 * x[2] for x in X]
ss_res = sum((y - p) ** 2 for y, p in zip(Y, pred))
ss_tot = sum((y - st.mean(Y)) ** 2 for y in Y)
put("SC.fit_prefactor", round(math.exp(b0), 2), "OLS in log space on scaling_law_data.json -> grid (16 points)")
put("SC.fit_chip_exp", round(b1, 4), "OLS")
put("SC.fit_len_exp", round(b2, 4), "OLS")
put("SC.fit_r2", round(1 - ss_res / ss_tot, 4), "OLS, log-log R^2")
put("SC.fit_in_file", fit_file, "scaling_law_data.json -> fitted_power_law")
# leave-one-length-out style checks: fit without chips (length only) and separate chip exponent per length
res = {}
for p, pr_ in zip(pts, pred):
    res[f"{p['chips']}x{p['length']}"] = round(math.exp(pr_) / p["throughput_per_sec"], 3)
put("SC.fit_over_measured", res, "exp(fit)/measured per grid point")
thr = {(p["chips"], p["length"]): p["throughput_per_sec"] for p in pts}
lengths = [100, 250, 500, 1000]
for L in lengths:
    put(f"SC.speedup_8over1_L{L}", round(thr[(8, L)] / thr[(1, L)], 3), "measured grid")
    put(f"SC.speedup_4over1_L{L}", round(thr[(4, L)] / thr[(1, L)], 3), "measured grid")
    put(f"SC.speedup_2over1_L{L}", round(thr[(2, L)] / thr[(1, L)], 3), "measured grid")
    put(f"SC.eff_8_L{L}_pct", pct(thr[(8, L)] / thr[(1, L)] / 8), "measured grid")
    # per-length chip exponent (OLS of log thr on log chips)
    xs_ = [math.log(cn) for cn in (1, 2, 4, 8)]
    ys_ = [math.log(thr[(cn, L)]) for cn in (1, 2, 4, 8)]
    mx_, my_ = st.mean(xs_), st.mean(ys_)
    put(f"SC.chip_exp_L{L}", round(sum((a - mx_) * (b - my_) for a, b in zip(xs_, ys_)) / sum((a - mx_) ** 2 for a in xs_), 4), "per-length OLS slope")
    for cn in (1, 2, 4, 8):
        put(f"SC.cost_c{cn}_L{L}", round(cn * 1.20 / (thr[(cn, L)] * 3600) * 1000, 4), "chips x $1.20 / (thr x 3600) x 1000")
    put(f"SC.cost_increase_1to8_L{L}_pct", pct(8 * thr[(1, L)] / thr[(8, L)] - 1), "cost(8)/cost(1)-1 from unrounded throughputs")
# length exponent per chip count
for cn in (1, 2, 4, 8):
    xs_ = [math.log(L) for L in lengths]
    ys_ = [math.log(thr[(cn, L)]) for L in lengths]
    mx_, my_ = st.mean(xs_), st.mean(ys_)
    put(f"SC.len_exp_c{cn}", round(sum((a - mx_) * (b - my_) for a, b in zip(xs_, ys_)) / sum((a - mx_) ** 2 for a in xs_), 4), "per-chip OLS slope")
put("SC.thr_8_L100", thr[(8, 100)], "grid")
put("SC.thr_8_L1000", thr[(8, 1000)], "grid")
put("SC.thr_1_L100", thr[(1, 100)], "grid")

# ---------------------------------------------------------------- cost (inputs from results/cost_analysis.md)
price_tpu_chip, price_gpu, price_cpu = 1.20, 0.35, 0.19
pod = 8 * price_tpu_chip
for tag, s in [("cpu", c), ("gpu", g), ("tpu", t)]:
    put(f"COST.{tag}.pred_per_h", round(3600 / s, 2), "3600 / steady")
put("COST.cpu.per_1k", round(price_cpu / (3600 / c) * 1000, 4), "$0.19/h")
put("COST.gpu.per_1k", round(price_gpu / (3600 / g) * 1000, 4), "$0.35/h")
put("COST.pod.per_1k", round(pod / (3600 / t) * 1000, 4), "$9.60/h")
put("COST.chip.per_1k", round(price_tpu_chip / (3600 / t) * 1000, 4), "$1.20/h hypothetical one chip")
put("COST.gpu_over_chip", round(R["COST.gpu.per_1k"] / R["COST.chip.per_1k"], 3), "")
put("COST.pod_over_gpu_hourly", round(pod / price_gpu, 3), "")
put("COST.pmap_pod_per_1k", round(pod / (R["MQ.thr"] * 3600) * 1000, 4), "pod / (MQ thr x 3600) x 1000")
put("COST.pmap_over_chip", round(R["COST.pmap_pod_per_1k"] / R["COST.chip.per_1k"], 3), "")
put("COST.gpu_sep.per_1k", round(price_gpu / (3600 / R["SEP.gpu.0915.mean"]) * 1000, 4), "September GPU mean")
put("COST.pod_over_gpu_sep", round(R["COST.pod.per_1k"] / R["COST.gpu_sep.per_1k"], 3), "")
put("COST.cpu_sep_per_1k_range", [round(price_cpu / (3600 / m) * 1000, 2) for m in cpu_means], "September CPU means")

# ---------------------------------------------------------------- inputs per experiment (sequence family), with script lines
def line_of(rel, pattern):
    for i, l in enumerate((ROOT / rel).read_text().splitlines(), 1):
        if re.search(pattern, l):
            return f"{rel}:{i}"
    return f"{rel}: pattern not found"


toy = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV" + "KALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWD"
put("INPUT.toy_length", len(toy), "src/spike_tpu_forward_pass.py:107-110")
inputs = {
    "baseline CPU/GPU (August)": ("fixed 118-residue sequence (TOY_SEQUENCE)", "notebook copy, git show faeaa4b:notebooks/alphafold_cpu_benchmark.ipynb cell 8 (TOY_SEQUENCE, NUM_RES = len(TOY_SEQUENCE))"),
    "baseline TPU, chip visibility, precision, model comparison, repeated runs, cache, trace": ("fixed 118-residue sequence", line_of("src/spike_tpu_forward_pass.py", r"if num_residues == len\(TOY_SEQUENCE_118\)") + "; configs pass --num_residues=118 or the default 118 (" + line_of("src/spike_tpu_forward_pass.py", r'"num_residues", 118') + ")"),
    "September CPU/GPU reruns": ("fixed 118-residue sequence", "environment_*.json -> flags.common (--num_residues=118); " + line_of("src/spike_tpu_forward_pass.py", r"if num_residues == len\(TOY_SEQUENCE_118\)")),
    "recycle-depth sweep": ("fixed 118-residue sequence", line_of("configs/af_spike_job_sweep.yaml", r"--num_residues=118 --num_recycle=\$RECYCLE")),
    "sequence-length sweep (60, 120, 250, 500)": ("alphabet ACDEFGHIKLMNPQRSTVWY repeated to the length (no length equals 118)", line_of("src/spike_tpu_forward_pass.py", r"return \(_AA_ALPHABET \*") + "; " + line_of("configs/af_spike_job_sweep.yaml", r"for LEN in 60 120 250 500")),
    "vmap batching (B = 1, 2, 4, 8; 118 residues)": ("alphabet repeated to 118, rotated by item index", line_of("src/spike_batch_forward_pass.py", r"build_sequence\(num_res, offset=i\)") + "; " + line_of("src/spike_batch_forward_pass.py", r"alphabet = _AA_ALPHABET\[offset")),
    "pmap multi-query (8 queries, 118 residues)": ("alphabet repeated to 118, rotated by item index", line_of("src/spike_pmap_forward_pass.py", r"build_sequence\(num_res, offset=i\)") + "; " + line_of("src/spike_pmap_forward_pass.py", r"alphabet = _AA_ALPHABET\[offset")),
    "scaling grid (chips 1-8 x length 100-1000)": ("alphabet repeated to the length, rotated by item index", line_of("configs/af_spike_scaling_grid.yaml", r"spike_pmap_forward_pass.py") + "; " + line_of("src/spike_pmap_forward_pass.py", r"build_sequence\(num_res, offset=i\)")),
    "auto-mesh (118 residues)": ("fixed 118-residue sequence", line_of("src/spike_meshshard_forward_pass.py", r"if num_residues == len\(TOY_SEQUENCE_118\)")),
    "external ensemble (118 residues, 8 seeds)": ("fixed 118-residue sequence, featurized with seeds 0-7", line_of("src/spike_ensemble_shard_forward_pass.py", r"sequence = TOY_SEQUENCE_118 if num_res == 118") + "; " + line_of("src/spike_ensemble_shard_forward_pass.py", r"random_seed=i\)")),
}
put("INPUT.per_experiment", inputs, "script lines")
timed = {
    "single-query script": line_of("src/spike_tpu_forward_pass.py", r"result2 = runner.predict") + " (predict: forward pass + host-side confidence metrics)",
    "pmap": line_of("src/spike_pmap_forward_pass.py", r"result2 = pmapped_apply") + " (apply only)",
    "vmap": line_of("src/spike_batch_forward_pass.py", r"result2 = batched_apply") + " (apply only)",
    "profiler on first call": line_of("src/spike_tpu_forward_pass.py", r"with \(jax.profiler.trace") + "; " + line_of("src/spike_batch_forward_pass.py", r"with jax.profiler.trace"),
}
put("TIMED.region", timed, "script lines")

# ---------------------------------------------------------------- experiment count
data_files = {
    "baseline CPU/GPU/TPU": ["results/result_cpu-colab.json", "results/result_gpu-t4.json", "results/result_tpu-v5e-podslice.json"],
    "sequence length": ["results/sweep/sequence_length_sweep.json"],
    "recycle depth": ["results/sweep/recycle_depth_sweep.json"],
    "chip visibility": ["results/sweep/chip_visibility_sweep.json"],
    "precision": ["results/sweep/precision_sweep.json"],
    "vmap batching": ["results/sweep/batching_sweep.json"],
    "compilation cache": ["results/sweep/compilation_cache.json"],
    "model comparison": ["results/sweep/model_comparison.json"],
    "repeated runs": ["results/sweep/repeated_runs.json"],
    "profiler trace": [],
    "pmap multi-query": ["results/sweep/sharding.json (pmap_data_parallelism)"],
    "auto-mesh": ["results/sweep/sharding.json (mesh_auto_sharding)"],
    "external ensemble": ["results/sweep/ensemble_shard.json"],
    "scaling grid": ["results/sweep/scaling_law_data.json"],
}
put("COUNT.experiments_with_data", {k: v for k, v in data_files.items()}, "listing")
put("COUNT.tpu_experiments_with_raw_data", sum(1 for k, v in data_files.items() if v and k != "baseline CPU/GPU/TPU"), "TPU experiments with a data file")
put("COUNT.readme_numbered_items", 12, "results/sweep/README.md mermaid nodes 1-12 (trace analysis is node 9)")

# ---------------------------------------------------------------- values that the text of version 3 needs (added 2026-11-23)
# One sentence on the minimum over all sessions, against the TPU baseline run.
gpu_sessions = [R["B.gpu.steady"], R["SEP.gpu.0915.mean"]]
cpu_sessions = [R["B.cpu.steady"]] + cpu_means
put("V2.min_tpu_over_t4_all_sessions", round(min(gpu_sessions) / R["B.tpu.steady"], 3), "min over T4 sessions (Aug 13.086, Sep 6.5672) / TPU baseline 0.47")
put("V2.min_tpu_over_cpu_all_sessions", round(min(cpu_sessions) / R["B.tpu.steady"], 2), "min over CPU sessions (Aug 212.113, Sep 348.861/351.8907/359.2726) / TPU baseline 0.47")
put("V2.min_tpu_over_t4_with_max_tpu", round(min(gpu_sessions) / max(R["TPU.single_query_steady_min_max"]), 3), "same, against the slowest single-query TPU value (0.471)")
put("V2.min_tpu_over_cpu_with_max_tpu", round(min(cpu_sessions) / max(R["TPU.single_query_steady_min_max"]), 2), "same, against the slowest single-query TPU value (0.471)")
put("V2.sep_cpu_steady_min", min(cpu_means), "min of the three September CPU means")
put("V2.sep_cpu_steady_max", max(cpu_means), "max of the three September CPU means")
sep_first = [R[f"SEP.cpu.{d}.first_noprofile"] for d in ("0915", "0916", "0917")]
put("V2.sep_cpu_first_min", min(sep_first), "min first call, September CPU, profiler off")
put("V2.sep_cpu_first_max", max(sep_first), "max first call, September CPU, profiler off")
sep_tails = [R[f"SEP.cpu.{d}.profiler_tail_s"] for d in ("0915", "0916", "0917")]
put("V2.sep_cpu_tail_min", min(sep_tails), "min profiler teardown, September CPU (profile runs)")
put("V2.sep_cpu_tail_max", max(sep_tails), "max profiler teardown, September CPU (profile runs)")
put("V2.sep_dates", ["2026-09-15", "2026-09-16", "2026-09-17"], "results/repro/<date>_<platform>/ folder names")
put("V2.tpu_first_n", len(R["TPU.single_query_first_all"]), "count of TPU.single_query_first_all")
put("V2.mq_mem_min", min(R["MQ.mem"]), "min of sharding.json -> memory_per_chip_mb")
put("V2.mq_mem_max", max(R["MQ.mem"]), "max of sharding.json -> memory_per_chip_mb")
put("V2.en_mem_min", min(R["EN.mem_in_use"]), "min of ensemble_shard.json -> bytes_in_use_mb")
put("V2.en_mem_max", max(R["EN.mem_in_use"]), "max of ensemble_shard.json -> bytes_in_use_mb")
fr = R["SC.fit_over_measured"]
put("V2.fit_over_measured_min", min(fr.values()), "min of SC.fit_over_measured")
put("V2.fit_over_measured_max", max(fr.values()), "max of SC.fit_over_measured")
put("V2.fit_under_pct_max", round((1 - min(fr.values())) * 100, 1), "largest under-prediction, %")
put("V2.fit_over_pct_max", round((max(fr.values()) - 1) * 100, 1), "largest over-prediction, %")
put("V2.speedup_8over1_min", min(R[f"SC.speedup_8over1_L{L}"] for L in (100, 250, 500, 1000)), "grid")
put("V2.speedup_8over1_max", max(R[f"SC.speedup_8over1_L{L}"] for L in (100, 250, 500, 1000)), "grid")
put("V2.chip_exp_min", min(R[f"SC.chip_exp_L{L}"] for L in (100, 250, 500, 1000)), "per-length OLS slope, min")
put("V2.chip_exp_max", max(R[f"SC.chip_exp_L{L}"] for L in (100, 250, 500, 1000)), "per-length OLS slope, max")
# Cost with the verified TPU price only (cloud.google.com/tpu/pricing, read 2026-11-23, bibliography entry
# googletpupricing). The slice is allocated as eight chips.
put("V2.price_chip_hour", price_tpu_chip, "cloud.google.com/tpu/pricing, TPU v5e on demand, us-west4, read 2026-11-23")
put("V2.price_slice_hour", round(8 * price_tpu_chip, 2), "8 chips x V2.price_chip_hour")
for L in (100, 250, 500, 1000):
    put(f"V2.cost_slice_1chip_L{L}", round(8 * price_tpu_chip / (thr[(1, L)] * 3600) * 1000, 4),
        "slice price / (1-chip grid throughput x 3600) x 1000, per 1000 queries")
    put(f"V2.cost_slice_8chip_L{L}", round(8 * price_tpu_chip / (thr[(8, L)] * 3600) * 1000, 4),
        "slice price / (8-chip grid throughput x 3600) x 1000, per 1000 queries")
put("V2.cost_slice_default_118", round(8 * price_tpu_chip / (3600 / t) * 1000, 4), "slice price, one query per call of 0.47 s (baseline), per 1000 queries")
put("V2.cost_slice_mq_118", R["COST.pmap_pod_per_1k"], "slice price / multi-query pmap throughput (14.719/s), per 1000 queries")
# Provenance facts from git (read only)
def git_out(*args):
    import os
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True,
                          env={**os.environ, "TZ": "UTC"}).stdout.strip()
put("V2.first_commit", git_out("show", "-s", "--format=%h %cd", "--date=format-local:%Y-%m-%dT%H:%MZ", "2755250"), "git show -s 2755250 (first commit, holds most TPU results)")
put("V2.arxiv_v1_tag_commit", git_out("rev-list", "-n", "1", "--abbrev-commit", "arxiv-v1"), "git rev-list -n 1 arxiv-v1")
put("V2.ensemble_commit", git_out("show", "-s", "--format=%h %cd", "--date=format-local:%Y-%m-%dT%H:%MZ", "625c5dd"), "git show -s 625c5dd (external ensemble results)")
put("V2.grid_commit", git_out("show", "-s", "--format=%h %cd", "--date=format-local:%Y-%m-%dT%H:%MZ", "cccd446"), "git show -s cccd446 (matched grid results)")
put("V2.sep_src_identical", git_out("diff", "--stat", "86fca03", "9e1a07e", "--", "src") == "", "git diff 86fca03 9e1a07e -- src is empty")
t4_runs = json.loads((ROOT / "results/repro/2026-09-15_gpu-t4/result_gpu-t4-repro_model_3_len118_recycle0_float32_noprofile.json").read_text())["steady_state_runs_seconds"]
put("V2.t4_sep_fastest_call", min(t4_runs), "min of steady_state_runs_seconds, T4, 15 September")
put("V2.tpu_over_t4_fastest_call", round(min(t4_runs) / R["B.tpu.steady"], 2), "fastest single T4 call / TPU baseline 0.47")
cpu_calls = [x for d in ("15", "16", "17") for x in json.loads((ROOT / f"results/repro/2026-09-{d}_cpu-colab/result_cpu-colab-repro_model_3_len118_recycle0_float32_noprofile.json").read_text())["steady_state_runs_seconds"]]
put("V2.sep_cpu_calls_min_max", [min(cpu_calls), max(cpu_calls)], "min, max of all 15 September CPU steady-state calls")
cfg = (ROOT / "configs/af_spike_job.yaml").read_text()
alpha = re.search(r'_AA_ALPHABET = "([A-Z]+)"', (ROOT / "src/spike_tpu_forward_pass.py").read_text()).group(1)
put("V2.alphabet_length", len(alpha), "src/spike_tpu_forward_pass.py -> _AA_ALPHABET")
ens = json.loads((ROOT / "results/sweep/ensemble_shard.json").read_text())
put("V2.ensemble_seeds", list(range(ens["num_ensemble"])), "src/spike_ensemble_shard_forward_pass.py:110-111 (random_seed=i for i in range(num_ensemble)); ensemble_shard.json -> num_ensemble")
put("V2.jax_pin", re.search(r'jax\[tpu\]==([0-9.]+)', cfg).group(1), "configs/af_spike_job.yaml")
put("V2.topology", re.search(r'tpu-topology:\s*"?([0-9x]+)', cfg).group(1), "configs/af_spike_job.yaml -> nodeSelector tpu-topology")

# ---------------------------------------------------------------- further values of the text (added 2026-11-23)
# The bounds against the slowest of the 11 single-query TPU runs (0.471 s), and against the upper end of the
# rounding of the recorded 0.47 s (0.4705 s). Both stay above 13.9 and 450.
put("V2.min_tpu_over_t4_rounding_upper", round(min(gpu_sessions) / 0.4705, 3), "min T4 session mean / 0.4705 (0.47 rounded to three decimals)")
put("V2.min_tpu_over_cpu_rounding_upper", round(min(cpu_sessions) / 0.4705, 2), "min CPU session mean / 0.4705")
# Call time of the matched grid, derived from the recorded throughput (queries per call / throughput).
for L in (100, 250, 500, 1000):
    c1, c8 = 1 / thr[(1, L)], 8 / thr[(8, L)]
    put(f"V2.call_1chip_L{L}", round(c1, 4), "1 / one-chip grid throughput (one query per call)")
    put(f"V2.call_8chip_L{L}", round(c8, 4), "8 / eight-chip grid throughput (eight queries per call)")
    put(f"V2.call_extra_8_L{L}", round(c8 - c1, 4), "eight-chip call minus one-chip call")
    put(f"V2.call_extra_8_L{L}_pct", pct((c8 - c1) / c1), "extra time as % of the one-chip call")
# At 1000 residues the one-chip throughput has two significant digits (0.066): the interval of the extra time.
lo1, hi1 = 1 / (thr[(1, 1000)] + 0.0005), 1 / (thr[(1, 1000)] - 0.0005)
lo8, hi8 = 8 / (thr[(8, 1000)] + 0.0005), 8 / (thr[(8, 1000)] - 0.0005)
put("V2.speedup_L1000_interval", [round((thr[(8, 1000)] - 0.0005) / (thr[(1, 1000)] + 0.0005), 2), round((thr[(8, 1000)] + 0.0005) / (thr[(1, 1000)] - 0.0005), 2)], "8-chip / 1-chip throughput at 1000 residues over the rounding of both")
put("V2.call_extra_8_L1000_interval", [round(lo8 - hi1, 3), round(hi8 - lo1, 3)], "extra time at 1000 residues over the rounding of both throughputs")
# The fit in two significant digits, and the local slopes of time on length.
put("V2.fit_prefactor_sci", {"mantissa": round(math.exp(b0) / 1000, 1), "base": 10, "exponent": 3}, "SC.fit_prefactor in scientific notation, two significant digits")
put("V2.fit_chip_exp_2d", round(b1, 2), "SC.fit_chip_exp, two decimals")
put("V2.fit_len_exp_2d", round(b2, 2), "SC.fit_len_exp, two decimals")
put("V2.local_slope_grid_1chip", [round(math.log(thr[(1, a)] / thr[(1, b)]) / math.log(b / a), 2) for a, b in zip(lengths, lengths[1:])],
    "slope of log call time on log length between neighbouring lengths, one-chip grid (call = 1 / throughput)")
_sl = [(r["num_residues"], r["steady_state_seconds"]) for r in load("results/sweep/sequence_length_sweep.json")]
put("V2.local_slope_sweep", [round(math.log(t2 / t1) / math.log(l2 / l1), 2) for (l1, t1), (l2, t2) in zip(_sl, _sl[1:])],
    "slope of log steady-state call on log length between neighbouring lengths, sequence-length sweep")
# One chip rented alone at the price per chip.
for L in (100, 1000):
    put(f"V2.cost_chip_alone_L{L}", round(price_tpu_chip / (thr[(1, L)] * 3600) * 1000, 4), "$1.20 / (one-chip grid throughput x 3600) x 1000, per 1000 queries")
# Cost section: three recycles against none (fixed sequence, one chip).
_rd = {r["num_recycle"]: r["steady_state_seconds"] for r in load("results/sweep/recycle_depth_sweep.json")}
put("V2.recycle3_over_0", round(_rd[3] / _rd[0], 3), "recycle_depth_sweep.json: steady-state call with 3 recycles / with 0")
# model_4 has the same configuration as model_3 in our runs.
put("V2.model_steady", {r["model_name"]: r["steady_state_seconds"] for r in load("results/sweep/model_comparison.json")}, "model_comparison.json -> [*].steady_state_seconds")

# ---------------------------------------------------------------- write
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "recomputed.json").write_text(json.dumps({"values": R, "sources": SRC}, indent=1, ensure_ascii=False, default=str))
with open(OUT / "recomputed.md", "w") as f:
    f.write("# Recomputed values (generated by recompute.py; do not edit)\n\n| key | value | source |\n|---|---|---|\n")
    for k_ in R:
        v = R[k_]
        f.write(f"| {k_} | {json.dumps(v, ensure_ascii=False, default=str)[:300]} | {SRC[k_]} |\n")
print(f"{len(R)} values written to {OUT/'recomputed.json'}")
