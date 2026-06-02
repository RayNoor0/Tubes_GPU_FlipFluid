#!/usr/bin/env python3
"""Generate speedup / scaling charts for the FLIP CPU-vs-CUDA benchmark report.

Run inside the conda env that has matplotlib:
    conda activate dsenv312
    python make_charts.py
Outputs PNGs into the same folder, ready to insert into the report.
"""
import os
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
RES = [50, 100, 150, 200]
STAGES = ["T1_integrate", "T2_pushApart", "T3_collisions", "T4_p2g",
          "T5_density", "T6_pressure", "T7_g2p", "T8_colors",
          "T9_render", "T10_transfer"]

# ms/frame, indexed [stage] -> [res50, res100, res150, res200]
cpu = {
    "T1_integrate": [0.0060, 0.0412, 0.0536, 0.0973],
    "T2_pushApart": [0.8273, 3.8638, 27.1076, 65.3851],
    "T3_collisions":[0.0085, 0.0339, 0.2338, 0.5069],
    "T4_p2g":       [0.1227, 0.4943, 3.4636, 8.1423],
    "T5_density":   [0.0270, 0.1158, 0.7995, 1.8964],
    "T6_pressure":  [0.6245, 2.6686, 28.0235, 89.3890],
    "T7_g2p":       [0.0865, 0.3770, 2.6437, 6.2982],
    "T8_colors":    [0.0344, 0.1346, 0.2863, 0.4965],
    "T9_render":    [3.7428, 12.9274, 23.8460, 31.6972],
    "T10_transfer": [0.0, 0.0, 0.0, 0.0],
}
cpu_total = [5.4802, 20.6573, 86.4592, 203.9105]

cuda = {
    "T1_integrate": [0.1960, 0.5682, 0.8083, 0.6379],
    "T2_pushApart": [0.1796, 0.2342, 0.7075, 1.1891],
    "T3_collisions":[0.0380, 0.0404, 0.0295, 0.0300],
    "T4_p2g":       [0.1043, 0.0994, 0.3185, 0.4498],
    "T5_density":   [0.0378, 0.0489, 0.1210, 0.1624],
    "T6_pressure":  [1.1490, 1.1459, 4.7101, 9.2152],
    "T7_g2p":       [0.0341, 0.0345, 0.1048, 0.1844],
    "T8_colors":    [0.0332, 0.0297, 0.0315, 0.0457],
    "T9_render":    [3.0832, 12.0728, 21.8693, 37.9822],
    "T10_transfer": [0.1162, 0.2147, 0.3988, 0.8497],
}
cuda_total = [5.1959, 14.8307, 29.5857, 51.3875]

# Simulation-only = total minus render (T9), which is NOT ported and dominates.
cpu_sim  = [cpu_total[i]  - cpu["T9_render"][i]  for i in range(4)]
cuda_sim = [cuda_total[i] - cuda["T9_render"][i] for i in range(4)]

sp_total = [cpu_total[i] / cuda_total[i] for i in range(4)]
sp_sim   = [cpu_sim[i]   / cuda_sim[i]   for i in range(4)]

# ---- Figure 1: T_total vs resolution (log-scale Y) -------------------------
plt.figure(figsize=(7, 4.5))
plt.plot(RES, cpu_total,  "o-", color="#c0392b", label="CPU (T_total)")
plt.plot(RES, cuda_total, "s-", color="#2471a3", label="CUDA (T_total)")
plt.plot(RES, cpu_sim,  "o--", color="#e08e0b", alpha=.8, label="CPU (tanpa render)")
plt.plot(RES, cuda_sim, "s--", color="#1abc9c", alpha=.8, label="CUDA (tanpa render)")
plt.yscale("log")
plt.xticks(RES)
plt.xlabel("Resolusi grid")
plt.ylabel("Waktu per frame (ms) — skala log")
plt.title("Skalabilitas: waktu per frame vs resolusi")
plt.grid(True, which="both", ls=":", alpha=.5)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_total_vs_res.png", dpi=150)
plt.close()

# ---- Figure 2: speedup vs resolution ---------------------------------------
plt.figure(figsize=(7, 4.5))
plt.plot(RES, sp_total, "o-", color="#2471a3", label="Speedup T_total")
plt.plot(RES, sp_sim,   "s-", color="#1abc9c", label="Speedup simulasi (tanpa render)")
plt.axhline(1.0, color="gray", ls="--", lw=1, label="break-even (1×)")
for x, y in zip(RES, sp_sim):
    plt.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=8)
plt.xticks(RES)
plt.xlabel("Resolusi grid")
plt.ylabel("Speedup (CPU / CUDA)")
plt.title("Speedup CUDA terhadap CPU vs resolusi")
plt.grid(True, ls=":", alpha=.5)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_speedup_vs_res.png", dpi=150)
plt.close()

# ---- Figure 3: per-stage speedup at res=200 --------------------------------
idx = 3  # res=200
bars = [s for s in STAGES if s != "T10_transfer"]  # CPU T10 = 0 -> skip
vals = [cpu[s][idx] / cuda[s][idx] for s in bars]
colors = ["#27ae60" if v >= 1 else "#c0392b" for v in vals]
plt.figure(figsize=(8, 4.5))
b = plt.bar(bars, vals, color=colors)
plt.axhline(1.0, color="gray", ls="--", lw=1)
plt.yscale("log")
plt.ylabel("Speedup (CPU / CUDA) — skala log")
plt.title("Speedup per tahap @ res=200  (hijau = lebih cepat, merah = lebih lambat)")
for rect, v in zip(b, vals):
    plt.text(rect.get_x() + rect.get_width()/2, v, f"{v:.1f}×",
             ha="center", va="bottom", fontsize=8)
plt.xticks(rotation=40, ha="right", fontsize=8)
plt.grid(True, axis="y", which="both", ls=":", alpha=.4)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_speedup_per_stage.png", dpi=150)
plt.close()

# ---- console summary --------------------------------------------------------
print("Resolusi          :", RES)
print("Speedup T_total   :", [f"{v:.2f}x" for v in sp_total])
print("Speedup sim-only  :", [f"{v:.2f}x" for v in sp_sim])
print("\nSpeedup per tahap @ res=200:")
for s in bars:
    print(f"  {s:14s}: {cpu[s][idx]/cuda[s][idx]:6.2f}x")
print("\nPNG tersimpan di:", OUT)
