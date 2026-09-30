# -*- coding: utf-8 -*-
"""N-24 全网格可视化: 1771 格的加权动量年化景观

用法: .venv/bin/python audit/n_mom_mix_plot.py [csv路径]
默认 audit/out/n_mom_mix_grid_A.csv; 输出两张 PNG 到 audit/out/。
注: 本机无 CJK 字体 (仅 DejaVu/Noto Sans/STIX), 图内标注一律用英文以免出现方块。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "audit", "out", "n_mom_mix_grid_A.csv")
OUT = os.path.join(HERE, "audit", "out")

BASE = dict(w5=0.0, w10=0.0, w15=0.0, w20=1.0)


def main():
    d = pd.read_csv(CSV)
    d["short"] = d.w5 + d.w10 + d.w15
    b = d[(d.w5 == 0) & (d.w10 == 0) & (d.w15 == 0)].iloc[0]
    base_ann, base_dd, base_cal = b.ann, b.dd, b.cal
    print("网格 %d 格 | 基线(纯mom20) ann=%.2f dd=%.2f cal=%.2f | 超过基线的格数 %d"
          % (len(d), base_ann, base_dd, base_cal, int((d.ann > base_ann).sum())))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("N-24  multi-period momentum weighting: full grid of 1771 weight combinations\n"
                 "window 2016-07-07 ~ 2026-08-20, narrow pool 3, ideal close / 1bp",
                 fontsize=13, y=0.985)

    # (A) scatter: w20 vs CAGR, colored by short-term weight sum
    a = ax[0, 0]
    sc = a.scatter(d.w20, d.ann, c=d.short, cmap="viridis", s=14, alpha=0.75, edgecolors="none")
    a.scatter([1.0], [base_ann], marker="*", s=420, color="red", zorder=5,
              edgecolors="white", linewidths=1.2, label="baseline (pure mom20)")
    a.axhline(base_ann, color="red", ls="--", lw=1, alpha=0.6)
    a.set_xlabel("weight on mom20  (w20)")
    a.set_ylabel("CAGR %")
    a.set_title("A. Every one of 1771 cells: CAGR vs mom20 weight\n"
                "(colour = w5+w10+w15, the short-horizon share)", fontsize=10)
    a.legend(loc="lower right", fontsize=9)
    a.grid(alpha=0.25)
    plt.colorbar(sc, ax=a, label="short-horizon weight")

    # (B) binned band over w20
    a = ax[0, 1]
    g = d.groupby("w20").ann
    xs = np.array(sorted(d.w20.unique()))
    med, p10, p90 = g.median().values, g.quantile(0.10).values, g.quantile(0.90).values
    a.fill_between(xs, p10, p90, alpha=0.25, color="steelblue", label="P10–P90 within bin")
    a.plot(xs, med, "o-", color="steelblue", lw=2, ms=4, label="median")
    a.scatter([1.0], [base_ann], marker="*", s=380, color="red", zorder=5,
              edgecolors="white", linewidths=1.2, label="baseline")
    a.axhline(base_ann, color="red", ls="--", lw=1, alpha=0.6)
    a.set_xlabel("weight on mom20  (w20)")
    a.set_ylabel("CAGR %")
    a.set_title("B. CAGR by mom20 weight (binned over all other weights)\n"
                "monotone: more mom20 is strictly better", fontsize=10)
    a.legend(loc="lower right", fontsize=9)
    a.grid(alpha=0.25)

    # (C) marginal curves for the four weights
    a = ax[1, 0]
    for col, c in (("w5", "#d62728"), ("w10", "#ff7f0e"), ("w15", "#2ca02c"), ("w20", "#1f77b4")):
        m = d.groupby(col).ann.median()
        a.plot(m.index, m.values, "o-", ms=4, lw=2, color=c, label="%s marginal median" % col)
    a.axhline(base_ann, color="red", ls="--", lw=1, alpha=0.6, label="baseline CAGR")
    a.set_xlabel("weight value")
    a.set_ylabel("median CAGR %  (over all cells with that weight)")
    a.set_title("C. Marginal effect of each horizon's weight\n"
                "w20 ↑ better, w5/w10 ↑ worse — and 1771/1771 cells lose to baseline", fontsize=10)
    a.legend(fontsize=9)
    a.grid(alpha=0.25)

    # (D) top 20 weights stacked + baseline
    a = ax[1, 1]
    top = d.sort_values("ann", ascending=False).head(20).reset_index(drop=True)
    y = np.arange(len(top))[::-1]
    left = np.zeros(len(top))
    cols = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    for col, c in zip(("w5", "w10", "w15", "w20"), cols):
        a.barh(y, top[col], left=left, color=c, height=0.72, label=col)
        left = left + top[col].values
    for i, r in top.iterrows():
        a.text(1.02, y[i], "%.2f%%  (dd %.1f%%)" % (r.ann, r.dd), va="center", fontsize=8)
    a.set_yticks(y)
    a.set_yticklabels(["#%d" % (i + 1) for i in range(len(top))], fontsize=8)
    a.set_xlim(0, 1.0)
    a.set_xlabel("weight")
    a.set_title("D. Top-20 cells by CAGR (bar = weight mix, text = CAGR / maxDD)\n"
                "#1 is the baseline itself", fontsize=10)
    a.legend(fontsize=8, ncol=4, loc="lower left")
    a.grid(alpha=0.25, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    p1 = os.path.join(OUT, "n_mom_mix_landscape.png")
    plt.savefig(p1, dpi=125)
    plt.close()
    print("saved ->", p1)

    # figure 2: 3D simplex
    fig = plt.figure(figsize=(13, 9))
    a = fig.add_subplot(111, projection="3d")
    pos = d[d.ann >= base_ann]
    neg = d[d.ann < base_ann]
    s1 = a.scatter(neg.w5, neg.w10, neg.w15, c=neg.ann, cmap="viridis", s=16, alpha=0.8)
    a.scatter(pos.w5, pos.w10, pos.w15, c="red", marker="*", s=500, edgecolors="white",
              linewidths=1.2, label="baseline (0,0,0)")
    a.set_xlabel("w5"); a.set_ylabel("w10"); a.set_zlabel("w15")
    a.set_title("N-24 simplex (w20 = 1 - w5 - w10 - w15), colour = CAGR %\n"
                "1771 cells; red star = pure mom20 = the global maximum", fontsize=11)
    plt.colorbar(s1, ax=a, shrink=0.6, label="CAGR %")
    a.legend()
    p2 = os.path.join(OUT, "n_mom_mix_simplex3d.png")
    plt.savefig(p2, dpi=125, bbox_inches="tight")
    plt.close()
    print("saved ->", p2)


if __name__ == "__main__":
    main()
