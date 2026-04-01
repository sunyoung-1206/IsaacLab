"""
sim-to-sim 비교 플롯 스크립트  (PNG, 한 장에 하나씩)

--plots 로 원하는 방법론만 선택:
  basic     IsaacLab / MuJoCo / Residual (I-M) 시계열 비교
  density   2D histogram  (x=IsaacLab, y=MuJoCo, 대각선=완벽일치)
  envelope  rolling mean ± std shading + 통계 텍스트
  psd       Welch Power Spectral Density

출력 파일 (방법론별 prefix 추가):
  basic    →  <out>_basic_00_base.png    <out>_basic_01_FL_hip.png  ...
  density  →  <out>_density_00_base.png  ...
  envelope →  <out>_envelope_00_base.png ...
  psd      →  <out>_psd_00_base.png      ...

t_start / t_end 는 모든 방법론에 공통 적용됩니다.

사용법:
  python scripts/mujoco/compare_sims.py \\
      --isaac  logs/isaac_data.npz \\
      --mujoco logs/mujoco_data.npz \\
      --plots  basic psd \\
      [--t_start 2.0] [--t_end 30.0] [--y_pct 99] [--fs 50] \\
      [--out   logs/comparison]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


JOINT_NAMES_SHORT = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]
PLOT_CHOICES = ["basic", "density", "envelope", "psd", "contact"]

C_ISAAC  = "#1f77b4"
C_MUJOCO = "#ff7f0e"
C_DIFF   = "#2ca02c"
ALPHA    = 0.9


# ── 데이터 유틸리티 ────────────────────────────────────────────────────────────

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _align_joint_order(data: dict, target_names: list) -> dict:
    src_names = list(data["joint_names"])
    if src_names == target_names:
        return data
    try:
        idx = [src_names.index(n) for n in target_names]
    except ValueError as e:
        print(f"[경고] 관절 이름 불일치: {e}  → 재정렬 생략")
        return data
    out = dict(data)
    for key in ("tau", "q", "qdot"):
        if key in data and data[key].ndim == 2:
            out[key] = data[key][:, idx]
    return out


def _trim(data: dict, t_start, t_end) -> dict:
    """t_start ~ t_end 범위만 남김. 모든 방법론에 공통 적용."""
    mask = np.ones(len(data["time"]), dtype=bool)
    if t_start is not None:
        mask &= data["time"] >= t_start
    if t_end is not None:
        mask &= data["time"] <= t_end
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == mask.shape[0] else v)
            for k, v in data.items()}


def _get_series(data, key, col):
    v = data[key]
    return v[:, col] if v.ndim == 2 else v


def _ylim_pct(arrays, pct):
    if pct >= 100:
        return None
    combined = np.concatenate([a.ravel() for a in arrays if len(a) > 0])
    lo = np.percentile(combined, 100 - pct)
    hi = np.percentile(combined, pct)
    margin = max((hi - lo) * 0.05, 1e-6)
    return (lo - margin, hi + margin)


def _rolling(v, win):
    mean = np.convolve(v, np.ones(win) / win, mode="same")
    v2   = np.convolve(v**2, np.ones(win) / win, mode="same")
    std  = np.sqrt(np.maximum(v2 - mean**2, 0))
    return mean, std


def _joint_name(j):
    return JOINT_NAMES_SHORT[j] if j < len(JOINT_NAMES_SHORT) else f"joint_{j}"


def _save(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ── subplot 헬퍼 (방법론별) ───────────────────────────────────────────────────

def _ax_basic(ax, isaac, mujoco, key, col_or_none, unit, label, y_pct):
    """IsaacLab / MuJoCo 시계열 + Residual — 3행으로 분리해서 호출."""
    pass  # 아래 _ax_basic_* 세 개로 분리


def _ax_ts(ax, t, v, color, src_label, ylabel, title, ylim=None):
    ax.plot(t, v, color=color, alpha=ALPHA, linewidth=1.2, label=src_label)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time [s]", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)


def _ax_residual(ax, t, diff, ylabel, title, ylim=None):
    ax.plot(t, diff, color=C_DIFF, alpha=ALPHA, linewidth=1.0)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.fill_between(t, diff, 0, alpha=0.15, color=C_DIFF)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time [s]", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)


def _ax_density(ax, vi, vm, unit, title):
    n = min(len(vi), len(vm))
    x, y = vi[:n], vm[:n]
    vmin, vmax = min(x.min(), y.min()), max(x.max(), y.max())
    h, xe, ye = np.histogram2d(x, y, bins=60, range=[[vmin, vmax], [vmin, vmax]])
    ax.imshow(h.T, origin="lower", aspect="auto",
              extent=[xe[0], xe[-1], ye[0], ye[-1]],
              cmap="YlOrRd", interpolation="nearest")
    ax.plot([vmin, vmax], [vmin, vmax], "b--", linewidth=1.2, label="y=x")
    ax.set_xlabel(f"IsaacLab [{unit}]", fontsize=8)
    ax.set_ylabel(f"MuJoCo [{unit}]",   fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def _ax_envelope(ax, t_i, vi, t_m, vm, unit, title, fs):
    win = max(1, int(1.0 * fs))  # 1초 윈도우
    mi, si = _rolling(vi, win)
    mm, sm = _rolling(vm, win)
    ax.plot(t_i, vi, color=C_ISAAC,  alpha=0.2,  linewidth=0.7)
    ax.plot(t_i, mi, color=C_ISAAC,  alpha=0.9,  linewidth=1.5, label="Isaac mean")
    ax.fill_between(t_i, mi-si, mi+si, color=C_ISAAC,  alpha=0.2, label="Isaac ±std")
    ax.plot(t_m, vm, color=C_MUJOCO, alpha=0.2,  linewidth=0.7)
    ax.plot(t_m, mm, color=C_MUJOCO, alpha=0.9,  linewidth=1.5, label="MuJoCo mean", linestyle="--")
    ax.fill_between(t_m, mm-sm, mm+sm, color=C_MUJOCO, alpha=0.2, label="MuJoCo ±std")
    ax.set_xlabel("time [s]", fontsize=8)
    ax.set_ylabel(unit, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    txt = (f"Isaac  μ={vi.mean():.3f}  σ={vi.std():.3f}\n"
           f"MuJoCo μ={vm.mean():.3f}  σ={vm.std():.3f}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7, va="top",
            family="monospace", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def _ax_psd(ax, vi, vm, unit, title, fs):
    nperseg = max(16, min(256, len(vi)//4, len(vm)//4))
    fi, pi = welch(vi, fs=fs, nperseg=nperseg)
    fm, pm = welch(vm, fs=fs, nperseg=nperseg)
    ax.semilogy(fi, pi, color=C_ISAAC,  alpha=0.9, linewidth=1.3, label="IsaacLab")
    ax.semilogy(fm, pm, color=C_MUJOCO, alpha=0.9, linewidth=1.3, label="MuJoCo", linestyle="--")
    ax.set_xlabel("Frequency [Hz]", fontsize=8)
    ax.set_ylabel(f"PSD [{unit}²/Hz]", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, fs / 2)


# ── 페이지 생성: basic ────────────────────────────────────────────────────────

def _run_basic_base(isaac, mujoco, prefix, y_pct):
    signals = [("lin_vel_b",0,"m/s","vx"),("lin_vel_b",1,"m/s","vy"),
               ("lin_vel_b",2,"m/s","vz"),("base_height",None,"m","height")]
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle("Base State  —  IsaacLab / MuJoCo / Residual (I−M)", fontsize=13, fontweight="bold")
    for ci, (key, dim, unit, label) in enumerate(signals):
        vi = isaac[key] if dim is None else _get_series(isaac, key, dim)
        vm = mujoco[key] if dim is None else _get_series(mujoco, key, dim)
        n  = min(len(vi), len(vm))
        diff, t_d = vi[:n] - vm[:n], isaac["time"][:n]
        ylim_s = _ylim_pct([vi, vm], y_pct)
        ylim_d = _ylim_pct([diff],   y_pct)
        _ax_ts(axes[0,ci], isaac["time"], vi, C_ISAAC,  "IsaacLab", unit, f"IsaacLab {label}", ylim_s)
        if label == "vx" and "cmd" in isaac:
            ref = float(isaac["cmd"][0, 0])
            axes[0,ci].axhline(ref, color="k", linestyle=":", linewidth=1.0, label=f"cmd={ref:.2f}")
            axes[0,ci].legend(fontsize=7)
        _ax_ts(axes[1,ci], mujoco["time"], vm, C_MUJOCO, "MuJoCo",   unit, f"MuJoCo {label}", ylim_s)
        _ax_residual(axes[2,ci], t_d, diff, unit, f"Residual (I−M) {label}", ylim_d)
    _save(fig, f"{prefix}_basic_00_base.png")


def _run_basic_joint(isaac, mujoco, j, prefix, y_pct):
    name = _joint_name(j)
    signals = [("tau","N·m","Torque"),("q","rad","Position"),("qdot","rad/s","Velocity")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(f"Joint [{j}] {name}  —  IsaacLab / MuJoCo / Residual (I−M)", fontsize=13, fontweight="bold")
    for ci, (key, unit, slabel) in enumerate(signals):
        vi = _get_series(isaac, key, j)
        vm = _get_series(mujoco, key, j)
        n  = min(len(vi), len(vm))
        diff, t_d = vi[:n] - vm[:n], isaac["time"][:n]
        ylim_s = _ylim_pct([vi, vm], y_pct)
        ylim_d = _ylim_pct([diff],   y_pct)
        _ax_ts(axes[0,ci], isaac["time"], vi, C_ISAAC,  "IsaacLab", unit, f"IsaacLab {slabel}", ylim_s)
        _ax_ts(axes[1,ci], mujoco["time"], vm, C_MUJOCO, "MuJoCo",  unit, f"MuJoCo {slabel}",  ylim_s)
        _ax_residual(axes[2,ci], t_d, diff, unit, f"Residual (I−M) {slabel}", ylim_d)
    _save(fig, f"{prefix}_basic_{j+1:02d}_{name}.png")


# ── 페이지 생성: density ──────────────────────────────────────────────────────

def _run_density_base(isaac, mujoco, prefix):
    signals = [("lin_vel_b",0,"m/s","vx"),("lin_vel_b",1,"m/s","vy"),
               ("lin_vel_b",2,"m/s","vz"),("base_height",None,"m","height")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Base State  —  Density (x=IsaacLab, y=MuJoCo)", fontsize=13, fontweight="bold")
    for ci, (key, dim, unit, label) in enumerate(signals):
        vi = isaac[key] if dim is None else _get_series(isaac, key, dim)
        vm = mujoco[key] if dim is None else _get_series(mujoco, key, dim)
        _ax_density(axes[ci], vi, vm, unit, f"Density {label}")
    _save(fig, f"{prefix}_density_00_base.png")


def _run_density_joint(isaac, mujoco, j, prefix):
    name = _joint_name(j)
    signals = [("tau","N·m","Torque"),("q","rad","Position"),("qdot","rad/s","Velocity")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Joint [{j}] {name}  —  Density (x=IsaacLab, y=MuJoCo)", fontsize=13, fontweight="bold")
    for ci, (key, unit, slabel) in enumerate(signals):
        _ax_density(axes[ci], _get_series(isaac,key,j), _get_series(mujoco,key,j), unit, f"Density {slabel}")
    _save(fig, f"{prefix}_density_{j+1:02d}_{name}.png")


# ── 페이지 생성: envelope ─────────────────────────────────────────────────────

def _run_envelope_base(isaac, mujoco, prefix, fs):
    signals = [("lin_vel_b",0,"m/s","vx"),("lin_vel_b",1,"m/s","vy"),
               ("lin_vel_b",2,"m/s","vz"),("base_height",None,"m","height")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Base State  —  Envelope (mean ± std)", fontsize=13, fontweight="bold")
    for ci, (key, dim, unit, label) in enumerate(signals):
        vi = isaac[key] if dim is None else _get_series(isaac, key, dim)
        vm = mujoco[key] if dim is None else _get_series(mujoco, key, dim)
        _ax_envelope(axes[ci], isaac["time"], vi, mujoco["time"], vm, unit, f"Envelope {label}", fs)
    _save(fig, f"{prefix}_envelope_00_base.png")


def _run_envelope_joint(isaac, mujoco, j, prefix, fs):
    name = _joint_name(j)
    signals = [("tau","N·m","Torque"),("q","rad","Position"),("qdot","rad/s","Velocity")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Joint [{j}] {name}  —  Envelope (mean ± std)", fontsize=13, fontweight="bold")
    for ci, (key, unit, slabel) in enumerate(signals):
        vi, vm = _get_series(isaac,key,j), _get_series(mujoco,key,j)
        _ax_envelope(axes[ci], isaac["time"], vi, mujoco["time"], vm, unit, f"Envelope {slabel}", fs)
    _save(fig, f"{prefix}_envelope_{j+1:02d}_{name}.png")


# ── 페이지 생성: psd ──────────────────────────────────────────────────────────

def _run_psd_base(isaac, mujoco, prefix, fs):
    signals = [("lin_vel_b",0,"m/s","vx"),("lin_vel_b",1,"m/s","vy"),
               ("lin_vel_b",2,"m/s","vz"),("base_height",None,"m","height")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Base State  —  PSD (Welch)", fontsize=13, fontweight="bold")
    for ci, (key, dim, unit, label) in enumerate(signals):
        vi = isaac[key] if dim is None else _get_series(isaac, key, dim)
        vm = mujoco[key] if dim is None else _get_series(mujoco, key, dim)
        _ax_psd(axes[ci], vi, vm, unit, f"PSD {label}", fs)
    _save(fig, f"{prefix}_psd_00_base.png")


def _run_psd_joint(isaac, mujoco, j, prefix, fs):
    name = _joint_name(j)
    signals = [("tau","N·m","Torque"),("q","rad","Position"),("qdot","rad/s","Velocity")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Joint [{j}] {name}  —  PSD (Welch)", fontsize=13, fontweight="bold")
    for ci, (key, unit, slabel) in enumerate(signals):
        vi, vm = _get_series(isaac,key,j), _get_series(mujoco,key,j)
        _ax_psd(axes[ci], vi, vm, unit, f"PSD {slabel}", fs)
    _save(fig, f"{prefix}_psd_{j+1:02d}_{name}.png")


# ── contact 통계 계산 ─────────────────────────────────────────────────────────

def _contact_stats(contact_bin, time):
    """
    binary contact 신호에서 stance/swing 통계 계산.
    양쪽 끝을 0으로 패딩해 t_start/t_end로 잘린 구간도 포함해서 있는 그대로 측정한다.
    """
    if len(time) < 2:
        return None
    dt = float(np.mean(np.diff(time)))
    total_time = time[-1] - time[0]

    c = np.concatenate([[0], contact_bin.astype(int), [0]])
    diff = np.diff(c)
    stance_starts = np.where(diff ==  1)[0]
    stance_ends   = np.where(diff == -1)[0]

    stance_dur = (stance_ends - stance_starts) * dt
    swing_dur  = (stance_starts[1:] - stance_ends[:-1]) * dt if len(stance_starts) > 1 else np.array([])

    # total / ratio: 윈도우 안의 모든 구간 포함
    stance_total = float(np.sum(stance_dur))
    swing_total  = total_time - stance_total

    # stance avg: edge 불완전 구간 제외 (zero-padding으로 포함된 것 제거)
    stance_complete = stance_dur.copy()
    if len(stance_complete) > 0 and contact_bin[0] == 1:
        stance_complete = stance_complete[1:]
    if len(stance_complete) > 0 and contact_bin[-1] == 1:
        stance_complete = stance_complete[:-1]

    # swing avg: 수식 자체가 stance 사이 구간만 → 이미 모두 완전한 주기
    swing_complete = swing_dur

    return {
        "stance_total": stance_total,
        "swing_total":  max(swing_total, 0.0),
        "stance_ratio": 100.0 * stance_total / total_time if total_time > 0 else 0.0,
        "stance_mean":  float(np.mean(stance_complete)) * 1000 if len(stance_complete) > 0 else 0.0,
        "swing_mean":   float(np.mean(swing_complete))  * 1000 if len(swing_complete)  > 0 else 0.0,
        "stance_count": len(stance_complete),
        "swing_count":  len(swing_complete),
    }


def _stats_text(stats, label, color):
    if stats is None:
        return f"{label}: N/A"
    return (
        f"{label}\n"
        f"  stance {stats['stance_total']:.2f}s ({stats['stance_ratio']:.1f}%)\n"
        f"  swing  {stats['swing_total']:.2f}s ({100-stats['stance_ratio']:.1f}%)\n"
        f"  avg stance {stats['stance_mean']:.0f}ms  x{stats['stance_count']}\n"
        f"  avg swing  {stats['swing_mean']:.0f}ms  x{stats['swing_count']}"
    )


# ── 페이지 생성: contact ──────────────────────────────────────────────────────

def _run_contact(isaac, mujoco, prefix):
    """발 contact binary + force magnitude + 통계. 2행 × 4열 (FL/FR/RL/RR)."""
    foot_labels = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    if "foot_names" in isaac:
        foot_labels = list(isaac["foot_names"])

    has_i = "foot_contact" in isaac and isaac["foot_contact"].ndim == 2
    has_m = "foot_contact" in mujoco and mujoco["foot_contact"].ndim == 2

    if not has_i and not has_m:
        print("  [contact] foot_contact 데이터 없음 — 건너뜀")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Foot Contact  —  binary + stats (row 1) / force magnitude (row 2)",
                 fontsize=13, fontweight="bold")

    for ci, fname in enumerate(foot_labels[:4]):
        ax_c = axes[0, ci]
        ax_f = axes[1, ci]
        short = fname.replace("_foot", "")

        # ── 행1: binary contact ───────────────────────────────────────────
        si, sm = None, None
        if has_i:
            yi = isaac["foot_contact"][:, ci]
            ax_c.step(isaac["time"], yi, where="post",
                      color=C_ISAAC, alpha=0.8, linewidth=2.0, label="IsaacLab")
            si = _contact_stats(yi, isaac["time"])
        if has_m:
            ym = mujoco["foot_contact"][:, ci]
            ax_c.step(mujoco["time"], ym, where="post",
                      color=C_MUJOCO, alpha=0.6, linewidth=1.5, linestyle="--", label="MuJoCo")
            sm = _contact_stats(ym, mujoco["time"])

        ax_c.set_yticks([0, 1])
        ax_c.set_yticklabels(["Off", "On"], fontsize=8)
        ax_c.set_ylim(-0.15, 1.3)
        ax_c.set_title(f"{short}  Contact", fontsize=9)
        ax_c.set_xlabel("time [s]", fontsize=8)
        ax_c.legend(fontsize=7, loc="upper right")
        ax_c.grid(True, alpha=0.3)

        # 통계 텍스트 (좌측 하단)
        # txt_i = _stats_text(si, "Isaac", C_ISAAC)
        # txt_m = _stats_text(sm, "MuJoCo", C_MUJOCO)
        # ax_c.text(0.01, 0.01, txt_i, transform=ax_c.transAxes,
        #           fontsize=6.5, va="bottom", family="monospace", color=C_ISAAC,
        #           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        # ax_c.text(0.50, 0.01, txt_m, transform=ax_c.transAxes,
        #           fontsize=6.5, va="bottom", family="monospace", color=C_MUJOCO,
        #           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # ── 행2: force magnitude ──────────────────────────────────────────
        if has_i and "foot_force" in isaac:
            ax_f.plot(isaac["time"], isaac["foot_force"][:, ci],
                      color=C_ISAAC, alpha=0.85, linewidth=1.2, label="IsaacLab")
        if has_m and "foot_force" in mujoco:
            ax_f.plot(mujoco["time"], mujoco["foot_force"][:, ci],
                      color=C_MUJOCO, alpha=0.85, linewidth=1.2, linestyle="--", label="MuJoCo")
        ax_f.set_title(f"{short}  Force [N]", fontsize=9)
        ax_f.set_xlabel("time [s]", fontsize=8)
        ax_f.set_ylabel("N", fontsize=8)
        ax_f.legend(fontsize=7)
        ax_f.grid(True, alpha=0.3)

    _save(fig, f"{prefix}_contact.png")


# ── 디스패처 ─────────────────────────────────────────────────────────────────

# (run_joint, run_base) — contact는 None으로 표시 (단일 페이지)
RUNNERS = {
    "basic":    (lambda i,m,j,p,**kw: _run_basic_joint(i,m,j,p,kw["y_pct"]),
                 lambda i,m,p,**kw:   _run_basic_base(i,m,p,kw["y_pct"])),
    "density":  (lambda i,m,j,p,**kw: _run_density_joint(i,m,j,p),
                 lambda i,m,p,**kw:   _run_density_base(i,m,p)),
    "envelope": (lambda i,m,j,p,**kw: _run_envelope_joint(i,m,j,p,kw["fs"]),
                 lambda i,m,p,**kw:   _run_envelope_base(i,m,p,kw["fs"])),
    "psd":      (lambda i,m,j,p,**kw: _run_psd_joint(i,m,j,p,kw["fs"]),
                 lambda i,m,p,**kw:   _run_psd_base(i,m,p,kw["fs"])),
    "contact":  (None,  # 관절별 페이지 없음
                 lambda i,m,p,**kw:   _run_contact(i,m,p)),
}


# ── 요약 출력 ─────────────────────────────────────────────────────────────────

def print_summary(label, data):
    n = len(data["time"])
    t_total = data["time"][-1] if n > 0 else 0.0
    mean_vx = np.mean(data["lin_vel_b"][:, 0]) if "lin_vel_b" in data else float("nan")
    cmd_vx  = float(data["cmd"][0, 0]) if "cmd" in data else float("nan")
    print(f"  [{label}]  steps={n}  t={data['time'][0]:.1f}~{t_total:.1f}s  mean_vx={mean_vx:.3f}  cmd_vx={cmd_vx:.2f}")
    if "tau" in data:
        print(f"    tau  max={np.abs(data['tau']).max():.2f}  mean={np.abs(data['tau']).mean():.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="sim-to-sim 비교 플롯",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예시:\n"
               "  --plots basic psd\n"
               "  --plots density envelope --t_start 2.0 --t_end 20.0\n"
    )
    parser.add_argument("--isaac",   required=True)
    parser.add_argument("--mujoco",  required=True)
    parser.add_argument("--plots",   nargs="+", choices=PLOT_CHOICES, default=["basic"],
                        metavar="METHOD",
                        help=f"출력할 방법론 (복수 선택 가능): {PLOT_CHOICES}  (기본: basic)")
    parser.add_argument("--t_start", type=float, default=2.0,
                        help="초반 transient 제거 [s] (기본 2.0) — 모든 방법론에 적용")
    parser.add_argument("--t_end",   type=float, default=None,
                        help="비교 종료 시각 [s]")
    parser.add_argument("--y_pct",   type=float, default=99,
                        help="basic y축 백분위수 줌 (기본 99, 100=전체)")
    parser.add_argument("--fs",      type=float, default=50.0,
                        help="샘플링 주파수 [Hz] (envelope/psd 용, 기본 50)")
    parser.add_argument("--out",     type=str, default="logs/comparison",
                        help="출력 파일 prefix")
    args = parser.parse_args()

    isaac  = load_npz(args.isaac)
    mujoco = load_npz(args.mujoco)

    # 관절 순서 통일
    ref_names = list(mujoco["joint_names"])
    isaac  = _align_joint_order(isaac,  ref_names)
    mujoco = _align_joint_order(mujoco, ref_names)

    # t_start / t_end 트리밍 — 모든 방법론에 공통 적용
    isaac  = _trim(isaac,  args.t_start, args.t_end)
    mujoco = _trim(mujoco, args.t_start, args.t_end)

    print("\n[데이터 요약 (trim 후)]")
    print_summary("IsaacLab", isaac)
    print_summary("MuJoCo",   mujoco)

    n_joints = isaac["tau"].shape[1] if "tau" in isaac else 12
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    kw = dict(y_pct=args.y_pct, fs=args.fs)

    n_files = 0
    for method in args.plots:
        run_joint, run_base = RUNNERS[method]
        print(f"\n[{method}]")
        run_base(isaac, mujoco, args.out, **kw)
        n_files += 1
        if run_joint is not None:
            for j in range(n_joints):
                run_joint(isaac, mujoco, j, args.out, **kw)
            n_files += n_joints
    print(f"\n[INFO] 완료  총 {n_files}개 파일  prefix={args.out}")


if __name__ == "__main__":
    main()
