"""Mockup of the expected Fig 2 output — generated independently from the
actual pipeline using the data points observed in the earlier successful
render of patient 14866589. Used as a visual target while waiting for the
regenerate_figures.py run to finish.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Observed data points from patient 14866589 (ACE/ARB → potassium)
days = np.array([-1, 0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 30, 31, 45], dtype=float)
vals = np.array([4.1, 4.9, 5.0, 5.2, 4.9, 4.6, 4.2, 4.6, 4.0, 4.9, 4.5, 4.5, 4.2, 4.5, 4.8, 5.3, 4.4, 5.3])

# Scenario constants
static_thr = 5.3
temporal_gate = 5.0
day_temporal_alert = 1
day_static_alert = 30
label_window = 30

# Axes (two-stage logic from the fixed code)
clinical_lo, clinical_hi = -30.0, label_window + 20.0  # -30, 50
mask = (days >= clinical_lo) & (days <= clinical_hi)
days_in = days[mask]
vals_in = vals[mask]
data_lo, data_hi = float(np.min(days_in)), float(np.max(days_in))
alerts = [day_temporal_alert, day_static_alert]
x_lo = max(min(data_lo - 2.0, -3.0), clinical_lo)
must_show_hi = max([data_hi] + alerts)
x_hi = min(must_show_hi + 5.0, clinical_hi)

# Line-break across gaps > 7 days
MAX_GAP = 7
plot_days = days_in.astype(float).copy()
plot_vals = vals_in.astype(float).copy()
gaps = np.diff(plot_days)
breaks = np.where(gaps > MAX_GAP)[0]
for i in reversed(breaks):
    plot_days = np.insert(plot_days, i + 1, np.nan)
    plot_vals = np.insert(plot_vals, i + 1, np.nan)

# Y-range with headroom for badges
y_lo = min(float(np.nanmin(vals_in)), temporal_gate) - 0.35
y_hi = max(float(np.nanmax(vals_in)), static_thr) + 0.65

# Palette
C_DATA      = "#1f4e79"
C_STATIC    = "#c0392b"
C_TEMPORAL  = "#e67e22"
C_DRUGSTART = "#1e8449"
C_TEXT      = "#2c3e50"

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.grid(True, color="#dadce0", linewidth=0.6, alpha=0.7, zorder=0)
ax.set_axisbelow(True)

ax.axhline(static_thr, ls=(0, (6, 3)), color=C_STATIC, lw=1.6, zorder=2,
           label=f"Static threshold ({static_thr} mmol/L)")
ax.axhline(temporal_gate, ls=(0, (1, 2)), color=C_TEMPORAL, lw=1.6, zorder=2,
           label=f"Temporal gate ({temporal_gate} mmol/L)")

ax.axvline(0, color=C_DRUGSTART, lw=2.2, zorder=3, label="Drug start (day 0)")
ax.axvline(day_temporal_alert, color=C_TEMPORAL, lw=1.8, alpha=0.85, zorder=3)
ax.axvline(day_static_alert, color=C_STATIC, lw=1.8, alpha=0.7, zorder=3)

ax.plot(plot_days, plot_vals, "-", color=C_DATA, lw=1.4, alpha=0.55, zorder=4)
ax.scatter(days_in, vals_in, color=C_DATA, s=55, edgecolors="white",
           linewidths=1.2, zorder=10, label="Potassium measurement")

badge = dict(boxstyle="round,pad=0.35", facecolor="white",
             edgecolor="none", alpha=0.92)
ax.annotate(f"Temporal alert\nday {day_temporal_alert}",
            xy=(day_temporal_alert, y_hi - 0.18),
            xytext=(8, 0), textcoords="offset points",
            ha="left", va="top",
            color=C_TEMPORAL, fontsize=9.5, fontweight="bold",
            bbox={**badge, "edgecolor": C_TEMPORAL, "linewidth": 1.0},
            zorder=15)
ax.annotate(f"Static alert\nday {day_static_alert}",
            xy=(day_static_alert, y_hi - 0.18),
            xytext=(-8, 0), textcoords="offset points",
            ha="right", va="top",
            color=C_STATIC, fontsize=9.5, fontweight="bold",
            bbox={**badge, "edgecolor": C_STATIC, "linewidth": 1.0},
            zorder=15)

gap_days = day_static_alert - day_temporal_alert
ax.set_title(f"Figure 2. Patient timeline (patient 14866589) "
             f"— temporal alert {gap_days} day(s) earlier",
             fontsize=11.5, color=C_TEXT, pad=10)
ax.set_xlabel("Days since drug start", fontsize=11, color=C_TEXT)
ax.set_ylabel("Potassium (mmol/L)", fontsize=11, color=C_TEXT)
ax.tick_params(colors=C_TEXT, labelsize=10)

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("bottom", "left"):
    ax.spines[s].set_color("#666666")
    ax.spines[s].set_linewidth(0.8)

leg = ax.legend(fontsize=9, loc="center", bbox_to_anchor=(0.55, 0.28),
                frameon=True, framealpha=0.95, edgecolor="#cccccc")
leg.get_frame().set_linewidth(0.6)

out = "paper/figures/fig2_mockup.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {out}  (x: {x_lo:.0f} to {x_hi:.0f},  y: {y_lo:.2f} to {y_hi:.2f})")
