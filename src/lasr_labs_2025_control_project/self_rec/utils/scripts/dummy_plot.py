import matplotlib.pyplot as plt

# Your data
safety_baseline = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_baseline = [1.0, 0.99, 0.99, 0.98, 0.975, 0.977, 0.92, 0.87]

safety_acausal = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_acausal = [1.0, 0.99, 0.99, 0.99, 0.98, 0.96, 0.95, 0.92]

safety_self_recognition = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_self_recognition = [1.0, 0.99, 0.99, 0.99, 0.98, 0.95, 0.94, 0.91]

safety_self_recognition_ft = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_self_recognition_ft = [1.0, 0.98, 0.98, 0.96, 0.94, 0.90, 0.83, 0.71]

safety_acausal_ft = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_acausal_ft = [1.0, 0.98, 0.99, 0.97, 0.94, 0.91, 0.87, 0.79]

safety_self_recognition_ft_bt = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_self_recognition_ft_bt = [1.0, 0.99, 0.99, 0.99, 0.98, 0.95, 0.92, 0.89]

safety_acausal_ft_bt = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_acausal_ft_bt = [1.0, 0.99, 0.99, 0.99, 0.98, 0.96, 0.93, 0.90]

# Create figure with 3 subplots in one row
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Baseline comparisons
axes[0].plot(
    safety_baseline,
    usefulness_baseline,
    "r-v",
    linewidth=2,
    markersize=6,
    label="Baseline",
)
axes[0].plot(
    safety_acausal,
    usefulness_acausal,
    "b-v",
    linewidth=2,
    markersize=6,
    label="Acausal",
)
axes[0].plot(
    safety_self_recognition,
    usefulness_self_recognition,
    "g-v",
    linewidth=2,
    markersize=6,
    label="Self-recognition",
)
axes[0].set_xlabel("Safety")
axes[0].set_ylabel("Usefulness")
axes[0].set_xlim(0.35, 1.0)
axes[0].set_ylim(0.65, 1.05)
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_title("Baseline Comparisons")

# Plot 2: Acausal variants
axes[1].plot(
    safety_acausal,
    usefulness_acausal,
    "b-v",
    linewidth=2,
    markersize=6,
    label="Acausal baseline",
)
axes[1].plot(
    safety_acausal_ft,
    usefulness_acausal_ft,
    "darkblue",
    linewidth=2,
    markersize=6,
    label="Acausal + FT",
)
axes[1].plot(
    safety_acausal_ft_bt,
    usefulness_acausal_ft_bt,
    "limegreen",
    linewidth=2,
    markersize=6,
    label="Acausal + FT + BT",
)
axes[1].set_xlabel("Safety")
axes[1].set_ylabel("Usefulness")
axes[1].set_xlim(0.35, 1.0)
axes[1].set_ylim(0.65, 1.05)
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_title("Acausal Variants")

# Plot 3: Self-recognition variants
axes[2].plot(
    safety_self_recognition,
    usefulness_self_recognition,
    "g-v",
    linewidth=2,
    markersize=6,
    label="Self-recognition baseline",
)
axes[2].plot(
    safety_self_recognition_ft,
    usefulness_self_recognition_ft,
    "forestgreen",
    linewidth=2,
    markersize=6,
    label="Self-recognition + FT",
)
axes[2].plot(
    safety_self_recognition_ft_bt,
    usefulness_self_recognition_ft_bt,
    "navy",
    linewidth=2,
    markersize=6,
    label="Self-recognition + FT + BT",
)
axes[2].set_xlabel("Safety")
axes[2].set_ylabel("Usefulness")
axes[2].set_xlim(0.35, 1.0)
axes[2].set_ylim(0.65, 1.05)
axes[2].grid(True, alpha=0.3)
axes[2].legend()
axes[2].set_title("Self-recognition Variants")

plt.tight_layout()
plt.show()
