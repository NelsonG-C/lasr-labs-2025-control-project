import matplotlib.pyplot as plt

# Approximate data points from the red line (eyeballed from your image)
safety_baseline = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_baseline = [
    1.0,
    0.99,
    0.99,
    0.98,
    0.975,
    0.977,
    0.92,
    0.87,
]  # approximate values

safety_acausal = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_acausal = [
    1.0,
    0.98,
    0.98,
    0.97,
    0.965,
    0.94,
    0.87,
    0.80,
]

safety_self_recognition = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_self_recognition = [
    1.0,
    0.99,
    0.99,
    0.98,
    0.97,
    0.96,
    0.88,
    0.80,
]  # approximate values

safety_self_recognition_ft = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
usefulness_self_recognition_ft = [
    1.0,
    0.99,
    0.99,
    0.98,
    0.97,
    0.96,
    0.88,
    0.80,
]  # approximate values

plt.figure(figsize=(8, 6))
plt.plot(
    safety_baseline,
    usefulness_baseline,
    "r-v",
    linewidth=2,
    markersize=8,
    label="Untrusted monitoring\n(baseline)",
)
plt.plot(
    safety_baseline,
    usefulness_acausal,
    "b-v",
    linewidth=2,
    markersize=8,
    label="Untrusted monitoring \n (acausal)",
)
plt.plot(
    safety_baseline,
    usefulness_self_recognition,
    "g-v",
    linewidth=2,
    markersize=8,
    label="Untrusted monitoring \n (self-recognition)",
)
plt.xlabel("Safety")
plt.ylabel("Usefulness")
plt.xlim(0.0, 1.05)
plt.ylim(0.0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
