# Probabilistic Network Latency Prediction with Mixture Density Networks


**Workshop for analyzing latency reliability in time-critical applications using 5G**

Samie Mostafavi  •  KTH Royal Institute of Technology  
samiemostafavi@gmail.com  •  2025-09-08  •  Stockholm


Reliable low-latency is the backbone of real-time 5G/6G applications. In this workshop, you’ll explore packet delay data from multiple experiments, build probabilistic ML models with Tensorflow to capture rare delay spikes that matter for time critical applications.

We start with unconditional (marginal) density estimation of latency in Part 1, then move to conditional density estimation in Part 2, predicting the latency distribution given the available features in the dataset.

References:
- [Data-Driven Latency Probability Prediction for Wireless Networks: Focusing on Tail Probabilities](https://ieeexplore.ieee.org/abstract/document/10437281)
- [EDAF: An End-to-End Delay Analytics Framework for 5G-and-Beyond Networks](https://ieeexplore.ieee.org/document/10620853)

---

## Datasets
- Measurements collected on the ExPECA testbed at KTH, using two OpenAirInterface 5G SDR nodes patched by the EDAF project to capture per-packet latency for analytics.
- A pickled container (`dataset.pkl`) with several experiments: **e4, e6, e7, e8, e19, e20**.
- Experiments were conducted by sending periodic traffic over the 5G uplink: every 50 ms, packets of varying sizes were transmitted under three channel-quality scenarios. Each experiment produces a Pandas DataFrame where each row records the measured transmission of a single packet on the 5G link. Experiment durations vary, ranging from ~15 minutes to ~1 hour.

---

## Environment & prerequisites
- Runs in **Google Colab** (CPU OK).
- Python: TensorFlow/Keras, TensorFlow Probability, NumPy, Pandas, Matplotlib, SciPy.

---

## Methods at a glance
In the first notebook (part 1), we only apply density estimation methods.
- **GMM (bulk):** Multi-modal fit of typical delays.  
- **EVM (bulk + tail):** GMM for the bulk + **GPD** for exceedances above a learned threshold `u`.  
  This improves estimation of **tail probabilities**.

Next (part 2), we model how the latency distribution depends on transmission conditions by learning a conditional density that links delays to the observed features.

---

## Quick navigation

Part 1:
- **Step 1: Setup — Install & Load Data**  
- **Step 2: Inspect Datasets**  
- **Step 3: Latency/Reliability Analysis**  
- **Step 4: Train GMM (bulk)**
- **Step 5: Train EVM (bulk + GPD tail) & Compare**

Part 2:
- **Step 1: Setup — Install & Load Data**  
- **Step 2: Inspect Datasets**
- 

---

## Learning outcomes

By the end, you’ll be able to:
1. Diagnose delay distributions and identify **multi-modality** and **heavy tails**.  
2. Fit and validate **GMM** and **EVM** models, including **tail metrics** like CCDF at service thresholds.  
3. Judge **sample size adequacy** for tail reliability claims (e.g., XR’s 0.999 at 50 ms).


