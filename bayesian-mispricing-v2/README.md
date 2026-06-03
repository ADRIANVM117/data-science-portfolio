## Information-Theoretic Market States

The final stage of the project investigated whether prediction markets naturally organize into distinct information-processing regimes.

Rather than imposing labels such as "efficient" or "inefficient", an unsupervised learning approach was used.

### Methodology

* Standardization
* Principal Component Analysis (PCA)
* K-Means Clustering
* Silhouette Analysis

### PCA Results

| Metric                   | Value |
| ------------------------ | ----: |
| PC1 Variance Explained   | 28.1% |
| PC2 Variance Explained   | 17.0% |
| Total Variance Explained | 45.1% |

### Cluster Selection

Several values of (K) were evaluated using the Silhouette Score.

| K | Silhouette Score |
| - | ---------------: |
| 2 |            0.226 |
| 3 |            0.178 |
| 4 |            0.186 |
| 5 |            0.223 |
| 6 |            0.217 |

The optimal solution was:

$ K = 2 $


---

## Market State Characteristics

### Cluster 0 — Information Processing Markets

Characteristics:

* High Drawdown
* High Reversals
* High Probability Range
* Low Entropy

| Feature           | Value |
| ----------------- | ----: |
| Max Drawdown      | 0.870 |
| Reversals         | 702.8 |
| Probability Range | 0.618 |
| Shannon Entropy   | 0.181 |

### Cluster 1 — Anchored / Noisy Markets

Characteristics:

* Low Drawdown
* Low Reversals
* Low Probability Range
* High Entropy

| Feature           | Value |
| ----------------- | ----: |
| Max Drawdown      | 0.448 |
| Reversals         |  96.7 |
| Probability Range | 0.218 |
| Shannon Entropy   | 0.568 |

---

## Forecast Accuracy by Market State

### Cluster 0

| Metric              | Value |
| ------------------- | ----: |
| Markets             |    32 |
| Mean Abs Surprise   | 0.040 |
| Median Abs Surprise | 0.005 |

### Cluster 1

| Metric              | Value |
| ------------------- | ----: |
| Markets             |    11 |
| Mean Abs Surprise   | 0.340 |
| Median Abs Surprise | 0.435 |

### Key Result

Markets belonging to the Information Processing Regime exhibit approximately:

* **8.5× lower mean forecast error**
* **83× lower median forecast error**

than markets belonging to the Anchored / Noisy Regime.

---

## Final Research Insight

Prediction markets do not appear to form a homogeneous population.

Instead, they naturally organize into distinct information-processing regimes.

The most accurate markets are characterized by:

* Frequent belief revisions
* Strong error-correction dynamics
* Large probability adjustments
* Low information entropy

while the least accurate markets exhibit:

* Limited belief updating
* Weak error correction
* Higher entropy
* Persistent forecast errors

These findings suggest that prediction market efficiency is fundamentally linked to how information is incorporated into prices rather than the topic being forecast.

---

## Project Summary

The project evolved through four major stages:

### Stage 1 — Market Categories

Question:

Do different market topics explain forecast accuracy?

Result: No meaningful evidence.

---

### Stage 2 — Information Dynamics

Question:

Do probability trajectories explain forecast accuracy?

Strong evidence.

Max Drawdown emerged as the dominant explanatory variable.

---

### Stage 3 — Early Warning System

Question:

Can efficiency be detected before market resolution?

Result: Yes. Using only the first 25% of market life:
* Accuracy = 67.5%
* F1 Score = 70.1%

---

### Stage 4  Market States

Question:

Do distinct information-processing regimes exist?

Result: Yes.

Two natural market states emerged:

1. Information Processing Markets
2. Anchored / Noisy Markets

with dramatically different forecast performance.

---

## Overall Conclusion

Prediction market accuracy is not primarily determined by market category.

Instead, forecast performance is driven by information-processing dynamics.

Markets that actively revise beliefs, aggressively correct mistakes, and maintain lower entropy consistently achieve superior forecasting performance.

These dynamics emerge early, remain interpretable, and ultimately define distinct market regimes.
