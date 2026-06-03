---

## Predicting Market Efficiency

A Logistic Regression model was trained to classify markets as:

- Efficient Markets
- Inefficient Markets

using only trajectory-based features.

### Features

- Realized Volatility
- Probability Range
- Trend
- Max Drawdown
- Reversals
- Shannon Entropy
- Skewness
- Kurtosis
- Autocorrelation

### Cross-Validation Results

| Metric | Score |
|----------|----------:|
| Accuracy | 74.7% |
| F1 Score | 77.1% |

These results indicate that information-dynamics features contain meaningful predictive information regarding future market efficiency.

---

## Early Warning System

A key practical question is whether market efficiency can be identified before market resolution.

To investigate this, trajectory features were recalculated using only the first portion of each market's life.

Three horizons were analyzed:

- First 25% of observations
- First 50% of observations
- First 75% of observations

### Results

| Horizon | Accuracy | F1 Score |
|----------|----------:|----------:|
| 25% | 67.5% | 70.1% |
| 50% | 67.5% | 70.1% |
| 75% | 67.2% | 72.7% |

### Key Insight

Most of the predictive signal appears extremely early.

Using only the first 25% of market life, the model retains most of the predictive power achieved using the complete trajectory.

This suggests that market quality is revealed surprisingly early.

---

## Early Signal Interpretability

The final stage of the analysis investigated which early trajectory features drive predictive performance.

Logistic Regression coefficients were estimated at each horizon.

### Feature Importance Across Horizons

| Feature | 25% | 50% | 75% |
|----------|----------:|----------:|----------:|
| Early Reversals | 0.966 | 0.779 | 0.725 |
| Early Max Drawdown | 0.604 | 0.690 | 0.802 |
| Early Entropy | -0.072 | -0.437 | -0.500 |
| Early Trend | -0.176 | -0.376 | 0.443 |
| Early Probability Range | 0.003 | 0.009 | 0.159 |
| Early Realized Volatility | -0.315 | -0.125 | 0.023 |

### Interpretation

Three signals consistently emerged:

#### Early Reversals

Markets that frequently revise beliefs during their early stages are more likely to become efficient.

#### Early Max Drawdown

Markets that aggressively correct mistakes converge to more accurate forecasts.

#### Early Entropy

Efficient markets exhibit lower entropy, suggesting less randomness and more structured information incorporation.

---

## Final Ranking of Signals

Across all analyses:

1. Early Reversals
2. Early Max Drawdown
3. Early Entropy
4. Trend
5. Probability Range
6. Realized Volatility

---

## Main Conclusion

Prediction market efficiency is not primarily determined by market category.

Instead, efficiency is strongly associated with the dynamics of information incorporation.

Efficient markets are characterized by:

- Frequent early belief revisions
- Strong error-correction dynamics
- Lower information entropy
- Early emergence of predictive signals

The evidence suggests that market quality can be identified long before market resolution using only trajectory-based information.