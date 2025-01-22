# **Adjusted Price Forecast: Chedraui**
This project applies probability, statistical techniques, and time series models to analyze and predict the adjusted prices of Chedraui's stock. Below is a summary of the analysis and models implemented.

---

## **1. Data Visualization**
- **Last 20 Historical Data Points:** Plotted adjusted closing prices (`Adj Close`) for the last 20 days to observe recent trends.  
- **Volatility:** Included transaction volume (`Volume`) as a key indicator to identify potential trend changes or significant movements.

---

## **2. Time Series Components**
- **Trend:** Assesses the overall direction of prices over time to identify long-term patterns.  
- **Seasonality:** Detects repetitive patterns (quarterly, monthly) potentially influenced by financial events.  
- **Residual Component:** Captures unpredictable variations (noise) not explained by trend or seasonality.

---

## **3. Preprocessing**
- **Stationarity:**  
  - The Dickey-Fuller test was used to verify whether the series was stationary.  
  - If the p-value was less than 0.05, the series was concluded to be stationary.  
- **Differencing:** Applied to remove trends and stabilize variance.  
- **Logarithmic Returns:** Calculated daily and monthly returns for further analysis.  

---

## **4. Time Series Models**
### **ARIMA**  
- **Parameter Optimization:**  
  - The best parameters (p, d, q) were selected based on AIC and BIC scores.  
  - The optimal model was ARIMA (0, 0, 2), which includes:  
    - No autoregressive component (`p = 0`).  
    - No additional differencing (`d = 0`).  
    - A moving average component of order 2 (`q = 2`).  
- **Evaluation:**  
  - Residuals were essentially white noise, indicating a good fit.  
  - RMSE was used to measure model performance on the test set.

### **SARIMA**  
- Included seasonal components to capture repetitive patterns over time.  
- Performed better on data with strong seasonality.

---

## **5. Results**
- **Forecast Visualization:**  
  - The plot compares the last historical data points with ARIMA and SARIMA predictions.  
  - **ARIMA:** Fits well to recent data and provides reliable short-term predictions.  
  - **SARIMA:** Underestimates future behavior, which could be adjusted with additional data.  

- **Summary of Findings:**  
  - The models successfully captured key historical patterns.  
  - The ARIMA model was the most suitable for the data due to its simplicity and short-term accuracy.  

---

## **6. Conclusion**
The analysis highlights the importance of combining visualization techniques, preprocessing, and probabilistic models to predict adjusted prices. Evaluation metrics and residual plots confirm that ARIMA and SARIMA models are powerful tools for this type of data.

---

## **Script Execution**
1. **Libraries Used:**  
   - `pandas`, `numpy`, `matplotlib`, `statsmodels`, `seaborn`, among others.  
2. **Script Steps:**  
   - Data cleaning.  
   - Exploratory analysis (visualization and time series components).  
   - Implementation of ARIMA and SARIMA.  
   - Visualization and evaluation of results.  

---

## **Forecast Plot**
![Forecast Plot](Prediction_of _Chedraui's_ stock_ using_ time_ series/Iamge/pronostico.jpg)

---

## **Next Steps**
- Test other advanced models, such as Prophet.  
- Adjust SARIMA hyperparameters for improved accuracy.  
- Implement hybrid models combining neural networks with time series approaches.
