# <b> Challenge goals </b>

The goal is to estimate the main direction that will occur during the last two hours of trading session, given the preceding history of the day.

To avoid to suffer of usual market noise, we only consider 3 states:

clear decreasing of price;
small evolution in both side;
clear increasing of price.


---

### Data description

- We provide input price evolutions (returns), with a granularity of 5 minutes, which leads to 53 values (4.5 hours) per day per equity

- As price movements are really small on such time windows, we give basis points (bps), so $ \frac{P_{t+5_{min} - P_t}}{P_t} * 10^{4} $

we given the rows: 
- 'ID' The unique input identifier 
-  'day' the day identifier (not unique inside dataset(s))
- 'equity', the equity identifier (not unique inside dataset(s))
- 'r0'. $ \frac{P_{09:35} - P_{09:30}}{P_{09:30}} * 10^4 $ <b> The returns of the first 5 min </b>
- 'r1'. $ \frac{P_{09:40} - P_{09:35}}{P_{09:35}} * 10^4 $ <b> The returns of the next 5 min </b>
- $ ... $
- 'r52'. $ \frac{P_{14:00} - P_{13:55}}{P_{13:55}} * 10^4 $ <b> The last returns </b>

To reduce the prediction task difficulty, we limit the prediction to the classification of the final returns, in 3 categories, limited by  ±25, so output is:

- -1 if $ \frac{P_{16PM} - P_{14PM} }{P_{14PM}} * 10^4 $  is below -25bps; 
- 0 if this ratio is between -25 and 25bps; 
- +1 if greater than 25 bps

'ID', the unique input identifier, correspond to the input ones;
'reod', the class of the returns during the the end of the day period, in [−1,0,1].

The training set and test set don't share the same days neither same equities. Nevertheless, all equities are of the same markets and share the same distribution. However, days of two datasets are from totally different periods, in order to reflect a real task of prediction, with fresh data coming from real world with potentially new characteristics.

#### <b> BENCHMARK description </b>

As this is a classification with only 3 potentials results, random (or even fixed!) responses might lead to a score around  33%. This is an easy way to test your solution.

The benchmark is less naive, and aggregates some basic characteristics of these 53 values, and try to detect pattern given their 2 main characteristics (day and equity). Then a basic state-of-art classifier leads to a (test) score of  41.74%