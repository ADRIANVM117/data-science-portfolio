# <b>Quant Credit Risk Modeler  </b>


LendingClub_Risk_Modeling/
│
├── data/
│   ├── raw/
│   │   └── accepted_2007_to_2018Q4.csv
│   ├── interim/
│   │   └── lendingclub_2016_2018.csv
│   └── processed/
│       └── datasets_ready_for_modeling.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_PD_Modeling.ipynb
│   ├── 03_LGD_Analysis.ipynb
│   ├── 04_EAD_Analysis.ipynb
│   └── 99_Report_Generation.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── eda_histograms.png
│   │   ├── pd_roc_curve.png
│   │   └── lgd_histogram.png
│   └── LendingClub_EDA_Report.pdf
│
├── src/
│   ├── data_preparation.py
│   ├── pd_modeling.py
│   ├── lgd_analysis.py
│   └── ead_analysis.py
│
├── requirements.txt
└── README.md
