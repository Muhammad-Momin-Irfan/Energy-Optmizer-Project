# Smart Hotel Energy Optimizer

A Machine Learning Pipeline and Interactive Web Dashboard for Predictive HVAC Management.

## Project Overview
The hospitality industry traditionally wastes significant energy by managing Heating, Ventilation, and Air Conditioning (HVAC) systems based on static schedules and external weather, entirely ignoring internal human occupancy. This project proposes a data-driven solution utilizing a dual-model Machine Learning architecture to bridge hospitality demand with building physics.

1. Demand Forecasting (Random Forest Regressor): Predicts hotel occupancy percentages by analyzing historical booking seasonality, stay duration, and demographic group sizes.
2. Building Physics Model (Multiple Linear Regression): Calculates the maximum physical thermodynamic load of a specific building based on standardized geometric features (Surface Area, Wall Area, Roof Area, and Building Height).
3. Integration Engine: A custom logic tier that dynamically scales the physical baseline energy load using the predicted human demand, outputting an optimized, real-time energy allocation.

## Technologies Used
* Language: Python 3.10+
* Data Engineering: pandas, numpy
* Machine Learning: scikit-learn, joblib
* Visualizations: matplotlib, seaborn
* Frontend Framework: streamlit

## Project Architecture
```text
Smart-Hotel-Optimizer/
├── app/
│   └── main.py                 # Interactive Streamlit web dashboard
├── data/
│   ├── raw/                    # Original UCI Energy and Hotel Booking datasets
│   └── processed/              # Cleaned and engineered datasets
├── models/                     # Serialized AI models (.pkl) for instant web loading
├── report/
│   └── figures/                # Automated EDA graphs generated during analysis
├── scripts/
│   ├── step2_preprocessing.py  # Data cleaning, imputation, and feature engineering
│   ├── step3_eda.py            # Graph generation for the Technical Report
│   ├── step4_modeling.py       # ML training pipeline and standard scaling
│   └── step5_6_prediction.py   # Terminal-based integration unit test
├── .gitignore                  # Git exclusion rules
├── requirements.txt            # Python environment dependencies
└── README.md                   # Project documentation