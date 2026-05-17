# Smart Hotel Energy Optimizer

A Machine Learning Pipeline and Interactive Web Dashboard for Predictive HVAC Management.

Live Demo: https://energy-optmizer-project-cjygzqwxtnsheg7qzscbhq.streamlit.app

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
Local Installation and Setup
Follow these steps to run the application locally on your machine.

1. Clone the repository

Bash
git clone [https://github.com/YourUsername/Smart-Hotel-Optimizer.git](https://github.com/YourUsername/Smart-Hotel-Optimizer.git)
cd Smart-Hotel-Optimizer
2. Create a virtual environment
It is highly recommended to use a virtual environment to prevent dependency conflicts.

Bash
# For Windows:
python -m venv venv
venv\Scripts\activate

# For macOS/Linux:
python3 -m venv venv
source venv/bin/activate
3. Install dependencies

Bash
pip install -r requirements.txt
4. Execute the Machine Learning Pipeline
If you need to reconstruct the processed data or retrain the models from scratch, run the pipeline scripts in the following sequence:

Bash
python scripts/step2_preprocessing.py
python scripts/step3_eda.py
python scripts/step4_modeling.py
5. Launch the Web Dashboard

Bash
streamlit run app/main.py
The application will automatically launch in your default web browser at http://localhost:8501.

System Constraints and Design Logic
This project was developed as a final-year Machine Learning academic capstone. To ensure algorithmic stability and prevent mathematical drift, the user interface inputs for guest capacity and stay duration are strictly capped (1 to 10 guests, 1 to 14 nights). This aligns the testing environment with the normal statistical boundaries established during the Exploratory Data Analysis phase.
