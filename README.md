# Credit-Risk-Probability-Model-for-Alternative-Data-Week4

Project: Credit Risk Probability Model 
This repository contains code, notebooks and documentation for Week 4 analysis tasks

## Used structure
.
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   ├── api/
│      ├── main.py
│      ├── pydantic_models.py
├── tests/
│   ├── test_data_processing.py
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md


## Quickstart
1. Clone repository
     git clone <repo-url>
2. Create environment
3. Creating src folder and creating the necessary .py files to include classes and functions involved in training and preprocessing.
4. Creating notebooks folder to create the EDA notebook for exploring the data and necessary visualizations.
5. Creating data folder for storing all related data



## Branches
| Branch     | Purpose                                |
| ---------- | -------------------------------------- |
| **task-1** | Understanding Credit Score business    |
| **task-2** | Understanding and Exploring Data       |


## Key Deliverables 

## Credit Scoring Business Understanding

Transparency and accountability, accurate risk measurement are areas Basel II Accord places a strong emphasis on. Financial institutions are not only required to quantify credit risk but also to demonstrate how risk estimates are produced, validated and governed. This makes interpretability and documentation critical. An interpretable model allows stakeholders to understand the drivers of risk, validate assumptions, and ensure compliance with capital adequacy requirements. A well-documented model helps in explaining and justifying results.

Given that a direct ‘default’ label doesn’t exist in our dataset, creating a proxy variable to simulate the ‘default’ label after considering other variables is helpful. Using indicators such as severe delinquency, write-offs, prolonged non-payment will allow supervised learning and model development. But this approach has its cons in introducing business risks. The model might produce wrong proxies which can lead to misclassification or bias (it might cause false positives and/or true negatives). This will ultimately lead to wrong decisions causing financial losses, regulatory scrutiny and unfair customer treatment. So careful proxy design along with validation and clear communication/disclosure of its limitations is important.

Given that various regulatory bodies require disclosure on how an individual is given his/her credit score, an interpretable model is preferred in such situations. Such models provide transparency, stability, ease of validation and clear economic interpretation of risk drivers. So even though their predictive power is limited, simple, interpretable models are suitable for reporting and long-term governance. Complex, high-performance models provide superior predictive accuracy by capturing non-linear patterns. When it comes to validating or explaining these models they become problematic. The choice of model should consider the factors: regulatory compliance, explainability, operational stability, and business performance.