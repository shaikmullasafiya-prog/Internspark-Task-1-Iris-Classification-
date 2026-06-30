InternSpark Machine Learning Internship Portfolio
Candidate Information
Name: Shaik Mulla Safiya
Domain: Machine Learning Intern
Repository Structure
This repository serves as the official project tracking portfolio for my engineering milestones during the InternSpark remote machine learning residency. The workspace is organized into modular task directories containing production notebooks, serialized weights, and performance briefs.

Task 1: Iris Flower Species Classification
Directory: /Task1_Iris_Classification
Core Objective: Implemented an end-to-end Supervised Classification Pipeline to classify Iris species based on morphometric features.
Pipeline Infrastructure: Built a baseline Logistic Regression model using scikit-learn. Implemented StandardScaler feature normalization to neutralize feature dominance.
Key Results: Achieved an overall classification accuracy of 96.67%. Serialized operational model states into portable iris_logistic_regression_model.pkl formats using joblib.
Task 2: California House Price Prediction
Directory: /Task2_House_Price_Prediction
Core Objective: Engineered a Supervised Continuous Regression workflow to predict median real estate property valuations using regional demographic and socioeconomic features.
Pipeline Infrastructure: Benchmarked an Ordinary Least Squares (OLS) Linear Regression baseline against a robust, non-linear Random Forest Regressor ensemble (max_depth=15).
Key Results: The Random Forest ensemble successfully captured spatial geographic coordinates and wealth density profiles, drastically outperforming the linear baseline by maximizing the variance explanation ratio (
R
2
). Exported compressed, deployment-ready production artifacts as house_price_random_forest_model.pkl.
Task 3: Titanic Survival Prediction
Directory: /Task3_Titanic_Survival
Core Objective: Engineered a binary Supervised Classification pipeline to predict passenger survivability by mapping structural attributes and custom engineered features.
Pipeline Infrastructure: Conducted categorical imputation and advanced feature engineering (extracting social titles, calculation of global family sizes, and mapping cabin presence indicators). Deployed a fine-tuned RandomForestClassifier ensemble to handle complex categorical feature interactions.
Key Results: Successfully evaluated predictive stability using precision, recall, and global feature importance metrics to verify transparent model explainability. Exported compressed, deployment-ready production model weights as titanic_random_forest_model.pkl.
