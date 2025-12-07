# E-Commerce Customer Churn Prediction 
## Saving At-Risk Buyers: Machine Learning Approach to Reduce Churn in E-Commerce

![alt text](Products/Simulator.gif)

### 🌐 Live App  
[Try it here! Customer Churn Predictor](https://e-commerce-customer-churn-prediction-25.streamlit.app/)

---

## 📌 Project Overview

In modern e-commerce, large volumes of customers make their first purchase during **flash sales, vouchers, or free shipping campaigns**. However, many of them don't return once the promotion ends — causing:

* Declining transactions despite high user acquisition
* Rising **Customer Acquisition Cost (CAC)**
* High competition in price, delivery speed, and promo intensity

This leads to a situation where **companies lose customers faster than they gain new ones**, making retention more valuable than constant acquisition.

This project applies **machine learning to predict customer churn**, identify the strongest churn drivers, and generate **strategic actions** that improve retention at scale.

---

## 📊 Problem Definition

* **Churn = 1** → Customer has left the platform
* **Churn = 0** → Customer is still active

Churn rate in this dataset is **~16.8% (about 1 in 6 customers)**.
Unmitigated, this results in:

* Lost revenue
* Lower customer lifetime value
* Inefficient marketing spend
* Loyalty campaigns wasted on the wrong customers

Machine learning can detect early warning trends before customers churn.

---

## 🧠 Model Summary

* **Model:** Tuned LightGBM
* **Resampling:** Random Over Sampling (ROS)
* **Feature Selection:** SelectKBest (f_classif)
* **Primary Metric:** **F2-Score = 0.901379**

| Prediction Result        | Count |
| -------------------------| ----- |
| True Negatives  (0 → 0)  | 921   |
| False Positives (0 → 1)  | 15    |
| True Positives  (1 → 1)  | 173   |
| False Negatives (1 → 0)  | 17    |

📌 Interpretation:

* Excellent at identifying non-churn users
* Strong recall in detecting churners
* Very low false positives and false negatives
* Suitable for **cost-efficient targeted retention campaigns**

---

## 👥 Stakeholder

**Customer Marketing & CRM Team**

Can use the model to:

* Proactively reach high-risk users
* Improve LTV
* Allocate retention budget more efficiently
* Enhance intervention timing

---


## 🔎 Key Insights

Churn is driven by:

* **Low cashback rewards**
* **Short customer tenure**
* **Recent unresolved complaints**
* **Higher churn in lower-tier cities**

Interestingly:

➡️ Customers with **Satisfaction Score = 5** show a surprising churn increase — suggesting hidden dissatisfaction or mismatched feedback behavior.

---

## 💡 Business Recommendations

1. **Boost Cashback Incentives**
   Especially for price-sensitive or low-tenure customers.

2. **Improve Delivery Experience**
   Service gaps may be causing geographic churn disparities.

3. **Investigate Satisfaction Score Anomalies**
   High ratings followed by churn hint at structural survey or expectation issues.

4. **Re-Engage Low-Order Customers**
   Personalized win-back vouchers and messaging are highly effective.

5. **Act Quickly on Complaints**
   Improve ticket resolution SLAs and track post-resolution churn.

---

## 🛠️ Tech Stack

<!-- Programming Language -->
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

<!-- Data Processing -->
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)

<!-- Visualization -->
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C8CB5?logo=seaborn&logoColor=white)

<!-- Machine Learning -->
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-2D3E50?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5B29?logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-056D50?logo=lightgbm&logoColor=white)

<!-- Notebook / App -->
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)


---

## 📄 Author
Ahmad Faik — Data Scientist
 
`Passionate about using machine learning to solve real-world business problems.`

[Portfolio](https://ahmadfaik.netlify.app/) | [LinkedIn](https://www.linkedin.com/in/ahmad-faik-5b64b0308/)

---

Ready to predict churn and turn insights into business impact? Let’s begin!
