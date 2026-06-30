# Exploratory Data Analysis Observations - Titanic Dataset

This report documents the key insights and analytical findings discovered during the Exploratory Data Analysis (EDA) of the Titanic dataset.

---

## 1. Dataset Overview and Completeness

- **Total Passengers in Dataset**: 891
- **Features (Columns)**: 12 (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)

### Missing Values Summary
- **Age**: 177 missing values (19.87%) - *High missingness; requires imputation or special care in modeling.*
- **Cabin**: 687 missing values (77.10%) - *Severely incomplete. May be converted to a binary indicator (Cabin recorded vs. not).*
- **Embarked**: 2 missing values (0.22%) - *Extremely minor missingness (only 2 records).*

---

## 2. Demographic Analysis & Survival Patterns

### Survival Rates by Gender (Sex)
- **Overall Survival Rate**: 38.38% (342 survived, 549 died)
- **Female Survival Rate**: 74.20%
- **Male Survival Rate**: 18.89%

> [!NOTE]
> There is a massive disparity in survival rate between genders, strongly aligning with the historic "women and children first" protocol. Females were nearly **4 times** more likely to survive than males.

### Survival Rates by Socioeconomic Status (Pclass)
- **1st Class (Pclass 1)**: 62.96% survival rate
- **2nd Class (Pclass 2)**: 47.28% survival rate
- **3rd Class (Pclass 3)**: 24.24% survival rate

> [!IMPORTANT]
> Ticket class was a highly significant factor in survival. Over 62% of 1st-class passengers survived, compared to only 24.24% of 3rd-class passengers. This indicates a strong socioeconomic bias in rescue accessibility.

### Survival Rates by Embarkation Port
- **Cherbourg (C)**: 55.36% survival rate
- **Queenstown (Q)**: 38.96% survival rate
- **Southampton (S)**: 33.70% survival rate

> [!NOTE]
> Passengers embarking from Cherbourg (C) had a higher survival rate. Further analysis shows this is strongly correlated with a larger proportion of 1st-class passengers boarding at Cherbourg.

### Survival Rates by Age Groups
- **Children (0-12)**: 57.97% survival rate
- **Teenagers (13-18)**: 42.86% survival rate
- **Young Adults (19-35)**: 38.27% survival rate
- **Adults (36-60)**: 40.00% survival rate
- **Seniors (60+)**: 22.73% survival rate

> [!TIP]
> Children had a significantly higher survival rate (nearly 58%), confirming priority was given to the young. Conversely, seniors (60+) had the lowest survival rate (26.9%), showing the physical difficulty or lack of priority in evacuating.

---

## 3. Numerical Feature Distributions and Correlations

### Age and Fare Distributions
- **Age**: Moderately normal distribution, peaking around 20-30 years of age. Median age is 28, mean is 29.7.
- **Fare**: Extremely right-skewed. The majority of tickets cost under $50, but a few outlier tickets exceeded $500. This requires log-transformation or scaling for machine learning.

### Correlation Analysis
Key relationships from the correlation heatmap:
1. **Pclass & Fare (-0.55)**: Strong negative correlation, confirming 1st class (lower numerical rank) corresponds to higher ticket fares.
2. **Pclass & Survived (-0.34)**: Moderate negative correlation, indicating higher-class passengers (lower numerical rank) had higher survival.
3. **Fare & Survived (0.26)**: Positive correlation, suggesting those who paid more had higher odds of survival.
4. **SibSp & Parch (0.41)**: Moderate positive correlation, representing families traveling together.
