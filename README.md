[README_Task2.md](https://github.com/user-attachments/files/26661175/README_Task2.md)
Task 2: Exploratory Data Analysis (EDA)

> **AI & ML Internship — Elevate Labs | Task 2**  
> Understand data using statistics and visualizations before building any ML model.

---

## 📌 Objective

Perform a complete Exploratory Data Analysis (EDA) on the Titanic dataset to understand the data distribution, relationships between features, patterns, trends, and anomalies — using statistical summaries and rich visualizations.

---

## 🗂️ Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Steps Covered](#steps-covered)
- [Key Visualizations](#key-visualizations)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [References](#references)

---

## 📖 Overview

EDA is the process of visually and statistically examining a dataset before applying any machine learning model. It helps answer questions like:

- What does the data look like? Are there missing values or outliers?
- How are individual features distributed?
- Which features are correlated with the target variable?
- Are there any patterns or trends that can guide feature selection?

In this task, EDA is performed on the **Titanic dataset** — a classic dataset with real-world messiness, mixed data types, and a binary survival target variable.

---

## 📊 Dataset

**Titanic Dataset** — Passenger information and survival labels from the RMS Titanic disaster.

| Feature | Type | Description |
|---------|------|-------------|
| `PassengerId` | int | Unique identifier |
| `Survived` | int | Target: 0 = No, 1 = Yes |
| `Pclass` | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | object | Passenger name |
| `Sex` | object | Gender |
| `Age` | float | Age in years |
| `SibSp` | int | Siblings/spouses aboard |
| `Parch` | int | Parents/children aboard |
| `Ticket` | object | Ticket number |
| `Fare` | float | Passenger fare |
| `Cabin` | object | Cabin number |
| `Embarked` | object | Port of embarkation (C, Q, S) |

**Download:** [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) | Core language |
| ![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas) | Data loading & manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy) | Numerical computations |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-orange) | Static visualizations |
| ![Seaborn](https://img.shields.io/badge/Seaborn-0.x-4C72B0) | Statistical plots |
| ![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly) | Interactive visualizations |

---

## 📁 Project Structure

```
task-2-eda/
│
├── data/
│   └── titanic.csv               # Raw dataset
│
├── notebooks/
│   └── eda_analysis.ipynb        # Full EDA notebook (step by step)
│
├── plots/
│   ├── histograms.png            # Distribution plots
│   ├── boxplots.png              # Outlier visualization
│   ├── correlation_heatmap.png   # Feature correlation
│   ├── pairplot.png              # Feature pair relationships
│   └── survival_by_features.png  # Pattern analysis charts
│
├── src/
│   └── eda.py                    # Reusable EDA functions
│
├── requirements.txt
└── README.md
```

---

## 🔢 Steps Covered

### Step 1 — Summary Statistics

Generate descriptive statistics to understand the data numerically before visualizing.

```python
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Basic stats: count, mean, std, min, max, quartiles
print(df.describe())

# Categorical column summary
print(df.describe(include='object'))

# Skewness and Kurtosis
print("Skewness:\n", df[['Age', 'Fare', 'SibSp', 'Parch']].skew())
print("Kurtosis:\n", df[['Age', 'Fare', 'SibSp', 'Parch']].kurtosis())

# Survival rate
print(f"Overall survival rate: {df['Survived'].mean()*100:.1f}%")
```

---

### Step 2 — Histograms & Boxplots

Visualize the distribution and spread of each numerical feature.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for all numeric columns
df[['Age', 'Fare', 'SibSp', 'Parch']].hist(
    bins=20, figsize=(12, 8), color='steelblue', edgecolor='white'
)
plt.suptitle('Histograms of Numeric Features', fontsize=14)
plt.tight_layout()
plt.show()

# Boxplots — Age and Fare vs Survival
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(x='Survived', y='Age', data=df, palette='Set2', ax=axes[0])
axes[0].set_title('Age vs Survival')
sns.boxplot(x='Survived', y='Fare', data=df, palette='Set2', ax=axes[1])
axes[1].set_title('Fare vs Survival')
plt.tight_layout()
plt.show()
```

---

### Step 3 — Pairplot & Correlation Matrix

Understand relationships between all numerical features at once.

```python
# Pairplot — scatter matrix colored by survival
sns.pairplot(
    df[['Age', 'Fare', 'SibSp', 'Parch', 'Survived']].dropna(),
    hue='Survived',
    palette={0: 'coral', 1: 'steelblue'},
    plot_kws={'alpha': 0.5}
)
plt.suptitle('Pairplot — Titanic Features', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 6))
corr = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

---

### Step 4 — Patterns, Trends & Anomalies

Dig into the data with grouped and category-based visualizations.

```python
# Survival rate by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=df, palette='Blues_d')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=df, palette='Set1')
plt.title('Survival Rate by Sex')
plt.show()

# Age distribution by Survival
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True,
             palette={0: 'coral', 1: 'steelblue'})
plt.title('Age Distribution: Survived vs Not Survived')
plt.show()

# Fare by Embarkation port
sns.boxplot(x='Embarked', y='Fare', data=df, palette='Set3')
plt.title('Fare Distribution by Port of Embarkation')
plt.show()

# Survival count by class
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
plt.title('Survival Count by Passenger Class')
plt.legend(['Did not survive', 'Survived'])
plt.show()
```

---

### Step 5 — Feature-Level Inferences

Summarize what the visuals tell you — these become your conclusions.

```python
# Cross-tabulation: Survival by Sex and Class
print(pd.crosstab(df['Sex'], df['Survived'], margins=True))
print(pd.crosstab(df['Pclass'], df['Survived'], margins=True))

# Survival rate by Age Group
df['AgeGroup'] = pd.cut(df['Age'],
                         bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

print("\nSurvival rate by age group:")
print(df.groupby('AgeGroup')['Survived'].mean().round(2))
```

---

## 📈 Key Visualizations

| Plot | What it Shows |
|------|--------------|
| Histogram | Distribution shape of Age, Fare, SibSp, Parch |
| Boxplot | Median, IQR, and outliers per feature |
| Pairplot | Pairwise relationships between all numeric features |
| Heatmap | Correlation strength between features and target |
| Barplot | Survival rates across categories (Sex, Pclass) |
| KDE Plot | Overlapping age distributions for survived vs not |

---

## 🔍 Key Findings

| Feature | Finding |
|---------|---------|
| **Sex** | Females survived at ~74% vs ~19% for males — strongest single predictor |
| **Pclass** | 1st class: 63% survival rate vs 3rd class: only 24% |
| **Age** | Children (< 12 years) had noticeably higher survival rates |
| **Fare** | Higher fare positively correlated with survival (r ≈ +0.26) |
| **Embarked** | Cherbourg (C) passengers had higher survival — more 1st class travellers |
| **SibSp/Parch** | Small families (1–2 members) survived more than solo or large family travellers |
| **Pclass vs Fare** | Strong negative correlation — higher class = lower class number = higher fare |

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/task-2-eda.git
cd task-2-eda
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook notebooks/eda_analysis.ipynb
```

### 4. Fix Plotly rendering (if needed)

If you get a `ValueError: Mime type rendering requires nbformat>=4.2.0`, run:

```bash
pip install --upgrade nbformat ipython plotly
```

Then restart the kernel. Or use the browser renderer:

```python
import plotly.io as pio
pio.renderers.default = "browser"
```

---

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.0.0
jupyter>=1.0.0
nbformat>=4.2.0
ipython>=8.0.0
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn plotly jupyter nbformat ipython
```

---

## 💡 EDA Quick Reference

| Goal | Best Plot |
|------|----------|
| Distribution of one feature | `histplot` / `displot` |
| Spread + outliers | `boxplot` / `violinplot` |
| Two features together | `scatterplot` / `pairplot` |
| All correlations at once | `heatmap` |
| Category vs numeric | `barplot` / `boxplot` |
| Count of categories | `countplot` |
| Interactive charts | `plotly.express` |

---

## 📚 References

- [Titanic Dataset — Kaggle](https://www.kaggle.com/competitions/titanic)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [Pandas Profiling](https://pandas-profiling.ydata.ai/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)

---

## 👤 Author

**Your Name**  
Atish Barik 

AI & ML Internship — Elevate Labs | Task 2 | 

---

## 📄 License

This project is for educational purposes as part of the Elevate Labs AI & ML Internship program.


