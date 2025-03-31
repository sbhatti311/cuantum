# 🧠 My Structured Approach to Data Wrangling & Descriptive Analysis

I follow a repeatable and thoughtful data pipeline process to ensure data is clean, meaningful, and actionable before modeling. Here's the end-to-end framework I typically follow:

---

## 1. 🧹 Initial Data Cleaning
- **Column Name Standardization**
```python
df.columns = df.columns.str.strip()
```
- **Date Formatting & Regex Handling**
```python
import re
match = re.match(r"([A-Za-z]+)\s*(\d+)", "CA 90210")
```
- **Check for Zero Values**
```python
df.isin([0]).sum()
```

---

## 2. ⚠️ Outlier Detection & Correction
> 📌 Catch outliers *before* imputation and scaling to avoid skewing statistics.

- **IQR Method**
```python
def outlier_iqr(col):
    Q1, Q3 = np.percentile(col, [25, 75])
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
```

- **Z-Score Method**
```python
from scipy import stats
z_scores = np.abs(stats.zscore(df['Income']))
df.loc[z_scores > 3, 'Income'] = df['Income'].median()
```

---

## 3. 🔍 Missing Data Handling
- **Count Nulls**
```python
print(df.isnull().sum())
```
- **Impute with Mean or Most Frequent**
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['Age']] = imputer.fit_transform(df[['Age']])
```

---

## 4. 📊 Descriptive Analytics & Visualization
- **Correlation Matrix & Heatmap**
```python
corr = df.corr()
sns.heatmap(corr, annot=True)
```

- **Scatterplots**
```python
ax.scatter(df['LSTAT'], df['MEDV'], color='green')
```

---

## 5. 🛠 Feature Engineering & Transformation
- **Log Transform**
```python
df['Log_Income'] = np.log(df['Income'])
```

- **Binning**
```python
age_bins = [20,30,40,50,60]
age_labels = ['20-29', '30-39', '40-49', '50-59']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
```

---

## 6. 🧮 Preprocessing: Scaling & Encoding
- **Standard Scaling**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- **Encoding Categorical Features**
```python
from sklearn.preprocessing import OneHotEncoder
encoded = OneHotEncoder().fit_transform(df[['Color']])
```

---

## 7. 🧱 Feature Selection
- **Remove Constant & Low Variance Features**
```python
from sklearn.feature_selection import VarianceThreshold
df = df.loc[:, VarianceThreshold(threshold=0.1).fit(df).get_support()]
```

- **Mutual Information for Feature Importance**
```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(df[features], df['Target'])
```

---

## 8. 🔗 Merging, Aggregating & Reshaping
- **GroupBy & Aggregation**
- **Merge & Join (inner, outer, left, right)**
- **Concatenation (vertical & horizontal stacking)**

---

## 9. 🚂 Train/Test Splitting & Model-Ready Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
```

- **Reproducibility**: `random_state=42`
- **Stratified Sampling** ensures balanced target classes

---

## 10. 🤖 Modeling & Evaluation
- **Train & Predict**
```python
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

- **Evaluation Metrics**
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

### ✨ Key Philosophies
- Reproducibility is sacred — always seed your splits and transformations
- Intuition is as critical as technique — visuals > metrics when exploring
- Keep everything modular and reusable — think pipelines
