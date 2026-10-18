# TCS-349: Responsible and Explainable AI Assignment

**Name:** Anubhav
**Course:** Responsible and Explainable AI(TCS-349)
**Program:** B.Tech CSE (AI/ML)
**Section:** ML1
**Roll No:** 12

## About this assignment

In this assignment I took one big dataset and trained 4 different machine
learning models on it. Then I compared them. Along with the normal scores, I
also checked whether the models treat younger and older people differently,
because that is the main point of Responsible AI.

## Dataset

I used the **Bank Account Fraud (BAF) dataset** from Kaggle. It comes from a
NeurIPS 2022 paper by Jesus et al. Each row is one bank account application
and the goal is to predict if the application is fraud (`fraud_bool` = 1) or
not (0).

- Link: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- File used: `Base.csv`
- Size: 1,000,000 rows and 32 columns
- Only about 1.1% of the applications are fraud

I picked this dataset because it is large and the authors made it for studying
fairness. The data is synthetic, so no real person's information is in it.
The dataset is not uploaded here because the file is too big for GitHub. To run
my code, download `Base.csv` from the link and put it in a folder named `data`.

## What I did

**1. Looking at the data**
First I checked the shape and how many rows are fraud. Since only 1.1% are
fraud, accuracy is not useful here. A model that says "not fraud" for everyone
would still get about 98.9% accuracy. So I used precision, recall and F1.

I also checked the fraud rate for each age group. It was about 0.35% for the
10s group and went up to about 5.3% for the 90s group. So older applicants have
more fraud in this data.

**2. Cleaning**
- `device_fraud_count` has the same value in every row, so I dropped it.
- There were no empty cells, but some columns have `-1` in them. I could not
  find in the paper what `-1` means, so I assumed it means "missing" for these
  5 columns: `prev_address_months_count`, `current_address_months_count`,
  `bank_months_count`, `session_length_in_minutes` and
  `device_distinct_emails_8w`. I changed `-1` to NaN and filled them with the
  median. This is my assumption and I may be wrong.
- Some other columns (like `credit_risk_score`) also have negative values, but I
  left them as they are because they look like real values.
- 5 columns have text in them. I changed them to 0/1 columns using
  `pd.get_dummies`.

**3. Splitting the data**
I used `train_test_split` two times to make 60% training, 20% validation and
20% test. I used `stratify` so the fraud percentage stays about the same in all
three parts. I used the training data to train, the validation data to compare
models, and the test data only at the end.

I filled the missing values using the median of the training data only, so the
validation and test data don't affect anything.

**4. Scaling**
I used `StandardScaler` so all columns are on a similar scale. It was fit on
training data only.

**5. Models**
I trained these 4 models:
1. Logistic Regression
2. Decision Tree (max_depth = 5)
3. Random Forest (50 trees, max_depth = 10)
4. Naive Bayes

For the first three I used `class_weight="balanced"` so the models don't ignore
the fraud cases. Naive Bayes doesn't have this option.

**6. How I checked the models**
- Precision, recall and F1 score.
- For fairness, I made two groups: age 50 and above (older) and below 50
  (younger). The age column comes in steps of 10 (10, 20, 30...), and 50 and
  above is about 18% of the data. For each group I found the **false positive
  rate**, which means honest customers who were wrongly flagged as fraud. Then
  I divided the smaller FPR by the bigger one (`FPR_ratio`). If it is close to
  1, both groups are treated equally. If it is close to 0, one group is flagged
  much more.

## Results

Validation set:

| Model | Precision | Recall | F1 | FPR older | FPR younger | FPR ratio |
|-------|-----------|--------|----|-----------|-------------|-----------|
| Logistic Regression | 0.043 | 0.790 | 0.081 | 0.365 | 0.161 | 0.440 |
| Decision Tree | 0.037 | 0.731 | 0.070 | 0.307 | 0.192 | 0.624 |
| Random Forest | 0.051 | 0.747 | 0.096 | 0.291 | 0.124 | 0.426 |
| Naive Bayes | 0.015 | 0.941 | 0.030 | 0.804 | 0.638 | 0.794 |

Test set (final check):

| Model | Precision | Recall | F1 | FPR older | FPR younger | FPR ratio |
|-------|-----------|--------|----|-----------|-------------|-----------|
| Logistic Regression | 0.043 | 0.787 | 0.081 | 0.364 | 0.159 | 0.436 |
| Decision Tree | 0.036 | 0.712 | 0.068 | 0.308 | 0.193 | 0.627 |
| Random Forest | 0.050 | 0.738 | 0.094 | 0.297 | 0.125 | 0.422 |
| Naive Bayes | 0.016 | 0.943 | 0.031 | 0.802 | 0.637 | 0.794 |

## What I found

- [Which model had the best recall / F1, and which was the worst.]
- [What happened to precision when recall went up. What does it mean for an
  honest customer?]
- [Which model was the most fair and which was the least, using FPR ratio.
  Were older people flagged more?]
- [Anything that surprised me.]

## Explainability

[Write this part after doing the explainability step: which method I used
(feature importance / SHAP / etc.), which features mattered most, and whether
the top features make sense for fraud.]

## Problems and limitations

- I split the data randomly. But this data has a `month` column and fraud
  changes over the months. The dataset authors suggest using the first 6
  months for training and the last 2 for testing. A random split can give
  better scores than real life, so my results may be a bit too optimistic.
- My idea that `-1` means missing is an assumption.
- I only used age to check fairness. The dataset also has income and employment
  status that could be checked.
- The data is synthetic and the authors say it is only for testing methods, so
  these models should not be used for real fraud detection.
- I did not tune the models much, so the scores can probably be improved.

## How to run

1. Clone this repo.
2. Install the libraries: `pip install -r requirements.txt`
3. Download `Base.csv` from the Kaggle link and put it in the `data/` folder.
4. Open `notebooks/assignment.ipynb` and run all the cells.

## Reference

Jesus, S. et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic
Tabular Datasets for ML Evaluation.* NeurIPS 2022.