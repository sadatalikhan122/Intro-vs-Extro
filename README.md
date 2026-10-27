# Introvert vs Extrovert Personality Prediction

A machine learning classification project built in a **Kaggle Notebook** for the **Playground Series — Season 5, Episode 7** competition, *Predict the Introverts from the Extroverts*. The project uses a **Random Forest Classifier** to predict whether a person is an **Introvert** or **Extrovert** based on social behavior and lifestyle features.

## Project Highlights

- **Algorithm:** Random Forest Classifier (`n_estimators=100`, `random_state=42`)
- **Validation accuracy:** **96.36%** (80/20 train/validation split)
- **5-fold cross-validation mean accuracy:** **96.12%**
- **5-fold cross-validation standard deviation:** **0.31 percentage points**
- **Training dataset:** 18,524 rows × 9 columns
- **Competition test dataset:** 6,175 rows × 8 columns
- **Output:** `submission.csv` containing predicted personality labels

> **Note:** These are local validation and cross-validation scores from the notebook, **not a Kaggle leaderboard score**. The preprocessing in the original notebook is fitted before the validation split, so these metrics should be considered preliminary; see [Potential Improvements](#potential-improvements).

## Dataset

**Competition:** [Predict the Introverts from the Extroverts — Kaggle](https://www.kaggle.com/competitions/playground-series-s5e7)

The model uses seven input features:

| Feature | Description |
| --- | --- |
| `Time_spent_Alone` | Time spent alone |
| `Stage_fear` | Whether the person experiences stage fear |
| `Social_event_attendance` | Frequency of attending social events |
| `Going_outside` | Frequency of going outside |
| `Drained_after_socializing` | Whether socializing causes exhaustion |
| `Friends_circle_size` | Size of the person's friend circle |
| `Post_frequency` | Social media posting frequency |

**Target:** `Personality` (`Introvert` or `Extrovert`). The `id` column is kept for submission but is not used as a model feature.

## Workflow

1. **Load data** from `train.csv` and `test.csv` using pandas.
2. **Select features** and the `Personality` target.
3. **Handle missing values** using `SimpleImputer(strategy='most_frequent')`.
4. **Encode categorical inputs** (`Stage_fear`, `Drained_after_socializing`) with `OrdinalEncoder`.
5. **Encode target labels** with `LabelEncoder`.
6. **Split the data** into training (80%) and validation (20%) sets (`random_state=42`).
7. **Train** a `RandomForestClassifier` with 100 trees.
8. **Evaluate** with validation accuracy, 5-fold cross-validation, a confusion matrix, and feature importance.
9. **Predict on the competition test data** and export `submission.csv`.

## Results

| Metric | Result |
| --- | ---: |
| Validation accuracy | **96.36%** |
| Mean 5-fold CV accuracy | **96.12%** |
| CV standard deviation | **0.31 percentage points** |

**5-fold cross-validation accuracy scores:**

```text
[0.95843455, 0.96248313, 0.95897436, 0.95924426, 0.96679266]
```

The notebook also displays:

- A horizontal bar chart of Random Forest feature importances.
- A confusion matrix for validation predictions.

## Tech Stack

- Python
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Kaggle Notebooks

## How to Run

### Run on Kaggle (recommended)

1. Open the [competition page](https://www.kaggle.com/competitions/playground-series-s5e7).
2. Create a new Kaggle Notebook and attach the competition dataset.
3. Paste the project code into the notebook or upload your `.ipynb` file.
4. Run all cells. The original code expects the following input paths:

   ```python
   train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
   test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
   ```

5. Download the generated `submission.csv` from the notebook output.

### Run locally

Install the required packages:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

Download the competition data according to Kaggle's access rules and **change the two CSV paths** in the notebook/script to wherever you saved `train.csv` and `test.csv`. Then run the notebook or Python script.

## Submission Format

The code saves `submission.csv` with two columns:

```csv
id,Personality
18524,Extrovert
18525,Introvert
```

*The rows above illustrate the required format and are not claims about this model's actual predictions.*

## Potential Improvements

- **Prevent data leakage:** Split before fitting imputers/encoders, and put preprocessing inside an `sklearn.pipeline.Pipeline` or `ColumnTransformer` so each cross-validation fold fits transforms only on its training partition.
- **Retrain on all labeled rows:** After evaluation and model selection, fit the final pipeline using the entire labeled training dataset before generating competition test predictions. The current notebook predicts test rows using the model trained on only 80% of the labeled data.
- **Tune hyperparameters:** Experiment with `max_depth`, `min_samples_split`, `min_samples_leaf`, and `n_estimators` using cross-validation.
- **Compare models:** Try Logistic Regression, Gradient Boosting, XGBoost, or CatBoost as additional baselines.
- **Add evaluation detail:** Report precision, recall, F1-score, class distribution, and a classification report.

## Acknowledgement

Dataset and task provided by Kaggle as part of the [Playground Series — Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7/data).

---

**Project type:** Supervised Machine Learning · Binary Classification · Kaggle Competition
