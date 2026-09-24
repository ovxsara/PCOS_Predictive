## PCOS & PMS Predictive Models

**Research question:** Can menstrual health and user-profile data reveal patterns associated with PCOS, PMS, and overall well-being?

This notebook merges per-cycle period logs with user lifestyle profiles and builds classification models to predict whether a user experiences PMS symptoms during the luteal phase. PCOS diagnosis, stress, energy, concentration and sleep are used as predictors.

### Data
Source: [Menstrual Health & Productivity Dataset](https://www.kaggle.com/datasets/puspitachowdhury2/menstrual-health-dataset/data) by Puspita Chowdhury on Kaggle. It pairs cycle tracking with lifestyle, stress, and health indicators.

- **Period_Log.csv**: one row per cycle (cycle phase, flow, pain, PMS symptoms, mood, stress, sleep, energy, concentration, hormone levels, and more)
- **User_Profile.csv**: one row per user (age, BMI, diet, exercise, sleep, caffeine, alcohol, smoking, birth control use, PCOS diagnosis, baseline stress)
- The two files are merged on `user_id` and aggregated to **one row per user (1,972 users)**, using luteal-phase cycles because that is when PMS typically appears.

### Workflow
1. **Data preparation**: merge the files, select features, check for missing values, and aggregate to user level. A follicular-phase dataset is also built for comparison.
2. **Exploratory analysis**
   - Luteal vs. follicular comparison of PMS rate, stress, energy, concentration and sleep
   - Energy, concentration and productivity loss for users with and without PCOS. Users with PCOS show lower energy (6.32 vs 6.88) and lower concentration (6.84 vs 7.40).
   - Correlation heatmap, plus box plots of stress, energy and sleep by PMS status and cycle phase
3. **Modeling**
   - Target: `pms_symptoms` (1,427 users with PMS vs 545 without, an imbalanced split)
   - Features: `pcos_diagnosed`, `stress_score_baseline`, `energy_level`, `concentration_score`, `sleep_hours`
   - Stratified 80/20 train-test split and 5-fold stratified cross-validation (macro F1)
   - Oversampling of the training data only, to balance the classes:
     - [**ADASYN**](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html) (Adaptive Synthetic Sampling) creates more       synthetic samples for minority-class examples that are harder to learn.
     - [**SMOTE**](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) (Synthetic Minority Over-sampling Technique) creates synthetic minority samples by interpolating between neighboring examples.
   - Models: **Logistic Regression** and **Random Forest** (200 trees)
4. **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, confusion matrices, a ROC curve and Random Forest feature importances

### Results (test set, n = 395)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| ADASYN + Logistic Regression | 0.57 | 0.75 | 0.60 | 0.67 | 0.56 |
| ADASYN + Random Forest | 0.65 | 0.74 | 0.80 | 0.77 | 0.53 |
| SMOTE + Logistic Regression | 0.56 | 0.75 | 0.58 | 0.66 | — |
| **SMOTE + Random Forest** | **0.65** | **0.74** | **0.80** | **0.77** | **0.57** |

Random Forest with SMOTE performed best, with a cross-validated macro F1 of 0.75 and a test F1 of 0.77 for the PMS class. However, ROC-AUC stayed between 0.53 and 0.57 for every model, and all models struggled to identify users without PMS. This suggests the selected features carry only a weak signal for PMS, and that richer inputs (hormone levels, pain, flow, lifestyle factors) could improve the models.

### Tech stack
Python · pandas · NumPy · scikit-learn · [imbalanced-learn](https://imbalanced-learn.org/) (ADASYN, SMOTE) · matplotlib · seaborn · Google Colab

### How to run
Open the notebook in Google Colab using the badge at the top. Then update the paths to `Period_Log.csv` and `User_Profile.csv`, which are also available in this repo.
