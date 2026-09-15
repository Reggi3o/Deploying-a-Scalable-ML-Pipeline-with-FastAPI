# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary logistic regression classifier trained to predict whether an individual’s annual income is above or below $50,000, using the target column `salary` in the census dataset. The model is implemented as a standard scikit-learn `LogisticRegression` classifier with one-hot encoding applied to categorical features before training. The feature engineering pipeline includes preprocessing for the following categorical fields: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`.

The model was developed as part of an ML DevOps training workflow and is intended to demonstrate a complete machine learning pipeline: data preprocessing, model training, validation, and model monitoring across population slices.

## Intended Use

This model is intended for educational and analytical use, specifically to evaluate how a salary prediction model performs across different segments of the population and to identify disparities in performance between demographic or categorical subgroups. It can help answer questions such as whether the classifier behaves differently for different occupations, education levels, or marital statuses.

This model should not be used as a decision-making system for high-stakes real-world decisions such as hiring, lending, housing, or credit eligibility. The target is a proxy for income bracket and may reflect historical inequities in the data.

## Training Data

The model is trained on the `data/census.csv` file, which is the Adult Census Income dataset (also known as the Census Income dataset). The data contains demographic and employment information used to predict whether a person earns more than $50,000 per year.

The training process uses an 80/20 train-test split with `random_state=42`, and the following columns are treated as categorical features during preprocessing:

- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

Continuous features are retained and concatenated with one-hot encoded categorical features to form the final feature matrix. The target variable `salary` is label-encoded using a `LabelBinarizer` to create a binary classification target.

## Evaluation Data

The model is evaluated on the held-out test split from the same census dataset, using data that was not included in training. Performance is assessed on the full test set and also on subgroup slices defined by individual categorical features to check for disparities in predictive behavior.

## Metrics

The model is evaluated using precision, recall, and F1 score. These metrics are computed on the binary salary label and are used both overall and within categorical slices of the data.

## Ethical Considerations

This model is trained on historical demographic data and may reflect social and economic biases present in the dataset. It should be used with caution and interpreted as a research or demonstration model rather than a production decision system.

## Caveats and Recommendations

- Interpret performance carefully across demographic subgroups.
- Review slice-level metrics before drawing conclusions about fairness or model behavior.
- Use the model only for exploratory analysis and educational purposes unless additional validation, fairness review, and domain-specific governance are in place.
