from timeit import main
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv(r"/home/muhammad/Documents/TechCrush/Assignments/Assignment - 2/cleaned_titanic_train.csv")
test_df = pd.read_csv(r"/home/muhammad/Documents/TechCrush/Assignments/Assignment - 2/cleaned_titanic_test.csv")

"""print(train_df.describe())
print(test_df.describe())"""

"""print(train_df.columns.tolist())
print(test_df.columns.tolist())"""

print(train_df["Survived"].value_counts())

"""## Correlation Heatmap
def plot_correlation_heatmap(df, name="dataset"):
    # Making a copy so original df is not altered
    df = df.copy()

    # Encode Sex and Embarked as numbers so they can be included
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Select only numeric columns relevant to modeling
    cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin']
    filtered_cols = []
    for c in cols:
        if c in df.columns:
            filtered_cols.append(c)
    cols = filtered_cols

    corr_matrix = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"Correlation Heatmap - {name}")
    plt.tight_layout()
    plt.savefig(f"{name}_correlation_heatmap.png")
    plt.show()

    print(f"Saved heatmap as {name}_correlation_heatmap.png")

plot_correlation_heatmap(train_df, name="train_df")"""

FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin']

## Preparing data for modeling
def prepare_features(df):

    df = df.copy()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

## Running models
def main():
    # Load your already-cleaned train data
    train_df = pd.read_csv('train_cleaned.csv')
    df = prepare_features(train_df)

    X = df[FEATURES]
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    best_name, best_model, best_acc = None, None, 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"{name} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest model: {best_name} ({best_acc:.4f}) — saving as titanic_model.pkl")
    joblib.dump(best_model, 'titanic_model.pkl')

if __name__ == "__main__":
    main()