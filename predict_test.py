import pandas as pd
import joblib

FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin']

def prepare_features(df):
    df = df.copy()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

def main():
    test_df = pd.read_csv('cleaned_titanic_test.csv')
    df = prepare_features(test_df)

    model = joblib.load('titanic_model.pkl')
    predictions = model.predict(df[FEATURES])

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('final_predictions.csv', index=False)
    print("Saved predictions to final_predictions.csv")

if __name__ == "__main__":
    main()