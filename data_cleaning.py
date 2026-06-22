import pandas as pd
import time

train_df = pd.read_csv(r"/home/muhammad/Documents/TechCrush/Assignments/Assignment - 2/titanic_train.csv")
test_df = pd.read_csv(r"/home/muhammad/Documents/TechCrush/Assignments/Assignment - 2/titanic_test.csv")

train_df_copy = train_df.copy()
test_df_copy = test_df.copy()

print(train_df.shape)
print(train_df.isnull().sum())
print(train_df.info())

print(test_df.shape)
print(test_df.isnull().sum())
print(test_df.info())

time.sleep(1)

### Data Cleaning

# Missing Age 
def fill_missing_age(df, reference_df, name="dataset"):
    """
    Fills missing Age values in df using median Age grouped by Pclass + Sex,
    calculated from reference_df (should always be the train set).
    """
    # Flag which rows had missing age, before filling
    df['Age_was_missing'] = df['Age'].isnull()

    # Calculate median age per Pclass+Sex group, from reference_df
    age_medians = reference_df.groupby(['Pclass', 'Sex'])['Age'].median()

    # Fill missing ages using the matching group's median
    for (pclass, sex), median_age in age_medians.items():
        mask = (df['Age'].isnull() & (df['Pclass'] == pclass) & (df['Sex'] == sex))
        df.loc[mask, 'Age'] = median_age

    print(f"Filled missing Age values in {name} using median ages from reference_df grouped by Pclass and Sex.")
    time.sleep(1)

    return df

train_df_copy = fill_missing_age(train_df_copy, reference_df=train_df, name="train_df_copy")
test_df_copy  = fill_missing_age(test_df_copy, reference_df=train_df, name="test_df_copy")

# Missing Embarked Value
def fill_missing_embarked(df):
    """
    Fills missing Embarked values in df with the most common value.
    """
    # Checking missing values in Embarked column
    df['Embarked_missing'] = df['Embarked'].isnull()
    # Calculating the mode (most common value)
    most_common_embarked = df['Embarked'].mode()[0]
    # Fill missing values with the mode value
    df['Embarked'] = df['Embarked'].fillna(most_common_embarked)

    print(f"Filled missing Embarked values with the most common value: {most_common_embarked}")
    time.sleep(1)
    
    return df

train_df_copy = fill_missing_embarked(train_df_copy)

# Missing Fare
def fill_missing_fare(df, reference_df):
    """
    Fills missing Fare values in df using median Fare grouped by Pclass,
    calculated from reference_df (should always be the train set).
    """
    # Flag which rows had missing fare, before filling
    df['Fare_was_missing'] = df['Fare'].isnull()

    # Calculate median fare per Pclass group, from reference_df
    fare_medians = reference_df.groupby('Pclass')['Fare'].median()

    # Fill missing fares using the matching group's median
    for pclass, median_fare in fare_medians.items():
        mask = (df['Fare'].isnull() & (df['Pclass'] == pclass))
        df.loc[mask, 'Fare'] = median_fare

    print(f"Filled missing Fare values using median fares from reference_df grouped by Pclass.")
    time.sleep(1)

    return df

test_df_copy  = fill_missing_fare(test_df_copy, reference_df=train_df)

# Missing Cabin
def fill_missing_cabin(df, name="dataset"):
    # Create a new column indicating whether the cabin was missing
    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    # Drop original cabin column
    df = df.drop('Cabin', axis=1)
    
    print(f"Filled missing Cabin values in {name} by creating a new binary column 'Has_Cabin' indicating presence of cabin information.")
    time.sleep(1)

    return df

train_df_copy= fill_missing_cabin(train_df_copy, name="train_df_copy")
test_df_copy  = fill_missing_cabin(test_df_copy, name="test_df_copy")

# Drop Ticket Column
def drop_ticket(df, name="dataset"):
    df = df.drop(columns=['Ticket'])
    print(f"Dropped Ticket column in {name}.")
    time.sleep(1)
    return df

train_df_copy = drop_ticket(train_df_copy, name="train_df_copy")
test_df_copy  = drop_ticket(test_df_copy, name="test_df_copy")

print(train_df_copy.shape)
print(train_df_copy.isnull().sum())
print(train_df_copy.info())

print(test_df_copy.shape)
print(test_df_copy.isnull().sum())
print(test_df_copy.info())

time.sleep(1)

train_df_copy.to_csv("cleaned_titanic_train.csv", index=False)
test_df_copy.to_csv("cleaned_titanic_test.csv", index=False)

print("Data cleaning completed. Cleaned datasets saved as 'cleaned_titanic_train.csv' and 'cleaned_titanic_test.csv'.")