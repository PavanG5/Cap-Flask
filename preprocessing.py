# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# def load_and_preprocess_data(file_path):
#     # Load dataset
#     df = pd.read_csv(file_path)

#     # Encoding the labels for prognosis
#     le = LabelEncoder()
#     df['prognosis'] = le.fit_transform(df['prognosis'])

#     # Splitting features (X) and labels (y)
#     X = df.drop(columns=['prognosis'])
#     y = df['prognosis']

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     return X_train, X_test, y_train, y_test, le

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# def load_and_preprocess_data(file_path):
#     # Load dataset
#     df = pd.read_csv(file_path)

#     # Encoding the labels for prognosis
#     le = LabelEncoder()
#     df['prognosis'] = le.fit_transform(df['prognosis'])

#     # Splitting features (X) and labels (y)
#     X = df.drop(columns=['prognosis'])
#     y = df['prognosis']

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Print dataset sizes
#     print("Training set size:", X_train.shape[0])
#     print("Test set size:", X_test.shape[0])

#     # Check for overlap between train and test sets
#     overlap = set(X_train.index).intersection(set(X_test.index))
#     print("Number of overlapping samples:", len(overlap))  # Should be 0 for no overlap

#     # # Inspect the label distribution in training and test sets
#     # print("Training label distribution:\n", y_train.value_counts(normalize=True))
#     # print("Test label distribution:\n", y_test.value_counts(normalize=True))

#     return X_train, X_test, y_train, y_test, le

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# def load_and_preprocess_data(file_path):
#     # Load dataset
#     df = pd.read_csv(file_path)

#     # Encoding the labels for prognosis
#     le = LabelEncoder()
#     df['prognosis'] = le.fit_transform(df['prognosis'])

#     # Splitting features (X) and labels (y)
#     X = df.drop(columns=['prognosis'])
#     y = df['prognosis']

#     # Ensure no target information is in features
#     assert 'prognosis' not in X.columns, "Target column found in features!"
    
#     # Check for potential feature leakage by examining high correlation with target
#     df_combined = pd.concat([X, y], axis=1)  # Temporarily combine for correlation check
#     correlations = df_combined.corr()['prognosis'].drop('prognosis')
#     high_corr_features = correlations[correlations.abs() > 0.8]  # Adjust threshold as needed
#     if not high_corr_features.empty:
#         print("Warning: Potential leakage detected. High correlation with target found in features:")
#         print(high_corr_features)

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Print dataset sizes
#     print("Training set size:", X_train.shape[0])
#     print("Test set size:", X_test.shape[0])

#     # Check for overlap between train and test sets
#     overlap = set(X_train.index).intersection(set(X_test.index))
#     print("Number of overlapping samples:", len(overlap))  # Should be 0 for no overlap

#     # # Inspect the label distribution in training and test sets
#     # print("Training label distribution:\n", y_train.value_counts(normalize=True))
#     # print("Test label distribution:\n", y_test.value_counts(normalize=True))

#     # Baseline model check to detect any strong single-feature prediction (possible leakage)
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.metrics import accuracy_score

#     print("Checking each feature for strong predictive power (potential leakage check)...")
#     for col in X.columns:
#         model = LogisticRegression(max_iter=1000)
#         model.fit(X_train[[col]], y_train)
#         predictions = model.predict(X_train[[col]])
#         accuracy = accuracy_score(y_train, predictions)
#         if accuracy > 0.7:  # Adjust threshold based on context
#             print(f"Potential leakage with feature '{col}': Training accuracy = {accuracy:.2f}")

#     return X_train, X_test, y_train, y_test, le

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Encode labels
    le = LabelEncoder()
    df['prognosis'] = le.fit_transform(df['prognosis'])

    # Split features and labels
    X = df.drop(columns=['prognosis'])
    y = df['prognosis']

    # Shuffle data and split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Print dataset sizes
    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])

    # Check for overlap (there should be none)
    overlap = set(X_train.index).intersection(set(X_test.index))
    print("Number of overlapping samples:", len(overlap))

    # # Label distribution in train and test sets
    # print("Training label distribution:\n", y_train.value_counts(normalize=True))
    # print("Test label distribution:\n", y_test.value_counts(normalize=True))

    # Check for potential data leakage by calculating mutual information
    mi_scores = mutual_info_classif(X, y, discrete_features=False)
    print("Mutual Information Scores (to check for leakage):\n", pd.Series(mi_scores, index=X.columns).sort_values(ascending=False))

    return X_train, X_test, y_train, y_test, le

