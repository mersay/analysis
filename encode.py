import pandas as pd

def one_hot_encode_runners(df, expected_columns=None):
    categorical_features = ['sex', 'country', 'colour']
    df_encoded = df.copy()

    for feature in categorical_features:
        dummies = pd.get_dummies(df_encoded[feature], prefix=feature)
        df_encoded = pd.concat([df_encoded.drop(columns=[feature]), dummies], axis=1)

    if expected_columns:
        # Add missing columns and sort
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_columns]
    
    return df_encoded