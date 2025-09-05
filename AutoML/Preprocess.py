import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings(action="ignore")


class Preprocess:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.type = 'categorical' if df[target].dtype == 'object' else 'numerical'
        print("Preprocessing Started ...")

    def handle_duplicate_rows(self):
        return self.df.drop_duplicates()

    def handle_missing_values(self):
        for column in self.df.columns:
            if self.df[column].isnull().any():
                if self.df[column].dtype == 'object':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:
                    self.df[column].fillna(self.df[column].median(), inplace=True)
        return self.df

    def handle_outliers(self):
        print("Handling Numerical Outliers ...")
        for column in self.df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df[column] = self.df[column].apply(lambda x: x if lower_bound <= x <= upper_bound else self.df[column].median())
        print("Handling Categrical Outliers ...")
        threshold = 0.05  # Minimum frequency threshold
        cat_columns = self.df.select_dtypes(include=['object']).columns
        for col in cat_columns:
            freq = self.df[col].value_counts(normalize=True)
            rare_categories = freq[freq < threshold].index
            self.df = self.df[~self.df[col].isin(rare_categories)]

        # Reset index after dropping rows
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def handle_categorical_data(self):
        le_dict = {}
        for column in self.df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            le_dict[column] = le
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        return X, y, le_dict

    def scale_data(self, X):
        scaler_dict = {}
        X_scaled = X.copy()
        for col in X.columns:
            scaler = StandardScaler()
            X_scaled[[col]] = scaler.fit_transform(X[[col]])
            scaler_dict[col] = scaler
        print('scaler_dict created ')    
        return X_scaled, scaler_dict
    
    def preprocess(self):
        self.df = self.handle_duplicate_rows()
        self.df = self.handle_missing_values()
        self.df = self.handle_outliers()
        X, y, le_dict = self.handle_categorical_data()
        X_scaled, scaler_dict = self.scale_data(X)
        print("Preprocessing Completed")
        return X_scaled, y, le_dict, scaler_dict, self.type

'''
Make Scalar like label encoder and add all scalar in dict for each column
'''