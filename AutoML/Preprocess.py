import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import numpy as np
warnings.filterwarnings(action="ignore")


class Preprocess:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.type = 'categorical' if df[target].dtype == 'object' else 'numerical'
        self.info={}
        print("Preprocessing Started ...")

    def handle_duplicate_rows(self):
        """
        Handle duplicate rows in the DataFrame.
        """
        return self.df.drop_duplicates()

    def handle_missing_values(self):
        """
        Handle missing values in the DataFrame.
        If Features has Missinge precentage > 20% : Drop the feature.
        Categorical columns: Fill with mode.
        Numerical columns: Fill with median.
        """
        print("Handling Missing Values ...")
        missing_percent = self.df.isnull().mean()
        cols_to_drop = missing_percent[missing_percent > 0.2].index
        if self.target in cols_to_drop:
            cols_to_drop = cols_to_drop.drop(self.target)
        self.df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns with >20% missing values: {list(cols_to_drop)}")
            # Fill missing values
        for column in self.df.columns:
            self.info[column] = self.df[column].dtype.name
            if self.df[column].isnull().any():
                if self.df[column].dtype == 'object':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:
                    self.df[column].fillna(self.df[column].median(), inplace=True)
        
        return self.df

    def remove_single_value_columns(self):
        """
        Remove columns with a single unique value.
        """
        print("Removing Single-Value Columns ...")
        single_value_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        self.df.drop(columns=single_value_cols, inplace=True)
        if single_value_cols:
            print(f"Removed single-value columns: {single_value_cols}")
            del self.info[single_value_cols]
        return self.df
    
    def handle_outliers(self):
        """
        Handle outliers in the DataFrame.
        """
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
        self.df = self.remove_single_value_columns()
        self.df = self.handle_outliers()
        X, y, le_dict = self.handle_categorical_data()
        X_scaled, scaler_dict = self.scale_data(X)
        print("Preprocessing Completed")
        return X_scaled, y, le_dict, scaler_dict, self.type, self.info

"""
To do :
1. Add two functions to remove low variance features and highly correlated features.
2. Have to do something about data imbalance for categorical target.
3. Make the outlier handling more robust.
4. Add ReadMe file.
5. Add ReadMe file for generated script.

"""