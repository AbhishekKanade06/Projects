import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings(action="ignore")

class Preprocess:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        if self.df[target].dtype == 'object':
            self.type = 'categorical'
        else:
            self.type = 'numerical'
        print("Preprocessing Started ...")    
    def handle_duplicate_rows(self):
        # Check for duplicate rows
        if self.df.duplicated().any():
            # Remove duplicate rows
            self.df = self.df.drop_duplicates()
        
        return self.df
    def handle_missing_values(self):
        for column in self.df.columns:
            if self.df[column].isnull().any():
                if self.df[column].dtype == 'object':
                    # Fill categorical missing values with the mode
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:
                    # Fill numerical missing values with the mean
                    self.df[column].fillna(self.df[column].median(), inplace=True)

        return self.df
    def handle_outliers(self):
        for column in self.df.select_dtypes(include=['float64', 'int64']).columns:
            # Calculate the IQR
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with the median
            self.df[column] = self.df[column].apply(lambda x: x if (x >= lower_bound and x <= upper_bound) else self.df[column].median())
        
        return self.df
    # def handel_categorical_data(self):
    #     # Convert categorical variables to numerical
    #     self.df = pd.get_dummies(self.df, drop_first=True)
        
    #     # Separate features and target variable
    #     X = self.df.drop(columns=[self.target])
    #     y = self.df[self.target]
        
    #     return X, y
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
        # Scale the data using StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, scaler
    def preprocess(self):
        self.df = self.handle_duplicate_rows()
        self.df = self.handle_missing_values()
        self.df = self.handle_outliers()
        X, y, le_dict = self.handle_categorical_data()
        X_scaled, scaler = self.scale_data(X)
        print("Preprocessing Completed")
        return X_scaled, y, le_dict, scaler, self.type