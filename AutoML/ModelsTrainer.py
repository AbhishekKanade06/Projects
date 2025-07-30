from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

class ModelsTrainer:
    
    def __init__(self,X_scaled, y, le, scaler ,type):
        self.X_scaled = X_scaled
        self.y = y
        self.le = le
        self.scaler = scaler
        self.type = type
        
    Regression_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR()
    }
    Classification_models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Random Forest Classifier': RandomForestClassifier(random_state=42),
        'Support Vector Classifier': SVC()
    }


    def train_models(self):
        results = {}
        models = []

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )

        if self.type == 'categorical':
            for name, model in self.Classification_models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                results[name] = score
                models.append(model)
        else:
            for name, model in self.Regression_models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                results[name] = score
                models.append(model)

        return results, models

        
    def best_model(self,results,models):
        # Find the best model based on the highest score
        best_model_name = max(results, key=results.get)
        best_model_score = results[best_model_name]
        best_model = models[list(results.keys()).index(best_model_name)]
        
        return best_model_name, best_model_score , best_model
    
    def hyperparameter_tuning(self, model):
        from sklearn.model_selection import GridSearchCV
        
        
        
        # Define hyperparameters to tune
        if isinstance(model, RandomForestRegressor) or isinstance(model, RandomForestClassifier):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'n_jobs' : [1]
            }
        elif isinstance(model, DecisionTreeRegressor) or isinstance(model, DecisionTreeClassifier):
            param_grid = {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        else:
            param_grid = {}
        
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(self.X_scaled, self.y)
        
        return grid_search.best_estimator_
    
    def train(self):
        trained_models, models = self.train_models()
        best_model_name, best_model_score, best_model = self.best_model(trained_models, models)
        print(f"Best Model: {best_model_name} with score: {best_model_score}")
        print(f"Hyperparameters Tuning started ...")
        tuned_model = self.hyperparameter_tuning(best_model)
        print(f"Tuned Model: {tuned_model}")
        
        return trained_models,tuned_model