from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import make_pipeline
import pandas as pd

class ModelTrainer:
    def __init__(self, csv_file='data.csv',data=None):
        # Reading the file
        self.df = pd.read_csv(csv_file)

        # Loading column names into a data dictionary
        self.data_dict = self.df.columns

        # Renaming columns
        for i in range(1, len(self.df.columns)):
            self.df.rename(columns={self.df.columns[i]: f"X{i}"}, inplace=True)

        self.model = self.train_model()

    def train_model(self):
        X = self.df.drop('Bankrupt?', axis=1)
        y = self.df['Bankrupt?']

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Over-sampling using RandomOverSampler
        over_sampler = RandomOverSampler(random_state=42)
        X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)

        # Creating a pipeline with SimpleImputer and RandomForestClassifier
        clf = make_pipeline(SimpleImputer(), RandomForestClassifier(random_state=42))

        # Parameter grid for GridSearchCV
        params = {'randomforestclassifier__max_depth': [40],
                  'randomforestclassifier__n_estimators': [50],
                  'simpleimputer__strategy': ['mean']}

        # GridSearchCV to find the best parameters
        model = GridSearchCV(clf, param_grid=params, verbose=1)
        model.fit(X_train_over, y_train_over)

        return model
