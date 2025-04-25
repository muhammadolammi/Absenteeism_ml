import pandas as np
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class CustomScaler():
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns_to_scale = [ 'Month', 'Transportation Expense', 'Age',  'Body Mass Index', 'Education', 'Children', 'Pets' ]
    def fit(self, X):
        self.scaler.fit(X[self.columns_to_scale])
        return self
    def transform(self,X):
        X[self.columns_to_scale] =self.scaler.transform(X[self.columns_to_scale])
        return X
    def fit_transform(self,X):
        X[self.columns_to_scale] =self.scaler.fit_transform(X[self.columns_to_scale])
        return X
        
class AbsenteeismModule():
    def __init__(self, model_file, scaler_file):
        with open(model_file, 'rb') as m:
            self.model = pickle.load(m)
        with open(scaler_file, 'rb') as s:
            self.scaler = pickle.load(s)
        self.data = None


    def load_and_clean_data(self, data_file):
        raw_data = pd.read_csv(data_file)
        df = raw_data.copy() 
        self.df_with_predictions = df.copy()
        ## drop id
        df = df.drop("ID", axis=1)
        # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
        df['Absenteeism Time in Hours'] = 'NaN'
        # get dummy for reasons 
        reasons_dummy = pd.get_dummies(df["Reason for Absence"], drop_first=True)
        reasons_column_1 = reasons_dummy.loc[:, 1:14]
        reasons_column_2 =reasons_dummy.loc[:, 15:17]
        reasons_column_3 =reasons_dummy.loc[:, 18:21]
        reasons_column_4 =reasons_dummy.loc[:, 22:]

        reasons_group_1 = reasons_column_1.sum(axis=1)
        reasons_group_2 =reasons_column_2.sum(axis=1)
        reasons_group_3 =reasons_column_3.sum(axis=1)
        reasons_group_4 =reasons_column_4.sum(axis=1) 
        df["reasons_group_1"]=reasons_group_1
        df["reasons_group_2"]=reasons_group_2
        df["reasons_group_3"]=reasons_group_3
        df["reasons_group_4"]=reasons_group_4
        df = df.drop("Reason for Absence", axis=1)
        columns_reordered= ['reasons_group_1',
       'reasons_group_2', 'reasons_group_3', 'reasons_group_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[columns_reordered]
        df_reasons_mod= df.copy()
        # convert the date to  datetime type
        df_reasons_mod["Date"] = pd.to_datetime(df_reasons_mod["Date"], format="%d/%m/%Y")
        def datetime_to_day(date_time):
            return date_time.day
        def datetime_to_month(date_time):
            return date_time.month
        df_reasons_mod["Day"] = df_reasons_mod["Date"].apply(datetime_to_day)
        df_reasons_mod["Month"] = df_reasons_mod["Date"].apply(datetime_to_month)
        df_reasons_mod= df_reasons_mod.drop("Date", axis=1)
        columns_reordered= ['reasons_group_1', 'reasons_group_2', 'reasons_group_3',
       'reasons_group_4', 'Day', 'Month','Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
       'Pets', 'Absenteeism Time in Hours']
        df_reasons_mod=df_reasons_mod[columns_reordered]
        df_datetime_mod = df_reasons_mod.copy()
        df_datetime_mod["Education"] = df_datetime_mod["Education"].map({1:0, 2:1, 3:1, 4:1})
        df_preprocessed = df_datetime_mod.copy()
        # replace the NaN values
        df_datetime_mod = df_datetime_mod.fillna(value=0)
        # drop the original absenteeism time
        df_datetime_mod = df_datetime_mod.drop(['Absenteeism Time in Hours'],axis=1)
        # drop the variables we decide we don't need
        df_datetime_mod = df_datetime_mod.drop(['Day','Daily Work Load Average','Distance to Work'],axis=1)
        self.preprocessed_data =df_preprocessed
        self.data = self.scaler.transform(df_datetime_mod)
        

    # a function that ouput 0 or 1 based on our model
    def predict(self):
        if self.data is not None and not self.data.empty:
            return self.model.predict(self.data)
    # a function that ouput the probability of our input being 1
    def predict_proba(self):
        if self.data is not None and not self.data.empty:
            return self.model.predict_proba(self.data)[:,1]
    def predict_outputs(self):
        if self.data is not None and not self.data.empty:
            self.preprocessed_data['Probability'] = self.predict_proba()
            self.preprocessed_data ['Prediction'] = self.predict()
            return self.preprocessed_data
        
    