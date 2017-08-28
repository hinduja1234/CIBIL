import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier


# reading Data for customer 
trainData = pd.read_csv("RBL_70_30/raw_data_70.csv")
testData = pd.read_csv("RBL_70_30/raw_data_30.csv")

# selecting columns 
selectingColumnsList = ['apprefno', 'cibil_score', 'marital_status', 'bad_flag_worst6']

trainData = trainData[selectingColumnsList]
testData = testData[selectingColumnsList]

# reading enquiry data
trainEnquiry = pd.read_csv('output/trainEnquiry.csv')
testEnquiry = pd.read_csv('output/testEnquiry.csv')

# reading account data
trainAccount = pd.read_csv('output/trainAccount.csv')
testAccount = pd.read_csv('output/testAccount.csv')

# join all three data to create train and test
train = pd.merge(trainData,trainEnquiry,on='apprefno', how='inner')
test  = pd.merge(testData,testEnquiry,on='apprefno', how='inner')
train = pd.merge(train,trainAccount,on='apprefno',how='inner')
test = pd.merge(test,testAccount,on='apprefno',how='inner')

# concat train and test
trainTestAppendDF = pd.concat([train,test]).reset_index(drop=True)

# utilization column creation
trainTestAppendDF['utilization'] = (trainTestAppendDF['total_currentlimit']/trainTestAppendDF['total_creditlimit'])/(trainTestAppendDF['mean_currentlimit']/(trainTestAppendDF['mean_creditlimit']+ trainTestAppendDF['cashlimit']))

# filling missing value
trainTestAppendDF['utilization'] = trainTestAppendDF['utilization'].fillna(0)
trainTestAppendDF['cibil_score'] = trainTestAppendDF['cibil_score'].fillna(0)
trainTestAppendDF['marital_status'] = trainTestAppendDF['marital_status'].fillna(2)

# features for training
trainTestAppendDFSelectingColumn = ['365_days','90_days','enq_purpose','openDays','ratio','mean_Days','total_Days','monthlength'
    ,'paymentBoolean','least_30dpd','utilization','max_30dpd'] + selectingColumnsList

trainTestAppendDF = trainTestAppendDF[trainTestAppendDFSelectingColumn]

# one hot encoding of enquiry purpose
oneHot = pd.get_dummies(trainTestAppendDF['enq_purpose'])
trainTestAppendDF  = trainTestAppendDF.drop('enq_purpose',axis =1)
trainTestAppendDF = trainTestAppendDF.join(oneHot)

# separate the train and test
trainTestAppendDF_train = trainTestAppendDF[0:len(train)]
trainTestAppendDF_test = trainTestAppendDF[len(train):len(trainTestAppendDF)]

# predict column for training and testing data
trainPredicted = trainTestAppendDF_train['bad_flag_worst6']
testPredicted = trainTestAppendDF_test['bad_flag_worst6']


# removing column for creation of features for training  and testing
trainTestAppendDF_train = trainTestAppendDF_train[trainTestAppendDF_train.columns.difference(['apprefno', 'bad_flag_worst6'])]
trainTestAppendDF_test = trainTestAppendDF_test[trainTestAppendDF_test.columns.difference(['apprefno', 'bad_flag_worst6'])]

# model initialize
model = GradientBoostingClassifier(learning_rate=0.1)
model.fit(trainTestAppendDF_train,trainPredicted)

# model prediction
predicted = model.predict(trainTestAppendDF_test)

# probs of model
probs = model.predict_proba(trainTestAppendDF_test)

# gain for each variable
model2 = ExtraTreesClassifier()
model2.fit(trainTestAppendDF_train,trainPredicted)
print(model2.feature_importances_)

print confusion_matrix(testPredicted,predicted)
print 2*(roc_auc_score(testPredicted, probs[:, 1])) -1

# rank ordering 
probs0 = probs[:, 0]
probs1 = probs[:, 1]
probs0 = sorted(probs0, reverse=True)
probs1 = sorted(probs1, reverse=True)
decile0 = np.percentile(probs0, np.arange(0, 100, 10))
decile1 = np.percentile(probs1, np.arange(0, 100, 10))

totalDecileRank = decile1 + decile0
print totalDecileRank

