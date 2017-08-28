import pandas as pd
from scipy.stats import mode

# reading the data
trainEnquiryDF = pd.read_csv("RBL_70_30/raw_enquiry_70.csv")
testEnquiryDF = pd.read_csv("RBL_70_30/raw_enquiry_30.csv")

# appending the train and test
trainTestAppendDF = pd.concat([trainEnquiryDF, testEnquiryDF]).reset_index(drop=True)

# delete columns List
deleteColumnsList = ['externalid', 'member_short_name']

# unsecured EnquiryList
unsecuredEnquiryList = [5, 8, 6, 9, 10, 12, 16, 35, 40, 41, 43, 0]

# drop the columns
trainTestAppendDF = trainTestAppendDF.drop(deleteColumnsList, 1)

# Days Calculation of 90 and 360
trainTestAppendDF['dt_opened']=pd.to_datetime(trainTestAppendDF['dt_opened']) # converted into datetime
trainTestAppendDF['enquiry_dt']=trainTestAppendDF['enquiry_dt'].fillna('01-Jan-2016') # fill missing date
trainTestAppendDF['enquiry_dt'] = pd.to_datetime(trainTestAppendDF['enquiry_dt'])
trainTestAppendDF['upload_dt']=trainTestAppendDF['upload_dt'].fillna('01-Jan-2016')
trainTestAppendDF['upload_dt'] = pd.to_datetime(trainTestAppendDF['upload_dt'])
trainTestAppendDF['days'] = trainTestAppendDF['upload_dt'].sub(trainTestAppendDF['enquiry_dt'], axis=0) # subtracted two date
trainTestAppendDF['days']= trainTestAppendDF['days'].astype('timedelta64[D]')
trainTestAppendDF['days'][trainTestAppendDF['days'] == 0.0] = 10000
trainTestAppendDF['365_days'] = (trainTestAppendDF['days']<=365)
trainTestAppendDF['90_days'] = (trainTestAppendDF['days']<=90)

# filling missing value for enq_purpose and enq_amt
trainTestAppendDF['enq_purpose']=trainTestAppendDF['enq_purpose'].fillna(0)
trainTestAppendDF['enq_amt']=trainTestAppendDF['enq_amt'].fillna(0)

# days calculation between enquiry_date and dt_opened
trainTestAppendDF['openDays'] = trainTestAppendDF['dt_opened'].sub(trainTestAppendDF['enquiry_dt'], axis=0)
trainTestAppendDF['openDays']= trainTestAppendDF['openDays'].astype('timedelta64[D]')
trainTestAppendDF['openDays'][trainTestAppendDF['openDays'] < 0.0] = 0.0

# initialize  the ratio column to False it is column to calculate the unsecured enquiry purpose
trainTestAppendDF['ratio'] = False

for i in range(0,len(trainTestAppendDF)):
    if(trainTestAppendDF.iloc[i]['enq_purpose'] in unsecuredEnquiryList):
        trainTestAppendDF.ix[i,'ratio'] = True
    else:
        trainTestAppendDF.ix[i,'ratio'] = False

# separate the train and test
trainTestAppendDF_train = trainTestAppendDF[0:len(trainEnquiryDF)]
trainTestAppendDF_test = trainTestAppendDF[len(trainEnquiryDF):len(trainTestAppendDF)]

# aggregations
aggregations = {
    '365_days':'sum',
    '90_days':'sum',
    'enq_purpose': lambda y:mode(y)[0],
    'openDays': lambda x : sum(x)/len(x),
    'enq_amt':'sum',
    'ratio':lambda x : sum(x)/len(x)
}

# aggregations in train and test
trainTestAppendDF_train = trainTestAppendDF_train.groupby('apprefno', as_index=False).agg(aggregations)
trainTestAppendDF_test = trainTestAppendDF_test.groupby('apprefno', as_index=False).agg(aggregations)

# saving train and test
trainTestAppendDF_train.to_csv('output/trainEnquiry.csv', index=False)
trainTestAppendDF_test.to_csv('output/testEnquiry.csv', index=False)



