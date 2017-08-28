import pandas as pd
import numpy as np

# list for cleaning the payment history string
n =3
acList = ['STD','XXX','SMA','SUB','DBT','LSS','"""','"""','Roh','it1']
acList2 = ['"""','"""','Roh','it1']
replaceList = ['STD','XXX','SMA','SUB','DBT','LSS']

# train and test reading
trainAccount = pd.read_csv("RBL_70_30/raw_account_70.csv")
testAccount = pd.read_csv("RBL_70_30/raw_account_30.csv")

# outlier function
def outlier(elements):
    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)
    for index,value in enumerate(elements):
        if(value <= (mean-2*sd)):
            elements[index] = (mean-2*sd)
        if (value >= (mean + 2 * sd)):
            elements[index] = (mean + 2 * sd)
    return elements

# appending train and test
trainTestAppendDF = pd.concat([trainAccount,testAccount]).reset_index(drop=True)

# converting variable into datetime
trainTestAppendDF['last_paymt_dt']=pd.to_datetime(trainTestAppendDF['last_paymt_dt'])
trainTestAppendDF['opened_dt']=pd.to_datetime(trainTestAppendDF['opened_dt'])
trainTestAppendDF['openDays'] = trainTestAppendDF['last_paymt_dt'].sub(trainTestAppendDF['opened_dt'], axis=0)
trainTestAppendDF['openDays']= trainTestAppendDF['openDays'].astype('timedelta64[D]')

# assigning the variable to new variable for further calculation
trainTestAppendDF['total_Days'] = trainTestAppendDF['openDays']
trainTestAppendDF['mean_Days'] = trainTestAppendDF['openDays']
trainTestAppendDF['total_creditlimit'] = trainTestAppendDF['creditlimit']
trainTestAppendDF['mean_creditlimit'] = trainTestAppendDF['creditlimit']
trainTestAppendDF['total_currentlimit'] = trainTestAppendDF['cur_balance_amt']
trainTestAppendDF['mean_currentlimit'] = trainTestAppendDF['cur_balance_amt']


# replacing missing value of creditlimit with high_credit_amt
trainTestAppendDF['creditlimit']= trainTestAppendDF['creditlimit'].fillna(trainTestAppendDF['high_credit_amt'])



# delete column list
deleteColumns = ['disputeremarkscode2','disputeremarkscode1','dateofentryforerrorordisputerema','cibilremarkscode',
                 'dateofentryforcibilremarkscode','errorcode','dateofentryforerrorcode','externalid','member_short_name','suitfiledorwilfuldefaultorwritte','dt_opened','upload_dt',
                 "closed_dt","reporting_dt","typeofcollateral","last_paymt_dt","opened_dt","high_credit_amt",'emiamount','repaymenttenure','creditlimit','cur_balance_amt']

# replacing missing value of a column with 0
fill_0_Columns = ['actualpaymentamount','paymentfrequency','settlementamount','writtenoffamountprincipal','writtenoffamounttotal',
                'cashlimit','valueofcollateral','amt_past_due','openDays',"total_creditlimit","rateofinterest","mean_creditlimit","total_Days","mean_Days"]

# drop the column
trainTestAppendDF = trainTestAppendDF.drop(deleteColumns,1)

# filling missing value with 0
for i in fill_0_Columns:
    trainTestAppendDF[i] = trainTestAppendDF[i].fillna(0)


# filling missing value of payment history
trainTestAppendDF["paymenthistory1"] = trainTestAppendDF['paymenthistory1'].fillna("Rohit1")
trainTestAppendDF["paymenthistory2"] = trainTestAppendDF['paymenthistory2'].fillna("Rohit1")

# payment boolean initialize,least_30dpd,max_30dpd
trainTestAppendDF['paymentBoolean'] = True
trainTestAppendDF['least_30dpd'] = 0.0
trainTestAppendDF['max_30dpd'] = 0.0



# loop for checking account in 0-29 dpd
for row in range(0,len(trainTestAppendDF)):
    string=trainTestAppendDF.iloc[row]['paymenthistory1'] + trainTestAppendDF.iloc[row]['paymenthistory2']
    output = [string[q:q+n] for q in range(0, len(string), n)]
    trainTestAppendDFOutput = [item for item in output if item not in acList]
    if(len(trainTestAppendDFOutput)==0):
        trainTestAppendDF.ix[row,'paymentBoolean'] = False
    else:
        int_output = [int(i) for i in trainTestAppendDFOutput]
        booleanList = []
        for j in range(0,len(int_output)):
            if(int_output[j]>=30):
                booleanList.append(True)
            else:
                booleanList.append(False)
        if(True in booleanList):
            trainTestAppendDF.ix[row, 'paymentBoolean'] = False
        else:
            trainTestAppendDF.ix[row, 'paymentBoolean'] = True

# loop for calculating first and last 30+ dpd month
for row3 in range(0, len(trainTestAppendDF)):
    string3 = trainTestAppendDF.iloc[row3]['paymenthistory1'] + trainTestAppendDF.iloc[row3]['paymenthistory2']
    output3 = [string3[q3:q3 + n] for q3 in range(0, len(string3), n)]
    trainTestAppendDFOutput3 = [item3 for item3 in output3 if item3 not in acList2]
    for z3, i3 in enumerate(trainTestAppendDFOutput3):
        if i3 in replaceList:
            trainTestAppendDFOutput3[z3] = '000'
    trainTestAppendDFOutput3 = trainTestAppendDFOutput3[::-1]
    if (len(trainTestAppendDFOutput3) == 0):
        trainTestAppendDF.ix[row3, 'least_30dpd'] = 0.0
    else:
        int_output3 = [int(i4) for i4 in trainTestAppendDFOutput3]
    indexList = []
    for index, value in enumerate(int_output3):
        if (value >= 30):
            indexList.append(index)

    if (len(indexList) == 0):
        trainTestAppendDF.ix[row3, 'least_30dpd'] = 0.0
    else:
        trainTestAppendDF.ix[row3, 'least_30dpd'] = indexList[0] + 1
        trainTestAppendDF.ix[row3,'max_30dpd'] = indexList[len(indexList)-1]+1


# payment history month length
trainTestAppendDF['paymenthistory1'] = trainTestAppendDF['paymenthistory1'].apply(len) -6
trainTestAppendDF['paymenthistory2'] = trainTestAppendDF['paymenthistory2'].apply(len) -6
trainTestAppendDF['monthlength'] = trainTestAppendDF['paymenthistory1']+ trainTestAppendDF['paymenthistory2']

# separate the train and test
trainTestAppendDF_train = trainTestAppendDF[0:len(trainAccount)]
trainTestAppendDF_test = trainTestAppendDF[len(trainAccount):len(trainTestAppendDF)]

# aggregations
aggregations = {
    "cashlimit": lambda x : sum(x)/float(len(x)),
    "total_creditlimit":'sum',
    "total_currentlimit":'sum',
    "mean_creditlimit":lambda x : sum(x)/float(len(x)),
    "mean_currentlimit":lambda x : sum(x)/float(len(x)),
    'mean_Days':lambda x : sum(x)/len(x),
    'total_Days': lambda x : sum(x),
    'monthlength': lambda x: (sum(x)/3)/len(x),
    'paymentBoolean':lambda x : sum(x)/len(x),
    'least_30dpd': lambda x: max(x),
    'max_30dpd':lambda x : max(x)
}

# aggregations in train and test
trainTestAppendDF_train = trainTestAppendDF_train.groupby('apprefno', as_index=False).agg(aggregations)
trainTestAppendDF_test = trainTestAppendDF_test.groupby('apprefno', as_index=False).agg(aggregations)

# saving train and test
trainTestAppendDF_train.to_csv('output/trainAccount.csv', index=False)
trainTestAppendDF_test.to_csv('output/testAccount.csv', index=False)







