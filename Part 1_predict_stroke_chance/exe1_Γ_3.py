import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LinearRegression

#,,keep_default_na=False
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
np.set_printoptions(threshold=np.inf)



########################################
#df1 contains the rows where bmi exists#
########################################
df1=df
df1=df1.dropna(axis=0) #drop rows with no bmi


temp=df1.copy(deep=True)
temp['ever_married']=temp['ever_married'].replace('Yes',1)
temp['ever_married']=temp['ever_married'].replace('No',0)

temp['gender']=temp['gender'].replace('Male',0)
temp['gender']=temp['gender'].replace('Female',1)
temp['gender']=temp['gender'].replace('Other',2)

temp['Residence_type']=temp['Residence_type'].replace('Urban',1)
temp['Residence_type']=temp['Residence_type'].replace('Rural',0)

temp['smoking_status']=temp['smoking_status'].replace('never smoked',0)
temp['smoking_status']=temp['smoking_status'].replace('Unknown',1)
temp['smoking_status']=temp['smoking_status'].replace('formerly smoked',2)
temp['smoking_status']=temp['smoking_status'].replace('smokes',3)

temp['work_type']=temp['work_type'].replace('Private',0)
temp['work_type']=temp['work_type'].replace('Self-employed',1)
temp['work_type']=temp['work_type'].replace('children',2)
temp['work_type']=temp['work_type'].replace('Govt_job',3)
temp['work_type']=temp['work_type'].replace('Never_worked',4)

sample1=temp[[ 'age','gender', 'hypertension', 'heart_disease','ever_married',
'work_type','Residence_type','avg_glucose_level','smoking_status',
 'stroke']].to_numpy()


model = LinearRegression()
#train model with existing values
model.fit(sample1, df1.bmi)


arr=np.array(sample1)


res=model.predict(arr)

bmi_col=df['bmi'].copy()

for k in range(len(arr)):
    if (math.isnan(df['bmi'][k])):
        #print("found",df['bmi'][k])
        bmi_col[k]=res[k]                 #replace missing values in temp column


temp.bmi=bmi_col                          #replace column in df with temp column
#temp column comtains values of bmi predicted by linear regression







target=temp.stroke
sample = temp.drop('stroke', axis='columns')


sample_train,sample_test,target_train,target_test\
=train_test_split(sample,target,test_size=0.25)



forest = RandomForestClassifier(n_estimators = 1000)


forest.fit(sample_train, target_train)

predictions = forest.predict(sample_test)

cnt=0
all=0
for key,value in target_test.items():
    #print("key is",key)
    if (predictions[all]==target_test[key]):
        if(predictions[all]==1):
            print(" its a one ")
        cnt+=1
    all+=1

print(len(target))
print("\n It found ",cnt," out of ",all)



f1_score = f1_score( target_test,predictions, average="weighted",zero_division =0)


recall = recall_score( target_test,predictions, average="weighted",zero_division =0)


precision = precision_score( target_test,predictions, average="weighted",zero_division =0)

print("\nScores for 'drop column method' are\n")
print("F1 Score: ", f1_score)
print("Recall: ", recall)
print("Precision: ", precision)
