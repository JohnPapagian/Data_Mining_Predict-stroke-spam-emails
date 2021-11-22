import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score



#,,keep_default_na=False
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
keys=df.columns
total_rows=len(df.index)

res=[]

for key in keys: # find column with N/A values
    for i in range(total_rows):
        #print(df[key][i])
        if (pd.isnull(df[key][i])):
            print("Missing value from "+key+" column")
            res.append(key)
            break



for key in res:
    df=df.fillna(df[key].mean()) #fill with mean of values
    #print(df[key].to_string())





df_mean=df
temp=df_mean.copy(deep=True)
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

target=temp.stroke
sample = temp.drop('stroke', axis='columns') #sample=all data except stroke


sample_train,sample_test,target_train,target_test\
=train_test_split(sample,target,test_size=0.25)



forest = RandomForestClassifier(n_estimators = 1000)


forest.fit(sample_train, target_train)

predictions = forest.predict(sample_test)


f1_score = f1_score( target_test,predictions, average="weighted",zero_division =0)


recall = recall_score( target_test,predictions, average="weighted",zero_division =0)


precision = precision_score( target_test,predictions, average="weighted",zero_division =0)

print("\nScores for 'drop column method' are\n")
print("F1 Score: ", f1_score)
print("Recall: ", recall)
print("Precision: ", precision)
