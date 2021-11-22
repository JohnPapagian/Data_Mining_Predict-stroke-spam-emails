import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#,,keep_default_na=False
df = pd.read_csv('healthcare-dataset-stroke-data.csv')


smoking1=df['smoking_status'].describe()
smoking2=df['heart_disease'].describe()
smoking3=df['ever_married'].describe()
smoking4=df['work_type'].describe()
smoking5=df['Residence_type'].describe()
smoking6=df['avg_glucose_level'].describe()
smoking7=df['bmi'].describe()
smoking8=df['smoking_status'].describe()
smoking9=df['stroke'].describe()
print(smoking1)
print("\n*******************************")
print(smoking2)
print("\n*******************************")
print(smoking3)
print("\n*******************************")
print(smoking4)
print("\n*******************************")
print(smoking5)
print("\n*******************************")
print(smoking6)
print("\n*******************************")
print(smoking7)
print("\n*******************************")
print(smoking8)
print("\n*******************************")
print(smoking9)
print("\n*******************************")

print("\n*******************************")
print("*******************************\n")
print(mean(df['smoking_status']))
print("\n*******************************")
print("*******************************\n")

un_smoking_status=pd.unique(df['smoking_status'])
smoking_status_un=df['smoking_status'].value_counts()

keys=smoking_status_un.keys()


plt.bar(keys,smoking_status_un,align='center')
plt.ylabel('Number of People')
plt.xlabel('Smoking habits')
plt.title('Smoking habits of people')


plt.show()


#print(id_des)
print("\n*******************************")
print("*******************************\n")
#print(un_work)
print(smoking_status_un)
print(keys)
