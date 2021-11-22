import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#,,keep_default_na=False
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

types=df.dtypes


print(types)
id_des=df['id'].describe()
gender_des=df['gender'].describe()
age_des=df['age'].describe()
hypertension_des=df['hypertension'].describe()
heart_disease_des=df['heart_disease'].describe()
married_des=df['ever_married'].describe()

work_des=df['work_type'].describe()
residence_des=df['Residence_type'].describe()
glucose_des=df['avg_glucose_level'].describe()
bmi_des=df['bmi'].describe()
smoking_des=df['smoking_status'].describe()
stroke_des=df['stroke'].describe()

un_glucose=pd.unique(df['avg_glucose_level'])
glucose_un=df['avg_glucose_level'].value_counts()

bins=range(20,300,20)



plt.hist(glucose_un.keys(),bins=bins,facecolor='blue',edgecolor='black')

plt.yticks(range(0,1200,50))
plt.xticks(bins)
plt.xlabel('Glucose Levels')
plt.ylabel('Number of people')
plt.title('Glucose Levels Distribution')
plt.show()


#print(id_des)
print("\n*******************************")
print("*******************************\n")
#print(un_work)
print(age_un.keys())
