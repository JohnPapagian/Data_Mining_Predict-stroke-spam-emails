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

un_work=pd.unique(df['work_type'])
num_un=df['work_type'].value_counts().plot(kind='bar')

sns.set(font_scale=1.4)
num_un=df['work_type'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Fields", labelpad=14)
plt.ylabel("Number of People", labelpad=14)
plt.title("People per occupation field", y=1.02);
plt.show()
#print(id_des)
#print("\n*******************************")
#print("*******************************\n")
#print(un_work)
#print(num_un)
