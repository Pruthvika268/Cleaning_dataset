# Load and inspect dataset
import pandas as pd

df = pd.read_csv('C:\\Users\\91808\\OneDrive\\Desktop\\ML\\Dataset\\sales_raw_data.csv')
df.head()
df.info()
df.isnull().sum()

# HANDLE MISSING DATA
# Fill missing numerical values
df['sales'] = df['sales'].fillna(df['sales'].mean())
df['profit'] = df['profit'].fillna(df['profit'].median())
#print(df)

#Fill categorical missing values
df['region'] = df['region'].fillna(df['region'].mode()[0])
df['category'] = df['category'].fillna(df['category'].mode()[0])
print(df)

#Remove Duplications
df.duplicated().sum()
df = df.drop_duplicates()
print(df)

#Encode Categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['region_encoded'] = le.fit_transform(df['region'])

#one_hot encoding
df = pd.get_dummies(df, columns=['category'], drop_first=True)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['sales','profit']] = scaler.fit_transform(df[['sales','profit']])

#final clean Dataset check
df.info()
df.head()