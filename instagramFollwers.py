import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data/Instagram_data.csv")



cols = ['Impression', 'From Home', 'From Hashtags', 'From Explore', 'From Other', 'Saves', 'Comments', 'Shares', 'Likes', 'Profile Visits']

############################################################

# 해시태그 개수를 세어 Hashtags_count 열을 만드는 함수
import csv
def count_question_marks(sentence):
    return sentence.count('?')

def process_csv(csv_file, column_name):
    new_column_name = f"{column_name}_count"

    temp_file = 'temp.csv'

    with open(csv_file, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + [new_column_name]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            sentence = row[column_name]
            question_marks_count = count_question_marks(sentence) + 1
            row[new_column_name] = question_marks_count
            writer.writerow(row)
    import os
    os.replace(temp_file, csv_file)


csv_file = 'data/Instagram_data.csv'
column_name = 'Hashtags'

process_csv(csv_file, column_name)


# 캡션의 문장 길이를 세어 Caption_length 열을 만드는 함수

import csv
def calculate_sentence_length(sentence):
    return len(sentence)
def process_csv(csv_file, column_name):
    new_column_name = f"{column_name}_length"

    temp_file = 'temp.csv'

    with open(csv_file, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + [new_column_name]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()


        for row in reader:
            sentence = row[column_name]
            sentence_length = calculate_sentence_length(sentence)
            row[new_column_name] = sentence_length
            writer.writerow(row)


    import os
    os.replace(temp_file, csv_file)

csv_file = "data/Instagram_data.csv"
column_name = 'Caption'

process_csv(csv_file, column_name)


with open('describe_result.txt', 'w') as f:
    f.write(df.describe().to_string())

print(df['Follows'].describe())


####################################################################


numerical_columns=['Impressions', 'From Home', 'From Hashtags', 'From Explore', 'From Other', 'Saves', 'Comments', 'Shares', 'Likes', 'Profile Visits', 'Caption_length', 'Hashtags_count']


fig = plt.figure(figsize = (16, 20))
ax = fig.gca()
df[numerical_columns].hist(ax=ax)
plt.show()



cols = ['Follows', 'Impressions', 'From Home', 'From Hashtags', 'From Explore', 'From Other', 'Saves', 'Comments', 'Shares', 'Likes', 'Profile Visits', 'Caption_length', 'Hashtags_count']

corr = df[cols].corr(method = 'pearson')

with open('correlation_matrix.txt', 'w') as f:
    f.write(corr.to_string())


fig = plt.figure(figsize = (16, 12))
ax = fig.gca()

sns.set(font_scale = 1.5)  # heatmap 안의 font-size 설정
heatmap = sns.heatmap(corr.values, annot = True, fmt='.2f', annot_kws={'size':15},
                      yticklabels = cols, xticklabels = cols, ax=ax, cmap = "OrRd")
plt.tight_layout()
plt.show()


sns.scatterplot(data=df, x='Caption_length', y='Follows', markers='o', color='red', alpha=0.6)
plt.title('Scatter Plot')
plt.show()


sns.scatterplot(data=df, x='Comments', y='Follows', markers='o', color='red', alpha=0.6)
plt.title('Scatter Plot')
plt.show()
############################################################################

print(df['Caption'].value_counts())




fig = plt.figure(figsize = (12, 20))
sns.boxplot(x='Follows', y='Impressions', data=df)
plt.show()


fig = plt.figure(figsize = (12, 20))
sns.boxplot(x='Hashtags_count', y='Impressions', data=df)
#####################################################################################

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # 평균 0, 분산 1
scale_columns = ['From Home', 'From Hashtags', 'From Explore', 'From Other', 'Comments', 'Likes', 'Profile Visits']
df[scale_columns] = scaler.fit_transform(df[scale_columns])

import seaborn as sns


sns.pairplot(df[['Comments', 'Shares', 'Likes', 'Profile Visits']])


with open('result.txt', 'w') as f:
    f.write(df.head().to_string())
with open('result2.txt', 'w') as f:
    f.write(df[scale_columns].head().to_string())
print(df.head())
print(df[scale_columns].head())



from sklearn.model_selection import train_test_split


X = df[scale_columns]
y = df['Follows']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_test.shape, y_test.shape

print(X_train.shape, y_train.shape)

print(y_train)

print(X_train)


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['features'] = X_train.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print(vif.round(1))



from sklearn import linear_model


lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
pred_test = lr.predict(X_test)


print(lr.coef_)


coefs = pd.DataFrame(zip(df[scale_columns].columns, lr.coef_), columns = ['feature', 'coefficients'])

coefs_new = coefs.reindex(coefs.coefficients.abs().sort_values(ascending=False).index)
print(coefs_new)



plt.figure(figsize = (8, 8))


plt.barh(coefs_new['feature'], coefs_new['coefficients'])
plt.title('"feature - coefficient" Graph')
plt.xlabel('coefficients')
plt.ylabel('features')
plt.show()


import statsmodels.api as sm

X_train2 = sm.add_constant(X_train)

model2 = sm.OLS(y_train, X_train2).fit()
print(model2.summary())



df = pd.DataFrame({'actual': y_test, 'prediction': pred_test})
df = df.sort_values(by='actual').reset_index(drop=True)
print(df.head())

plt.figure(figsize=(12, 9))
plt.scatter(df.index, df['prediction'], marker='x', color='r')
plt.scatter(df.index, df['actual'], alpha=0.3, marker='o', color='black')
plt.title("Prediction Result in Test Set", fontsize=20)
plt.legend(['prediction', 'actual'], fontsize=12)
plt.show()


print(model.score(X_train, y_train))  # training set
print(model.score(X_test, y_test))  # test set

### RMSE(Root Mean Squared Eror)
from sklearn.metrics import mean_squared_error
from math import sqrt

### training set
pred_train = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, pred_train)))

### test set
print(sqrt(mean_squared_error(y_test, pred_test)))


# 회귀식 출력
from sklearn import linear_model

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)


print(lr.coef_)
