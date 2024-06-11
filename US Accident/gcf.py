import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from category_encoders import TargetEncoder
from GCForest import gcForest
from sklearn.metrics import accuracy_score

# 读取数据并解析日期时间列
accidents = pd.read_csv("US_Accidents_March23_del.csv", parse_dates=['Start_Time', 'End_Time'])

# 解析日期时间列并处理格式不一致的问题
accidents['Start_Time'] = pd.to_datetime(accidents['Start_Time'], errors='coerce')
accidents['End_Time'] = pd.to_datetime(accidents['End_Time'], errors='coerce')

# 提取时间相关的数据
accidents['Hour_of_Day'] = accidents['Start_Time'].dt.hour
accidents['Day_of_Week'] = accidents['Start_Time'].dt.dayofweek
accidents['Month'] = accidents['Start_Time'].dt.month
accidents['Time_of_Year'] = accidents['Start_Time'].dt.quarter
accidents['Duration'] = accidents['End_Time'] - accidents['Start_Time']
accidents['Duration_in_minutes'] = accidents['Duration'].dt.total_seconds() / 60

# 将数据集中的特定列转换为分类数据类型
categories = ['Sunrise_Sunset', 'Weather_Condition', 'Wind_Direction', 'State', 'Month', 'Time_of_Year',
              'Day_of_Week', 'City']
accidents[categories] = accidents[categories].astype('category')

# 调整严重性等级
accidents['Severity'] = accidents['Severity'] - 1

accidents[['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
           'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']] = accidents[['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
           'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']].astype('int')

# 分离特征和目标变量
y = accidents.pop('Severity')
X = accidents.copy()

numeric_cols = ['Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                'Precipitation(in)', 'Hour_of_Day', 'Duration_in_minutes', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
                'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                'Turning_Loop']

categorical_cols = categories

# 数值特征处理
numeric_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(numeric_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_cols)

# 分类特征处理
categorical_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
X_categorical = pd.DataFrame(categorical_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

encoder = TargetEncoder()

# 确保索引一致性
X_categorical.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# 使用 TargetEncoder 进行编码
X_categorical_encoded = pd.DataFrame(encoder.fit_transform(X_categorical, y), columns=categorical_cols)

# 合并处理后的特征
X_processed = pd.concat([X_numeric_scaled, X_categorical_encoded], axis=1)

# Initialize and train the gcForest model
gc = gcForest(shape_1X=X_processed.shape[1],  # shape_1X should match the number of features
              n_mgsRFtree=30, window=[3, 7], stride=1, cascade_test_size=0.2,
              n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=np.inf, min_samples_mgs=0.1,
              min_samples_cascade=0.05, tolerance=0.0, n_jobs=1)

# Fit the model
gc.fit(X_processed.values, y.values)

# Predictions
y_pred = gc.predict(X_processed.values)

# Evaluation
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='gnuplot', xticklabels=['1', '2', '3', '4'], yticklabels=['1', '2', '3', '4'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plotting True vs Predicted Values
plt.figure(figsize=(10, 6))
sns.histplot(y, label='True Values', kde=False, color='blue', alpha=0.6)
sns.histplot(y_pred, label='Predicted Values', kde=False, color='green', alpha=0.6)
plt.xlabel('Severity')
plt.ylabel('Count')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()

# Optionally, you can also print classification reports
print("\nClassification Report:")
print(classification_report(y, y_pred))

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
# from category_encoders import TargetEncoder
# from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# from sklearn.metrics import accuracy_score, classification_report
# from GCForest import *
#
# # 读取数据并解析日期时间列
# accidents = pd.read_csv("US_Accidents_March23.csv", parse_dates=['Start_Time', 'End_Time'])
# accidents=accidents.loc[0:80000,:]
#
# # 解析日期时间列并处理格式不一致的问题
# accidents['Start_Time'] = pd.to_datetime(accidents['Start_Time'], errors='coerce')
# accidents['End_Time'] = pd.to_datetime(accidents['End_Time'], errors='coerce')
#
# # 提取时间相关的数据
# accidents['Hour_of_Day'] = accidents['Start_Time'].dt.hour
# accidents['Day_of_Week'] = accidents['Start_Time'].dt.dayofweek
# accidents['Month'] = accidents['Start_Time'].dt.month
# accidents['Time_of_Year'] = accidents['Start_Time'].dt.quarter
# accidents['Duration'] = accidents['End_Time'] - accidents['Start_Time']
# accidents['Duration_in_minutes'] = accidents['Duration'].dt.total_seconds() / 60
#
# # 将数据集中的特定列转换为分类数据类型
# categories = ['Sunrise_Sunset', 'Weather_Condition', 'Wind_Direction', 'State', 'Month', 'Time_of_Year',
#               'Day_of_Week', 'City']
# accidents[categories] = accidents[categories].astype('category')
#
# # 调整严重性等级
# accidents['Severity'] = accidents['Severity'] - 1
#
# accidents[['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
#            'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']] = accidents[['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
#            'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']].astype('int')
#
# # 分离特征和目标变量
# y = accidents.pop('Severity')
# X = accidents.copy()
#
# # 划分训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#
# numeric_cols = ['Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
#                 'Precipitation(in)', 'Hour_of_Day', 'Duration_in_minutes', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
#                 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
#                 'Turning_Loop']
#
# categorical_cols = categories
#
# # 数值特征处理
# numeric_imputer = SimpleImputer(strategy='mean')
# X_train_numeric = pd.DataFrame(numeric_imputer.fit_transform(X_train[numeric_cols]), columns=numeric_cols)
# X_val_numeric = pd.DataFrame(numeric_imputer.transform(X_val[numeric_cols]), columns=numeric_cols)
#
# scaler = StandardScaler()
# X_train_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_train_numeric), columns=numeric_cols)
# X_val_numeric_scaled = pd.DataFrame(scaler.transform(X_val_numeric), columns=numeric_cols)
#
# # 分类特征处理
# categorical_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
# X_train_categorical = pd.DataFrame(categorical_imputer.fit_transform(X_train[categorical_cols]), columns=categorical_cols)
# X_val_categorical = pd.DataFrame(categorical_imputer.transform(X_val[categorical_cols]), columns=categorical_cols)
#
# encoder = TargetEncoder()
#
# # 确保索引一致性
# X_train_categorical.reset_index(drop=True, inplace=True)
# y_train.reset_index(drop=True, inplace=True)
#
# # 使用 TargetEncoder 进行编码
# X_train_categorical_encoded = pd.DataFrame(encoder.fit_transform(X_train_categorical, y_train), columns=categorical_cols)
# X_val_categorical_encoded = pd.DataFrame(encoder.transform(X_val_categorical), columns=categorical_cols)
#
# # 合并处理后的特征
# X_train_processed = pd.concat([X_train_numeric_scaled, X_train_categorical_encoded], axis=1)
# X_val_processed = pd.concat([X_val_numeric_scaled, X_val_categorical_encoded], axis=1)
#
#
#
# # 定义GCForest模型
# gc = gcForest(shape_1X=X_train_processed.shape[1],  # shape_1X should match the number of features
#               n_mgsRFtree=30, window=[3, 7], stride=1, cascade_test_size=0.2,
#               n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=np.inf, min_samples_mgs=0.1,
#               min_samples_cascade=0.05, tolerance=0.0, n_jobs=1)
#
# #拟合模型
# gc.fit(X_train_processed.values, y_train.values)
#
# #预测
# y_pred_train = gc.predict(X_train_processed.values)
# y_pred_val = gc.predict(X_val_processed.values)
#
#
# train_accuracy = accuracy_score(y_train, y_pred_train)
# val_accuracy = accuracy_score(y_val, y_pred_val)
#
# print(f"Training Accuracy: {train_accuracy:.4f}")
# print(f"Validation Accuracy: {val_accuracy:.4f}")
#
# #混淆矩阵
# cm_train = confusion_matrix(y_train, y_pred_train)
# cm_val = confusion_matrix(y_val, y_pred_val)
# plt.figure(figsize=(12, 5))
# #训练集混淆矩阵
# plt.subplot(1, 2, 1)
# sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title('Training Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# #验证集混淆矩阵
# plt.subplot(1, 2, 2)
# sns.heatmap(cm_val, annot=True, fmt='d', cmap='gnuplot', xticklabels=['1', '2', '3', '4'], yticklabels=['1', '2', '3', '4'])
# plt.title('Validation Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.tight_layout()
# plt.show()
#
# #真实值与预测值对比图
# plt.figure(figsize=(10, 6))
# sns.histplot(y_val, label='True Values', kde=False, color='blue', alpha=0.6)
# sns.histplot(y_pred_val, label='Predicted Values', kde=False, color='green', alpha=0.6)
# plt.xlabel('Severity')
# plt.ylabel('Count')
# plt.title('True vs Predicted Values (Validation Set)')
# plt.legend()
# plt.show()
#
# #分类报告
# print("\nTraining Classification Report:")
# print(classification_report(y_train, y_pred_train))
#
# print("\nValidation Classification Report:")
# print(classification_report(y_val, y_pred_val))