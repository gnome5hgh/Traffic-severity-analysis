from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_cols = ['Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                'Precipitation(in)', 'Hour_of_Day', 'Duration_in_minutes', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
                'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                'Turning_Loop']

categorical_cols = categories

# 数值特征处理
numeric_imputer = SimpleImputer(strategy='mean')
X_train_numeric = pd.DataFrame(numeric_imputer.fit_transform(X_train[numeric_cols]), columns=numeric_cols)
X_val_numeric = pd.DataFrame(numeric_imputer.transform(X_val[numeric_cols]), columns=numeric_cols)

scaler = StandardScaler()
X_train_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_train_numeric), columns=numeric_cols)
X_val_numeric_scaled = pd.DataFrame(scaler.transform(X_val_numeric), columns=numeric_cols)

# 分类特征处理
categorical_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
X_train_categorical = pd.DataFrame(categorical_imputer.fit_transform(X_train[categorical_cols]), columns=categorical_cols)
X_val_categorical = pd.DataFrame(categorical_imputer.transform(X_val[categorical_cols]), columns=categorical_cols)

encoder = TargetEncoder()

# 确保索引一致性
X_train_categorical.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

# 使用 TargetEncoder 进行编码
X_train_categorical_encoded = pd.DataFrame(encoder.fit_transform(X_train_categorical, y_train), columns=categorical_cols)
X_val_categorical_encoded = pd.DataFrame(encoder.transform(X_val_categorical), columns=categorical_cols)

# 合并处理后的特征
X_train_processed = pd.concat([X_train_numeric_scaled, X_train_categorical_encoded], axis=1)
X_val_processed = pd.concat([X_val_numeric_scaled, X_val_categorical_encoded], axis=1)

# 创建SVM分类器
svm_classifier = SVC(kernel='rbf', random_state=42)

# 在训练集上训练模型
svm_classifier.fit(X_train_processed, y_train)

# 在验证集上进行预测
y_pred = svm_classifier.predict(X_val_processed)

# 评估模型性能
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_val, y_pred))

# 绘制混淆矩阵
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='gnuplot', xticklabels=['1', '2', '3', '4'], yticklabels=['1', '2', '3', '4'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 绘制真实值与预测值对比图
plt.figure(figsize=(10, 6))
sns.histplot(y_val, label='True Values', kde=False, color='blue', alpha=0.6)
sns.histplot(y_pred, label='Predicted Values', kde=False, color='green', alpha=0.6)
plt.xlabel('Severity')
plt.ylabel('Count')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()