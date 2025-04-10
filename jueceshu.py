import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 假设文件名为 fault_1.csv, fault_2.csv, ..., fault_10.csv
file_names = [f'data/{i}.csv' for i in range(10)]
data_list = []
labels_list = []

# 读取每个CSV文件并添加标签
for i, file in enumerate(file_names):
    data = pd.read_csv(file, header=None)
    data['label'] = f'{i+1}'
    data_list.append(data)

# 合并所有数据
df = pd.concat(data_list, axis=0).reset_index(drop=True)

# 分离特征和标签
X_data = df.iloc[:, 0].values.reshape(-1, 1)  # 只有一列数据
Y_data = df['label']

# 标准化处理
scaler = StandardScaler()
X_data_scaled = scaler.fit_transform(X_data)

# 标签编码
labels = LabelEncoder()
labels = labels.fit_transform(Y_data)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, labels, test_size=0.3, random_state=42)
X_val, X_test_final, y_val, y_test_final = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print('size of x train:', X_train.shape)
print('size of x test:', X_test.shape)
print('size of y train:', y_train.shape)
print('size of y test:', y_test.shape)
print('size of x val:', X_val.shape)
print('size of x test final:', X_test_final.shape)
print('size of y val:', y_val.shape)
print('size of y test final:', y_test_final.shape)

# 训练决策树模型
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=50, min_samples_split=10, min_samples_leaf=5, random_state=0)
clf = clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_val = clf.predict(X_val)
acc_train = accuracy_score(y_train, y_pred_train)
acc_val = accuracy_score(y_val, y_pred_val)
print('训练集准确率:', acc_train)
print('验证集准确率:', acc_val)

# 绘制验证集上的混淆矩阵
confusion_matrix_val = confusion_matrix(y_val, y_pred_val)
sns.set(font_scale=1.5)
fig = plt.figure(figsize=(10, 8))
ax = sns.heatmap(confusion_matrix_val, annot=True, cmap='YlGnBu', fmt='g')
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize=22)
ax.set_ylabel("Actual Label", fontsize=22)
plt.yticks(rotation=0)
plt.show()

# 不同深度决策树的准确率曲线
dept_max = range(1, 10, 1)
train_scores_1, val_scores_1 = list(), list()
for i in dept_max:
    clf_over = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=i, random_state=0)
    clf_over = clf_over.fit(X_train, y_train)
    y_pred_train = clf_over.predict(X_train)
    y_pred_val_over = clf_over.predict(X_val)
    acc_train_decision = accuracy_score(y_train, y_pred_train)
    acc_val_decision = accuracy_score(y_val, y_pred_val_over)
    train_scores_1.append(acc_train_decision)
    val_scores_1.append(acc_val_decision)
plt.figure(figsize=(8, 6))
plt.plot(dept_max, train_scores_1, '-o', label='Train')
plt.plot(dept_max, val_scores_1, '-o', label='Val')
plt.xlabel('Number of tree depths', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.legend()
plt.show()

# 在最终测试集上的评估
y_pred_test = clf.predict(X_test_final)
acc_test_decision = accuracy_score(y_test_final, y_pred_test)
print('测试集准确率:', acc_test_decision)

# 绘制测试集上的混淆矩阵
confusion_matrix = confusion_matrix(y_test_final, y_pred_test)
sns.set(font_scale=1.5)
fig = plt.figure(figsize=(10, 8))
ax = sns.heatmap(confusion_matrix, annot=True, cmap='magma', fmt='g')
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize=22)
ax.set_ylabel("Actual Label", fontsize=22)
plt.yticks(rotation=0)
plt.show()
fig.savefig('confusion_matrix_1.png')
