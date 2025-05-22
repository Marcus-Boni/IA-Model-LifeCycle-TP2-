import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score, mean_squared_error

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Acurácia: {acc:.4f}')

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.show()

np.random.seed(42)
X_train_noise = X_train + np.random.normal(0, 0.1, X_train.shape)

knn_noise = KNeighborsClassifier(n_neighbors=5)
knn_noise.fit(X_train_noise, y_train)
y_pred_noise = knn_noise.predict(X_test)
acc_noise = accuracy_score(y_test, y_pred_noise)
print(f'Acurácia com ruído: {acc_noise:.4f}')

k_range = range(1, 21)
accuracies = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    y_k_pred = knn_k.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_k_pred))

plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o')
plt.xlabel('Número de Vizinhos K')
plt.ylabel('Acurácia')
plt.title('Acurácia variando o valor de K')
plt.grid()
plt.show()

X_reg = df.drop(['mean area', 'target'], axis=1)
y_reg = df['mean area']
X_reg_scaled = scaler.fit_transform(X_reg)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

r2 = r2_score(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f'R²: {r2:.4f}')
print(f'MSE: {mse:.4f}')

residuals = y_test_reg - y_pred_reg

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histograma dos Resíduos')
plt.xlabel('Resíduo')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel('Valor Real')
plt.ylabel('Valor Predito')
plt.title('Valor Real x Valor Predito')
plt.show()
