import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import norm
from sklearn.decomposition import KernelPCA

# Номер 1.1
A = [5.6, 4.8, 5.5, 6.4, 5.6, 6.7, 4.9, 5.5]
B = [-15.2, -11.6, -17.8, -8.9, -17.5, -17.4, -10.7, -10.2]
n = len(A)

plt.scatter(A, B)
for i in range(n):
    plt.text(A[i], B[i], str(i + 1))

plt.xlabel('A')
plt.ylabel('B')
plt.title('Диаграмма рассеяния')
plt.show()

# Номер 1.2
print('Номер 1.2')
D = np.stack([A, B]).T
K = np.ndarray((n, n))

for i in range(n):
    for j in range(n):
        K[i][j] = np.linalg.norm(D[i] - D[j]) ** 2

print('Ядерная матрица:\n', K)

# Номер 2.1
X1 = [-16, -18, 47, 6, -71, 31, 173, 45]
X2 = [60, 56, 64, 63, 68, 65, 88, 65]
n = len(X1)
name = [str(i + 1) for i in range(n)]

plt.scatter(X1, X2)
for i in range(len(X1)):
    plt.text(X1[i], X2[i], name[i])

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Диаграмма рассеяния')
plt.show()

# Номер 2.2
print('\nНомер 2.2')
X = np.vstack((X1, X2))

mean = np.array([np.mean(X1), np.mean(X2)])
print('Среднее значение матрицы D:\n', mean)

Xc = np.vstack((X[0] - mean[0], X[1] - mean[1]))

C1 = np.cov(X)
print('Ковариационная матрица матрицы D:\n', np.round(C1, 2))

C2 = np.cov(Xc)
print('Ковариационная матрица матрицы Dc:\n', np.round(C2, 2))

# Номер 2.3
print('\nНомер 2.3')
l, v = np.linalg.eig(C2)
print('Собственные числа матрицы Σc:\n', np.round(l, 4))
print('Собственные векторы матрицы Σc:\n', np.round(v, 4))

# Номер 2.4
print('\nНомер 2.4')
d = {l[i]: i for i in range(len(l))}
ind = d[max(l)]
print('Индекс, соответствующий первой главной компоненте:', ind)

# Номер 2.5
print('\nНомер 2.5')
vec = v[:, ind]
print('Первый главный компонент:\n', np.round(vec, 4))

k = vec[1] / vec[0]
X1new = Xc.transpose().dot(vec)  ###
X2new = X1new * k

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.scatter(X1new, X2new, color='blue')

for i in range(len(X1new)):
    plt.text(X1new[i], X2new[i], name[i])

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Проекция точек (ручная)')
# plt.show()

# Номер 2.6
print('\nНомер 2.6')
pca = PCA(n_components=1)
X1PCA = pca.fit_transform(np.array(X).transpose())

X2PCA = X1PCA * k
print('Первый главный компонент sklearn:\n', np.round(pca.components_, 4))

plt.subplot(1, 2, 2)
plt.scatter(X1PCA, X2PCA, color='green')

for i in range(len(X1PCA)):
    plt.text(X1PCA[i], X2PCA[i], name[i])

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Проекция точек (sklearn)')
plt.show()

# Номер 2.7
print('\nНомер 2.7')
print('Название оси графика из пункта 1, проекция данных на которую сравнима с результатами PCA преобразований:',
      end=' ')
if (abs(vec[0]) > abs(vec[1])):
    print('X1')
else:
    print('X2')

# Номер 2.8
a, b = np.random.multivariate_normal(mean, C1, 1000).T
plt.scatter(a, b)
plt.title('Диаграмма рассеяния двумерной случайной величины')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

val = np.linspace(np.mean(X1) - 3 * np.std(X1), np.mean(X1) + 3 * np.std(X1), num=1000)

pdfA = norm.pdf(val, np.mean(X1), np.std(X1))
pdfB = norm.pdf(val, np.mean(X2), np.std(X2))

plt.plot(val, pdfA, color='green')
plt.plot(val, pdfB, color='orange')

plt.ylabel('Вероятность')
plt.title('Функции плотности')
plt.legend(['f(X1)', 'f(X2)'])
plt.show()

# Номер 3.1
print('\nНомер 3.1')
n = len(A)

E = np.eye(n, n)
one = np.ones((n, n))
Kc = np.array(np.array(E - one / n).dot(K)).dot(E - one / n)

Kl, Kv = np.linalg.eig(Kc)
d = {np.abs(Kl[i]): i for i in range(n)}
ind = d[max(np.abs(Kl))]

Kvec = Kv[:, ind]
print('Первая главная компонента:\n ', np.round(Kvec.real, 4))

Xker = K.dot(np.array(Kvec).transpose()).real
null = np.zeros((1, n))

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.scatter(Xker, null, color='green')
for i in range(n):
    plt.text(Xker[i], 0, str(i + 1))
plt.title('Linear kernel')

# Номер 3.2
print('\nНомер 3.2')
X = np.stack((A, B))
t = KernelPCA(n_components=1, kernel='rbf', gamma=1)
#t = KernelPCA(n_components=1, kernel='linear')
X_t = t.fit_transform(np.array(X).transpose())
print('Первая главная компонента:\n ', np.round(t.eigenvectors_, 4))

plt.subplot(1, 2, 2)
plt.scatter(X_t, null, color='orange')
for i in range(n):
    plt.text(X_t[i], 0, str(i + 1))
plt.title('RBF kernel')
plt.show()
