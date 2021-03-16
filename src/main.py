from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import pandas as pd
os.chdir(os.path.dirname(__file__))

IMG_DIR = "../docs/src/img".replace('/', os.sep)
DATA_DIR = "../docs/src/data".replace('/', os.sep)

# kNN test
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))

# plot testt
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
fig.savefig(os.path.join(IMG_DIR, 'test.png'))

# csv test
test_data = pd.DataFrame([[1, 22], [3, 4]])
test_data.to_csv(
    os.path.join(DATA_DIR, 'test.csv'),
    index=False,
    header=False
)
