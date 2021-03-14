from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd

IMG_DIR = "docs/src/img"
DATA_DIR = "docs/src/data"

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
fig.savefig(join(IMG_DIR, 'test.png'))

# csv test
test_data = pd.DataFrame([[1, 2], [3, 4]])
test_data.to_csv(
    join(DATA_DIR, 'test.csv'),
    index=False,
    header=False
)
