from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import DistanceMetric
import matplotlib.ticker as mtick
os.chdir(os.path.dirname(__file__))

# OPISYWANKO
OUT_IMG_DIR = "../docs/src/img".replace('/', os.sep)
OUT_DATA_DIR = "../docs/src/data".replace('/', os.sep)
IN_DATA_DIR = "../data".replace('/', os.sep)

data_columns = [
    'Age',
    'Sex',
    # Pain
    'Pain location',
    'Chest pain radiation',
    'Pain character',
    'Onset of pain',
    'Number of hours since onset',
    'Duration of the last episode',
    # Associated symptoms
    'Nausea',
    'Diaphoresis_AS',
    'Palpitations',
    'Dyspnea',
    'Dizziness/syncope',
    'Burping',
    # Palliative factors
    'Palliative factors',
    # History of similar pain
    'Prior chest pain of this type',
    'Physician consulted for prior pain',
    'Prior pain related to heart',
    'Prior pain due to MI',
    'Prior pain due to angina prectoris',
    # Past medical history
    'Prior MI',
    'Prior angina prectoris',
    'Prior atypical chest pain',
    'Congestive heart failure',
    'Peripheral vascular disease',
    'Hiatal hernia',
    'Hypertension',
    'Diabetes',
    'Smoker',
    # Current medication usage
    'Diuretics',
    'Nitrates',
    'Beta blockers',
    'Digitalis',
    'Nonsteroidal anti-inflammator',
    'Antacids/H2 blockers',
    # Physical examinations
    'Systolic blood pressure',
    'Diastolic blood pressure',
    'Heart rate',
    'Respiration rate',
    'Rales',
    'Cyanosis',
    'Pallor',
    'Systolic murmur',
    'Diastolic murmur',
    'Oedema',
    'S3 gallop',
    'S4 gallop',
    'Chest wall tenderness',
    'Diaphoresis_PE',
    # ECG examination
    'New Q wave',
    'Any Q wave',
    'New ST segment elevation',
    'Any ST segment elevation',
    'New ST segment depression',
    'Any ST segment depression',
    'New T wave inversion',
    'Any T wave inversion',
    'New intraventricular conduction defect',
    'Any intraventricular conduction defect',
]

data_classes = {
    1: "Pain of non-heart origin",
    2: "Angina prectoris",
    3: "Angina prectoris - prizmental variant",
    4: "Myocardial infraction (transmural)",
    5: "Myocardial infraction (subendocardial)"
}


# WCZYTYWANKO
data_mi = pd.read_csv(join(IN_DATA_DIR, 'mi.txt'),
                      sep='\t', header=None).transpose()

data_mi_np = pd.read_csv(join(IN_DATA_DIR, 'mi_np.txt'),
                         sep='\t', header=None).transpose()

data_others = pd.read_csv(join(IN_DATA_DIR, 'inne.txt'),
                          sep='\t', header=None).transpose()

data_ang_prect = pd.read_csv(join(IN_DATA_DIR, 'ang_prect.txt'),
                             sep='\t', header=None).transpose()

data_ang_prect_2 = pd.read_csv(join(IN_DATA_DIR, 'ang_prct_2.txt'),
                               sep='\t', header=None).transpose()

# ŁĄCZONKO
data_all = pd.concat([
    data_mi, data_mi_np, data_others, data_ang_prect, data_ang_prect_2
])

count = data_all.shape[0]
pd.DataFrame([
    ['Pełnościenny zawał serca', data_mi.shape[0], "%.2f" %
        (data_mi.shape[0] / count * 100)],
    ['Podwsierdziowy zawał serca', data_mi_np.shape[0],  "%.2f" %
        (data_mi_np.shape[0] / count * 100)],
    ['Dusznica bolesna – dławica piersiowa', data_ang_prect.shape[0], "%.2f" %
        (data_ang_prect.shape[0] / count * 100)],
    ['Dusznica Prinzmetala – dławica naczynioskurczowa', data_ang_prect_2.shape[0], "%.2f" %
        (data_ang_prect_2.shape[0] / count * 100)],
    ['Ból niepochodzący z serca', data_others.shape[0], "%.2f" %
        (data_others.shape[0] / count * 100)],
]).to_csv(join(OUT_DATA_DIR, 'class_distribution.csv'), header=None)


data_classes = pd.concat([
    pd.Series([4]).repeat(data_mi.shape[0]),
    pd.Series([5]).repeat(data_mi_np.shape[0]),
    pd.Series([1]).repeat(data_others.shape[0]),
    pd.Series([2]).repeat(data_ang_prect.shape[0]),
    pd.Series([3]).repeat(data_ang_prect_2.shape[0]),

])

# RANKINGOWANIE
K = 30
kBest = SelectKBest(f_classif, k=K).fit(data_all, data_classes)
data_columns_score = pd.concat(
    [pd.DataFrame(data_columns), pd.DataFrame(kBest.scores_)], axis=1)
data_columns_score.columns = ['name', 'score']
data_columns_scores_sorted = data_columns_score.sort_values(
    'score', ascending=True
)

# WYKRESIK
fig, ax = plt.subplots()
rects = ax.barh(
    data_columns_scores_sorted.name,
    data_columns_scores_sorted.score,
)

ax.set_title(f'Ranking oparty o metodę ANOVA')
ax.grid(axis='x')

for i, rect in enumerate(rects):
    width = rect.get_width()
    ax.text(rect.get_width()+5, rect.get_y()+rect.get_height()*0.25,
            '%.2f' % data_columns_scores_sorted.score.tolist()[i],
            ha='left', va='bottom')

fig.set_figheight(16)
fig.set_figwidth(10)
ax.set_xlim((0, 340))
plt.ylim(-1, data_columns_scores_sorted.name.shape[0])

fig.tight_layout()
fig.savefig(join(OUT_IMG_DIR, 'by_score.png'))

# ANOVA ręcznie dla zabawy
""" i = 1  # analizowana cecha
anova_classes = [data_mi[i], data_mi_np[i],
                 data_others[i], data_ang_prect[i], data_ang_prect_2[i]]
y_mean = data_all[i].mean()

ss_factor = 0

for cl in anova_classes:
    cl_mean = cl.mean()
    n_i = cl.shape[0]
    ss_factor += n_i*((cl_mean - y_mean)**2)
ss_factor /= len(anova_classes) - 1

ss_error = 0
df = 0
for cl in anova_classes:
    cl_mean = cl.mean()
    n_i = cl.shape[0]
    df += n_i-1
    for value in cl:
        ss_error += (value - cl_mean)**2

ss_error /= df

print(ss_factor/ss_error) # Wynik """

rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=2137)
metrics = {
    'euclidean': 'euklidesową',
    'manhattan': 'Manhattan',
    'chebyshev': 'Czebyszewa',
}
for metric in ['euclidean', 'manhattan', 'chebyshev']:  # , 'manhattan', 'chebyshev'
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(16)
    fig.tight_layout(pad=3)

    ax.set_title('kNN z metryką %s' % metrics[metric])
    ax.grid()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Liczba cech')

    legends = []
    for N in [3, 5, 8]:  # , 5, 8
        k_scores = []
        k_stdevs = []
        legends.append('%i-NN' % N)
        for K in range(1, data_all.shape[1]):  #
            scores = []

            kBest.set_params(k=K)
            data_k_best = kBest.transform(data_all)
            for train_index, test_index in rkf.split(data_k_best):
                X_train = data_k_best[train_index]
                X_test = data_k_best[test_index]

                y_train = data_classes.iloc[train_index]
                y_test = data_classes.iloc[test_index]

                neigh = KNeighborsClassifier(
                    n_neighbors=N, metric=metric)
                neigh.fit(X_train, y_train)

                predict = neigh.predict(X_test)

                scores.append(accuracy_score(y_test, predict))

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            k_scores.append(mean_score)
            k_stdevs.append(std_score)
            # print("Score: %.3f (%.3f)" % (mean_score, std_score))

        x_ticks = np.arange(1, len(k_scores)+1)
        ax.set_xticks(x_ticks)
        print('metric %s, n %f, max %f' % (metric, N, max(k_scores)))
        ax.errorbar(x_ticks,
                    list(map(lambda x: x*100, k_scores)),
                    yerr=list(map(lambda x: x*100, k_stdevs)),
                    fmt='--o', capsize=2
                    )

    ax.legend(legends, loc='lower right')

    fig.savefig(join(OUT_IMG_DIR, 'plot_%s.png' % metric))


""" # kNN test
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
test_data = pd.DataFrame([[1, 22], [3, 4]])
test_data.to_csv(
    join(DATA_DIR, 'test.csv'),
    index=False,
    header=False
)
 """
