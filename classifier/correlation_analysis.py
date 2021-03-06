import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils


plt.rcParams.update({'font.size': 16})

X, _ = utils.get_dataset('dataset_threshold_100.txt')
# X, _ = utils.get_dataset('dataset_threshold_100_shift_05_2.txt')
features = utils.get_feature_names()
d = pd.DataFrame(data=X, columns=features)

# corr = d.corr('pearson')
corr = d.corr('kendall')
# corr = d.corr('spearman')

plt.figure()
sns.set(style="dark")

mask = np.triu(np.ones_like(corr, dtype=np.bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

points = [i + 0.5 for i in range(len(features))]
xticks = [i + 1 for i in range(len(features))]
yticks = [features[i] + '    ' + str(i + 1) for i in range(len(features))]
plt.xticks(points, xticks, rotation=0)
plt.yticks(points, yticks)
plt.title('Correlation matrix')
plt.show()