import seaborn as sns
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

labels = {
    'kvr': ['Nav','Sch','Wea'],
    'mwoz':['Att','Hot','Res']
}


confusion = {
    "kvr": [[0.840, 0.030, 0.130], [0.048, 0.798, 0.154], [0.020, 0.010, 0.970]],
    "kvr_p": [[0.667, 0.333, 0.000], [0.000, 1.000, 0.000], [0.000, 0.000, 1.000]],
    "mwoz": [[0.333, 0.583, 0.083], [0.119, 0.791, 0.090], [0.000, 0.097, 0.903]],
    "mwoz_p": [[0.667, 0.333, 0.000], [0.000, 1.000, 0.000], [0.333, 0.000, 0.667]],
}

name = sys.argv[1]


fig = plt.figure(figsize=(8, 6))
sns.set_context("poster")
heatmap = sns.heatmap(confusion[name], annot=True, cmap='Blues' )
heatmap.set_yticklabels(labels[name.replace("_p","")], rotation=30) 
heatmap.set_xticklabels(labels[name.replace("_p","")], rotation=30) 
fig.tight_layout()
plt.show()