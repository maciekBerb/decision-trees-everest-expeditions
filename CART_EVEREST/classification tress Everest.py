# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:36:29 2019

@author: Maciek
"""

import pandas as pd
import numpy as np
import seaborn as sns
"""
import graphviz
import pydotplus
"""
import io

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from IPython.display import Image
from scipy import misc
from matplotlib import pyplot as plt

data = pd.read_excel("Everest_data.xlsx")
train, test = train_test_split(data, test_size = 0.15)
print("training size: {}, test size: {}". format(len(train), len(test)))

# custom colors
red_blue = ["#19B5FE", "#EF4836"]
palette = sns.color_palette(red_blue)
sns.set_palette(palette)
sns.set_style("white")

# create variable for variables of songs liked and not liked
pos_year = data[data['Y'] == 1]['year']
neg_year = data[data['Y'] == 0]['year']

pos_season = data[data['Y'] == 1]['season']
neg_season = data[data['Y'] == 0]['season']

pos_nation = data[data['Y'] == 1]['nation']
neg_nation = data[data['Y'] == 0]['nation']

pos_totdays = data[data['Y'] == 1]['totdays']
neg_totdays = data[data['Y'] == 0]['totdays']

pos_traverse = data[data['Y'] == 1]['traverse']
neg_traverse = data[data['Y'] == 0]['traverse']

pos_ski = data[data['Y'] == 1]['ski']
neg_ski = data[data['Y'] == 0]['ski']

pos_camps = data[data['Y'] == 1]['camps']
neg_camps = data[data['Y'] == 0]['camps']

pos_rope = data[data['Y'] == 1]['rope']
neg_rope = data[data['Y'] == 0]['rope']

pos_totmembers = data[data['Y'] == 1]['totmembers']
neg_totmembers = data[data['Y'] == 0]['totmembers']

pos_instrumentalness = data[data['Y'] == 1]['instrumentalness']
neg_instrumentalness = data[data['Y'] == 0]['instrumentalness']

fig = plt.figure(figsize = (12, 8))
plt.title("Song Tempo Like/Dislike Distribution")
pos_tempo.hist(alpha = 0.7, bins = 30, label = 'positive')
neg_tempo.hist(alpha = 0.7, bins = 30, label = 'negative')
plt.legend(loc = 'upper right')
