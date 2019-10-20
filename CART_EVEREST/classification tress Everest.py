# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:36:29 2019

@author: Maciek
"""

# Importig libraries
import pandas as pd
import numpy as np
import seaborn as sns
import graphviz
import pydotplus
import io

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from IPython.display import Image
from scipy import misc
from matplotlib import pyplot as plt


#Reading datastet, splitting it  to train set and test set
data = pd.read_excel("Everest_data2.xlsx")
train, test = train_test_split(data, test_size = 0.15)
print("training size: {}, test size: {}". format(len(train), len(test)))

# custom colors
red_blue = ["#19B5FE", "#EF4836"]
palette = sns.color_palette(red_blue)
sns.set_palette(palette)
sns.set_style("white")

# create variable for variables of success and failure
pos_year = data[data['Y'] == 1]['year']
neg_year = data[data['Y'] == 0]['year']

pos_autoumn = data[data['Y'] == 1]['autoumn']
neg_autoumn = data[data['Y'] == 0]['autoumn']

pos_spring = data[data['Y'] == 1]['spring']
neg_spring = data[data['Y'] == 0]['spring']

pos_winter = data[data['Y'] == 1]['winter']
neg_winter = data[data['Y'] == 0]['winter']

pos_summer = data[data['Y'] == 1]['summer']
neg_summer = data[data['Y'] == 0]['summer']

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

pos_tothired = data[data['Y'] == 1]['tothired']
neg_tothired = data[data['Y'] == 0]['tothired']

pos_hratio = data[data['Y'] == 1]['hratio']
neg_hratio = data[data['Y'] == 0]['hratio']

pos_smthired = data[data['Y'] == 1]['smthired']
neg_smthired = data[data['Y'] == 0]['smthired']

pos_Nepal = data[data['Y'] == 1]['Nepal']
neg_Nepal = data[data['Y'] == 0]['Nepal']

pos_o2used = data[data['Y'] == 1]['o2used']
neg_o2used = data[data['Y'] == 0]['o2used']

pos_comrte = data[data['Y'] == 1]['comrte']
neg_comrte = data[data['Y'] == 0]['comrte']

pos_stdrte = data[data['Y'] == 1]['stdrte']
neg_stdrte = data[data['Y'] == 0]['stdrte']


#Plots 
fig = plt.figure(figsize = (8, 6))
plt.title("Distribution of succes or failure among the years")
pos_year.hist(alpha = 0.7, bins = 30, label = 'succes')
neg_year.hist(alpha = 0.7, bins = 30, label = 'failure')
plt.legend(loc = 'upper right')


# create plot to view distribution of success/failure for other variables
fig2 = plt.figure(figsize = (15, 15))


# Totdays
ax7 = fig2.add_subplot(333) # grid location
ax7.set_xlabel('totdays')
ax7.set_ylabel('count')
ax7.set_title('Distribution of succes or failure on different lenght of trip')
pos_totdays.hist(alpha = 0.5, bins = 30)
ax7 = fig2.add_subplot(333)
neg_totdays.hist(alpha = 0.5, bins = 30)


# Traverse
ax9 = fig2.add_subplot(334) # grid location
ax9.set_xlabel('traverse')
ax9.set_ylabel('count')
ax9.set_title('Traverse succes or failure distribution')
pos_traverse.hist(alpha = 0.5, bins = 30)
ax9 = fig2.add_subplot(334)
neg_traverse.hist(alpha = 0.5, bins = 30)

# Ski
ax11 = fig2.add_subplot(335) # grid location
ax11.set_xlabel('ski')
ax11.set_ylabel('count')
ax11.set_title('Ski succes or failure distribution')
pos_ski.hist(alpha = 0.5, bins = 30)
ax11 = fig2.add_subplot(335)
neg_ski.hist(alpha = 0.5, bins = 30)

# Camps 
ax13 = fig2.add_subplot(336) # grid location
ax13.set_xlabel('camps')
ax13.set_ylabel('count')
ax13.set_title('Camps succes or failure distribution')
pos_camps.hist(alpha = 0.5, bins = 30)
ax13 = fig2.add_subplot(336)
neg_camps.hist(alpha = 0.5, bins = 30)


# Rope
ax15 = fig2.add_subplot(337) # grid location
ax15.set_xlabel('rope')
ax15.set_ylabel('count')
ax15.set_title('Rope succes or failure distribution')
pos_rope.hist(alpha = 0.5, bins = 30)
ax16 = fig2.add_subplot(337)
neg_rope.hist(alpha = 0.5, bins = 30)

# Totmembers
ax17 = fig2.add_subplot(338) # grid location
ax17.set_xlabel('totmembers')
ax17.set_ylabel('count')
ax17.set_title('Totmembers succes or failure distribution')
pos_totmembers.hist(alpha = 0.5, bins = 30)
ax18 = fig2.add_subplot(338)
neg_totmembers.hist(alpha = 0.5, bins = 30)

# Hratio
ax19 = fig2.add_subplot(339) # grid location
ax19.set_xlabel('hratio')
ax19.set_ylabel('count')
ax19.set_title('Hratio succes or failure distribution')
pos_hratio.hist(alpha = 0.5, bins = 30)
ax20 = fig2.add_subplot(339)
neg_hratio.hist(alpha = 0.5, bins = 30)

# Smthired
ax19 = fig2.add_subplot(341) # grid location
ax19.set_xlabel('smthired')
ax19.set_ylabel('count')
ax19.set_title('Smthired succes or failure distribution')
pos_smthired.hist(alpha = 0.5, bins = 30)
ax20 = fig2.add_subplot(341)
neg_smthired.hist(alpha = 0.5, bins = 30)

# Nepal
ax19 = fig2.add_subplot(342) # grid location
ax19.set_xlabel('Nepal')
ax19.set_ylabel('count')
ax19.set_title('Nepal succes or failure distribution')
pos_Nepal.hist(alpha = 0.5, bins = 30)
ax20 = fig2.add_subplot(342)
neg_Nepal.hist(alpha = 0.5, bins = 30)

# o2used
ax19 = fig2.add_subplot(343) # grid location
ax19.set_xlabel('o2used')
ax19.set_ylabel('count')
ax19.set_title('O2used succes or failure distribution')
pos_o2used.hist(alpha = 0.5, bins = 30)
ax20 = fig2.add_subplot(343)
neg_o2used.hist(alpha = 0.5, bins = 30)

# Comrte
ax19 = fig2.add_subplot(344) # grid location
ax19.set_xlabel('Comrte')
ax19.set_ylabel('count')
ax19.set_title('Comrte succes or failure distribution')
pos_comrte.hist(alpha = 0.5, bins = 30)
ax20 = fig2.add_subplot(344)
neg_comrte.hist(alpha = 0.5, bins = 30)

# Stdrte
ax19 = fig2.add_subplot(345) # grid location
ax19.set_xlabel('Stdrte')
ax19.set_ylabel('count')
ax19.set_title('Stdrte or failure distribution')
pos_stdrte.hist(alpha = 0.5, bins = 30)
ax20 = fig2.add_subplot(345)
neg_stdrte.hist(alpha = 0.5, bins = 30)




#Create decission tree
c = DecisionTreeClassifier(min_samples_split = 100)
features = ["year", "autoumn", "spring", "winter", "summer", "totdays",	"traverse",	"ski",	"camps",	"rope",	"totmembers",	"tothired",	"hratio",	"smthired",	"Nepal",	"o2used",	"comrte",	"stdrte"]

x_train = train[features]
y_train = train["Y"]

x_test = test[features]
y_test = test["Y"]

dt = c.fit(x_train, y_train)

def show_tree(tree, features, path):
     f = io.StringIO()
     export_graphviz(tree, out_file = f, feature_names = features)
     pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
     img = misc.imread(path)
     plt.rcParams["figure.figsize"] = (20, 20)
     fig2 = plt.figure(figsize = (10, 8))
     plt.imshow(img)
    
     
show_tree(dt, features, 'dec_tree_1.png')
     
# Create DOT data
dot_data = export_graphviz(dt, out_file=None, feature_names=features)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())
