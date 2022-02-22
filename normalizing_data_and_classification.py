# -*- coding: utf-8 -*-
"""normalizing data and classification.ipynb


#importing important libraries
import pandas as pd
import numpy as np

#loading the datasets
df_red=pd.read_excel('/content/drive/MyDrive/red_data.xlsx')
df_green=pd.read_excel('/content/drive/MyDrive/green_file.xlsx')
df_white=pd.read_excel('/content/drive/MyDrive/white_file.xlsx')

#lets visualize some data
df_green.sample(8)

#we will be using the z score to do the data normalization 
df_green['x_zs_green']=(df_green.x-df_green.x.mean())/df_green.x.std()
df_green['y_zs_green']=(df_green.y-df_green.y.mean())/df_green.y.std()
df_green.head()

df_red['x_zs_red']=(df_red.x-df_red.x.mean())/df_red.x.std()
df_red['y_zs_red']=(df_red.y-df_red.y.mean())/df_red.y.std()
df_red.head()

df_white['x_zs_white']=(df_white.x-df_white.x.mean())/df_white.x.std()
df_white['y_zs_white']=(df_white.y-df_white.y.mean())/df_white.y.std()
df_white.head()

#transforming the data into list values
x_green=[i for i in df_green['x_zs_green']]
y_green=[i for i in df_green['y_zs_green']]
x_green=[i for i in df_green['x']]
y_green=[i for i in df_green['y']]

x_red=[i for i in df_red['x_zs_red']]
y_red=[i for i in df_red['y_zs_red']]
x_red=[i for i in df_red['x']]
y_red=[i for i in df_red['y']]

x_white=[i for i in df_white['x_zs_white']]
y_white=[i for i in df_white['y_zs_white']]
x_white=[i for i in df_white['x']]
y_white=[i for i in df_white['y']]

#iterating through each list
for i in x_white:
  for j in  x_green:
    if i == j:
      print (f'This white x value {i} belongs to green area ')

for i in y_white:
  for j in  y_green:
    if i == j:
      print (f'This white y value {i} belongs to green area ')

for i in x_white:
  for j in  x_red:
    if i == j:
      print (f'This white x value {i} belongs to red area ')

for i in y_white:
  for j in  y_red:
    if i == j:
      print (f'This white y value {i} belongs to red area ')

"""END OF IMPLEMENTATION AND TESTING.THANK YOU!!!"""

