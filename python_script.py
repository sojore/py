# -*- coding: utf-8 -*-
"""python_script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fEQKww2Iiry3-mFwH6FLVbbMr5HwptK1
"""

#importing neccesary library
import pandas as pd

#reading the dataset
data = pd.read_csv("/content/drive/MyDrive/extreme_changes_post.csv")
data.head(10)

"""Title column -most common words """

#writing a code to run through the dataset TITLE column to rank most common words

#initializing empty list to store the split words from the dataframe
words_list = []
new_list=[]
final_list=[]
most_common_words1 = []
dictionary_value_count = {}

#iterating through the Title column and replacing all any words split by . and  / with '', for easy splitting
for word in data['title']:
    words_list.append(word.replace('.', ''))
    words_list.append(word.replace('/', ''))
    words_list.append(word.replace('-', ''))

#iterating through the word_list and converting all words to a string
for val in words_list:
  output1=str(val)
  new_list.append(output1)

#iterating through the new_list and splitting all the words and storing them in a new list called final_list
for n_val in new_list:
  output2=n_val.split()
  for res in output2:
    final_list.append(res)

#iterating through final_list and storing all split words into a dictionary
for word in final_list:
    if word not in dictionary_value_count:
        dictionary_value_count[word] = 1
    else:
        num = dictionary_value_count[word]
        dictionary_value_count[word] = (num + 1)

#function for obtaining the largest word count and aggreating them all
def largest_value(value):
    res = 0
    new_str = ''
    for key, val in value.items():
        if val > res:
            res = val
            new_str = key
    return [res, new_str]

#function for grouping all common words and ranking them together
def common_words( value,target):
    while len(most_common_words1) < target:
        most_common_words1.append(largest_value(value))
        del value[(largest_value(value)[1])]

common_words( dictionary_value_count,935)

print(most_common_words1)

"""PRO column -most common words """

#writing a code to run through the dataset pro column to rank most common words

#initializing empty list to store the split words from the dataframe
words_list = []
new_list=[]
final_list=[]
most_common_words2 = []
dictionary_value_count = {}

#iterating through the Title column and replacing all any words split by . and  / with '', for easy splitting
for word in data['pro']:
    words_list.append(word.replace('.', ''))
    words_list.append(word.replace('/', ''))
    words_list.append(word.replace('-', ''))

#iterating through the word_list and converting all words to a string
for val in words_list:
  output1=str(val)
  new_list.append(output1)

#iterating through the new_list and splitting all the words and storing them in a new list called final_list
for n_val in new_list:
  output2=n_val.split()
  for res in output2:
    final_list.append(res)

#iterating through final_list and storing all split words into a dictionary
for word in final_list:
    if word not in dictionary_value_count:
        dictionary_value_count[word] = 1
    else:
        num = dictionary_value_count[word]
        dictionary_value_count[word] = (num + 1)

#function for obtaining the largest word count and aggreating them all
def largest_value(value):
    res = 0
    new_str = ''
    for key, val in value.items():
        if val > res:
            res = val
            new_str = key
    return [res, new_str]

#function for grouping all common words and ranking them together
def common_words( value,target):
    while len(most_common_words2) < target:
        most_common_words2.append(largest_value(value))
        del value[(largest_value(value)[1])]

common_words( dictionary_value_count,935)

print(most_common_words2)

"""CONS column -most common words """

#writing a code to run through the dataset cons column to rank most common words

#initializing empty list to store the split words from the dataframe
words_list = []
new_list=[]
final_list=[]
most_common_words3 = []
dictionary_value_count = {}

#iterating through the Title column and replacing all any words split by . and  / with '', for easy splitting
for word in data['cons']:
    words_list.append(word.replace('.', ''))
    words_list.append(word.replace('/', ''))
    words_list.append(word.replace('-', ''))


#iterating through the word_list and converting all words to a string
for val in words_list:
  output1=str(val)
  new_list.append(output1)

#iterating through the new_list and splitting all the words and storing them in a new list called final_list
for n_val in new_list:
  output2=n_val.split()
  for res in output2:
    final_list.append(res)

#iterating through final_list and storing all split words into a dictionary
for word in final_list:
    if word not in dictionary_value_count:
        dictionary_value_count[word] = 1
    else:
        num = dictionary_value_count[word]
        dictionary_value_count[word] = (num + 1)

#function for obtaining the largest word count and aggreating them all
def largest_value(value):
    res = 0
    new_str = ''
    for key, val in value.items():
        if val > res:
            res = val
            new_str = key
    return [res, new_str]

#function for grouping all common words and ranking them together
def common_words( value,target):
    while len(most_common_words3) < target:
        most_common_words3.append(largest_value(value))
        del value[(largest_value(value)[1])]

common_words( dictionary_value_count,935)

print(most_common_words3)

import pandas
df = pandas.DataFrame(data={"Title_col_most_common_words": most_common_words1, "PRO_col_most_common_words": most_common_words2,
                             "CONS_col_most_common_words": most_common_words3})
df.to_csv("./most_common_words.csv", sep=',',index=False)

most_common_words_file=pd.read_csv('/content/most_common_words.csv')
most_common_words_file

"""THE END OF IMPLEMENTATION.THANK YOU!!!"""

