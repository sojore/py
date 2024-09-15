# -*- coding: utf-8 -*-
"""decision tree classifier implementation.ipynb

#in this project we are going to implement the DT classifier from scratch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math 
import operator

#defining the DTC class 
class DecisionTreeClassifier1:

	def __init__(self, max_depth = None, min_sample_leaf = None):

		self.depth = 0  ##the actuals trees depth
		self.max_depth = max_depth	 ##the maximum depth of the tree
		self.min_sample_leaf = min_sample_leaf	 ##each nodes minimum samples number

		self.features = list
		self.X_train = np.array
		self.y_train = np.array
		self.num_feats = int 
		self.train_size = int 

  ## defining function for buildig the tree
	def _build_tree(self, df, tree = None):
		feature, cutoff = self._find_best_split(df) #this will get the feature that has the maximum information gain so to build tree based on it

		##lets initialize the tree
		if tree is None:
			tree = {}
			tree[feature] = {}

		if df[feature].dtypes == object:
			for feature_value in np.unique(df[feature]):

				new_df = self._split_rows(df, feature, feature_value, operator.eq)
				targets, count = np.unique(new_df['target'], return_counts = True)

				if(len(count) == 1):
					tree[feature][feature_value] = targets[0]
				else:
					self.depth += 1
					if self.max_depth is not None and self.depth >= self.max_depth:
						tree[feature][feature_value] = targets[np.argmax(count)]
					else:
						tree[feature][feature_value] = self._build_tree(new_df)
					
		else:
			##defining the left child node
			new_df = self._split_rows(df, feature, cutoff, operator.le)
			targets, count = np.unique(new_df['target'], return_counts = True)

			self.depth += 1
			if(len(count) == 1):
				tree[feature]['<=' + str(cutoff)] = targets[0]
			else:
				if self.max_depth is not None and self.depth >= self.max_depth:
					tree[feature]['<=' + str(cutoff)] = targets[np.argmax(count)]
				else:
					tree[feature]['<=' + str(cutoff)] = self._build_tree(new_df)

			##defining the right child node
			new_df = self._split_rows(df, feature, cutoff, operator.gt)
			targets, count = np.unique(new_df['target'], return_counts = True)

			if(len(count) == 1): 
				tree[feature]['>' + str(cutoff)] = targets[0]
			else:
				if self.max_depth is not None and self.depth >= self.max_depth:
					tree[feature]['>' + str(cutoff)] = targets[np.argmax(count)]
				else:
					tree[feature]['>' + str(cutoff)] = self._build_tree(new_df)

		return tree

 #function for splitting the tree
	def _split_rows(self, df, feature, feat_val, operation ):

		return df[operation(df[feature], feat_val)].reset_index(drop = True)

    ##defining the training function
	def fit(self, X, y):
		self.X_train = X 
		self.y_train = y
		self.features = list(X.columns)
		self.train_size = X.shape[0]
		self.num_feats = X.shape[1]

		df = X.copy()
		df['target'] = y.copy()

		#Builds Decision Tree
		self.tree = self._build_tree(df)

    #function for predicting the test
	def predict(self, X):
		results = []
		feature_lookup = {key: i for i, key in enumerate(list(X.columns))}
		
		for index in range(len(X)):

			results.append(self._predict_target(feature_lookup, X.iloc[index], self.tree))

		return np.array(results)

  #function for defining parents entropy
	def _get_entropy(self, df):
		entropy = 0
		for target in np.unique(df['target']):
			fraction = df['target'].value_counts()[target] / len(df['target'])
			entropy += -fraction * np.log2(fraction)

		return entropy


  ##function for getting the total sum of the children entropy
	def _get_entropy_feature(self, df, feature):
		entropy = 0
		threshold = None

		if(df[feature].dtypes == object):

			#sum of entropies of children
			for feat_val in np.unique(df[feature]):
				entropy_feature = 0

				# the entropy for every distinct feature value
				for target in np.unique(df['target']):
					num = len(df[feature][df[feature] == feat_val][df['target'] == target])
					den = len(df[feature][df[feature] == feat_val])

					fraction = num / (den+eps)

					if(fraction > 0):
						entropy_feature += -fraction * np.log2(fraction)

				weightage = len(df[feature][df[feature] == feat_val])/len(df)
				entropy += weightage * entropy_feature
		else:
			entropy = 1 #this defines the maximum value

			prev = 0
			for feat_val in np.unique(df[feature]):
				cur_entropy = 0
				cutoff = (feat_val + prev)/2
        
				#sum of entropies of both left child and right child
				for operation in [operator.le, operator.gt]:
					entropy_feature = 0

					for target in np.unique(df['target']):
						num = len(df[feature][operation(df[feature], cutoff)][df['target'] == target])
						den = len(df[feature][operation(df[feature], cutoff)])

						fraction = num / (den+eps)
						if(fraction > 0):
							entropy_feature += -fraction * np.log2(fraction)

					weightage = den/len(df)
					cur_entropy += weightage * entropy_feature

				if cur_entropy < entropy:
					entropy = cur_entropy
					threshold = cutoff
				prev = feat_val

		return entropy, threshold


    #function to find the best slit with maximum information gain
	def _find_best_split(self, df):
		listing = []
		thresholds = []

		for ftr in list(df.columns[:-1]):
      
			parent_entropy = self._get_entropy(df) 
			feature_entropy_split, threshold = self._get_entropy_feature(df, ftr)

			info_gain = parent_entropy - feature_entropy_split
			listing.append(info_gain)
			thresholds.append(threshold)

		return df.columns[:-1][np.argmax(listing)], thresholds[np.argmax(listing)] #Returns feature with max information gain 


  #function for predicting the target node value
	def _predict_target(self, feature_lookup, x, tree):
		for node in tree.keys():
			val = x[node]
			if type(val) == str:
				tree = tree[node][val]
			else:
				cutoff = str(list(tree[node].keys())[0]).split('<=')[1]

				if(val <= float(cutoff)):	
					tree = tree[node]['<='+cutoff]
				else:						
					tree = tree[node]['>'+cutoff]
			prediction = str

			if type(tree) is dict:
				prediction = self._predict_target(feature_lookup, x, tree)
			else:
				predicton = tree 
				return predicton

		return prediction

from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/drive/MyDrive/Housing-data.csv')
df.head()

#aim is to predict if the votes is gonna be for republican or democrat based on given data
#exploring the dataset
df.shape

df[' republican ']

#creating replica dataframe to change columns heading
df['target']=df[' republican ']
df['first']=df['n']
df['second']=df['y']
df['third']=df['n.1']
df['fourth']=df['y.1']
df['fifth']=df['y.2']
df['sixth']=df['y.3']
df['seventh']=df['n.2']
df['eighth']=df['n.3']
df['ninght']=df['n.4']
df['tenth']=df['y.4']
df['eleventh']=df['?']
df['twelvth']=df['y.5']
df['thirteeth']=df['y.6']
df['fouteeth']=df['y.7']
df['fifteeth']=df['n.5']
df['sixteeth']=df['y.8']

df1=df.drop(['n','y','n.1','y.1','y.2','y.3','y.4','n.2','n.3','n.4','?','y.5','y.6','y.7','n.5','y.8',' republican '],axis='columns')
df1.head(5)

#this kinda looks good
df1['first'].mode()

df2=df1.replace('?','y')
df2.sample(5)

df3=df2.replace([' republican ',' democrat '],[1,0])
df3.sample(3)

df4=df3.replace(['y','n'],[1,0])
df4.head()

df4.dtypes.unique()

df4.columns[0]

#Split Features and target
X, y =  df4.drop([df4.columns[0]],axis=1),df4[df4.columns[0]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

custom_model = DecisionTreeClassifier1(max_depth = 400, min_sample_leaf=3)
custom_model.fit(X_train,y_train)

custom_model.predict(X_test)

y_predicted=custom_model.predict(X_test)
y_predicted[:20]

#lets print some metrics for better visualization so we can compare it the DecisionTreeClassifier from sklearn library
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_predicted)
cm

##using seaborn library 
import seaborn as sn
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sn.heatmap(cm,annot=True)
plt.ylabel('Truth')
plt.xlabel('Predicted')

#interpretation is that with the custom_model ,56 times the true value was 0 and the model actually predicted is correct,25 times the true
#value was 1 and the model predicted it to be 1,only 6 times the true value was 1 and the model predicted it ot be 0,
#so overall the custom_model is doing pretty good

classification_report(y_test,y_predicted) #so the custom_model has an accuracy of 93% which is pretty good

##next we gonna compare our custom_model perfomance  based on accuracy with the DecisionTreeClasssifier model form sklearn library
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(X_train,y_train)

model.predict(X_test)

y_pred=model.predict(X_test)
y_pred[:20]

#plotting both confussion matrix and classification report and checking the accuracy
cm=confusion_matrix(y_test,y_pred)
cm

##using seaborn library 
import seaborn as sn
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sn.heatmap(cm,annot=True)
plt.ylabel('Truth')
plt.xlabel('Predicted')

classification_report(y_test,y_predicted) #so the model also has the same  accuracy of 93% which is pretty good
#implying that our custom DecisionTreeClassifier1 algorithm is working exceendingly great



"""END OF DecisionTreeClassifier1 ALGORITHM IMPLEMENTATION AND TESTING . THANK YOU!!"""
