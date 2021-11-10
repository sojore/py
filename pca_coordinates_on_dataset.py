# -*- coding: utf-8 -*-
"""PCA_coordinates_on_dataset.ipynb

#in this project  we will be proccessing a given dataset by computing its leading principle components
#and the creating a scatter plot with 2 of the PCA coordinates
#so we will be using the digit_recognizer dataset from kaggle to do the analysis
#link to the dataset  link...https://www.kaggle.com/c/digit-recognizer/data?select=train.csv

# Commented out IPython magic to ensure Python compatibility.
#importing important libraries that we gonna be using 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

#importing the sklean PCA library
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

#loading the dataset
df=pd.read_csv('/content/drive/MyDrive/digit-recognizer/train.csv')  
df.shape

df.sample(5)

#next we gonna be separating all the other data features from the target columns
cols=df.columns   
df_features=cols.tolist()  
df_feature=df_features[1:]  
df_target=df_features[0]  
 
df_bkp=df.copy()  #taking a backup of the dataframe 
df_label=df[df_target]  
df_data=df[df_feature]

#spliting the dataset for training and testing
X=df_data.loc[:,df_feature].values  
y=df_label.loc[:].values 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=45,random_state=10)

X_train_original=X_train.copy()

#standadizing the data using the sklearn standadizer function
standr=StandardScaler()

standr.fit(X_train)   #fitting only on Training data  
 
X_train=standr.fit_transform(X_train)   #we will also apply the transformation to training and test data as well  
X_test=standr.fit_transform(X_test)

#next we gonna plot componets number and variance
pca_mod=PCA()  
pca_mod.n_components=784  
pca_mod_data=pca_mod.fit_transform(X_train)
percentage_var_explained = pca_mod.explained_variance_ratio_;  
cum_var_explained=np.cumsum(percentage_var_explained)
#plot PCA spectrum   
plt.figure(1,figsize=(15,10))  #plotting the  PCA spectrum  
plt.clf()  
plt.plot(cum_var_explained,linewidth=2)  
plt.axis('tight')  
plt.grid() 
plt.xlabel('no_of_components') 
plt.ylabel('Cumulative_Variance')  
plt.show()

#we now going to get the Explained Variance Table
pca_mod=PCA()  
pca_mod.n_components=784  
pca_mod_data=pca_mod.fit_transform(X_train)  
exp_var_cumsum=pd.Series(np.round(pca_mod.explained_variance_ratio_.cumsum(),4)*100)

#next we will be passing the variance percentage as Input parameter to PCA() this is to alter the dataset dimension
pca_mod=PCA(.80)  
pca_mod.fit(X_train)

#lets plot the above in 2-d space
pca_mod=PCA(n_components=2) 
pca_mod_data_visualization=pca_mod.fit_transform(X_train)

#plotting a scatter plot using 2 of the PCA coordinates
#plotting a scatter plot for visualizing the data with 2 PCA coordinates
pca_mod_data_visualization = np.vstack((pca_mod_data_visualization.T,y_train)).T
pca_visuaization_df = pd.DataFrame(data=pca_mod_data_visualization,columns=("1st_principal","2nd_principal","label"))  
sns.FacetGrid(pca_visuaization_df,hue="label",size=10).map(plt.scatter,'1st_principal','2nd_principal').add_legend()
plt.show()

#improving the variance to test the perfomance and training the model
pca_mod=PCA(.80) 
pca_mod.fit(X_train) 
X_train=pca_mod.transform(X_train) 
X_test=pca_mod.transform(X_test)
from sklearn.linear_model  import LogisticRegression  
model=LogisticRegression(class_weight='balanced')  
model.fit(X_train,y_train)
y_predicted=model.predict(X_test)

#lets see the perfomace of the model on the test data
from sklearn.metrics import precision_score,recall_score,confusion_matrix,classification_report,accuracy_score,f1_score
print(f'The accuracy of the model on test dataset is: {round(accuracy_score(y_test,y_predicted),2)*100}%')     
print(f"The precision of the model on test dataset is: {round(precision_score(y_test,y_predicted,average='weighted'),2)*100}%")  
print(f'The classification report: {classification_report(y_test,y_predicted)}')  
cm=confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(20,10))
sns.heatmap(cm,annot=True)
plt.xlabel('Truth')
plt.ylabel('Predicted')

"""END OF PROJECT IMPLEMENTATION WITH 2-PCA COORDINATES.  THANK YOU!!!"""