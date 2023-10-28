import numpy as np
import pandas as pd
random_seed=np.random.RandomState(12)

#generating train dataset(normal observations)
X_train= 0.5*random_seed.randn(500,2)
X_train= np.r_[X_train+3,X_train]
X_train= pd.DataFrame(X_train,columns=["x","y"])

#generating testing set(normal observations)
X_test= 0.5*random_seed.randn(500,2)
X_test= np.r_[X_test+3,X_test]
X_test= pd.DataFrame(X_test,columns=["x","y"])

#generating set of outlier observations(different from normal observations)
X_outliers = random_seed.uniform(low=-5,high=5,size=(50,2))
X_outliers = pd.DataFrame(X_outliers,columns=["x","y"])

#look the data generated
import matplotlib.pyplot as plt
p1=plt.scatter(X_train.x,X_train.y,c="white",s=50,edgecolor="black")
p2=plt.scatter(X_test.x,X_test.y,c="green",s=50,edgecolor="black")
p3=plt.scatter(X_outliers.x,X_outliers.y,c="blue",s=50,edgecolor="black")

plt.xlim((-6,6))
plt.ylim((-6,6))
plt.legend([p1,p2,p3],["training set","normal testing set","anomalous testing set"],loc="lower right",)
plt.show()

#train isolation forest model on training data
from sklearn.ensemble import IsolationForest
clf=IsolationForest()
clf.fit(X_train)
y_pred_train=clf.predict(X_train)
y_pred_test=clf.predict(X_test)
y_pred_outliers=clf.predict(X_outliers)

#append the labels to x_outliers
X_outliers=X_outliers.assign(pred=y_pred_outliers)
X_outliers.head()
p1 = plt.scatter(X_train.x,X_train.y,c="white",s=50,edgecolor="black")
p2 = plt.scatter(X_outliers.loc[X_outliers.pred==-1,["x"]],X_outliers.loc[X_outliers.pred==-1,["y"]],c="blue",s=50,edgecolor="black", )
p3 = plt.scatter(X_outliers.loc[X_outliers.pred==1,["x"]],X_outliers.loc[X_outliers.pred==1,["y"]],c="red",s=50,edgecolor="black" ,)
plt.xlim((-6,6))
plt.ylim((-6,6))
plt.legend([p1,p2,p3],["training observations","detected outliers","incorrectly labeled outliers"],loc="lower right",)
plt.show()

#now to see its performance on normal testing data, append the predicted label to x_test
X_test=X_test.assign(pred=y_pred_test)
X_test.head()

#plot the results to see whether our classifier labelled the normal testing data correctly
p1 = plt.scatter(X_train.x,X_train.y,c="white",s=50,edgecolor="black")
p2 = plt.scatter(X_test.loc[X_test.pred==1,["x"]],X_test.loc[X_test.pred==1,["y"]],c="blue",s=50,edgecolor="black",)
p3 = plt.scatter(X_test.loc[X_test.pred==-1,["x"]],X_test.loc[X_test.pred==-1,["y"]],c="red",s=50,edgecolor="black", )
plt.xlim((-6,6))
plt.ylim((-6,6))
plt.legend([p1,p2,p3],["training observations","correctly labelled test observations","incorrectly labelled test observations"],loc="lower right",)
plt.show()
