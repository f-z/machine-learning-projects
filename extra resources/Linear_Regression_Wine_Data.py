""" Predict the quality of wine based on 
its physiochemical features
Dataset downloaded from: 
http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
"""

#Load the dataset using pandas
import pandas as pd
df = pd.read_csv('winequality-red.csv', sep = ';')
# Present the dataset
print df.describe()
# Calculate pairwise correlation matrix to see how 
#different variables are related to quality
print df.corr()

#Split the data into training and testing sets
from sklearn.cross_validation import train_test_split
Features = df[list(df.columns)[:-1]]
Quality = df['quality']
Features_train, Features_test, Quality_train, Quality_test = train_test_split(Features, Quality)

#Create and fit the model on the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Features_train, Quality_train)

#Evaluate the predictions of the model
Quality_predictions = regressor.predict(Features_test)
# print 'R squared:', regressor.score(Features_test, Quality_test)
# for i, prediction in enumerate(Quality_predictions):
# 	print 'Predicted: %s, True: %s' %(prediction, Quality_test[i])

# Create scatterplot of Predicted Quality against True Quality 
import matplotlib.pylab as plt
plt.scatter(Quality_test, Quality_predictions)
plt.xlabel('True Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted Quality Against True Quality ')
plt.show()

#Use cross-validation to produce a better estimate
# of the regressor's performance. Returns the value of the 
# regressor's score(r-squared) for each round
# from sklearn.cross_validation import cross_val_score
# cv=5 means that each instance will be randomly assigned
# to one of the five partitions. Each partition will be used to 
# train and test the model 
# scores = cross_val_score(regressor, Features, Quality, cv=5) 
# print scores.mean(), scores
