"""
Implement the linear regression model using python and numpy in the following class.
The method fit() should take inputs like,
x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
"""

class LinearRegression(object):
  """
  An implementation of linear regression model
  """
  
  def fit(self,X_train,y):
    y_train=np.array(y)
    y_train=y_train[np.newaxis,:]
    X_train=np.array(X_train).T
    num_features,num_examples=np.shape(X_train)
    bias_terms=np.ones((1,num_examples))
    X_train=np.vstack((bias_terms,X_train))
    weights=np.random.randn(1,num_features+1)
    num_iterations=1000
    alpha=0.01
    lambd=0.001

    for i in range(num_iterations):
      predictions=np.dot(weights,X_train)
      error=predictions-y_train
      weights=weights - (alpha/num_examples) * np.dot(error,X_train.T) - (lambd/num_examples) * weights

    pass
    
  def predict(self,X_test):
    X_test=np.array(X_test).T
    num_features,num_examples=np.shape(X_test)
    bias_terms=np.ones((1,num_examples))
    X_test=np.vstack((bias_terms,X_test))
    return np.dot(weights,X_test)

