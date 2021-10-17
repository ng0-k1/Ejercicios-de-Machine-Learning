import numpy as np
import pandas as pd

dataset = pd.read_csv('Fish.csv')

x = dataset.iloc[:, [0,2,3,4,5]].values         
y = dataset.iloc[:, 1].values

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = preprocessing.LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])


ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories = "auto"),[0])],
    remainder = 'passthrough')

x = np.array(ct.fit_transform(x), dtype = np.float)
x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

#Creando clase de eliminación hacia atras
import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 

SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5,6,7,8,9]]
X_Modeled = backwardElimination(X_opt, SL)









import numpy as np
import pandas as pd

dataset = pd.read_csv('Fish.csv')
x = dataset.iloc[:, [3,4]].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)

from sklearn.linear_model import 

regression = LinearRegression()
regression.fit(x_train, y_train)
y_pred = regression.predict(x_test)




























