import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# same data set as decision tree
main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)
req_column = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
my_columns = data[req_column]

#Model 
y = data['SalePrice']
model_predictors = req_column
X = data[model_predictors]
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 0)
forest_model = RandomForestRegressor()
forest_model.fit(train_X,train_y)
forest_pred = forest_model.predict(val_X)
mae_forest = mean_absolute_error(val_y, forest_pred)
print ("mae for random forest is:",mae_forest) 
