import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)
req_column = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
my_columns = data[req_column]

#Model 
y = data['SalePrice']
model_predictors = req_column
X = data[model_predictors]
iowa_model = DecisionTreeRegressor()
iowa_model.fit(X,y)

predicted_prices = iowa_model.predict(X)
mean_absolute_error(y,predicted_prices) # This error is an example of in-sample score. We are calculating MAE with the training data predictions. 
#Therefore we need to find MAE on data that hasn't been used for training. This is known as validation data.
# This point has given rise to the train and test split of data.
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 0)

# The following is to select max leaf nodes in order to avoid underfitting and overfitting
def get_mae(max_leaf_nodes):
    new_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)
    new_model.fit(train_X,train_y)
    act_prediction = new_model.predict(val_X)
    mae = mean_absolute_error(val_y, act_prediction)
    return mae

# iterate over a few values of max_leaf_nodes to find out corresponding mae values. The least mae corresponds to the optimal number of max leaf nodes.
for max_leaf_nodes in [50, 100, 200, 400, 500, 1000]:
    our_mae = get_mae(max_leaf_nodes)
    print("The mae is %d for the leaf nodes%d"%(our_mae,max_leaf_nodes) )
