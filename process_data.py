import pandas as pd
import numpy as np


class DataProcess:

    def __init__(self):
        self.to_drop = []

    def processTrain():
        pass
        
    def processTest(self,test):
        pass


def drop_columns(data, names):
    for name in names:
        try:
            data.drop(name, axis = 1, inplace = True)
        except:
            continue
    
def process_country(data):
    for i in data.index:
        if data.ix[i,'var3'] == 2:
            data.set_value(i,'var3', 0)
        else:
            data.set_value(i,'var3', 1)
    # Make it into a categorical variable.
    data['var3'] = data['var3'].astype('category')
    # Transform it into indicator variable.
            
def process(data):
    # ID column is just an identifier.
    columns_to_drop = ['ID']
    # In the EAD, these columns were found to be duplicated.
    duplicates = ['ind_var18_0', 'ind_var25_0', 'ind_var26_0', 'ind_var32_0', 'ind_var34_0', 'ind_var37_0', 'ind_var13_medio_0']
    columns_to_drop+= duplicates
    # In the EAD, these columns are trivial in the training data.
    zero_columns = ['ind_var2_0', 'ind_var2',  'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0',  'ind_var46',  'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46', 'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3', 'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3']
    columns_to_drop+= zero_columns
    # We drop all the features above.
    drop_columns(data, columns_to_drop)
    data.rename(columns={'saldo_var13_medio': 'saldo_medio_var13'}, inplace=True)
    #var38 and the saldo variables have a log-normal distribution
    data['var38'] = data['var38'].apply(np.log)
    # var2 has some strange values -999999 that don't fit the rest of the data.
    # We assume it is the country and categorize the variable as 0 being domestic (which corresponds to the value 2) and 1 for international or unknown.
    process_country(data)
    
    # New feature counting zero entries. It represents the customer activity.
    original_features = data.columns[:-1]
    data.insert(len(original_features),'SumZeros',(data[original_features] == 0).sum(axis=1))

    # New feature describing the number of assets.
    asset_features = [name for name in data.columns if 'ind' in name]
    temp = data[asset_features].sum(axis=1)
    data.insert(data.shape[1]-1, 'NumAssets', temp)

def extra_process(data):
    pass
    
def create_submission(test_id, Y_prob):
    ver = 1
    while True:
        try:
            open('submissions/rforest_ensemble{}.csv'.format(ver), 'x')
            break
        except:
            ver+=1
    path = 'submissions/rforest_ensemble{}.csv'.format(ver)
    pd.DataFrame({"ID": test_id, "TARGET": Y_prob}).to_csv(path, index=False, mode='w')
    print("The submission file is " + path)
