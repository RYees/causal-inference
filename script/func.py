from sklearn import preprocessing
import pandas as pd
from logger import Logger
import sys 

class dataHandler:
    def __init__(self):            
        """Initilize class."""
        try:
            self.logger = Logger("data.log").get_app_logger()
            self.logger.info(
                'Successfully Instantiated Preprocessing Class Object')
        except Exception:
            self.logger.exception(
                'Failed to Instantiate Preprocessing Class Object')
            sys.exit(1)
    def standardize_columns(df, column) -> pd.DataFrame:
        try:
            std_column_df = pd.DataFrame(df[column])
            std_column_values = std_column_df.values
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled_array = min_max_scaler.fit_transform(std_column_values)
            df[column] = scaled_array
            print('Successfull data scaling')
            return df.head()
        except:
            print('error in scaling data')

    

