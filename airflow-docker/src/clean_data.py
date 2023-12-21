from pathlib import Path
import pandas as pd
from datetime import datetime
import os

def get_latest_scraped_data_csv(data_dir):
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob('scraped_data_*.csv'))

    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        return latest_csv
    else:
        return None

# Define the directory containing scraped data CSV files
data_directory = '/opt/airflow/Data'

# Get the path to the latest scraped_data CSV file
latest_scraped_data_csv = get_latest_scraped_data_csv(data_directory)

if latest_scraped_data_csv:
    # Define the input file path using the latest scraped_data CSV file
    input_file_path = latest_scraped_data_csv
else:
    print("No scraped_data CSV file found. Please run the scraper script first.")
    exit(1)

# Define the output file path with a datestamp
current_datetime = datetime.now()
datestamp = current_datetime.strftime("%Y%m%d%H%M%S")
output_file_path = Path(f'/opt/airflow/Data/cleaned_data_{datestamp}.csv')


drop_columns=[
            'street', 'housenumber', 'box', 'city', 'subtype', 'location_area', 'region', 
            'district', 'province', 'type_of_sale', 'garden', 'kitchen_type', 'epc_score', 'latitude',
            'longitude', 'property_url'
            ]

def trans_to_bool(df):
    """
    kitchen equipped column is created with 1 (equipped) or 0
    other columns are transformed in place with 1 or 0
    terrace gains 1 status if terrace surface has a value > 0
    all these columns are set to boolean afterwards.

    returns: df

    """
    print('step 1 trans_to_bool:')
    df['kitchen_equipped'] = df['kitchen_type'].apply(lambda x: True if x != '0' else False).astype('bool') #create a new column as boolean.
    df['furnished'] = df['furnished'].apply(lambda x: 1 if x != 0 else 0).astype('bool') #transform column to boolean.
    df['fireplace'] = df['fireplace'].apply(lambda x: 1 if x != 0 else 0).astype('bool') #transform column to boolean.
    df['terrace'] = ((df['terrace'] == True) | (df['terrace_surface'] > 0)).astype('bool') #transform column to boolean, if any condition is true: value will be 1.
    print(f'df rows after step 1: {df.shape[0]}')
    return df

def convert_to_num(df):
    """
    Converts any column from the given dataframe into a numerical datatype so it can be further preprocessed or
    is usable in the model.

    Returns: df
    """
    print('step 2 convert_to_num:')
    columns_to_float = ['price', 'construction_year', 'postalcode', 'bedroom_count', 'facades', 'kitchen_equipped']
    for column in columns_to_float:
        if df[column].dtype == 'object':
            # Remove non-numeric characters and keep numerical values
            df[column] = df[column].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
            # Replace empty fields with -1
            df[column] = df[column].replace('', '-1')
            # Convert the column to float
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
        elif df[column].dtype == 'bool':
            df[column] = df[column].astype('int64')
        else:
            df[column] = df[column].astype('float64')  # Convert numeric column to integers
    print(f'df rows after step 2: {df.shape[0]}')
    return df


def clean_postalcodes(df):
    """
    removes all postalcodes above 9992 (= max number belgian postalcodes)
    reset the postalcode column to str datatype, so the onehotencoder will handle it correctly.
    
    parameters:
        df (pd.dataframe): input dataframe containing the data

    returns:
        pd.dataframe: df
    """
    print('step 3 clean_postalcodes:')
    df= df[df['postalcode'] < 9993]
    df.loc[:, 'postalcode'] = df['postalcode'].astype('str')
    #df['postalcode']= df['postalcode'].astype('str')
    print(f'df rows after step 3: {df.shape[0]}')
    return df

def drop_zero_rows(df):
    """
    drop rows with zero values (none, nan from scrape) if any left.
    
    returns: df

    """
    print('step 4 drop_zero_rows:')
    columns_to_check = ['city', 'price', 'habitable_surface']
    for column in columns_to_check:
        if df[column].dtype == 'object':
            df = df[df[column] != '0']
        elif df[column].dtype == 'int64':
            df = df[df[column] != 0]
        else:
            df = df[df[column] != 0.0]
    print(f'df rows after step 4: {df.shape[0]}')
    return df

def remove_duplicates(df):
    """
    removes duplicates by subsetting
    prints the amount of duplicates present in the dataframe

    returns: df
        
    """
    print('step 5 remove_duplicates:')
    duplicates = df[df.duplicated(subset=['latitude', 'longitude', 'type', 'subtype', 'price', 'district', 'city', 'street', 'housenumber', 'box', 'floor', 'habitable_surface'], keep=False)]
    print(f'amount of duplicates: {duplicates.shape[0]}')
    df = df.drop_duplicates(subset=['latitude', 'longitude', 'type', 'subtype', 'price', 'district', 'city', 'street', 'housenumber', 'box', 'floor', 'habitable_surface'], keep='first')
    print(f'amount of rows after step 5: {df.shape[0]}')
    return df

def remove_outliers(df):
    """
    removes the outliers from the df (manually set specifically for this dataset)
    
    returns: df
    
    """
    print('step 6 remove_outliers:')
    df = df[df['habitable_surface'] <= 2000]
    df = df[df['price'] <= 4000000]
    df = df[df['terrace_surface'] <= 250]
    df = df[df['bedroom_count'] <= 10]
    print(f'amount of rows after step 6: {df.shape[0]}')
    return df

def drop_outside_belgium(df):
    """
    create dataframes for each region
    checks for listings outside of belgium
    drops those listings from the main dataframe

    args:
        df (pd.dataframe): dataframe that contains the data.
        
    returns:
        pd.dataframe: df
    """
    print('step 7 drop_outside_belgium:')
    regions = ['FLANDERS', 'BRUSSELS', 'WALLONIE']
    FLANDERS = df.loc[df['region'] == 'FLANDERS']
    WALLONIE = df.loc[df['region'] == 'WALLONIE']
    BRUSSELS = df.loc[df['region'] == 'BRUSSELS']
    other = df.query('region not in @regions')
    df= df.drop(other.index)
    print(FLANDERS.shape[0])
    print(WALLONIE.shape[0])
    print(BRUSSELS.shape[0])
    print(other.shape[0])
    print(f'total amount = {FLANDERS.shape[0] + BRUSSELS.shape[0] + WALLONIE.shape[0] + other.shape[0]}')
    print(f'amount of rows after step 7: {df.shape[0]}')
    return df

def ordinal_encode_condition_column(df):
    """
    perform ordinal encoding on the 'condition' column of a dataframe.

    parameters:
        df (pd.dataframe): the dataframe containing the 'condition' column to be encoded.

    returns:
        pd.dataframe: the original dataframe with an additional column containing the encoded values.
    """
    print('step 8 ordinal_encode_condition_column:')
    condition_mapping = {
        'TO_RESTORE': 1,
        'TO_RENOVATE': 2,
        'TO_BE_DONE_UP': 3,
        'JUST__RENOVATED': 4,
        'GOOD': 5,
        'AS_NEW': 6
    }

    # apply the custom mapping to the 'condition' column and modify the original dataframe
    df['condition_encoded'] = df['condition'].map(condition_mapping)

    # fill the missing values (0) with a value that indicates 'missing information'
    missing_value = -1
    df['condition_encoded'].fillna(missing_value, inplace=True)
    df['condition_encoded']= df['condition_encoded'].astype('int64')
    print(f'amount of rows after step 8: {df.shape[0]}')
    return df

def check_nan(df):
    print('step 9 check_nan:')
    nan_values = df.isna().any()
    # print the columns with nan values, if any
    print(nan_values[nan_values].index)
    total_nan_values = df.isna().sum().sum()
    print(f"total nan values in the dataframe: {total_nan_values}")
    
# def split_df_on_type(df):
#     """
#     splits the df on type, the dataframe is split into a house and apartment df

#     returns:
#         df_house
#         df_apt
#     """
#     print('step 10 split_df_on_type:')
#     df_house = df[df['type'] == 'house']
#     df_apt = df[df['type'] == 'apartment']
#     df_apt_group = df[df['type'] == 'apartment_group']
#     df_house_group = df[df['type'] == 'house_group']
#     print(f'amount of rows after step 10: {df.shape[0]}')
#     print(f'amount of rows in df_house: {df_house.shape[0]}')
#     print(f'amount of rows in df_apt: {df_apt.shape[0]}')
#     return df_house, df_apt

def run_cleanup():
    """
    main function to call upon all functions in order to clean the given dataframe
    returns:
        int: the version number of the csv file.
    """
    
    df = pd.read_csv(input_file_path, index_col='id', skip_blank_lines=True)
    df = trans_to_bool(df)
    df = convert_to_num(df)
    df = clean_postalcodes(df)
    df = drop_zero_rows(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = drop_outside_belgium(df)
    df = ordinal_encode_condition_column(df)
    check_nan(df)
    #df_house, df_apt = split_df_on_type(df)
    print(f'Total records left: {df.shape[0]} ')
    df.to_csv(output_file_path, index='id')
    return df
    #, df_house, df_apt
    # save the cleaned dataframe to the /data directory
    

# if __name__ == "__main__":
    
#     df = run_cleanup()
    