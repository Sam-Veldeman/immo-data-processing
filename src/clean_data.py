import sys
import pandas as pd
sys.path.insert(0, '../')
from sklearn.preprocessing import OneHotEncoder

drop_columns=[
            'Street', 'Housenumber', 'Box', 'City', 'Subtype', 'Location area', 'Region', 
            'District', 'Province', 'Type of sale', 'Garden', 'Kitchen type', 'EPC score', 'Latitude',
            'Longitude', 'Property url'
            ]

def trans_to_bool(df):
    """
    Kitchen equipped column is created with 1 (equipped) or 0
    Other columns are transformed in place with 1 or 0
    Terrace gains 1 status if Terrace surface has a value > 0
    All these columns are set to boolean afterwards.

    Returns: df

    """
    print('Step 1 trans_to_bool:')
    df['Kitchen equipped'] = df['Kitchen type'].apply(lambda x: True if x != '0' else False).astype('bool') #create a new column as boolean.
    df['Furnished'] = df['Furnished'].apply(lambda x: 1 if x != 0 else 0).astype('bool') #transform column to boolean.
    df['Fireplace'] = df['Fireplace'].apply(lambda x: 1 if x != 0 else 0).astype('bool') #transform column to boolean.
    df['SwimmingPool'] = df['SwimmingPool'].apply(lambda x: 1 if x != 0 else 0).astype('bool') #transform column to boolean.
    df['Terrace'] = ((df['Terrace'] == True) | (df['Terrace surface'] > 0)).astype('bool') #transform column to boolean, if any condition is True: value will be 1.
    print(f'Df rows after step 1: {df.shape[0]}')
    return df

def convert_to_num(df):
    """
    Converts any column from the given dataframe into numerical datatype so it can be further preprocessed or
    is useable in the model.

    Returns: df
    """
    print('Step 2 convert_to_num:')
    columns_to_float = ['Price', 'Construction year', 'Postalcode', 'Bedroom count', 'Facades', 'Kitchen equipped']
    for column in columns_to_float:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace('\D', '', regex=True)  # Remove non-numeric characters
            df = df[df[column] != '']  # Drop rows with empty values
            df[column] = df[column].astype('float64')  # Convert column to integers
        elif df[column].dtype == 'bool':
            df[column] = df[column].astype('int64')
        else:
            df[column] = df[column].astype('float64')  # Convert numeric column to integers
    print(f'Df rows after step 2: {df.shape[0]}')
    return df

def clean_postalcodes(df):
    """
    Removes all postalcodes above 9992 (= max number Belgian postalcodes)
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data

    Returns:
        pd.DataFrame: df
    """
    print('Step 3 clean_postalcodes:')
    df= df[df['Postalcode'] < 9993]
    print(f'Df rows after step 3: {df.shape[0]}')
    return df

def drop_zero_rows(df):
    """
    Drop rows with zero values (none, nan from scrape) if any left.
    
    Returns: df

    """
    print('Step 4 drop_zero_rows:')
    columns_to_check = ['City', 'Price', 'Habitable surface']
    for column in columns_to_check:
        if df[column].dtype == 'object':
            df = df[df[column] != '0']
        elif df[column].dtype == 'int64':
            df = df[df[column] != 0]
        else:
            df = df[df[column] != 0.0]
    print(f'Df rows after step 4: {df.shape[0]}')
    return df

def remove_duplicates(df):
    """
    Removes duplicates by subsetting
    Prints the amount of duplicates present in the DataFrame

    Returns: df
        
    """
    print('Step 5 remove_duplicates:')
    duplicates = df[df.duplicated(subset=['Latitude', 'Longitude', 'Type', 'Subtype', 'Price', 'District', 'City', 'Street', 'Housenumber', 'Box', 'Floor', 'Habitable surface'], keep=False)]
    print(f'Amount of duplicates: {duplicates.shape[0]}')
    df = df.drop_duplicates(subset=['Latitude', 'Longitude', 'Type', 'Subtype', 'Price', 'District', 'City', 'Street', 'Housenumber', 'Box', 'Floor', 'Habitable surface'], keep='first')
    print(f'Amount of rows after step 5: {df.shape[0]}')
    return df

def remove_outliers(df):
    """
    Removes the outliers from the df (manually set specifically for this dataset)
    
    Returns: df
    
    """
    print('Step 6 remove_outliers:')
    df = df[df['Habitable surface'] <= 2000]
    df = df[df['Price'] <= 4000000]
    df = df[df['Terrace surface'] <= 250]
    df = df[df['Bedroom count'] <= 10]
    print(f'Amount of rows after step 6: {df.shape[0]}')
    return df

def drop_outside_belgium(df):
    """
    Create dataframes for each region
    Checks for listings outside of Belgium
    Drops those listings from the main DataFrame

    Args:
        df (pd.DataFrame): DataFrame that contains the data.
        
    Returns:
        pd.DataFrame: df
    """
    print('Step 7 drop_outside_belgium:')
    regions = ['FLANDERS', 'BRUSSELS', 'WALLONIE']
    flanders = df.loc[df['Region'] == 'FLANDERS']
    wallonie = df.loc[df['Region'] == 'WALLONIE']
    brussels = df.loc[df['Region'] == 'BRUSSELS']
    other = df.query('Region not in @regions')
    df= df.drop(other.index)
    print(flanders.shape[0])
    print(wallonie.shape[0])
    print(brussels.shape[0])
    print(other.shape[0])
    print(f'total amount = {flanders.shape[0] + brussels.shape[0] + wallonie.shape[0] + other.shape[0]}')
    print(f'Amount of rows after step 7: {df.shape[0]}')
    return df

def ordinal_encode_condition_column(df):
    """
    Perform ordinal encoding on the 'Condition' column of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'Condition' column to be encoded.

    Returns:
        pd.DataFrame: The original DataFrame with an additional column containing the encoded values.
    """
    print('Step 8 ordinal_encode_condition_column:')
    condition_mapping = {
        'TO_RESTORE': 1,
        'TO_RENOVATE': 2,
        'TO_BE_DONE_UP': 3,
        'JUST_RENOVATED': 4,
        'GOOD': 5,
        'AS_NEW': 6
    }

    # Apply the custom mapping to the 'Condition' column and modify the original DataFrame
    df['Condition_encoded'] = df['Condition'].map(condition_mapping)

    # Fill the missing values (0) with a value that indicates 'Missing Information'
    missing_value = -1
    df['Condition_encoded'].fillna(missing_value, inplace=True)
    df['Condition_encoded']= df['Condition_encoded'].astype('int64')
    print(f'Amount of rows after step 8: {df.shape[0]}')
    return df

def check_nan(df):
    print('Step 9 check_nan:')
    nan_values = df.isna().any()
    # Print the columns with NaN values, if any
    print(nan_values[nan_values].index)
    total_nan_values = df.isna().sum().sum()
    print(f"Total NaN values in the DataFrame: {total_nan_values}")
    
def split_df_on_type(df):
    """
    Splits the df on type, the dataframe is split into a house and apartment df

    Returns:
        df_house
        df_apt
    """
    print('Step 10 split_df_on_type:')
    df_house = df[df['Type'] == 'HOUSE']
    df_apt = df[df['Type'] == 'APARTMENT']
    df_apt_group = df[df['Type'] == 'APARTMENT_GROUP']
    df_house_group = df[df['Type'] == 'HOUSE_GROUP']
    print(f'Amount of rows after step 10: {df.shape[0]}')
    print(f'Amount of rows in df_house: {df_house.shape[0]}')
    print(f'Amount of rows in df_apt: {df_apt.shape[0]}')
    return df_house, df_apt






    
def run_cleanup(df):
    """
    Main function to call upon all functions in order to clean the given dataframe
    Returns:
        int: The version number of the CSV file.
    """
    df = trans_to_bool(df)
    df = convert_to_num(df)
    df = clean_postalcodes(df)
    df = drop_zero_rows(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = drop_outside_belgium(df)
    df = ordinal_encode_condition_column(df)
    check_nan(df)
    df_house, df_apt = split_df_on_type(df)
    return df, df_house, df_apt