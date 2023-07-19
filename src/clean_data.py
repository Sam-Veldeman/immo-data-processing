import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
def import_data(path):
    file_path = r'c:\Users\samve\OneDrive\0BeCode\repos\immo-data-processing\Data\Filtered_Data\house_details_v1.csv'
    df = pd.read_csv(file_path, index_col='id', skip_blank_lines=True)
    return df

def convert_to_num(df):
    """
    Converts any column from the given dataframe into numerical datatype so it can be further preprocessed or
    is useable in the model.

    Returns: df
    """
    columns_to_float = ['Price', 'Postalcode', 'Construction year', 'Bedroom count', 'Facades']
    for column in columns_to_float:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace('\D', '', regex=True)  # Remove non-numeric characters
            df = df[df[column] != '']  # Drop rows with empty values
            df[column] = df[column].astype('float64')  # Convert column to integers
        else:
            df[column] = df[column].astype('float64')  # Convert numeric column to integers
    print(f'Df rows after step 1: {df.shape[0]}\n{df.info()}')
    return df
    
def drop_zero_rows(df):
    """
    Drop rows with zero values (none, nan from scrape) if any left.
    
    Returns: df

    """
    
    columns_to_check = ['City', 'Postalcode', 'Region', 'Price', 'Construction year', 'Habitable surface']
    for column in columns_to_check:
        if df[column].dtype == 'object':
            df = df[df[column] != '0']
        elif df[column].dtype == 'int64':
            df = df[df[column] != 0]
        else:
            df = df[df[column] != 0.0]
    print(f'Df rows after step 2: {df.shape[0]}')
    return df

def remove_duplicates(df):
    """
    Removes duplicates by subsetting

    Returns: df
        
    """
    print(f'Df rows: {df.shape[0]}')
    duplicates = df[df.duplicated(subset=['Latitude', 'Longitude', 'Type', 'Subtype', 'Price', 'District', 'City', 'Street', 'Housenumber', 'Box', 'Floor', 'Habitable surface'], keep=False)]
    print(f'Amount of duplicates: {duplicates.shape[0]}')
    df = df.drop_duplicates(subset=['Latitude', 'Longitude', 'Type', 'Subtype', 'Price', 'District', 'City', 'Street', 'Housenumber', 'Box', 'Floor', 'Habitable surface'], keep='first')
    print(f'Amount of rows after step 3: {df.shape[0]}')
    return df

def trans_to_bool(df):
    """
    Changes certain selected categorical features to numerical values

    Returns: df

    """
    df['Kitchen equiped'] = df['Kitchen type'].apply(lambda x: 1 if x !=0 else 0).astype('bool') #create a new column as boolean.
    df['Furnished'] = df['Furnished'].apply(lambda x: 1 if x != 0 else 0).astype('bool') #transform column to boolean.
    df['Fireplace'] = df['Fireplace'].apply(lambda x: 1 if x != 0 else 0).astype('bool') #transform column to boolean.
    df['SwimmingPool'] = df['SwimmingPool'].apply(lambda x: 1 if x != 0 else 0).astype('bool') #transform column to boolean.
    df['Terrace'] = ((df['Terrace'] == True) | (df['Terrace surface'] > 0)).astype('bool') #transform column to boolean, if any condition is True: value will be 1.
    return df


def split_df_on_type(df):
    """
    Splits the df on type, the dataframe is split into a house and apartment df

    Returns:
        df_house
        df_apt
    """
    df_apt, df_house = [x for _, x in df.groupby(df['Type'] == 'HOUSE')]
    #df_apt.drop(column= 'Type', inplact= True)
    #df_house.drop(column='Type', inplace= True)
    return df_house, df_apt

def remove_outliers(df):
    """
    Removes the outliers from the df (manually set specifically for this dataset)
    
    Returns: df
    
    """
    df = df[df['Habitable surface'] <= 2000]
    df = df[df['Price'] <= 4000000]
    df = df[df['Terrace surface'] <= 250]
    df = df[df['Bedroom count'] <= 10]
    print(f'Number of rows (listings) final df: {df.shape[0]}')
    return df
def encode_column(df):
    # Define the order of the conditions
    condition_order = ['AS_NEW', 'GOOD', '0', 'JUST_RENOVATED', 'TO_RENOVATE', 'TO_BE_DONE_UP', 'TO_RESTORE']
    # Use OrdinalEncoder to transform the 'Condition' column
    encoder = OrdinalEncoder(categories=[condition_order])
    df['Condition'] = encoder.fit_transform(df[['Condition']])
    return df

def dummies(df, columns_to_dum):
    if columns_to_dum != []:
        for column_dum in columns_to_dum:
            dummies = pd.get_dummies(df[column_dum], dtype=int)
            df = pd.concat([df, dummies], axis='columns')
            df.drop(columns=[column_dum, dummies.columns[0]], inplace=True)
    return df
def drop_columns(df, columns):
    """

    Drops all columns that are not needed for the model to run.
    Small increase in speed to be expected.

    Returns: df
    
    """
    df = df.drop(columns= columns)
    return df

def run_cleanup(df, columns, columns_to_dum= []):
    """
    Main function to call upon all functions in order to clean the given dataframe
    Returns:
        int: The version number of the CSV file.
    """
    df = convert_to_num(df)
    df = drop_zero_rows(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = trans_to_bool(df)
    df = encode_column(df)
    df = drop_columns(df, columns)
    df = dummies(df, columns_to_dum)
    return df
