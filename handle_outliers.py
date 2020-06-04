import pandas as pd

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.
    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix upper_outliers for all the numeric columns
    in the given dataframe.
    '''

    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)

    return df

def remove_outliers(df, k):
    """
    removes rows of df that are outliers
    """
    # df = add_upper_outlier_columns(df, k)
    drop_list = list(df.select_dtypes('object').columns)
    # IQR to detect and remove outliers
    Q1 = df.drop(columns=drop_list).quantile(0.25)
    Q3 = df.drop(columns=drop_list).quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df.drop(columns=drop_list) < (Q1 - k * IQR)) | (df.drop(columns=drop_list) > (Q3 + k * IQR))).any(axis=1)]
       
    return df