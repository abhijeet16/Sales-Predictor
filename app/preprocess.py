import numpy as np
import pandas as pd

def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess sales data for modeling."""

    # Accumulate Dailey total sales
    df['Date'] = pd.to_datetime(df['Date'])
    full_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    Dailey_sales = df.groupby('Date')['Total Amount'].sum()

    # Fill the missing days with min sale value
    min_sale = Dailey_sales.min()
    Dailey_sales = df.groupby('Date')['Total Amount'].sum().reindex(full_range).fillna(min_sale)

    # Detect outliers based on IQR Method
    q1 = Dailey_sales.quantile(0.25)
    q3 = Dailey_sales.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Clip outliers with windowing method
    clipped_sales = Dailey_sales.clip(lower=lower_bound, upper=upper_bound)

    # Create final DataFrame with Date as column
    aggregated_df = pd.DataFrame({
        'Date': Dailey_sales.index,
        'DaileySales': Dailey_sales.values,
        'ClippedSales': clipped_sales.values
    })

    # Log transform the clidded sales add in the dataframe
    clipped_sales_log = np.log1p(aggregated_df['ClippedSales'])
    aggregated_df['ClippedSalesLog'] = clipped_sales_log

    return aggregated_df


def preprocess_prediciion_data(df: pd.DataFrame) -> pd.DataFrame:
    """Returns preprocessed data after removing outliers and log transforming the target variable."""

    # Load or use existing df
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by date
    df = df.sort_values('Date')

    # Make sure 'Total Amount' is numeric
    df['Total Amount'] = pd.to_numeric(df['Total Amount'], errors='coerce')

    # Calculate Z-scores
    mean = df['Total Amount'].mean()
    std = df['Total Amount'].std()
    df['Z_Score'] = (df['Total Amount'] - mean) / std

    # Define outliers
    z_threshold = 2
    df['IsOutlierZ'] = df['Z_Score'].abs() > z_threshold

    # Find bounds from non-outliers
    non_outliers = df[~df['IsOutlierZ']]['Total Amount']
    min_valid = non_outliers.min()
    max_valid = non_outliers.max()

    # Replace outliers
    df['AdjustedTotalAmount'] = df.apply(
        lambda row: max_valid if row['Z_Score'] > z_threshold 
        else min_valid if row['Z_Score'] < -z_threshold 
        else row['Total Amount'],
        axis=1
    )

    # Drop 'Z_Score' and 'Is_Outlier_Z' columns
    df.drop(columns=['Z_Score', 'IsOutlierZ'], inplace=True)

    # Create new DataFrame with selected and engineered features
    processed_df = pd.DataFrame({
        'Gender': df['Gender'],
        'Age': df['Age'],
        'Product Category': df['Product Category'],
        'Day': df['Date'].dt.day,
        'Month': df['Date'].dt.month,
        'Weekday': df['Date'].dt.weekday,
        'isWeekend': df['Date'].dt.weekday.isin([5, 6]).astype(int),
        'isAfter25': (df['Date'].dt.day > 25).astype(int),
        'LogTotalAmount': np.log1p(df['AdjustedTotalAmount'])
    })

    return processed_df
