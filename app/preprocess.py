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

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add more features for modeling."""

    # Feature engineering
    df['Date'] = pd.to_datetime(df['Date'])

    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['isWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['isAfter25'] = (df['Day'] > 25).astype(int)
    df['LogTotalAmount'] = np.log1p(df['Total Amount'])

    return df
