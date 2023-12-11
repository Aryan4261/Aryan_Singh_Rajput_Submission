import pandas as pd
import numpy as np


def generate_car_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'id_1', 'id_2', and 'car'.

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    for index in car_matrix.index:
        car_matrix.at[index, index] = 0

    return car_matrix


def get_type_count(df: pd.DataFrame) -> dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame): Input DataFrame with a column named 'car'.

    Returns:
        dict: Dictionary with car types as keys and their counts as values.
    """
    df['car_type'] = pd.cut(df['car'],
                            bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'],
                            right=False)

    type_count = df['car_type'].value_counts().to_dict()
    sorted_type_count = dict(sorted(type_count.items()))

    return sorted_type_count


def get_bus_indexes(df: pd.DataFrame) -> list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame): Input DataFrame with a column named 'bus'.

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    mean_bus = df['bus'].mean()

    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()
    bus_indexes.sort()

    return bus_indexes


def filter_routes(df: pd.DataFrame) -> list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'route' and 'truck'.

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    route_avg_truck = df.groupby('route')['truck'].mean()
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    return selected_routes


def multiply_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    modified_matrix = modified_matrix.round(1)

    return modified_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    df['start_day'] = df['start_timestamp'].dt.day_name()
    df['start_time'] = df['start_timestamp'].dt.time

    completeness_series = df.groupby(['id', 'id_2']).apply(check_timestamp_completeness)

    return completeness_series

def check_timestamp_completeness(group):
    """
    Check the completeness of timestamps for a specific (id, id_2) pair.

    Args:
        group: A group of rows corresponding to a unique (id, id_2) pair.

    Returns:
        bool: True if timestamps are complete, False otherwise.
    """
    days_present = set(group['start_day']) == set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    time_ranges_present = set(group['start_time']) == set(pd.date_range('00:00:00', '23:59:59', freq='1S').time)

    return days_present and time_ranges_present
