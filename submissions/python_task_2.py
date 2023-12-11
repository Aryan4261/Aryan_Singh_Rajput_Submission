import pandas as pd
import networkx as nx
from datetime import datetime, timedelta, time


def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'ID_1', 'ID_2', and 'Distance'.

    Returns:
        pandas.DataFrame: Distance matrix
    """
    g = nx.DiGraph()

    for _, row in df.iterrows():
        g.add_edge(row['id_start'], row['id_end'], weight=row['distance'])
        g.add_edge(row['id_end'], row['id_start'], weight=row['distance'])

    distance_matrix = nx.floyd_warshall_numpy(g, weight='weight')
    distance_df = pd.DataFrame(distance_matrix, index=g.nodes, columns=g.nodes)

    return distance_df


def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix DataFrame to a format with columns 'id_start', 'id_end', and 'distance'.

    Args:
        df (pandas.DataFrame): Input DataFrame representing a distance matrix.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Extract unique pairs of id_start and id_end from the upper triangle (excluding diagonal)
    pairs = [(start, end) for start in df.index for end in df.index if start != end]

    # Create a DataFrame with id_start, id_end, and distance values
    unrolled_df = pd.DataFrame(pairs, columns=['id_start', 'id_end'])
    unrolled_df['distance'] = unrolled_df.apply(lambda row: df.at[row['id_start'], row['id_end']], axis=1)

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): Reference ID for calculating the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for the reference ID
    average_distance = df[df['id_start'] == reference_id]['distance'].mean()

    # Calculate the threshold range
    lower_threshold = 0.9 * average_distance
    upper_threshold = 1.1 * average_distance

    # Filter values within the threshold range
    filtered_df = df[(df['id_start'] != reference_id) & (df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    return filtered_df



def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates based on vehicle types and add corresponding columns to the input DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with added columns 'moto', 'car', 'rv', 'bus', and 'truck' representing toll rates.
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


def calculate_time_based_toll_rates(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for different time intervals within a day and add columns to the input DataFrame.

    Args:
        input_df (pandas.DataFrame): Input DataFrame with columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with additional columns 'start_day', 'start_time', 'end_day', 'end_time',
                          and modified values of vehicle columns based on time intervals.
    """
    weekday_discounts = [(time(0, 0, 0), time(10, 0, 0), 0.8),
                         (time(10, 0, 0), time(18, 0, 0), 1.2),
                         (time(18, 0, 0), time(23, 59, 59), 0.8)]

    weekend_discount = 0.7

    new_columns_data = []

    for _, group_df in input_df.groupby(['id_start', 'id_end']):
        for day in range(7):
            for start_time, end_time, discount_factor in (weekday_discounts if day < 5 else [(time(0, 0, 0), time(23, 59, 59), weekend_discount)]):
                start_datetime = datetime.combine(datetime.today(), start_time) + timedelta(days=day)
                end_datetime = datetime.combine(datetime.today(), end_time) + timedelta(days=day)

                new_columns_data.append({
                    'id_start': group_df['id_start'].iloc[0],
                    'id_end': group_df['id_end'].iloc[0],
                    'start_day': start_datetime.strftime('%A'),
                    'start_time': start_datetime.time(),
                    'end_day': end_datetime.strftime('%A'),
                    'end_time': end_datetime.time(),
                    'distance': group_df['distance'].mean() * discount_factor,
                })

    result_df = pd.DataFrame(new_columns_data)

    return result_df
