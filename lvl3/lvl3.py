import pandas as pd
from itertools import combinations

df0 = pd.read_csv("lvl3/src/level_3_c.in")
df1 = pd.read_csv("lvl3/src/all_data_from_level_1.in")

is_fahrenheit = (df1['Temperature [°C]'] > 60)
df1.loc[is_fahrenheit, 'Temperature [°C]'] = (df1.loc[is_fahrenheit, 'Temperature [°C]'] - 32) * 5 / 9

# Function to check symmetry
def is_symmetric(path_str):
    # Convert the string to a list of ints
    path = list(map(int, path_str.split()))
    return path == path[::-1]

# Apply the function and filter symmetric paths
df0['Symmetric'] = df0['BOP Path'].apply(is_symmetric)
symmetric_flocks = df0[df0['Symmetric']]['Flock ID'].tolist()
print("Symmetric paths found in Flocks:", symmetric_flocks)


def average_temperature(df_paths, df_bop):
    """
    For each path in df_paths, compute the average temperature from df_bop
    and add it as a new column 'Avg_temp'.

    df_paths: DataFrame with columns ['Flock ID', 'BOP Path']
    df_bop: DataFrame with columns ['BOP', 'Temperature [°C]', ...]

    Returns: df_paths with a new column 'Avg_temp'
    """
    # Convert BOP to temperature mapping for fast lookup
    temp_dict = df_bop.set_index('BOP')['Temperature [°C]'].to_dict()

    # Function to compute average temp for a single path
    def compute_avg_temp(path_str):
        path_bops = list(map(int, path_str.split()))
        temps = [temp_dict[bop] for bop in path_bops if bop in temp_dict]
        return sum(temps) / len(temps) if temps else None

    # Add new column to df_paths
    df_paths = df_paths.copy()  # avoid modifying original df
    df_paths['Avg_temp'] = df_paths['BOP Path'].apply(compute_avg_temp)

    return df_paths

df_avg_temp = average_temperature(df0, df1)
df_avg_per_flock = df_avg_temp.groupby('Flock ID', as_index=False)['Avg Temperature'].mean()
print(df_avg_per_flock)

## CHECK FLOCK SIMILARIT<
def lcs_length(seq1, seq2):
    """Compute length of Longest Common Subsequence (LCS) between two sequences"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def flock_path_similarity(df_paths):
    """
    For each flock, compute average similarity between all pairs of paths.
    Returns DataFrame with ['Flock ID', 'Avg Path Similarity']
    """
    results = []

    for flock_id, group in df_paths.groupby('Flock ID'):
        paths = [list(map(int, row.split())) for row in group['BOP Path']]

        if len(paths) < 2:
            avg_similarity = 1.0  # Only one path, consider it fully similar
        else:
            sims = []
            for p1, p2 in combinations(paths, 2):
                lcs_len = lcs_length(p1, p2)
                avg_len = (len(p1) + len(p2)) / 2
                sims.append(lcs_len / avg_len)
            avg_similarity = sum(sims) / len(sims)

        results.append({'Flock ID': flock_id, 'Avg Path Similarity': avg_similarity})

    return pd.DataFrame(results)

df_similarity = flock_path_similarity(df0)
print(df_similarity)

## afterwards use ground truth information on species, result was created manually