import pandas as pd

# Read file
df = pd.read_csv("lvl1/src/level_1_e.in")

# Identify invalid temperature values BEFORE conversion
invalid_temp_mask = ~df["Temperature [°C]"].str.replace('.', '', 1).str.isnumeric()
errors = df.loc[invalid_temp_mask, "Temperature [°C]"].tolist()
print("Errors:", errors if errors else "OK")

word_to_num = {
    "seventeen": 17
}

df["Temperature [°C]"] = df["Temperature [°C]"].replace(word_to_num)

# Convert temperature to numeric, invalid become NaN
df["Temperature [°C]"] = pd.to_numeric(df["Temperature [°C]"], errors="coerce")
df["Humidity [%]"] = pd.to_numeric(df["Humidity [%]"], errors="coerce")

# Sort: Temperature descending, Humidity ascending
df_sorted = df.sort_values(
    by=["Temperature [°C]", "Humidity [%]"],
    ascending=[False, True]
)

# Extract only BOP values with valid temperatures
sorted_bops = df_sorted.loc[df_sorted["Temperature [°C]"].notna(), "BOP"].tolist()

# Print results
print("Sorted BOPs:", " ".join(str(x) for x in sorted_bops))

with open("D:/Blanka/random - érdekes/cc_contest/lvl1/src/level_1_a.out", "w") as f:
    f.write(" ".join(str(x) for x in sorted_bops))


## For further conversions without error values:
# Read file
df = pd.read_csv("lvl1/src/level_1_e.in")
# Convert temperature to numeric, invalid become NaN
df["Temperature [°C]"] = pd.to_numeric(df["Temperature [°C]"], errors="coerce")
df["Humidity [%]"] = pd.to_numeric(df["Humidity [%]"], errors="coerce")

df_sorted = df.sort_values(
    by=["Temperature [°C]", "Humidity [%]"],
    ascending=[False, True]
)

# Extract only BOP values with valid temperatures
sorted_bops = df_sorted.loc[df_sorted["Temperature [°C]"].notna(), "BOP"].tolist()

# Print results
print("Sorted BOPs:", " ".join(str(x) for x in sorted_bops))

with open("lvl1/src/level_1_e.out", "w") as f:
    f.write(" ".join(str(x) for x in sorted_bops))
