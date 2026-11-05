
# Import libraries
import pandas as pd
from pathlib import Path

# Specify the directory
data_dir = Path("../data")

# Load the datasets
df_helpdesk = pd.read_csv(data_dir / "mtlformer_helpdesk.csv")
df_a = pd.read_csv(data_dir / "mtlformer_BPIC2012_A.csv")
df_o = pd.read_csv(data_dir / "mtlformer_BPIC2012_O.csv")
df_w = pd.read_csv(data_dir / "mtlformer_BPIC2012_W.csv")

dfs = [df_helpdesk, df_a, df_o, df_w]

dfs_names = [
    "mtlformer_helpdesk_formatted",
    "mtlformer_BPIC2012_A_formatted",
    "mtlformer_BPIC2012_O_formatted",
    "mtlformer_BPIC2012_W_formatted"
]

# Specify the activity column
activity_column = "Activity"

# Convert timestamps and encode activities
for i in range(len(dfs)):

    # Convert timestamps to the desired format
    if i == 0:
        dfs[i]["Complete Timestamp"] = pd.to_datetime(
            dfs[i]["Complete Timestamp"],
            format="%Y/%m/%d"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    elif i == 1:
        dfs[i]["Complete Timestamp"] = pd.to_datetime(
            dfs[i]["Complete Timestamp"],
            format="%m/%d/%Y %H:%M"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    else:
        dfs[i]["Complete Timestamp"] = pd.to_datetime(
            dfs[i]["Complete Timestamp"],
            format="%Y/%m/%d %H:%M"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Encode activity names as integers (starting at 0)
    activity_codes, activity_names = pd.factorize(
        dfs[i][activity_column],
        sort=True
    )

    dfs[i][activity_column] = activity_codes

    # Save the activity mapping
    activity_mapping = pd.DataFrame({
        "Activity": activity_names,
        "Activity_ID": range(len(activity_names))
    })

    activity_mapping.to_csv(
        data_dir / f"{dfs_names[i]}_activity_mapping.csv",
        index=False
    )

    # Save the modified dataset
    dfs[i].to_csv(
        data_dir / f"{dfs_names[i]}.csv",
        index=False
    )

print("All datasets successfully formatted and encoded!")