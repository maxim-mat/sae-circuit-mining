import pm4py
import pandas as pd

# 1. Load the data from your optimized Parquet file
df_events = pd.read_parquet("sae_events.parquet")

# 2. Hack pm4py's Validator: Convert layers to an actual datetime format
# Layer 0 becomes 1970-01-01 00:00:00, Layer 1 becomes 1970-01-01 00:00:01, etc.
# This preserves the exact order relation while matching the expected data type.
df_events["fake_timestamp"] = pd.to_datetime(df_events["timestamp"], unit="s", origin="1970-01-01")

# 3. Establish the CPN Token Identity (The Position Color)
# This treats each unique token position in each prompt run as an independent thread
df_events["case_id"] = df_events["trace_id"] + "_pos_" + df_events["token_idx"].astype(str)

# 4. Standardize the log with pm4py's formatting tool
# This maps our clean keys to pm4py's internal attributes (concept:name, time:timestamp, etc.)
event_log = pm4py.format_dataframe(
    df_events, 
    case_id='case_id', 
    activity_key='activity', 
    timestamp_key='fake_timestamp' # Passing the validated datetime column
)

print("Discovering process architecture via Inductive Miner...")

# 5. Execute Inductive Miner Discovery
# This returns the formal Petri Net components: Net object, Initial Marking, Final Marking
net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)

# 6. Visualize the resulting CPN Process Graph
pm4py.view_petri_net(net, initial_marking, final_marking)