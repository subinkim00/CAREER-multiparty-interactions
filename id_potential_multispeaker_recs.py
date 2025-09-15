import pandas as pd
import re

# load annotation file tracker
df = pd.read_excel("Annotation file tracker_CAREER.xlsx")

# we kept track of the number of speakers in each recording, so can easily filter for # speakers > 2
# format: "xxmins, xspkrs, xutts"
def is_multispeaker(value):
    if isinstance(value, str) and value.count(",") == 2:
        match = re.search(r',\s*(\d+)spkrs', value)
        if match:
            return int(match.group(1)) > 2
    return False

multispeaker_df = df[df["Time/speaker/utterance tracking"].apply(is_multispeaker)]

multispeaker_df.to_excel("multispeaker_output.xlsx", index=False)