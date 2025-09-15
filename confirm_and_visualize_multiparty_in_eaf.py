import os
import re
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pympi.Elan import Eaf # pip install pympi-ling
from scipy.stats import chisquare # pip install scipy
from matplotlib.patches import Rectangle
from collections import defaultdict

# defining directory with downloaded eafs and max silence for interaction burst id
EAF_DIR = 'downloaded_eafs'
MAX_GAP = 10  # seconds

# load age mapping from multispeaker recordings excel sheet
age_df = pd.read_excel("multispeaker_output.xlsx")

# make sure 'filename' is a string + strip any leading/trailing spaces
age_df['Recording'] = age_df['Recording'].astype(str).str.strip()

# create dictionary mapping between filename and age (in months)
age_map = dict(zip(age_df['Recording'], age_df['Age']))

# helper: get non chi speaker tiers from eaf
def get_non_chi_speaker_tiers(eaf):
    return [tier for tier in eaf.get_tier_names() if re.match(r'^[A-Z]{2}[0-9]$', tier)]

# helper: get all annotations in eaf
def get_all_annotations(eaf, include_chi=True):
    annotations = []
    for tier in eaf.get_tier_names():
        if include_chi and tier == 'CHI':
            valid = True
        elif re.match(r'^[A-Z]{2}[0-9]$', tier):
            valid = True
        else:
            valid = False

        if valid:
            for ann in eaf.get_annotation_data_for_tier(tier):
                start, end, _ = ann
                annotations.append({
                    'tier': tier,
                    'start': start / 1000.0,
                    'end': end / 1000.0,
                    'duration': (end - start) / 1000.0
                })
    return sorted(annotations, key=lambda x: x['start'])

#################################################################################
# IDENTIFY POTENTIAL MULTIPARTY RECORDINGS
# 2 or more non-chi speaker utts w/in 10 secs, no speaker continues for > 5 utts
# merge interaction bursts separated by < 10 secs
# return interaction bursts
#################################################################################
def find_interaction_windows(non_chi_annotations, max_gap):
    # sort annotations by time
    anns = sorted(non_chi_annotations, key=lambda x: x['start'])

    # build initial interaction bursts
    bursts = []
    current_burst = [anns[0]]

    for prev, curr in zip(anns, anns[1:]):
        gap = curr['start'] - prev['end']
        if gap <= max_gap:
            current_burst.append(curr)
        else:
            bursts.append(current_burst)
            current_burst = [curr]
    bursts.append(current_burst)

    # filter for bursts that include 2 or more unique non-CHI speakers and no speaker has > 5 consecutive turns (active particiapants)
    def is_valid_burst(burst):
        speakers = [a['tier'] for a in burst]
        if len(set(speakers)) < 2:
            return False
        count = 1
        for i in range(1, len(speakers)):
            if speakers[i] == speakers[i-1]:
                count += 1
                if count > 5:
                    return False
            else:
                count = 1
        return True

    valid_bursts = [burst for burst in bursts if is_valid_burst(burst)]

    # merge interaction bursts that are separated by < 10s
    merged_bursts = []
    for cluster in valid_bursts:
        if not merged_bursts:
            merged_bursts.append(cluster)
            continue

        last_cluster = merged_bursts[-1]
        gap = cluster[0]['start'] - last_cluster[-1]['end']

        if gap < 10:  # merge!
            merged_bursts[-1].extend(cluster)
            merged_bursts[-1].sort(key=lambda x: x['start'])
        else:
            merged_bursts.append(cluster)

    return merged_bursts

#################################################################################
# EXTEND OUTWARDS TO FORM INTERACTION BURST
# continue expanding until silence is >= 10 secs
# return interaction bursts
#################################################################################
def extend_boundaries(all_annotations, interaction, silence_threshold=10):
    idxs = [all_annotations.index(ann) for ann in interaction]
    start_idx = min(idxs)
    end_idx = max(idxs)

    # extend backward
    while start_idx > 0:
        prev_ann = all_annotations[start_idx - 1]
        curr_ann = all_annotations[start_idx]
        gap = curr_ann['start'] - prev_ann['end']
        if gap >= silence_threshold:
            break
        start_idx -= 1

    # extend forward
    while end_idx < len(all_annotations) - 1:
        curr_ann = all_annotations[end_idx]
        next_ann = all_annotations[end_idx + 1]
        gap = next_ann['start'] - curr_ann['end']
        if gap >= silence_threshold:
            break
        end_idx += 1

    return all_annotations[start_idx:end_idx + 1]

# helper: get speaker groups (FA, MA, XC, CHI)
def get_speaker_group(speaker):
    speaker = str(speaker) 
    if re.match(r'^FA\d+$', speaker): return "Female Adult"
    elif re.match(r'^MA\d+$', speaker): return "Male Adult"
    elif re.match(r'^[A-Z]C\d+$', speaker): return "Other Child"
    elif speaker == "CHI": return "Target Child"
    else: return "Other" 

#################################################################################
# CREATE INTERACTION PLOTS FOR INTERACTION BURST VISUALIZATION W/IN 5 MIN REC
#################################################################################
def plot_timeline(all_annotations, interactions, filename):
    fig, ax = plt.subplots(figsize=(12, 6))

    tiers = sorted(set(a['tier'] for a in all_annotations))
    tier_to_y = {tier: i for i, tier in enumerate(tiers)}

    # plot each annotation
    for ann in all_annotations:
        y = tier_to_y[ann['tier']]
        ax.add_patch(Rectangle(
            (ann['start'], y - 0.3),
            ann['end'] - ann['start'],
            0.6,
            color='skyblue'
        ))

    # highlight multi-party interaction windows (orange)
    for i, interaction in enumerate(interactions):
        extended = extend_boundaries(all_annotations, interaction)
        start = min(a['start'] for a in extended)
        end = max(a['end'] for a in extended)
        ax.axvspan(start, end, color='orange', alpha=0.2)

    # highlight silence ≥10s across speakers (gray)
    sorted_anns = sorted(all_annotations, key=lambda x: x['start'])
    for a1, a2 in zip(sorted_anns, sorted_anns[1:]):
        if a2['start'] - a1['end'] >= 10:
            ax.axvspan(a1['end'], a2['start'], color='gray', alpha=0.1)

    # axis labels and ticks
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(tiers)
    ax.set_xlabel('Time in recording (seconds)', fontsize=12)
    ax.set_title(f'Multi-party Interactions: {filename}', fontsize=14)

    max_time = max(a['end'] for a in all_annotations)
    ax.set_xlim(0, max_time + 10)
    ax.set_xticks(range(0, int(max_time) + 60, 60))
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    os.makedirs("interaction_plots", exist_ok=True)
    output_path = os.path.join("interaction_plots", f"{filename}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved plot to {output_path}")

#################################################################################
# MAIN EAF PROCESSING FUNCTION THAT CALLS ABOVE HELPER FUNCTIONS
# plots saved to interaction_plots
# returns results and utterance_results (interaction-level and utterance-level)
#################################################################################
def process_eaf_file(filepath):
    eaf = Eaf(filepath)
    all_anns = get_all_annotations(eaf)
    non_chi_anns = [a for a in all_anns if re.match(r'^[A-Z]{2}[0-9]$', a['tier'])]

    interaction_windows = find_interaction_windows(non_chi_anns, MAX_GAP)
    results = []
    utterance_results = []

    for interaction_idx, interaction in enumerate(interaction_windows):
        extended = extend_boundaries(all_anns, interaction, MAX_GAP)
        start_time = min(a['start'] for a in extended)
        end_time = max(a['end'] for a in extended)
        interaction_duration = end_time - start_time

        # speech per speaker
        speech = {}
        for ann in extended:
            speaker = ann['tier']
            speech[speaker] = speech.get(speaker, 0) + ann['duration']

        # create interaction summary record
        interaction_id = f"{os.path.basename(filepath)}_{interaction_idx}"
        base_filename = os.path.basename(filepath)
        
        result = {
            'interaction_id': interaction_id,
            'filename': base_filename,
            'age': age_map.get(re.sub(r'\.eaf$', '', base_filename), 'NA'),
            'interaction_start': round(start_time, 3),
            'interaction_end': round(end_time, 3)
        }
        result.update(speech)
        results.append(result)
        
        # create individual utterance record
        for ann in extended:
            # calculate normalized start and end times within the interaction
            norm_start = (ann['start'] - start_time) / interaction_duration
            norm_end = (ann['end'] - start_time) / interaction_duration
            
            utterance_results.append({
                'interaction_id': interaction_id,
                'filename': base_filename,
                'age': age_map.get(re.sub(r'\.eaf$', '', base_filename), 'NA'),
                'interaction_start': round(start_time, 3),
                'interaction_end': round(end_time, 3),
                'speaker': ann['tier'],
                'utterance_start': round(ann['start'], 3),
                'utterance_end': round(ann['end'], 3),
                'duration': round(ann['duration'], 3),
                'norm_start': round(norm_start, 3),
                'norm_end': round(norm_end, 3)
            })

    # make a timeline plot
    if interaction_windows:
        plot_timeline(all_anns, interaction_windows, os.path.basename(filepath))

    return results, utterance_results

def main():
    all_results = []
    all_utterance_results = [] # utterance level

    for file in os.listdir(EAF_DIR):
        if file.endswith('.eaf'):
            filepath = os.path.join(EAF_DIR, file)
            print(f"Processing {file}...")
            results, utterance_results = process_eaf_file(filepath)
            all_results.extend(results)
            all_utterance_results.extend(utterance_results)  # add utterance data

    # create summary df and save
    df = pd.DataFrame(all_results)
    df = df.fillna(0)  # fill missing speaker columns with 0
    df.to_csv('multi_party_interactions.csv', index=False)
    print("✅ Output saved to multi_party_interactions.csv")
    
    # create utterance-level df and save
    utterance_df = pd.DataFrame(all_utterance_results)
    utterance_df.to_csv('utterance_timing.csv', index=False)
    print("✅ Output saved to utterance_timing.csv")

if __name__ == '__main__':
    main()
