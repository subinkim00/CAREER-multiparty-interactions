import pandas as pd
from scipy import stats
import numpy as np
import math
import re

# using CAREER age bins
def bin_age_months(age_series):
    numeric_age = pd.to_numeric(age_series, errors='coerce') 
    bins = [-float("inf"), 10, 17, 23, 30, float("inf")]
    labels = ['<10 months', '11-17 months', '18-23 months', '24-30 months', '>30 months']
    binned_data = pd.cut(numeric_age, bins=bins, labels=labels, right=True)
    return binned_data

# calculate mean, median, mode, std dev, SEM, range, num multiparty bouts
def calculate_descriptives_by_group(data_series, group_name):
    if data_series.empty or data_series.isnull().all():
        print(f"Warning: No valid data for '{group_name}' to calculate descriptives.")
        stats_dict = {
            'Mean': np.nan, 'Median': np.nan, 'Mode': np.nan, 
            'Std Dev': np.nan, 'SEM': np.nan, 
            'Range (Min)': np.nan, 'Range (Max)': np.nan,
            'N_bouts': 0
        }
        return stats_dict

    # if multiple modes, it returns the smallest; if series is all NaN or empty after dropping NaNs, mode calculation can fail.
    mode_result = stats.mode(data_series.dropna(), keepdims=False)
    mode_val = mode_result.mode if mode_result.count > 0 else np.nan

    stats_dict = {
        'Mean': data_series.mean(),
        'Median': data_series.median(),
        'Mode': mode_val,
        'Std Dev': data_series.std(),
        'SEM': stats.sem(data_series, nan_policy='omit'),
        'Range (Min)': data_series.min(),
        'Range (Max)': data_series.max(),
        'N_bouts': data_series.count() # count non-NaN values
    }
    return stats_dict

# using above helper function, report interaction descriptives by age bin
def report_interaction_descriptives_by_age(df):
    # basic data validation
    required_cols = ['interaction_id', 'speaker', 'interaction_start', 'interaction_end', 'age', 'filename']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: CSV is missing required columns: {missing}")
        return None, None # Return None for both summary and recordings count

    # convert relevant columns to numeric, just in case :p
    for col in ['interaction_start', 'interaction_end', 'age']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Info: Column '{col}' is not numeric. Attempting conversion.")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # drop rows where essential data is missing (should be okay, though)
    df.dropna(subset=['interaction_id', 'speaker', 'interaction_start', 'interaction_end', 'age', 'filename'], inplace=True)
    if df.empty:
        print("Error: DataFrame is empty after removing rows with missing essential data.")
        return None, None
        
    print(f"Processing {df['interaction_id'].nunique()} unique interaction bouts.")

    # age binning
    df['age_bin'] = bin_age_months(df['age'])
    unbinned_count = df['age_bin'].isnull().sum()
    if unbinned_count > 0:
        print(f"Warning: {unbinned_count} rows had ages that didn't fall into defined bins. These will be excluded from age-binned stats.")
        # (will be kept for overall stats but they won't appear in age_bin groups)

    # calculate per-interaction statistics
    # group by interaction_id first to get one value per interaction
    # then, we can merge the age_bin associated with that interaction
    
    # INTERACTION BOUT DURATION
    # calculate duration for each unique interaction_id
    interaction_summary = df.groupby('interaction_id').agg(
        interaction_start=('interaction_start', 'first'),
        interaction_end=('interaction_end', 'first'),
        age_bin=('age_bin', 'first'), # Get age_bin for interaction
        num_speakers=('speaker', 'nunique'),
        num_turns=('speaker', 'count'), # speaker 'count' is equivalent to .size() for turns
        filename=('filename', 'first') # Get filename for distinct recording count
    ).reset_index()

    interaction_summary['duration'] = interaction_summary['interaction_end'] - interaction_summary['interaction_start']
    
    # filter out invalid durations
    interaction_summary = interaction_summary[interaction_summary['duration'] >= 0]
    if interaction_summary.empty:
        print("Warning: No valid interaction bouts found after duration calculation.")
        return None, None

    # DESCRIPTIVE STATS
    all_stats_dfs = []
    
    # define groups to iterate over: overall + each age bin
    # Ensure 'Overall' is a string and age bins are strings for consistent comparison
    age_groups_to_analyze = ['Overall'] + sorted([str(b) for b in interaction_summary['age_bin'].dropna().unique()])

    # This will store the N_recordings for each age bin
    n_recordings_data = []

    for group_label in age_groups_to_analyze:
        print(f"\n--- Statistics for: {group_label} ---")
        
        if group_label == 'Overall':
            current_data = interaction_summary
        else:
            # filter for current age_bin (need to handle if 'age_bin' is categorical)
            current_data = interaction_summary[interaction_summary['age_bin'].astype(str) == group_label]

        if current_data.empty:
            print(f"  No data for group: {group_label}")
            n_recordings_data.append({'Age Bin': group_label, 'N_recordings': 0})
            continue

        # Calculate distinct recordings for the current group
        n_distinct_recordings = current_data['filename'].nunique()
        n_recordings_data.append({'Age Bin': group_label, 'N_recordings': n_distinct_recordings})

        descriptives = {}
        descriptives['Bout Duration (seconds)'] = calculate_descriptives_by_group(current_data['duration'], f"{group_label} - Duration")
        descriptives['Number of Speakers per Bout'] = calculate_descriptives_by_group(current_data['num_speakers'], f"{group_label} - Speakers")
        descriptives['Number of Turns per Bout'] = calculate_descriptives_by_group(current_data['num_turns'], f"{group_label} - Turns")
        
        group_summary_df = pd.DataFrame(descriptives).T
        group_summary_df['Age Bin'] = group_label # Changed from 'Group' to 'Age Bin'
        all_stats_dfs.append(group_summary_df)

    if not all_stats_dfs:
        print("No statistics were calculated for any group.")
        return None, None

    # combine all summary dfs
    final_summary_df = pd.concat(all_stats_dfs)
    final_summary_df = final_summary_df.reset_index().rename(columns={'index': 'Measure'})
    
    # Merge N_recordings into the final_summary_df
    n_recordings_df = pd.DataFrame(n_recordings_data)
    
    # Ensure 'Age Bin' in final_summary_df is of consistent type with n_recordings_df for merging
    final_summary_df['Age Bin'] = final_summary_df['Age Bin'].astype(str)

    # We need to reshape final_summary_df before merging to have 'Age Bin' as a primary key
    # Pivot the DataFrame to have measures as columns for each age bin
    reshaped_df = final_summary_df.pivot_table(index='Age Bin', columns='Measure', values=['Mean', 'Median', 'Mode', 'Std Dev', 'SEM', 'Range (Min)', 'Range (Max)', 'N_bouts'])
    
    # Flatten the multi-index columns
    reshaped_df.columns = ['_'.join(col).strip() for col in reshaped_df.columns.values]
    reshaped_df = reshaped_df.reset_index()

    # Now merge with N_recordings_df
    final_descriptive_stats_output = pd.merge(n_recordings_df, reshaped_df, on='Age Bin', how='left')

    # Reorder columns to have Age Bin, N_recordings first
    # Dynamic column ordering for measures
    measure_cols = [col for col in final_descriptive_stats_output.columns if col not in ['Age Bin', 'N_recordings']]
    final_descriptive_stats_output = final_descriptive_stats_output[['Age Bin', 'N_recordings'] + measure_cols]
    
    # display all cols
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n\n--- Combined Descriptive Statistics Summary (Overall and by Age Group) ---")
    print(final_descriptive_stats_output.round(2).to_string()) # Use .to_string() for full display in console

    return final_descriptive_stats_output, n_recordings_df # Return the modified DataFrame for saving

# helper: get speaker groups (FA, MA, XC, CHI)
def get_speaker_group(speaker):
    speaker = str(speaker)
    if re.match(r'^FA\d+$', speaker): return "Female Adult"
    elif re.match(r'^MA\d+$', speaker): return "Male Adult"
    elif re.match(r'^[A-Z]C\d+$', speaker): return "Other Child"
    elif speaker == "CHI": return "Target Child"
    else: return "Other"

#################################################################################
# CHI-SQUARED GOODNESS-OF-FIT TEST
# see if a specific speaker group's utterance distribution across time bins is
# significantly different from uniform.
# args:
#   df (pd.DataFrame): df with 'speaker' and 'norm_start'
#   speaker_group_to_test (str): name of the speaker group to test (e.g., "Male Adult")
#   bins (int): number of time bins
#   alpha (float): significance level
# returns:
#   tuple: (chi2_stat, p_value, interpretation) if the test is successful, otherwise (0, 1, message)
#################################################################################
def test_speaker_group_temporal_pattern(
    df,
    speaker_group_to_test,
    bins=20,
    alpha=0.05
    ):
    # This function expects 'age_bin' and 'norm_start' to be present if testing by age group
    if not all(col in df.columns for col in ['speaker', 'norm_start']):
        raise ValueError("Input DataFrame must contain 'speaker' and 'norm_start' columns.")

    # No need to print "--- Testing Temporal Pattern for Speaker Group" here as it's handled by main loop
    
    df_copy = df.copy() 
    df_copy['speaker_group'] = df_copy['speaker'].apply(get_speaker_group)
    
    # isolate Data for the specific group
    df_group = df_copy[df_copy['speaker_group'] == speaker_group_to_test]
    
    n_total = len(df_group)
    # print(f"Total utterances for {speaker_group_to_test}: {n_total}") # Removed from here, handled in main loop
    
    result_interpretation = ""
    if n_total == 0:
        # print("  No utterances found for this group. Cannot perform test.") # Removed from here
        result_interpretation = "No utterances found for this group."
        return (np.nan, np.nan, result_interpretation) # Use np.nan for stats when no data
        
    # bin time and get observed counts
    time_bins_def = np.linspace(0, 1, bins + 1)
    time_bins_def[0] = -0.001 
    
    # create all possible bin labels using the definition
    all_bin_labels = pd.IntervalIndex.from_breaks(time_bins_def, closed='right')

    # assign bins to filtered data
    df_group['time_bin'] = pd.cut(df_group['norm_start'], bins=time_bins_def, right=True)
    
    # count occurrences in each bin defined by pd.cut
    observed_counts_series = df_group['time_bin'].value_counts()
    
    # ensure all bins are present, filling missing ones with 0 counts
    observed_counts_series = observed_counts_series.reindex(all_bin_labels, fill_value=0)
    
    # convert to numpy array for chisquare function
    observed_counts = observed_counts_series.values

    # check if all counts are in a single bin (or zero bins have counts)
    if np.sum(observed_counts > 0) <= 1:
         # print(f"  Warning: All observed utterances fall into <= 1 bin. Goodness-of-fit test is not meaningful here.") # Removed from here
         result_interpretation = "All observed utterances fall into <= 1 bin; test not meaningful."
         return (np.nan, np.nan, result_interpretation) # Use np.nan for stats when not meaningful

    # print(f"\nObserved Counts across {bins} bins:") # Removed from here
    # print(observed_counts_series.to_string())

    try:
        # Ensure that there are enough degrees of freedom (k-1 > 0 where k is number of non-zero bins)
        if (observed_counts > 0).sum() <= 1:
            result_interpretation = "Insufficient data points across bins for a meaningful chi-squared test."
            return (np.nan, np.nan, result_interpretation)

        chi2_stat, p_value = stats.chisquare(f_obs=observed_counts)
        
        # print(f"\nChi-Squared Goodness-of-Fit Results:") # Removed from here
        # print(f"  Chi-Squared Statistic (χ²): {chi2_stat:.4f}")
        # print(f"  Degrees of Freedom (dof): {len(observed_counts) - 1}") 
        # print(f"  P-value: {p_value:.4g}")
        # print(f"  Significance Level (α): {alpha}")

        if p_value < alpha:
            result_interpretation = f"Significant deviation from uniform (p < {alpha})."
        else:
            result_interpretation = f"No significant deviation from uniform (p >= {alpha})."
            
        return chi2_stat, p_value, result_interpretation

    except ValueError as e:
        # print(f"\n❌ Error during Chi-Squared Goodness-of-Fit test: {e}") # Removed from here
        result_interpretation = f"Error during test: {e}"
        return np.nan, np.nan, result_interpretation

#################################################################################
# tests if a speaker group's utterance distribution differs across defined epochs
# roughly doing early, middle, late
#################################################################################
def test_pattern_by_epoch(df, speaker_group_to_test, epoch_boundaries=[0, 0.3, 0.7, 1.0], alpha=0.05):
    if len(epoch_boundaries) != 4 or not all(0 <= x <= 1 for x in epoch_boundaries):
        raise ValueError("epoch_boundaries must be a list of 4 values between 0 and 1, e.g., [0, 0.2, 0.8, 1.0]")
        
    epoch_labels = ['Beginning', 'Middle', 'End']
    bins = epoch_boundaries

    # print(f"\n--- Testing Epoch Pattern for: {speaker_group_to_test} ---") # Removed from here
    # print(f"Epochs: Beginning ({bins[0]:.2f}-{bins[1]:.2f}), Middle ({bins[1]:.2f}-{bins[2]:.2f}), End ({bins[2]:.2f}-{bins[3]:.2f})")

    df_copy = df.copy()
    df_copy['speaker_group'] = df_copy['speaker'].apply(get_speaker_group)
    df_group = df_copy[df_copy['speaker_group'] == speaker_group_to_test]

    n_total = len(df_group)
    # print(f"Total utterances: {n_total}") # Removed from here

    result_interpretation = ""
    if n_total == 0:
        # print("No utterances, cannot test.") # Removed from here
        result_interpretation = "No utterances found for this group."
        return np.nan, np.nan, result_interpretation

    # assign epochs (use right=False for first bin to include 0, but handle edge case for 1.0)
    # ensure labels match the number of intervals defined by bins
    df_group['epoch'] = pd.cut(df_group['norm_start'], bins=bins, labels=epoch_labels, right=False, include_lowest=True)
    # handle exact 1.0 case if right=False
    if df_group['norm_start'].max() == 1.0:
         df_group.loc[df_group['norm_start'] == 1.0, 'epoch'] = epoch_labels[-1]

    # get observed counts, ensuring all epoch labels are present
    observed_counts = df_group['epoch'].value_counts().reindex(epoch_labels, fill_value=0).values

    # print(f"Observed Counts: Beginning={observed_counts[0]}, Middle={observed_counts[1]}, End={observed_counts[2]}") # Removed from here

    if sum(observed_counts > 0) <= 1:
        # print("Warning: All utterances fall into <= 1 epoch. Test not meaningful.") # Removed from here
        result_interpretation = "All utterances fall into <= 1 epoch; test not meaningful."
        return np.nan, np.nan, result_interpretation # Not significant, use nan for stats

    # test goodness-of-fit against uniform distribution across epochs
    try:
        # Ensure that there are enough degrees of freedom (k-1 > 0 where k is number of non-zero bins)
        if (observed_counts > 0).sum() <= 1:
            result_interpretation = "Insufficient data points across epochs for a meaningful chi-squared test."
            return (np.nan, np.nan, result_interpretation)

        chi2_stat, p_value = stats.chisquare(f_obs=observed_counts) # default compares to uniform

        # print(f"Chi-Squared Goodness-of-Fit vs Uniform Epoch Distribution:") # Removed from here
        # print(f"  Chi2 Stat: {chi2_stat:.4f}, P-value: {p_value:.4g}")
        
        if p_value < alpha:
            result_interpretation = f"Significant deviation from uniform (p < {alpha}). "
            # print("  Result: Significant deviation from uniform distribution across epochs.")
            if (observed_counts[0] > observed_counts[2]) and  (observed_counts[0]> observed_counts[1]):
                result_interpretation += "Pattern Suggests: More utterances at the beginning."
                # print("  Pattern Suggests: More utterances at the beginning than middle/end.")
            elif (observed_counts[1] > observed_counts[0]) and  (observed_counts[1] > observed_counts[2]):
                result_interpretation += "Pattern Suggests: More utterances in the middle."
                # print("  Pattern Suggests: More utterances in the middle than beginning/end.")
            elif (observed_counts[2] > observed_counts[0]) and (observed_counts[2] >  observed_counts[1]):
                result_interpretation += "Pattern Suggests: More utterances in the end."
                # print("  Pattern Suggests: More utterances in the end than beginning/middle.")
            else:
                result_interpretation += "Pattern Suggests: Other deviation from uniform."
                # print("  Pattern Suggests: Other deviation from uniform.")
                
        else:
            result_interpretation = f"No significant deviation from uniform (p >= {alpha})."
            # print("  Result: No significant evidence that distribution across epochs is non-uniform.")
        
        return chi2_stat, p_value, result_interpretation
    except ValueError as e:
        result_interpretation = f"Error during test: {e}"
        return np.nan, np.nan, result_interpretation

if __name__ == "__main__":
    utterance_df = pd.read_csv("utterance_timing.csv")

    # Add age_bin to the main DataFrame for filtering during statistical tests
    utterance_df['age_bin'] = bin_age_months(utterance_df['age'])

    final_descriptive_stats_output, _ = report_interaction_descriptives_by_age(utterance_df)

    if final_descriptive_stats_output is not None:
        print("\nFull descriptive statistics table generated. Saving to 'descriptive_stats.csv'.")
        final_descriptive_stats_output.round(2).to_csv("descriptive_stats.csv", index=False)
    else:
        print("\nFailed to generate summary statistics. 'descriptive_stats.csv' will not be created.")

    ################
    #  stat tests by age group
    ################
    speaker_groups_to_analyze = ["Female Adult", "Male Adult", "Target Child", "Other Child"] # Specify groups to test
    
    # Get all unique age bins, including 'Overall' for a general test
    all_age_bins = ['Overall'] + sorted([str(b) for b in utterance_df['age_bin'].dropna().unique()])
    
    all_test_results = []

    for age_group_label in all_age_bins:
        print(f"\n======== Running Statistical Tests for Age Group: {age_group_label} ========")

        # Filter the DataFrame for the current age group
        if age_group_label == 'Overall':
            current_age_group_df = utterance_df
        else:
            # Ensure age_bin is treated as string for comparison
            current_age_group_df = utterance_df[utterance_df['age_bin'].astype(str) == age_group_label]
        
        if current_age_group_df.empty:
            print(f"No data for age group '{age_group_label}'. Skipping statistical tests.")
            for group in speaker_groups_to_analyze:
                all_test_results.append({
                    'Age Bin': age_group_label,
                    'Test': 'Chi-Squared Goodness-of-Fit (Temporal Bins)',
                    'Speaker Group': group,
                    'Chi2 Statistic': np.nan,
                    'P-value': np.nan,
                    'Significance Level': 0.05,
                    'Interpretation': "No data for this age group."
                })
                all_test_results.append({
                    'Age Bin': age_group_label,
                    'Test': 'Chi-Squared Goodness-of-Fit (Epochs)',
                    'Speaker Group': group,
                    'Chi2 Statistic': np.nan,
                    'P-value': np.nan,
                    'Significance Level': 0.05,
                    'Interpretation': "No data for this age group."
                })
            continue

        # --- Chi-squared temporal pattern test for each speaker group within the current age group ---
        for group in speaker_groups_to_analyze:
            print(f"\n--- Temporal Pattern Test for {group} in {age_group_label} ---")
            chi2_stat, p_val, interpretation = test_speaker_group_temporal_pattern(
                df=current_age_group_df,
                speaker_group_to_test=group,
                bins=10,
                alpha=0.05
            )
            print(f"  Result: Chi2 Stat={chi2_stat:.4f} (or NaN), P-value={p_val:.4g} (or NaN) - {interpretation}")

            all_test_results.append({
                'Age Bin': age_group_label,
                'Test': 'Chi-Squared Goodness-of-Fit (Temporal Bins)',
                'Speaker Group': group,
                'Chi2 Statistic': chi2_stat,
                'P-value': p_val,
                'Significance Level': 0.05,
                'Interpretation': interpretation
            })

        # --- Chi-squared epoch pattern test for each speaker group within the current age group ---
        for group in speaker_groups_to_analyze:
            print(f"\n--- Epoch Pattern Test for {group} in {age_group_label} ---")
            chi2_stat, p_val, interpretation = test_pattern_by_epoch(
                df=current_age_group_df,
                speaker_group_to_test=group,
                epoch_boundaries=[0, 0.3, 0.6, 1.0],
                alpha=0.05
            )
            print(f"  Result: Chi2 Stat={chi2_stat:.4f} (or NaN), P-value={p_val:.4g} (or NaN) - {interpretation}")

            all_test_results.append({
                'Age Bin': age_group_label,
                'Test': 'Chi-Squared Goodness-of-Fit (Epochs)',
                'Speaker Group': group,
                'Chi2 Statistic': chi2_stat,
                'P-value': p_val,
                'Significance Level': 0.05,
                'Interpretation': interpretation
            })

    # Save all statistical test results to CSV
    statistical_test_results_df = pd.DataFrame(all_test_results)
    if not statistical_test_results_df.empty:
        print("\nSaving all statistical test results to 'statistical_test_results.csv'.")
        # Ensure numerical columns are rounded for cleaner output
        for col in ['Chi2 Statistic', 'P-value', 'Significance Level']:
            if col in statistical_test_results_df.columns:
                statistical_test_results_df[col] = statistical_test_results_df[col].round(4)
        statistical_test_results_df.to_csv("statistical_test_results.csv", index=False)
    else:
        print("\nNo statistical test results to save.")