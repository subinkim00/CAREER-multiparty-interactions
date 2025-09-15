import os
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns

# preprocess multi_party_interactions.csv file produced by confirm_and_visualize_multiparty_in_eaf.py
def pre_process_multi_party(multi_party_csv):
    multi_party_df = pd.read_csv(multi_party_csv)
    multi_party_df['filename'] = multi_party_df['filename'].astype(str).str.strip()
    filename_pattern = r'\.eaf$'
    multi_party_df['filename'] = multi_party_df['filename'].apply(lambda x: re.sub(filename_pattern, '', x))
    multi_party_df['duration'] = multi_party_df['interaction_end'] - multi_party_df['interaction_start']
    return multi_party_df

# preprocess annotation file tracker
def pre_process_annotation_file_tracker(annotation_tracker_xlsx):
    annotation_tracker_df = pd.read_excel(annotation_tracker_xlsx)
    annotation_tracker_df = annotation_tracker_df[annotation_tracker_df['Annotator'].notna()]
    annotation_tracker_df = annotation_tracker_df[['Age', 'Recording']].copy()
    annotation_tracker_df['Age'] = pd.to_numeric(annotation_tracker_df['Age'], errors='coerce')
    annotation_tracker_df = annotation_tracker_df.dropna(subset=['Age'])
    return annotation_tracker_df

# using CAREER age bins
def bin_age_months(age_series):
    bins = [-float("inf"), 10, 17, 23, 30, float("inf")]
    labels = ['<10 months', '11-17 months', '18-23 months', '24-30 months', '>30 months']
    return pd.cut(age_series, bins=bins, labels=labels, right=True)

# speaker group regex
def get_speaker_group(speaker):
    speaker = str(speaker) # just making sure :p
    if speaker == "CHI":
        return "Target Child"
    elif re.match(r'^MA\d+$', speaker):
        return "Male Adult"
    elif re.match(r'^FA\d+$', speaker):
        return "Female Adult"
    elif re.match(r'^[A-Z]C\d+$', speaker):
        return "Other Child"
    else:
        return "Other"

# pie charts for age bin composition within all CAREER recordings vs. identified multiparty recordings
def plot_recordings_vs_multiparty(annotation_tracker_df, multi_party_df, output_file):
    multi_party_df = multi_party_df.drop_duplicates(subset='filename')

    # bin ages for both dataframes
    annotation_tracker_df['age_bin'] = bin_age_months(annotation_tracker_df['Age'])
    multi_party_df.loc[:, 'age_bin'] = bin_age_months(multi_party_df['age'])

    # count # recordings per age bin
    tracker_counts = annotation_tracker_df['age_bin'].value_counts().sort_index()
    multi_counts = multi_party_df['age_bin'].value_counts().sort_index()

    # set up subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colors = plt.cm.Paired.colors[:len(tracker_counts)]

    # percentage label formatter
    def func(pct, allvalues):
        absolute = int(round(pct / 100. * sum(allvalues)))
        return f"{pct:.1f}%\n({absolute})"

    # pies 🥧
    wedges1, _, autotexts1 = axes[0].pie(tracker_counts, autopct=lambda pct: func(pct, tracker_counts), startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    _, _, autotexts2 = axes[1].pie(multi_counts, autopct=lambda pct: func(pct, multi_counts), startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    
    axes[0].set_title('All Annotated Recordings')
    axes[1].set_title('Multi-party Recordings')

    fig.legend(wedges1, tracker_counts.index.tolist(), title="Age Group", loc='center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    plt.savefig(output_file)
    plt.close()

# bar chart of average multiparty interaction duration within each age bin
def plot_average_multiparty_inter_duration(multi_party_df, output_file):
    # standardize columns and bin ages
    multi_party_df.columns = multi_party_df.columns.str.strip().str.lower()
    multi_party_df['age_bin'] = bin_age_months(multi_party_df['age'])

    multi_party_df = multi_party_df.groupby('filename', as_index=False).agg({
        'age': 'first',         # assumes age is consistent within recording
        'duration': 'sum'       # total interaction duration per recording
    })
    multi_party_df['age_bin'] = bin_age_months(multi_party_df['age'])

    # group by age bin and compute stats
    grouped = multi_party_df.groupby('age_bin')['duration']
    means = grouped.mean().sort_index()
    stds = grouped.apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x))).sort_index()

    # 📊
    plt.figure(figsize=(10, 6))
    plt.bar(means.index, means.values, yerr=stds.values, capsize=5, color='skyblue', edgecolor='black')
    plt.title('Average Duration of Multi-party Interactions by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Average Duration (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()

# pie charts for adult vs. child participants within age bins
def plot_adult_vs_child_per_agebin(df, output_file):
    # create age_bin column
    df['age_bin'] = bin_age_months(df['age'])

    # get only speaker columns
    metadata_cols = ['filename', 'age', 'age_bin', 'interaction_start', 'interaction_end']
    speaker_cols = [col for col in df.columns if col not in metadata_cols]

    # check for naming conflicts
    if 'speaker_duration' in df.columns:
        raise ValueError("Column name 'speaker_duration' already exists. Please rename or drop it.")

    # melt to long format
    long_df = df.melt(
        id_vars=['filename', 'age', 'age_bin'],
        value_vars=speaker_cols,
        var_name='speaker',
        value_name='speaker_duration'
    )

    # classify speaker
    def classify_speaker(speaker):
        if re.match(r'.*A\d+', speaker):
            return 'Adult'
        elif re.match(r'.*C\d+', speaker):
            return 'Child'
        else:
            return 'Unknown'

    long_df['speaker_type'] = long_df['speaker'].apply(classify_speaker)
    long_df = long_df[long_df['speaker_type'] != 'Unknown']

    # sum durations
    grouped = long_df.groupby(['age_bin', 'speaker_type'])['speaker_duration'].sum().reset_index()

    # pivot to plot
    pivot_df = grouped.pivot(index='age_bin', columns='speaker_type', values='speaker_duration').fillna(0)

    fig, axes = plt.subplots(1, len(pivot_df), figsize=(5 * len(pivot_df), 5))
    if len(pivot_df) == 1:
        axes = [axes]

    for i, (age_bin, row) in enumerate(pivot_df.iterrows()):
        ax = axes[i]
        total = row.sum()
        # percentages = (row / total * 100).round(1) # This line is not used for direct label
        ax.pie(
            row,
            labels=None, # Use None to let autopct handle all labeling
            autopct=lambda pct: f"{pct:.1f}%",
            startangle=90,
            colors=sns.color_palette("pastel"),
        )
        ax.set_title(f"Age {age_bin} mo")

    fig.legend(row.index, loc='lower center', ncol=len(row.index), title='Speaker Type')
    plt.savefig(output_file)
    plt.close()

# don't find this super interesting :p but pie charts for adult gender breakdown within age bins
def plot_adult_gender_breakdown(df, output_file):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re

    # create age_bin column
    df['age_bin'] = bin_age_months(df['age'])

    # split metadata vs. speaker columns
    metadata_cols = ['filename', 'age', 'age_bin', 'interaction_start', 'interaction_end']
    speaker_cols = [col for col in df.columns if col not in metadata_cols]

    # filter adult speaker columns only (FA#, MA#, UA#)
    adult_speaker_cols = [col for col in speaker_cols if re.match(r'(FA|MA|UA)\d+', col)]

    # melt to long format
    long_df = df.melt(
        id_vars=['filename', 'age', 'age_bin'],
        value_vars=adult_speaker_cols,
        var_name='speaker',
        value_name='speaker_duration'
    )

    # classify adult gender
    def classify_adult_gender(speaker):
        if speaker.startswith('FA'):
            return 'Female'
        elif speaker.startswith('MA'):
            return 'Male'
        elif speaker.startswith('UA'):
            return 'Unknown'
        else:
            return 'Other'

    long_df['adult_gender'] = long_df['speaker'].apply(classify_adult_gender)

    # group and sum durations
    grouped = long_df.groupby(['age_bin', 'adult_gender'])['speaker_duration'].sum().reset_index()

    # pivot for plotting
    pivot_df = grouped.pivot(index='age_bin', columns='adult_gender', values='speaker_duration').fillna(0)

    # plot pie charts per age bin
    fig, axes = plt.subplots(1, len(pivot_df), figsize=(5 * len(pivot_df), 5))
    if len(pivot_df) == 1:
        axes = [axes]

    for i, (age_bin, row) in enumerate(pivot_df.iterrows()):
        ax = axes[i]
        total = row.sum()
        # percentages = (row / total * 100).round(1) # This line is not used for direct label
        ax.pie(
            row,
            labels=None, # Use None to let autopct handle all labeling
            autopct=lambda pct: f"{pct:.1f}%",
            startangle=90,
            colors=sns.color_palette("pastel"),
        )
        ax.set_title(f"Age {age_bin} mo")

    fig.legend(row.index, loc='lower center', ncol=len(row.index), title='Adult Gender')
    plt.savefig(output_file)
    plt.close()

# KDE of utterance start times :)
def plot_speaker_density_lines(
    df, 
    include_groups=None, 
    order=None,          
    min_utterances=10,   
    bw_adjust=0.75,      
    clip=(0, 1),         
    palette=None,        
    custom_title=None,   
    save_path='speaker_density_lines.png' 
    ):
    # check utterance df has speaker and norm_start columns
    if not all(col in df.columns for col in ['speaker', 'norm_start']):
        raise ValueError("Input DataFrame must contain 'speaker' and 'norm_start' columns.")
        
    df_copy = df.copy() 
    # apply speaker grouping
    if 'speaker_group' not in df_copy.columns: 
        df_copy['speaker_group'] = df_copy['speaker'].apply(get_speaker_group)
    
    # determine groups to process based on include_groups and what's found
    present_groups = df_copy['speaker_group'].unique()
    if include_groups is None:
        groups_to_process = sorted([g for g in present_groups if g != 'Other'])
        base_filter_description = "All Groups (Excluding 'Other')" 
    elif len(include_groups) == 0:
        groups_to_process = sorted(list(present_groups))
        base_filter_description = "All Groups (Including 'Other')"
    else:
        groups_to_process = sorted([g for g in include_groups if g in present_groups])
        base_filter_description = f"Filtered: {', '.join(groups_to_process)}" if groups_to_process else "Filtered: (None)"
        if not groups_to_process:
            print(f"⚠️ Warning: None of the specified include_groups found in this data subset. No plot generated ({custom_title or save_path}).")
            return None
            
    # determine final order for plotting/legend
    if order:
        plot_order = [g for g in order if g in groups_to_process]
    else:
        default_order_all = ["Target Child", "Other Child", "Female Adult", "Male Adult", "Other"]
        plot_order = [g for g in default_order_all if g in groups_to_process]
        # add any remaining groups_to_process not covered by default order
        for g in groups_to_process:
             if g not in plot_order: plot_order.append(g)
             
    # filter data for included groups and check minimum utterances
    df_plot = df_copy[df_copy['speaker_group'].isin(plot_order)].copy()
    group_counts = df_plot['speaker_group'].value_counts()
    
    groups_to_remove = group_counts[group_counts < min_utterances].index.tolist()
    if groups_to_remove:
        print(f"⚠️ Warning: Removing groups <{min_utterances} utterances: {groups_to_remove} (Plot: {custom_title or save_path})")
        df_plot = df_plot[~df_plot['speaker_group'].isin(groups_to_remove)]
        plot_order = [g for g in plot_order if g not in groups_to_remove] # update plot_order

    if df_plot.empty or not plot_order:
         print(f"⚠️ Warning: No data remaining after filtering (min utterances/groups). No plot generated ({custom_title or save_path}).")
         return None

    # create plot
    print(f"--- Plotting ({custom_title or base_filter_description}) --- Groups: {plot_order}")
    plt.figure(figsize=(10, 5)) 
    the_ax = None # initialize axis variable
    try:
        ax = sns.kdeplot(
            data=df_plot, x='norm_start', hue='speaker_group', hue_order=plot_order, 
            common_norm=False, bw_adjust=bw_adjust, clip=clip, fill=False,           
            linewidth=2, palette=palette       
        )

        for line, label in zip(ax.get_lines(), plot_order):
            if label == "Target Child":
                line.set_linestyle("--")

        the_ax = ax # assign axis if plotting is successful

        # axis labels
        ax.set_xlabel("Normalized Time (0 = start, 1 = end)")
        ax.set_ylabel("Utterance Density")
        
        # Use custom title if provided, otherwise generate default
        if custom_title:
             plot_title = custom_title
        else:
             plot_title = f"Utterance Density Over Interaction Timeline\n(Groups: {base_filter_description})"
        ax.set_title(plot_title)
             
        ax.set_xlim(clip) 
        ax.set_ylim(bottom=0) 
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        print("--- Legend Handling ---") 
        try:
            handles, labels = ax.get_legend_handles_labels()
            print(f"DEBUG (Attempt 1): Found {len(handles)} handles, {len(labels)} labels: {labels}")

            if handles or labels:
                print("DEBUG: Using automatic handles/labels.")
                ax.legend(handles=list(handles), labels=list(labels), 
                          title='Speaker Group', bbox_to_anchor=(1.02, 1), loc='upper left')
                print("DEBUG: Legend created (Auto).")
            elif not (handles or labels) and plot_order: 
                print(f"DEBUG (Attempt 2): Manual legend for groups: {plot_order}")
                lines = ax.get_lines()
                if len(lines) >= len(plot_order):
                    print(f"DEBUG: Found {len(lines)} lines.")
                    manual_handles = lines[:len(plot_order)] 
                    manual_labels = plot_order 
                    ax.legend(handles=manual_handles, labels=manual_labels, 
                              title='Speaker Group', bbox_to_anchor=(1.02, 1), loc='upper left')
                    print("DEBUG: Legend created (Manual).")
                else:
                     print(f"DEBUG: Mismatch lines ({len(lines)}) vs groups ({len(plot_order)}). No manual legend.")
                     if ax.get_legend() is not None: ax.get_legend().remove()
            else: 
                print("DEBUG: No handles/labels/order. No legend.")
                if ax.get_legend() is not None: ax.get_legend().remove()
        except Exception as legend_e:
            print(f"❌ Error during legend creation: {legend_e}")
            if ax.get_legend() is not None: ax.get_legend().remove()
        
        # tight layout
        try:
            plt.tight_layout(rect=[0, 0, 0.80, 1])
            print("DEBUG: Applied tight_layout.")
        except Exception as layout_e:
             print(f"⚠️ Warning: Could not apply tight_layout: {layout_e}")
             
        # save if given filepath, show otherwise
        if save_path:
            final_path = save_path
            output_dir = os.path.dirname(final_path)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)
                 print(f"Created output directory: {output_dir}")
            try:
                plt.savefig(final_path, dpi=300) 
                print(f"✅ Plot saved: {final_path}")
            except Exception as e:
                 print(f"❌ Error saving (path: {final_path}): {e}")
        else:
            plt.show() 

    except Exception as e:
        print(f"\n❌ Error during plotting for ({custom_title or save_path}): {e}")
        # print traceback for unexpected plotting errors
        traceback.print_exc() 
    finally:
        # ensure plot is closed!!
        plt.close() 
        
    return the_ax # return the axis object, None if plot failed

if __name__ == "__main__":
    # filenames
    multi_party_csv = "multi_party_interactions.csv"
    annotation_tracker_xlsx = "Annotation file tracker_CAREER.xlsx"
    utterance_timing_csv = "utterance_timing.csv"

    # df's with helper functions
    multi_party_df = pre_process_multi_party(multi_party_csv)
    annotation_tracker_df = pre_process_annotation_file_tracker(annotation_tracker_xlsx)
    utterance_df = pd.read_csv(utterance_timing_csv)
    age_dist_plot = "total_vs_multiparty_recordings_by_age_bin.png"
    age_dur_plot = "interaction_duration_by_age_bin.png"
    speaker_dist_plot = "speaker_dist.png"
    adult_gender_breakdown_plot = "adult_gender_breakdown.png"
    speaker_timeline_dist_plot = "speaker_timeline_dist.png"
    
    # metadata graphs; uncomment to regenerate :)
    # plot_recordings_vs_multiparty(annotation_tracker_df, multi_party_df, age_dist_plot)
    # plot_average_multiparty_inter_duration(multi_party_df, age_dur_plot)
    # plot_adult_vs_child_per_agebin(multi_party_df, speaker_dist_plot)
    # plot_adult_gender_breakdown(multi_party_df, adult_gender_breakdown_plot)
    # plot_speaker_timeline_distribution(multi_party_df, speaker_timeline_dist_plot)
    # plot_speaker_density_lines(df=utterance_df, save_path=speaker_timeline_dist_plot)

    # density plots by age bin!
    output_dir = "density_plots_by_age"
    
    # Parameters to control the appearance and content of each plot
    plot_params = {
        "include_groups": None, # default: All groups except 'Other' (do [] to include 'Other')
        "bw_adjust": 0.75,    # smoothness control (0.5=detailed, 1.5=smoother).
        "min_utterances": 10, # min utterances for a group line to appear in a specific plot.
        "palette": "tab10",   # color scheme (also played around with "viridis", "Set2", "Paired").
        "order": ["Target Child", "Other Child", "Female Adult", "Male Adult"] # consistent legend order.
    }

    try:
        required_cols = ['speaker', 'norm_start', 'age'] 
        if not all(col in utterance_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in utterance_df.columns]
             raise ValueError(f"CSV file is missing required columns: {missing}")
             
        # convert crucial columns to numeric, coercing errors to NaN
        # age is handled within bin_age_months if needed, but check norm_start
        if not pd.api.types.is_numeric_dtype(utterance_df['norm_start']):
            print(f"Info: 'norm_start' column is not numeric. Attempting conversion...")
            utterance_df['norm_start'] = pd.to_numeric(utterance_df['norm_start'], errors='coerce')
        if utterance_df['norm_start'].isnull().any():
            print(f"Info: Found {utterance_df['norm_start'].isnull().sum()} missing/non-numeric values in 'norm_start'. Related rows will be dropped.")
        # also check 'age' if bin_age_months doesn't handle non-numeric robustly
        if not pd.api.types.is_numeric_dtype(utterance_df['age']):
            print(f"Info: 'age' column is not numeric. bin_age_months will attempt conversion.")
            
        # drop rows with missing essential info AFTER potential conversion
        initial_rows = len(utterance_df)
        utterance_df.dropna(subset=['speaker', 'norm_start', 'age'], inplace=True)
        if len(utterance_df) < initial_rows:
             print(f"Info: Dropped {initial_rows - len(utterance_df)} rows due to missing essential data (speaker, norm_start, age).")

        if utterance_df.empty:
            raise ValueError("DataFrame is empty after handling missing values. Cannot proceed.")

        # apply age binning
        print("\nApplying age binning using bin_age_months function...")
        utterance_df['age_bin'] = bin_age_months(utterance_df['age']) 
        
        # check distribution and handle potential NaNs from binning
        print("\nDistribution across age bins:")
        # display counts for each bin, including potentially empty ones if function returns Categorical
        print(utterance_df['age_bin'].value_counts(dropna=False).sort_index()) 
        unbinned_count = utterance_df['age_bin'].isnull().sum()
        if unbinned_count > 0:
             print(f"⚠️ Warning: {unbinned_count} rows resulted in NaN 'age_bin' (check age values / binning function). Dropping these rows.")
             utterance_df.dropna(subset=['age_bin'], inplace=True) 
             if utterance_df.empty:
                 raise ValueError("DataFrame is empty after dropping rows that failed age binning.")

        # prep for plotting loop
        # get unique bins present in the data, convert category/object to string for reliable iteration/sorting
        unique_age_bins = utterance_df['age_bin'].astype(str).unique() 
        
        # create output directory if it doesn't exist
        if not os.path.exists(output_dir):
             os.makedirs(output_dir)
             print(f"Created output directory: {output_dir}")

        # loop through each age bin and plot
        print("\n--- Generating Plots per Age Bin ---")
        if len(unique_age_bins) == 0:
            print("No valid age bins found in the data after processing.")
            
        # sort the string labels for consistent processing order
        for age_bin_label in sorted(unique_age_bins): 
            
            print(f"\nProcessing Age Bin: {age_bin_label}")
            
            # filter data for the current bin (comparing string representations) (be super safe using .copy())
            df_for_this_bin = utterance_df[utterance_df['age_bin'].astype(str) == age_bin_label].copy()
            
            if df_for_this_bin.empty:
                print("  No data in this bin, skipping plot.")
                continue

            # create filename and title
            # replace characters potentially problematic in filenames
            safe_bin_label = str(age_bin_label).replace('+', 'plus').replace('-', '_').replace(' ', '').replace('<', 'lt').replace('>', 'gt').replace('/','_')
            output_filename = os.path.join(output_dir, f"density_plot_age_{safe_bin_label}.png")
            plot_title = f"Utterance Density - Age Bin: {age_bin_label}"

            # call the plotting function
            plot_speaker_density_lines(
                df=df_for_this_bin,
                custom_title=plot_title,
                save_path=output_filename,
                **plot_params
            )

        print("\n--- Processing Complete ---")

    # basic error handling
    except FileNotFoundError:
        print(f"❌ Error: Utterance timing CSV file not found at '{utterance_timing_csv}'. Please check the path.")
    except ValueError as ve: # specific errors like missing columns or empty df
        print(f"❌ Data Error: {ve}")
    except Exception as e: # any other unexpected errors
        print(f"❌ An unexpected error occurred in the main script: {e}")
        traceback.print_exc() # print detailed traceback for debugging mysterious errors