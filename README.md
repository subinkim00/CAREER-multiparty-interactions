## scripts (under root directory)
### id_potential_multispeaker_recs.py
- USE: iterates through `Annotation file tracker_CAREER.xlsx` to identify recordings where tot_num_speakers > 2
- OUTPUT: `multispeaker_output.xlsx`

### search_download_box_files.py
- USE: facilitate downloading of potential multiparty recordings (`multispeaker_output.xlsx`) from Box
- OUTPUT: downloaded eaf files to `downloaded_eafs`

### confirm_and_visualize_multiparty_in_eaf.py
- USE: iterates through `multispeaker_output.xlsx` to find recordings with multiparty interactions through the following logic:
    1. identify center of interaction by finding a point in time in the recording where the time between the offset of the previous annotation and the onset of the next annotation in non-CHI tiers is < 30 seconds
    2. from there, extend outwards to find the boundaries of the interaction. as we extend backwards and forwards, we've reached an interactional boundary if there is no annotation from ANY participant (including CHI) within 10 seconds from the current annotation
- OUTPUT: interaction plots for each recording are saved to `interaction_plots` to visualize participant turns over the course of the recording. `multi_party_interactions.csv` stores this in csv form

### graphs_analyses_multiparty_recs.py
- USE: given `multi_party_interactions.csv` and `utterance_timing.csv` produced by `confirm_and_visualize_multiparty_in_eaf.py`, produce graphs for basic stats and density plots for main analysis
- OUTPUT: main analysis density plots get saved in the density_plots_by_age directory. other basic stat graphs save to root folder

### stats.py
- USE: run basic descriptive stats for multiparty interaction bouts (overall and by age group) + chi-square goodness-of-fit tests
- OUTPUT: saves to `descriptive_stats.csv` and `statistical_tests_results.csv`

## processed data (under root directory)
### Annotation file tracker_CAREER.xlsx
- metadata file for annotated recordings as of may 1st, 2025

### utterance_timing.csv
- csv file used for generating descriptive stats and analysis graphs

## example_raw_data directory
### eaf file
- example annotated eaf file of publicly-accesible homebank daylong recording ([VanDam-Daylong BN32](https://sla.talkbank.org/TBB/homebank/Public/VanDam-Daylong/BN32/BN32_010007.cha))
### interaction plot
- example interaction plot of above eaf file highlighting potential multiparty interaction bouts in orange and silences in gray

## visualizations
### density_plots_by_age directory
- main analysis density plots produced by `graphs_analyses_multiparty_recs.py`
### metadata_plots directory
- metadata produced by `graphs_analyses_multiparty_recs.py`
