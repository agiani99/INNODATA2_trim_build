# INNODATA2_trim_build
Streamlit apps to trim protacs and rebuild in virtual libraries.

## Optimized Protac_app
Shredder of Protac smiles. It needs an input CSV file with protac smiles. User needs to decide which two bonds be cut and the app generates the fragments.
To make user's lief easier unout CSV is divided in 50 rows chunks which are then saved in relative output CSV files

## 


## protac_streamlit_app.py to build a user-defined virtual library of Protacs

Target-Specific Fragment Selection

When user selects a compound, the app identifies the target protein (e.g., ER, AKT1)
E3 binders & Warheads: Only uses fragments from compounds targeting the SAME protein
Linkers: Uses ALL available linkers from the entire dataset (as requested)

Smart Combinatorial Strategy

For ER (many compounds): Uses only ER-specific warheads/E3binders → drastically reduces combinations
For AKT1 (few compounds): Uses AKT1-specific fragments → manageable library size
Formula: Target_E3binders × All_Linkers × Target_Warheads

Enhanced User Interface

Clear target identification in the interface
Shows which target was selected (e.g., "🎯 Selected Target: ER")
Updated help text to clarify the new logic
Better labeling (Reference fragments vs Virtual library)

Practical Benefits

Reduces candidate explosion: Instead of 34,314 candidates, you'll get target-relevant libraries
Maintains chemical relevance: Only combines fragments known to work for the same target
Explores linker diversity: Still allows exploration of different linkers across all PROTACs
Handles edge cases: Works for both high-frequency targets (ER) and low-frequency ones (AKT1)
