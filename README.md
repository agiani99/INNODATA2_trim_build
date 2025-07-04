# INNODATA2_trim_build
Streamlit apps to trim protacs and rebuild in virtual libraries.

## Optimized Protac_app
Shredder of Protac smiles. It needs an input CSV file with protac smiles. User needs to decide which two bonds be cut and the app generates the fragments.
To make user's lief easier unout CSV is divided in 50 rows chunks which are then saved in relative output CSV files. The user does not have to precisely identify
where warhards or E3binders lay (see below), but only cut the protac proposed using bond_id. Fragments are then collected.

## batch protac mapper.py
This script takes all chunks generated above and tries to identify the class (Warhead or E3binder) of fragments 1 and 3 to generated a mapped CSV file with a 
clever concatenation in one big files to be used in the builder app below. I used evo_ecbd_von16.csv as reference for fragments and MACSS keys for similarity rankings.
The final concatenated csv file is then reordered to have fragment_1 as warhead and _2 for linker.

## protac_streamlit_app.py

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

## protac_conformer_hbond_app.py

This app allows user to upload virtual library designed and after calculations of special mean descriptors on 
50 conformers of each virtual compound, to predict pDC50 of the virtual Protac for ranking. It check also for multiprocessing
for batch calculations. Roughly 33mins for 100 compounds.

