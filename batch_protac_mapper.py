import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
import warnings
import os
import glob
from pathlib import Path
warnings.filterwarnings('ignore')

def calculate_tanimoto_similarity(smiles1, smiles2):
    """
    Calculate Tanimoto similarity between two SMILES using RDKit MACCS keys
    
    Args:
        smiles1 (str): First SMILES string
        smiles2 (str): Second SMILES string
    
    Returns:
        float: Tanimoto similarity score (0-1), or 0 if molecules can't be processed
    """
    try:
        # Convert SMILES to molecules
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Generate MACCS keys
        fp1 = rdMolDescriptors.GetMACCSKeysFingerprint(mol1)
        fp2 = rdMolDescriptors.GetMACCSKeysFingerprint(mol2)
        
        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return similarity
    
    except Exception as e:
        print(f"Error calculating similarity for {smiles1} vs {smiles2}: {e}")
        return 0.0

def find_most_similar_compound(query_smiles, reference_df):
    """
    Find the most similar compound in reference dataframe using Tanimoto similarity
    
    Args:
        query_smiles (str): Query SMILES string
        reference_df (pd.DataFrame): Reference dataframe with SMILES and TYPE columns
    
    Returns:
        tuple: (max_similarity, corresponding_type)
    """
    max_similarity = 0.0
    corresponding_type = None
    
    for _, row in reference_df.iterrows():
        similarity = calculate_tanimoto_similarity(query_smiles, row['SMILES'])
        if similarity > max_similarity:
            max_similarity = similarity
            corresponding_type = row['TYPE']
    
    return max_similarity, corresponding_type

def count_deuterium(smiles):
    """
    Count the number of deuterium atoms ([2H]) in a SMILES string
    
    Args:
        smiles (str): SMILES string
    
    Returns:
        int: Number of deuterium atoms
    """
    if not smiles or not isinstance(smiles, str):
        return 0
    import re
    matches = re.findall(r'\[2H\]', smiles)
    return len(matches)

def identify_single_deuterium_columns(fragments_df):
    """
    Identify which fragment columns have exactly one deuterium atom consistently
    
    Args:
        fragments_df (pd.DataFrame): Fragment dataframe
    
    Returns:
        list: List of column names that have exactly 1 deuterium consistently
    """
    fragment_columns = ['Fragment_1_SMILES', 'Fragment_2_SMILES', 'Fragment_3_SMILES']
    single_deuterium_columns = []
    
    print("  Analyzing deuterium patterns...")
    
    for col in fragment_columns:
        if col in fragments_df.columns:
            deuterium_counts = fragments_df[col].apply(count_deuterium)
            unique_counts = deuterium_counts.unique()
            
            print(f"    {col}: Unique deuterium counts = {sorted(unique_counts)}")
            
            # Check if all values are exactly 1
            if len(unique_counts) == 1 and unique_counts[0] == 1:
                single_deuterium_columns.append(col)
                print(f"    ✓ {col} has exactly 1 deuterium consistently")
            else:
                print(f"    ✗ {col} does not have consistent single deuterium")
    
    return single_deuterium_columns

def assign_fragment_types(fragments_df, reference_df, chunk_num):
    """
    Assign types to fragment columns with single deuterium based on similarity
    
    Args:
        fragments_df (pd.DataFrame): Fragment dataframe
        reference_df (pd.DataFrame): Reference dataframe filtered for E3 Binder and Warhead
        chunk_num (int): Chunk number for progress reporting
    
    Returns:
        pd.DataFrame: Updated dataframe with TYPE columns
    """
    result_df = fragments_df.copy()
    
    # Identify columns with exactly one deuterium
    single_d_columns = identify_single_deuterium_columns(fragments_df)
    
    if len(single_d_columns) != 2:
        raise ValueError(f"Chunk {chunk_num}: Expected exactly 2 columns with single deuterium, found {len(single_d_columns)}: {single_d_columns}")
    
    print(f"  Using columns for similarity mapping: {single_d_columns}")
    
    # Use the second column for similarity search (arbitrary choice)
    query_column = single_d_columns[1]
    reference_column = single_d_columns[0]
    
    type_lists = {single_d_columns[0]: [], single_d_columns[1]: []}
    
    print(f"  Processing {len(fragments_df)} fragments...")
    
    for idx, row in fragments_df.iterrows():
        if (idx + 1) % 10 == 0:  # Progress update every 10 rows
            print(f"    Processing row {idx + 1}/{len(fragments_df)}")
        
        # Find most similar compound for the query column
        query_smiles = row[query_column]
        max_sim, found_type = find_most_similar_compound(query_smiles, reference_df)
        
        # Assign the found type to query column and opposite to reference column
        if found_type == "Warhead":
            query_type = "Warhead"
            reference_type = "E3 Binder"
        elif found_type == "E3 Binder":
            query_type = "E3 Binder"
            reference_type = "Warhead"
        else:
            # Fallback if no match found
            query_type = "Unknown"
            reference_type = "Unknown"
        
        type_lists[query_column].append(query_type)
        type_lists[reference_column].append(reference_type)
    
    # Add type columns to result dataframe
    for col in single_d_columns:
        type_col_name = col.replace('_SMILES', '_TYPE')
        result_df[type_col_name] = type_lists[col]
    
    return result_df, single_d_columns

def standardize_fragment_assignments(df, single_d_columns):
    """
    Standardize fragment assignments globally:
    - Fragment_1 should always be Warhead
    - Fragment_3 should always be E3 Binder
    
    Args:
        df (pd.DataFrame): Dataframe with TYPE columns
        single_d_columns (list): List of fragment column names with single deuterium
    
    Returns:
        pd.DataFrame: Standardized dataframe with consistent fragment assignments
    """
    result_df = df.copy()
    
    # Get type column names
    type_columns = [col.replace('_SMILES', '_TYPE') for col in single_d_columns]
    
    print(f"  Standardizing fragment assignments for columns: {single_d_columns}")
    print(f"  Corresponding type columns: {type_columns}")
    
    # Count current distributions
    for i, type_col in enumerate(type_columns):
        if type_col in result_df.columns:
            type_counts = result_df[type_col].value_counts()
            print(f"    {type_col} current distribution: {dict(type_counts)}")
    
    # Determine target assignments based on Fragment column names
    # We want Fragment_1 = Warhead, Fragment_3 = E3 Binder
    target_assignments = {}
    
    for i, smiles_col in enumerate(single_d_columns):
        type_col = type_columns[i]
        
        if 'Fragment_1' in smiles_col:
            target_assignments[smiles_col] = 'Warhead'
            target_assignments[type_col] = 'Warhead'
        elif 'Fragment_3' in smiles_col:
            target_assignments[smiles_col] = 'E3 Binder'
            target_assignments[type_col] = 'E3 Binder'
        else:
            # Default fallback - shouldn't happen with Fragment_1 and Fragment_3
            target_assignments[smiles_col] = 'Unknown'
            target_assignments[type_col] = 'Unknown'
    
    print(f"  Target assignments: {target_assignments}")
    
    # Check if we need to swap to achieve target assignments
    swaps_needed = []
    
    for i, type_col in enumerate(type_columns):
        if type_col not in result_df.columns:
            continue
            
        target_type = target_assignments.get(type_col, 'Unknown')
        
        # Count how many rows already have the correct assignment
        current_correct = (result_df[type_col] == target_type).sum()
        current_incorrect = (result_df[type_col] != target_type).sum()
        
        print(f"    {type_col}: {current_correct} correct ({target_type}), {current_incorrect} incorrect")
        
        # If more rows are incorrect than correct, we should swap this column
        if current_incorrect > current_correct:
            swaps_needed.append(i)
    
    # Perform swaps if needed
    if swaps_needed:
        print(f"  Swapping columns to achieve target assignments...")
        
        # Swap between the two single deuterium columns
        smiles_col_1, smiles_col_2 = single_d_columns[0], single_d_columns[1]
        type_col_1, type_col_2 = type_columns[0], type_columns[1]
        
        # Find rows that need swapping based on the first column that needs swapping
        swap_col_idx = swaps_needed[0]
        type_col_to_check = type_columns[swap_col_idx]
        target_type = target_assignments[type_col_to_check]
        
        # Swap rows where the type doesn't match the target
        swap_mask = result_df[type_col_to_check] != target_type
        
        if swap_mask.any():
            print(f"    Swapping {swap_mask.sum()} rows to standardize assignments")
            
            # Swap SMILES columns
            temp_smiles = result_df.loc[swap_mask, smiles_col_1].copy()
            result_df.loc[swap_mask, smiles_col_1] = result_df.loc[swap_mask, smiles_col_2]
            result_df.loc[swap_mask, smiles_col_2] = temp_smiles
            
            # Swap TYPE columns
            temp_types = result_df.loc[swap_mask, type_col_1].copy()
            result_df.loc[swap_mask, type_col_1] = result_df.loc[swap_mask, type_col_2]
            result_df.loc[swap_mask, type_col_2] = temp_types
    
    # Verify final assignments
    print(f"  Final distributions after standardization:")
    for type_col in type_columns:
        if type_col in result_df.columns:
            type_counts = result_df[type_col].value_counts()
            print(f"    {type_col}: {dict(type_counts)}")
    
    return result_df

def process_single_chunk(chunk_file, reference_df, output_dir="output"):
    """
    Process a single chunk file
    
    Args:
        chunk_file (str): Path to chunk file
        reference_df (pd.DataFrame): Reference dataframe
        output_dir (str): Output directory
    
    Returns:
        pd.DataFrame: Processed dataframe with types assigned
    """
    # Extract chunk number from filename
    chunk_num = int(chunk_file.split('_')[-1].split('.')[0])
    
    print(f"\n{'='*60}")
    print(f"Processing Chunk {chunk_num}: {chunk_file}")
    print(f"{'='*60}")
    
    try:
        # Load fragment data
        print(f"Loading fragment data from {chunk_file}...")
        fragments_df = pd.read_csv(chunk_file)
        print(f"  Loaded {len(fragments_df)} fragment records")
        
        # Assign fragment types based on similarity
        print(f"Assigning fragment types...")
        result_df, single_d_columns = assign_fragment_types(fragments_df, reference_df, chunk_num)
        
        # Standardize fragment assignments (replaces balance_fragment_types)
        print(f"Standardizing fragment assignments globally...")
        standardized_df = standardize_fragment_assignments(result_df, single_d_columns)
        
        # Get type column names for final reporting
        type_columns = [col.replace('_SMILES', '_TYPE') for col in single_d_columns]
        
        # Display final results
        print(f"Final type distributions:")
        for type_col in type_columns:
            if type_col in standardized_df.columns:
                distribution = standardized_df[type_col].value_counts()
                print(f"  {type_col}: {dict(distribution)}")
        
        # Save individual results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'protac_fragments_with_types_{chunk_num}.csv')
        standardized_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        return standardized_df
        
    except Exception as e:
        print(f"ERROR processing chunk {chunk_num}: {str(e)}")
        return None

def main():
    """
    Main function to process all PROTAC fragment chunks and assign types
    """
    print("Batch PROTAC Fragment Type Mapping Script")
    print("=" * 60)
    
    # Load reference data once
    print("Loading reference data...")
    reference_file = 'evo_ecbd_nov16.csv'
    
    if not os.path.exists(reference_file):
        print(f"ERROR: Reference file '{reference_file}' not found!")
        print("Please ensure the file is in the current directory.")
        return
    
    reference_df = pd.read_csv(reference_file)
    print(f"Loaded {len(reference_df)} reference records")
    
    # Filter reference data for E3 Binder and Warhead only
    filtered_reference = reference_df[reference_df['TYPE'].isin(['E3 Binder', 'Warhead'])].copy()
    print(f"Filtered to {len(filtered_reference)} E3 Binder and Warhead records")
    
    type_distribution = filtered_reference['TYPE'].value_counts()
    print(f"Type distribution: {dict(type_distribution)}")
    
    # Find all chunk files
    chunk_pattern = 'protac_fragments_chunk_*.csv'
    chunk_files = sorted(glob.glob(chunk_pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not chunk_files:
        print(f"ERROR: No files found matching pattern '{chunk_pattern}'")
        print("Please ensure your chunk files are in the current directory.")
        return
    
    print(f"\nFound {len(chunk_files)} chunk files:")
    for f in chunk_files:
        print(f"  {f}")
    
    # Process all chunks
    print(f"\nStarting batch processing...")
    all_results = []
    successful_chunks = []
    failed_chunks = []
    
    for chunk_file in chunk_files:
        result_df = process_single_chunk(chunk_file, filtered_reference)
        
        if result_df is not None:
            all_results.append(result_df)
            successful_chunks.append(chunk_file)
        else:
            failed_chunks.append(chunk_file)
    
    # Concatenate all results and perform final global standardization
    if all_results:
        print(f"\n{'='*60}")
        print("Creating concatenated results with global standardization...")
        print(f"{'='*60}")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Perform final global standardization to ensure consistency
        print("Performing final global standardization...")
        
        # Identify the single deuterium columns in the combined data
        fragment_columns = ['Fragment_1_SMILES', 'Fragment_2_SMILES', 'Fragment_3_SMILES']
        combined_single_d_columns = []
        
        for col in fragment_columns:
            if col in combined_df.columns:
                deuterium_counts = combined_df[col].apply(count_deuterium)
                unique_counts = deuterium_counts.unique()
                
                if len(unique_counts) == 1 and unique_counts[0] == 1:
                    combined_single_d_columns.append(col)
        
        print(f"Combined single deuterium columns: {combined_single_d_columns}")
        
        # Apply final standardization
        if len(combined_single_d_columns) == 2:
            final_combined_df = standardize_fragment_assignments(combined_df, combined_single_d_columns)
        else:
            print("Warning: Could not identify exactly 2 single deuterium columns in combined data")
            final_combined_df = combined_df
        
        # Save combined results
        combined_output = 'protac_fragments_with_types_all_chunks_combined.csv'
        final_combined_df.to_csv(combined_output, index=False)
        
        print(f"Combined results saved to {combined_output}")
        print(f"Total records in combined file: {len(final_combined_df)}")
        
        # Display overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Successfully processed: {len(successful_chunks)} chunks")
        print(f"  Failed: {len(failed_chunks)} chunks")
        print(f"  Total records: {len(final_combined_df)}")
        
        # Show type distributions in combined data
        type_columns = [col for col in final_combined_df.columns if col.endswith('_TYPE')]
        if type_columns:
            print(f"\nFinal combined type distributions:")
            for type_col in type_columns:
                distribution = final_combined_df[type_col].value_counts()
                print(f"  {type_col}: {dict(distribution)}")
        
        # Verify standardization worked
        print(f"\nStandardization verification:")
        if 'Fragment_1_TYPE' in final_combined_df.columns:
            warhead_in_frag1 = (final_combined_df['Fragment_1_TYPE'] == 'Warhead').sum()
            total_frag1 = len(final_combined_df)
            print(f"  Fragment_1_TYPE = Warhead: {warhead_in_frag1}/{total_frag1} ({warhead_in_frag1/total_frag1*100:.1f}%)")
        
        if 'Fragment_3_TYPE' in final_combined_df.columns:
            e3_in_frag3 = (final_combined_df['Fragment_3_TYPE'] == 'E3 Binder').sum()
            total_frag3 = len(final_combined_df)
            print(f"  Fragment_3_TYPE = E3 Binder: {e3_in_frag3}/{total_frag3} ({e3_in_frag3/total_frag3*100:.1f}%)")
        
        # Display sample of combined results
        print(f"\nSample of combined results:")
        display_columns = ['Compound_ID', 'Target'] + [col for col in final_combined_df.columns if 'Fragment' in col and ('SMILES' in col or 'TYPE' in col)]
        available_columns = [col for col in display_columns if col in final_combined_df.columns]
        print(final_combined_df[available_columns].head(10))
        
    else:
        print("ERROR: No chunks were processed successfully!")
    
    if failed_chunks:
        print(f"\nFailed chunks:")
        for chunk in failed_chunks:
            print(f"  {chunk}")
    
    print(f"\n{'='*60}")
    print("Batch processing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
