#!/usr/bin/env python3
"""
PROTAC Fragment Connector Streamlit App
Reconstructs PROTAC compounds from fragment SMILES with virtual library generation
"""

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Draw
from rdkit.Chem.rdchem import BondType
import re
import io
import base64
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import itertools

# Set page config
st.set_page_config(
    page_title="PROTAC Fragment Connector",
    page_icon="🧬",
    layout="wide"
)

def parse_tagged_smiles(smiles):
    """
    Parse SMILES with [2H] or [*] tags to identify connection points
    Returns the molecule and connection point information
    """
    if pd.isna(smiles) or not smiles:
        return None, []
    
    # Convert [2H] to [*] for consistency
    working_smiles = smiles.replace('[2H]', '[*]')
    
    try:
        mol = Chem.MolFromSmiles(working_smiles)
        if mol is None:
            return None, []
        
        # Find dummy atoms (connection points)
        dummy_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atom
                dummy_atoms.append(atom.GetIdx())
        
        return mol, dummy_atoms
    
    except Exception as e:
        st.error(f"Error parsing SMILES {smiles}: {e}")
        return None, []

def determine_bond_type(mol1, dummy1_idx, mol2, dummy2_idx):
    """
    Determine the appropriate bond type based on the atoms connected to dummy atoms
    """
    atom1_neighbors = [n for n in mol1.GetAtomWithIdx(dummy1_idx).GetNeighbors()]
    atom2_neighbors = [n for n in mol2.GetAtomWithIdx(dummy2_idx).GetNeighbors()]
    
    if not atom1_neighbors or not atom2_neighbors:
        return None, False, "Dummy atoms must be connected to at least one real atom"
    
    connect_atom1 = atom1_neighbors[0]
    connect_atom2 = atom2_neighbors[0]
    
    atom1_symbol = connect_atom1.GetSymbol()
    atom2_symbol = connect_atom2.GetSymbol()
    atom1_aromatic = connect_atom1.GetIsAromatic()
    atom2_aromatic = connect_atom2.GetIsAromatic()
    
    # Define connection patterns
    connection_types = {
        ('C', 'O'): ('ether', 'C-O etherification'),
        ('c', 'O'): ('ether', 'Ar-O etherification'),
        ('O', 'C'): ('ether', 'O-C etherification'),
        ('O', 'c'): ('ether', 'O-Ar etherification'),
        ('C', 'N'): ('amine', 'C-N alkylation'),
        ('c', 'N'): ('amine', 'Ar-N arylation'),
        ('N', 'C'): ('amine', 'N-C alkylation'),
        ('N', 'c'): ('amine', 'N-Ar arylation'),
        ('C', 'C'): ('alkyl', 'C-C alkylation'),
        ('c', 'C'): ('alkyl', 'Ar-C alkylation'),
        ('C', 'c'): ('alkyl', 'C-Ar alkylation'),
        ('c', 'c'): ('alkyl', 'Ar-Ar coupling'),
    }
    
    # Convert aromatic carbons to lowercase
    symbol1 = 'c' if atom1_aromatic and atom1_symbol == 'C' else atom1_symbol
    symbol2 = 'c' if atom2_aromatic and atom2_symbol == 'C' else atom2_symbol
    
    connection_key = (symbol1, symbol2)
    
    if connection_key in connection_types:
        bond_type, description = connection_types[connection_key]
        return bond_type, True, description
    else:
        return None, False, f"Unsupported connection: {symbol1}-{symbol2}"

def connect_fragments_with_chemistry(mol1, mol2, dummy1_idx, dummy2_idx):
    """
    Connect two molecules using chemically appropriate bond formation
    """
    # Determine bond type
    bond_type, is_valid, description = determine_bond_type(mol1, dummy1_idx, mol2, dummy2_idx)
    
    if not is_valid:
        raise ValueError(f"Invalid connection: {description}")
    
    # Get the atoms connected to the dummy atoms
    atom1_neighbors = [n for n in mol1.GetAtomWithIdx(dummy1_idx).GetNeighbors()]
    atom2_neighbors = [n for n in mol2.GetAtomWithIdx(dummy2_idx).GetNeighbors()]
    
    connect_atom1 = atom1_neighbors[0].GetIdx()
    connect_atom2 = atom2_neighbors[0].GetIdx()
    
    return connect_with_simple_bond(mol1, mol2, dummy1_idx, dummy2_idx, 
                                  connect_atom1, connect_atom2)

def connect_with_simple_bond(mol1, mol2, dummy1_idx, dummy2_idx, connect_atom1, connect_atom2):
    """
    Create a simple single bond connection between fragments
    """
    # Create editable molecules
    mol1_rw = Chem.RWMol(mol1)
    mol2_rw = Chem.RWMol(mol2)
    
    # Remove dummy atoms
    mol1_rw.RemoveAtom(dummy1_idx)
    mol2_rw.RemoveAtom(dummy2_idx)
    
    # Adjust indices if dummy atom removal affected the connection atom
    if dummy1_idx < connect_atom1:
        connect_atom1 -= 1
    if dummy2_idx < connect_atom2:
        connect_atom2 -= 1
    
    # Get final molecules
    mol1_final = mol1_rw.GetMol()
    mol2_final = mol2_rw.GetMol()
    
    # Combine molecules
    combined = Chem.CombineMols(mol1_final, mol2_final)
    combined_rw = Chem.RWMol(combined)
    
    # Add bond between connection atoms
    offset = mol1_final.GetNumAtoms()
    combined_rw.AddBond(connect_atom1, connect_atom2 + offset, BondType.SINGLE)
    
    # Sanitize the molecule
    try:
        final_mol = combined_rw.GetMol()
        Chem.SanitizeMol(final_mol)
        return final_mol
    except Exception as e:
        raise ValueError(f"Error sanitizing combined molecule: {e}")

def build_protac_from_fragments(e3binder_smiles, linker_smiles, warhead_smiles):
    """
    Build PROTAC by connecting fragments: E3binder -> Linker -> Warhead
    """
    try:
        # Parse fragments
        e3binder, e3_dummies = parse_tagged_smiles(e3binder_smiles)
        linker, linker_dummies = parse_tagged_smiles(linker_smiles)
        warhead, warhead_dummies = parse_tagged_smiles(warhead_smiles)
        
        if not all([e3binder, linker, warhead]):
            return None, "Failed to parse one or more fragments"
        
        # Check connection points
        if len(e3_dummies) != 1 or len(warhead_dummies) != 1:
            return None, "E3binder and Warhead should each have exactly 1 connection point"
        
        if len(linker_dummies) != 2:
            return None, "Linker should have exactly 2 connection points"
        
        # Step 1: Connect E3binder to Linker
        e3_linker = connect_fragments_with_chemistry(
            e3binder, linker, 
            e3_dummies[0], linker_dummies[0]
        )
        
        if e3_linker is None:
            return None, "Failed to connect E3binder to Linker"
        
        # Find remaining dummy atom in the E3binder-Linker intermediate
        remaining_dummies = []
        for atom in e3_linker.GetAtoms():
            if atom.GetAtomicNum() == 0:
                remaining_dummies.append(atom.GetIdx())
        
        if len(remaining_dummies) != 1:
            return None, f"Expected 1 remaining connection point, found {len(remaining_dummies)}"
        
        # Step 2: Connect E3binder-Linker to Warhead
        final_protac = connect_fragments_with_chemistry(
            e3_linker, warhead,
            remaining_dummies[0], warhead_dummies[0]
        )
        
        if final_protac is None:
            return None, "Failed to connect to Warhead"
        
        return final_protac, "Success"
        
    except Exception as e:
        return None, str(e)

def calculate_molecular_properties(mol):
    """Calculate molecular properties for the PROTAC"""
    try:
        properties = {
            'Molecular Weight (Da)': round(rdMolDescriptors.CalcExactMolWt(mol), 2),
            'LogP': round(rdMolDescriptors.CalcCrippenDescriptors(mol)[0], 2),
            'H-bond Donors': rdMolDescriptors.CalcNumHBD(mol),
            'H-bond Acceptors': rdMolDescriptors.CalcNumHBA(mol),
            'TPSA (Ų)': round(rdMolDescriptors.CalcTPSA(mol), 2),
            'Rotatable Bonds': rdMolDescriptors.CalcNumRotatableBonds(mol)
        }
        return properties
    except Exception as e:
        st.error(f"Error calculating properties: {e}")
        return {}

def mol_to_image(mol, size=(300, 300)):
    """Convert RDKit molecule to PIL Image"""
    try:
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception as e:
        st.error(f"Error generating molecular image: {e}")
        return None

def calculate_similarity(target_props, candidate_props, property_names):
    """Calculate Euclidean distance similarity between property vectors"""
    target_vector = [target_props.get(prop, 0) for prop in property_names]
    candidate_vector = [candidate_props.get(prop, 0) for prop in property_names]
    
    # Normalize the vectors
    scaler = StandardScaler()
    combined = np.array([target_vector, candidate_vector])
    normalized = scaler.fit_transform(combined)
    
    # Calculate Euclidean distance (lower = more similar)
    distance = euclidean_distances([normalized[0]], [normalized[1]])[0][0]
    
    # Convert to similarity score (higher = more similar)
    similarity = 1 / (1 + distance)
    
    return similarity

def create_dimensionality_reduction_plot(virtual_df, original_protacs_df, target_protein, method='t-SNE'):
    """
    Create t-SNE or UMAP plot showing virtual library vs original PROTACs
    """
    try:
        # Combine virtual library and original PROTACs data
        property_cols = ['Molecular Weight (Da)', 'LogP', 'H-bond Donors', 'H-bond Acceptors', 'TPSA (Ų)', 'Rotatable Bonds']
        
        # Prepare virtual library data
        virtual_props = virtual_df[property_cols].values
        virtual_hover = [f"Virtual ID: {row['Virtual_ID']}<br>Similarity: {row['Similarity_Score']:.3f}" for _, row in virtual_df.iterrows()]
        
        # Prepare original PROTACs data (filter by target if specified)
        if target_protein and target_protein != 'Unknown':
            if 'Original_Target' in original_protacs_df.columns:
                target_protacs = original_protacs_df[original_protacs_df['Original_Target'] == target_protein]
            else:
                target_protacs = original_protacs_df
        else:
            target_protacs = original_protacs_df
        
        # Get successful reconstructions only
        target_protacs = target_protacs[target_protacs['Status'] == 'Success']
        
        if len(target_protacs) == 0:
            return None, "No successful original PROTACs found for comparison"
        
        original_props = target_protacs[property_cols].values
        original_hover = [f"Original Row: {row['Row_Index']}" for _, row in target_protacs.iterrows()]
        
        # Combine all data
        all_props = np.vstack([virtual_props, original_props])
        all_hover = virtual_hover + original_hover
        
        # Create labels for plotting
        virtual_indices = list(range(len(virtual_props)))
        original_indices = list(range(len(virtual_props), len(virtual_props) + len(original_props)))
        
        # Standardize the data
        scaler = StandardScaler()
        all_props_scaled = scaler.fit_transform(all_props)
        
        # Apply dimensionality reduction
        if method.lower() == 't-sne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_props_scaled)//4))
            embedding = reducer.fit_transform(all_props_scaled)
        else:  # UMAP
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_props_scaled)//3))
                embedding = reducer.fit_transform(all_props_scaled)
            except ImportError:
                return None, "UMAP not installed. Please install with: pip install umap-learn"
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add virtual library points (background)
        fig.add_trace(go.Scatter(
            x=embedding[virtual_indices, 0],
            y=embedding[virtual_indices, 1],
            mode='markers',
            marker=dict(
                size=8,
                color='lightblue',
                opacity=0.6,
                line=dict(width=0.5, color='darkblue')
            ),
            name='Virtual Library',
            text=virtual_hover,
            hovertemplate='<b>Virtual Library</b><br>%{text}<extra></extra>'
        ))
        
        # Add original PROTAC points (foreground)
        fig.add_trace(go.Scatter(
            x=embedding[original_indices, 0],
            y=embedding[original_indices, 1],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                opacity=0.9,
                symbol='diamond',
                line=dict(width=2, color='darkred')
            ),
            name='Original PROTACs',
            text=original_hover,
            hovertemplate='<b>Original PROTAC</b><br>%{text}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{method} Visualization: Virtual Library vs Original PROTACs<br>Target: {target_protein if target_protein else 'All'}",
            xaxis_title=f"{method} Component 1",
            yaxis_title=f"{method} Component 2",
            width=800,
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig, f"Successfully created {method} plot with {len(virtual_props)} virtual compounds and {len(original_props)} original PROTACs"
        
    except Exception as e:
        return None, f"Error creating {method} plot: {str(e)}"

def generate_virtual_library(df, fragment_cols, target_compound_idx, max_library_size):
    """Generate virtual library by combining fragments for a specific target"""
    
    # Get the target compound
    target_row = df.iloc[target_compound_idx]
    target_e3binder = target_row[fragment_cols['Fragment_3']]
    target_linker = target_row[fragment_cols['Fragment_2']]
    target_warhead = target_row[fragment_cols['Fragment_1']]
    
    # Identify the target (use Target column if available, otherwise use the selected compound info)
    target_protein = None
    if 'Target' in df.columns:
        target_protein = target_row['Target']
    elif 'Original_Target' in df.columns:
        target_protein = target_row['Original_Target']
    
    # Build original PROTAC to get reference properties
    original_mol, _ = build_protac_from_fragments(target_e3binder, target_linker, target_warhead)
    if not original_mol:
        return None, "Could not build original PROTAC"
    
    original_props = calculate_molecular_properties(original_mol)
    property_names = ['Molecular Weight (Da)', 'LogP', 'H-bond Donors', 'H-bond Acceptors', 'TPSA (Ų)', 'Rotatable Bonds']
    
    # Filter dataset for the same target
    if target_protein:
        target_col = 'Target' if 'Target' in df.columns else 'Original_Target'
        target_df = df[df[target_col] == target_protein]
        st.info(f"Found {len(target_df)} compounds for target: {target_protein}")
    else:
        target_df = df
        st.info(f"No target column found, using all {len(target_df)} compounds")
    
    # Get unique fragments for this target
    unique_e3binders = target_df[fragment_cols['Fragment_3']].dropna().unique()
    unique_warheads = target_df[fragment_cols['Fragment_1']].dropna().unique()
    # Always use ALL unique linkers from the entire dataset
    unique_linkers = df[fragment_cols['Fragment_2']].dropna().unique()
    
    st.info(f"Target-specific fragments: {len(unique_e3binders)} E3 binders, {len(unique_warheads)} warheads")
    st.info(f"Using all {len(unique_linkers)} available linkers")
    
    # Generate all combinations: E3binder × Linker × Warhead
    combinations = list(itertools.product(unique_e3binders, unique_linkers, unique_warheads))
    
    st.info(f"Total possible combinations: {len(combinations)}")
    
    # If we have more combinations than max size, we'll sample all and then select top similar ones
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (e3binder, linker, warhead) in enumerate(combinations):
        progress = (idx + 1) / len(combinations)
        progress_bar.progress(progress)
        status_text.text(f'Generating virtual compound {idx + 1} of {len(combinations)}...')
        
        try:
            # Build PROTAC
            protac_mol, message = build_protac_from_fragments(e3binder, linker, warhead)
            
            if protac_mol:
                protac_smiles = Chem.MolToSmiles(protac_mol)
                properties = calculate_molecular_properties(protac_mol)
                
                # Calculate similarity to original
                similarity = calculate_similarity(original_props, properties, property_names)
                
                result = {
                    'Virtual_ID': idx,
                    'PROTAC_SMILES': protac_smiles,
                    'E3binder_SMILES': e3binder,
                    'Linker_SMILES': linker,
                    'Warhead_SMILES': warhead,
                    'Target_Protein': target_protein if target_protein else 'Unknown',
                    'Similarity_Score': round(similarity, 4),
                    **properties
                }
                
                results.append(result)
        
        except Exception as e:
            continue  # Skip failed combinations
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return None, "No valid virtual compounds generated"
    
    # Convert to DataFrame and sort by similarity
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Similarity_Score', ascending=False)
    
    # Select top compounds up to max_library_size
    if len(results_df) > max_library_size:
        results_df = results_df.head(max_library_size)
    
    return results_df, f"Generated {len(results_df)} virtual compounds"

def process_csv_data(df, fragment_cols):
    """Process the uploaded CSV data and reconstruct PROTACs"""
    results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f'Processing compound {idx + 1} of {total_rows}...')
        
        # Extract fragment SMILES
        try:
            e3binder_smiles = row[fragment_cols['Fragment_3']]  # E3binder
            linker_smiles = row[fragment_cols['Fragment_2']]    # Linker
            warhead_smiles = row[fragment_cols['Fragment_1']]   # Warhead
            
            # Build PROTAC
            protac_mol, message = build_protac_from_fragments(
                e3binder_smiles, linker_smiles, warhead_smiles
            )
            
            if protac_mol:
                protac_smiles = Chem.MolToSmiles(protac_mol)
                properties = calculate_molecular_properties(protac_mol)
                
                result = {
                    'Row_Index': idx,
                    'Status': 'Success',
                    'PROTAC_SMILES': protac_smiles,
                    'E3binder_SMILES': e3binder_smiles,
                    'Linker_SMILES': linker_smiles,
                    'Warhead_SMILES': warhead_smiles,
                    **properties
                }
                
                # Add original columns if they exist
                for col in df.columns:
                    if col not in fragment_cols.values():
                        result[f'Original_{col}'] = row[col]
                
            else:
                result = {
                    'Row_Index': idx,
                    'Status': 'Failed',
                    'Error_Message': message,
                    'E3binder_SMILES': e3binder_smiles,
                    'Linker_SMILES': linker_smiles,
                    'Warhead_SMILES': warhead_smiles
                }
            
            results.append(result)
            
        except Exception as e:
            result = {
                'Row_Index': idx,
                'Status': 'Error',
                'Error_Message': str(e)
            }
            results.append(result)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def main():
    st.title("🧬 PROTAC Fragment Connector & Virtual Library Generator")
    st.markdown("**Reconstruct PROTAC compounds and generate virtual libraries from molecular fragments**")
    
    # Main tabs
    tab1, tab2 = st.tabs(["📋 PROTAC Reconstruction", "🧪 Virtual Library Generation"])
    
    with tab1:
        st.header("📁 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with fragment SMILES",
            type=['csv'],
            key="main_upload",
            help="CSV should contain columns with SMILES strings and attachment points marked as [2H] or [*]"
        )
        
        if uploaded_file is not None:
            try:
                # Load the CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                
                # Show preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10))
                
                # Column mapping
                st.subheader("🔗 Column Mapping")
                st.markdown("Map your CSV columns to the required fragment types:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fragment_3_col = st.selectbox(
                        "E3binder (Fragment_3)",
                        options=df.columns.tolist(),
                        help="Column containing E3 ligase binder SMILES",
                        key="frag3_main"
                    )
                
                with col2:
                    fragment_2_col = st.selectbox(
                        "Linker (Fragment_2)",
                        options=df.columns.tolist(),
                        help="Column containing linker SMILES",
                        key="frag2_main"
                    )
                
                with col3:
                    fragment_1_col = st.selectbox(
                        "Warhead (Fragment_1)",
                        options=df.columns.tolist(),
                        help="Column containing warhead/target binder SMILES",
                        key="frag1_main"
                    )
                
                fragment_cols = {
                    'Fragment_3': fragment_3_col,
                    'Fragment_2': fragment_2_col,
                    'Fragment_1': fragment_1_col
                }
                
                # Store data in session state for use in virtual library tab
                st.session_state.df = df
                st.session_state.fragment_cols = fragment_cols
                
                # Validation
                st.subheader("🔬 Fragment Validation")
                sample_row = df.iloc[0] if len(df) > 0 else None
                
                if sample_row is not None:
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (frag_name, col_name) in enumerate(fragment_cols.items()):
                        with [col1, col2, col3][i]:
                            smiles = sample_row[col_name]
                            st.write(f"**{frag_name} ({col_name}):**")
                            st.code(smiles, language="text")
                            
                            # Parse and validate
                            mol, dummies = parse_tagged_smiles(smiles)
                            if mol and dummies:
                                st.success(f"✅ Valid SMILES with {len(dummies)} connection point(s)")
                                
                                # Show molecular image
                                img = mol_to_image(mol, size=(200, 150))
                                if img:
                                    st.image(img, width=200)
                            else:
                                st.error("❌ Invalid SMILES or no connection points found")
                
                # Processing
                st.subheader("⚡ Process Fragments")
                
                if st.button("🚀 Process All Fragments", type="primary"):
                    with st.spinner("Processing fragments and reconstructing PROTACs..."):
                        results_df = process_csv_data(df, fragment_cols)
                    
                    # Store results in session state
                    st.session_state.results_df = results_df
                    
                    # Display results
                    st.subheader("📈 Results Summary")
                    
                    success_count = len(results_df[results_df['Status'] == 'Success'])
                    failed_count = len(results_df[results_df['Status'] == 'Failed'])
                    error_count = len(results_df[results_df['Status'] == 'Error'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Processed", len(results_df))
                    col2.metric("Successful", success_count)
                    col3.metric("Failed", failed_count)
                    col4.metric("Errors", error_count)
                    
                    # Show results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_data,
                        file_name="protac_reconstruction_results.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
        
        else:
            # Show example format
            st.subheader("📝 Expected CSV Format")
            st.markdown("Your CSV should contain columns with SMILES strings where attachment points are marked with `[2H]` or `[*]`:")
            
            example_data = {
                'Compound_ID': [1, 2, 3],
                'Fragment_1_SMILES': ['[*]Cc1ccc(C(=O)c2ccc(O)cc2)cc1', '[*]Cc1ccc(C#N)cc1', '[*]Cc1ccc(F)cc1'],
                'Fragment_2_SMILES': ['[*]CCOCCOCCN(CC)CC[*]', '[*]CCOCCOCC[*]', '[*]CCNCC[*]'],
                'Fragment_3_SMILES': ['[*]C1CC(O)CC1C(=O)Nc1ccc(C)cc1', '[*]C1CCC(O)CC1', '[*]C1CCNCC1'],
                'Target': ['BRD4', 'CDK2', 'EGFR']
            }
            
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df)
            
            st.info("💡 **Tip:** Attachment points can be marked with either `[2H]` or `[*]` - both formats are supported!")
    
    with tab2:
        st.header("🧪 Virtual Library Generation")
        st.markdown("Generate new PROTAC candidates by combining different warheads and E3 binders with available linkers")
        
        # Check if data is available
        if 'df' not in st.session_state or 'fragment_cols' not in st.session_state:
            st.warning("⚠️ Please upload and process a CSV file in the 'PROTAC Reconstruction' tab first")
            return
        
        df = st.session_state.df
        fragment_cols = st.session_state.fragment_cols
        
        # Target selection
        st.subheader("🎯 Select Target Compound")
        
        # Show available compounds
        if 'Target' in df.columns:
            target_col = 'Target'
            compound_options = [f"Row {i}: {row[target_col]} (ID: {row.get('Compound_ID', i)})" 
                              for i, row in df.iterrows()]
        elif 'Original_Target' in df.columns:
            target_col = 'Original_Target'
            compound_options = [f"Row {i}: {row[target_col]} (ID: {row.get('Compound_ID', i)})" 
                              for i, row in df.iterrows()]
        else:
            compound_options = [f"Row {i}: Compound {row.get('Compound_ID', i)}" for i, row in df.iterrows()]
        
        selected_compound = st.selectbox(
            "Choose target compound for virtual library generation:",
            options=range(len(df)),
            format_func=lambda x: compound_options[x],
            help="All warheads and E3 binders for this target will be used with all available linkers"
        )
        
        # Show selected compound details and target info
        target_row = df.iloc[selected_compound]
        
        # Display target information
        if 'Target' in df.columns:
            st.info(f"🎯 **Selected Target:** {target_row['Target']}")
        elif 'Original_Target' in df.columns:
            st.info(f"🎯 **Selected Target:** {target_row['Original_Target']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Reference E3 Binder:**")
            st.code(target_row[fragment_cols['Fragment_3']], language="text")
        with col2:
            st.write("**Reference Linker:**")
            st.code(target_row[fragment_cols['Fragment_2']], language="text")
        with col3:
            st.write("**Reference Warhead:**")
            st.code(target_row[fragment_cols['Fragment_1']], language="text")
        
        # Library size setting
        st.subheader("⚙️ Library Configuration")
        max_library_size = st.number_input(
            "Maximum library size (top X most similar compounds):",
            min_value=1,
            max_value=10000,
            value=100,
            help="Maximum number of virtual compounds to generate, ranked by similarity to the original"
        )
        
        # Generate virtual library
        if st.button("🚀 Generate Virtual Library", type="primary"):
            with st.spinner("Generating virtual library..."):
                result = generate_virtual_library(df, fragment_cols, selected_compound, max_library_size)
                
                if result is not None and len(result) == 2:
                    virtual_df, message = result
                    if virtual_df is not None:
                        # Store virtual library in session state
                        st.session_state.virtual_df = virtual_df
                        st.session_state.virtual_message = message
                        st.session_state.virtual_target_idx = selected_compound
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("❌ Failed to generate virtual library")
        
        # Display virtual library results if they exist
        if 'virtual_df' in st.session_state and st.session_state.virtual_df is not None:
            virtual_df = st.session_state.virtual_df
            
            # Display results
            st.subheader("📊 Virtual Library Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Generated Compounds", len(virtual_df))
            col2.metric("Highest Similarity", f"{virtual_df['Similarity_Score'].max():.3f}")
            col3.metric("Average Similarity", f"{virtual_df['Similarity_Score'].mean():.3f}")
            col4.metric("Unique E3/Warhead Pairs", len(virtual_df))
            
            # Show top compounds
            st.subheader("🏆 Top Virtual Compounds")
            
            # Create display dataframe
            display_df = virtual_df[['Virtual_ID', 'Similarity_Score', 'Molecular Weight (Da)', 
                                   'LogP', 'H-bond Donors', 'H-bond Acceptors', 'TPSA (Ų)', 'Rotatable Bonds']].copy()
            st.dataframe(display_df)
            
            # Show detailed view of top 3
            st.subheader("🔍 Top 3 Candidates Detailed View")
            
            for idx, (_, row) in enumerate(virtual_df.head(3).iterrows()):
                with st.expander(f"Rank #{idx + 1} - Virtual ID {row['Virtual_ID']} (Similarity: {row['Similarity_Score']:.3f})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Virtual PROTAC SMILES:**")
                        st.code(row['PROTAC_SMILES'], language="text")
                        
                        st.write("**Fragment Composition:**")
                        st.write(f"• E3 Binder: `{row['E3binder_SMILES'][:50]}...`")
                        st.write(f"• Linker: `{row['Linker_SMILES'][:50]}...`")
                        st.write(f"• Warhead: `{row['Warhead_SMILES'][:50]}...`")
                        
                        # Show molecular properties
                        st.write("**Molecular Properties:**")
                        props = {k: v for k, v in row.items() 
                                if k in ['Molecular Weight (Da)', 'LogP', 'H-bond Donors', 
                                        'H-bond Acceptors', 'TPSA (Ų)', 'Rotatable Bonds']}
                        st.json(props)
                    
                    with col2:
                        # Show molecular structure
                        try:
                            mol = Chem.MolFromSmiles(row['PROTAC_SMILES'])
                            if mol:
                                img = mol_to_image(mol, size=(300, 200))
                                if img:
                                    st.image(img, caption=f"Virtual PROTAC #{idx + 1}")
                        except:
                            st.error("Could not generate structure image")
            
            # Download virtual library
            csv_buffer = io.StringIO()
            virtual_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Virtual Library CSV",
                data=csv_data,
                file_name=f"virtual_library_target_{st.session_state.get('virtual_target_idx', 'unknown')}.csv",
                mime="text/csv"
            )
            
            # Chemical Space Visualization
            st.subheader("🎯 Chemical Space Visualization")
            
            # Check if we have original PROTACs data for comparison
            if 'results_df' in st.session_state:
                original_protacs_df = st.session_state.results_df
                
                st.write("Compare your virtual library with original PROTAC compounds in chemical space:")
                
                col1, col2 = st.columns(2)
                with col1:
                    viz_method = st.selectbox(
                        "Choose dimensionality reduction method:",
                        options=['t-SNE', 'UMAP'],
                        help="t-SNE preserves local structure, UMAP preserves both local and global structure",
                        key="viz_method_select"
                    )
                
                with col2:
                    if st.button("🗺️ Generate Chemical Space Plot", type="primary", key="generate_plot_btn"):
                        with st.spinner(f"Generating {viz_method} visualization..."):
                            try:
                                fig, plot_message = create_dimensionality_reduction_plot(
                                    virtual_df, 
                                    original_protacs_df, 
                                    virtual_df.iloc[0]['Target_Protein'] if 'Target_Protein' in virtual_df.columns else None,
                                    method=viz_method
                                )
                                
                                if fig is not None:
                                    # Store the plot in session state
                                    st.session_state.plot_fig = fig
                                    st.session_state.plot_message = plot_message
                                    st.session_state.plot_method = viz_method
                                    st.success(plot_message)
                                else:
                                    st.error(f"❌ {plot_message}")
                            except Exception as e:
                                st.error(f"❌ Error generating plot: {str(e)}")
                
                # Display the plot if it exists in session state
                if 'plot_fig' in st.session_state and st.session_state.plot_fig is not None:
                    st.markdown("---")
                    st.subheader(f"📊 {st.session_state.get('plot_method', 'Chemical Space')} Visualization")
                    
                    # Display the plot
                    st.plotly_chart(st.session_state.plot_fig, use_container_width=True)
                    
                    # Add interpretation guide
                    st.markdown("""
                    **📋 Plot Interpretation Guide:**
                    - 🔵 **Light Blue Circles**: Virtual library compounds (generated candidates)
                    - 🔴 **Red Diamonds**: Original PROTAC compounds (from your dataset)
                    - **Proximity**: Points close together have similar molecular properties
                    - **Clusters**: Groups of compounds with similar chemical characteristics
                    - **Outliers**: Isolated points represent compounds with unique properties
                    
                    **💡 Tips for Analysis:**
                    - Virtual compounds near red diamonds are similar to known PROTACs
                    - Virtual compounds in empty regions explore new chemical space
                    - Hover over points to see compound details and similarity scores
                    """)
                    
                    # Option to clear the plot
                    if st.button("🗑️ Clear Plot", key="clear_plot_btn"):
                        if 'plot_fig' in st.session_state:
                            del st.session_state.plot_fig
                        if 'plot_message' in st.session_state:
                            del st.session_state.plot_message
                        if 'plot_method' in st.session_state:
                            del st.session_state.plot_method
                        st.rerun()
            
            else:
                st.info("💡 **To enable Chemical Space Visualization:** Process compounds in the 'PROTAC Reconstruction' tab first to create reference data for comparison.")
                st.markdown("""
                **What you'll get with Chemical Space Visualization:**
                - Interactive t-SNE or UMAP plots
                - Virtual library compounds vs original PROTACs
                - Molecular property-based clustering
                - Exploration of chemical space coverage
                """)

            
            # Property Distribution Analysis
            st.subheader("📈 Property Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Molecular weight distribution
                st.write("**Molecular Weight Distribution**")
                mw_data = virtual_df['Molecular Weight (Da)'].values
                mw_hist, mw_bins = np.histogram(mw_data, bins=10)
                mw_df = pd.DataFrame({
                    'MW_Range': [f"{mw_bins[i]:.0f}-{mw_bins[i+1]:.0f}" for i in range(len(mw_hist))],
                    'Count': mw_hist
                })
                st.bar_chart(mw_df.set_index('MW_Range'))
            
            with col2:
                # LogP distribution
                st.write("**LogP Distribution**")
                logp_data = virtual_df['LogP'].values
                logp_hist, logp_bins = np.histogram(logp_data, bins=10)
                logp_df = pd.DataFrame({
                    'LogP_Range': [f"{logp_bins[i]:.1f}-{logp_bins[i+1]:.1f}" for i in range(len(logp_hist))],
                    'Count': logp_hist
                })
                st.bar_chart(logp_df.set_index('LogP_Range'))
            
            # Similarity vs properties scatter plot
            st.subheader("🎯 Similarity vs Molecular Properties")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Similarity vs MW
                chart_data = pd.DataFrame({
                    'Similarity Score': virtual_df['Similarity_Score'],
                    'Molecular Weight': virtual_df['Molecular Weight (Da)']
                })
                st.scatter_chart(chart_data, x='Molecular Weight', y='Similarity Score')
            
            with col2:
                # Similarity vs LogP
                chart_data = pd.DataFrame({
                    'Similarity Score': virtual_df['Similarity_Score'],
                    'LogP': virtual_df['LogP']
                })
                st.scatter_chart(chart_data, x='LogP', y='Similarity Score')

if __name__ == "__main__":
    main()