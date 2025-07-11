import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, inchi
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="PROTAC Bond Cutting Tool",
    page_icon="🧬",
    layout="wide"
)

def load_and_process_data(uploaded_file):
    """Load CSV and extract unique SMILES"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Extract important columns
        important_cols = ['Compound ID', 'Target', 'Name', 'Smiles', 
                         'DC50 (nM)', 'Dmax (%)', 'Assay (DC50/Dmax)', 
                         'Percent degradation (%)', 'InChI Key']
        
        # Filter for valid SMILES
        valid_data = df[df['Smiles'].notna() & (df['Smiles'].str.strip() != '')]
        
        # Get unique SMILES with their first occurrence data
        unique_smiles_data = valid_data.drop_duplicates(subset=['Smiles']).copy()
        
        return unique_smiles_data, len(valid_data)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, 0

def draw_molecule_with_indices(smiles, highlight_bonds=None):
    """Draw molecule with bond indices highlighted"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
            
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(900, 900)
        
        # Set drawing options
        opts = drawer.drawOptions()
        opts.addBondIndices = True
        opts.addAtomIndices = False  # Remove atom indices
        opts.annotationFontScale = 1.2
        
        # Draw molecule with proper parameter handling
        if highlight_bonds:
            # Filter valid bond indices
            valid_bonds = [bond_idx for bond_idx in highlight_bonds if bond_idx < mol.GetNumBonds()]
            if valid_bonds:
                # Create highlight colors dictionary
                highlight_bond_colors = {bond_idx: (1.0, 0.0, 0.0) for bond_idx in valid_bonds}
                
                # Draw with highlights - use proper parameter order
                drawer.DrawMolecule(mol, 
                                  highlightAtoms=[], 
                                  highlightBonds=valid_bonds,
                                  highlightAtomColors={}, 
                                  highlightBondColors=highlight_bond_colors)
            else:
                drawer.DrawMolecule(mol)
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        
        # Convert to image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        return img, mol
    
    except Exception as e:
        st.error(f"Error drawing molecule: {str(e)}")
        return None, None

def cut_bonds_and_add_deuterium(smiles, bond_ids):
    """Cut specified bonds and add deuterium attachment points"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Validate bond IDs
        valid_bond_ids = [bid for bid in bond_ids if 0 <= bid < mol.GetNumBonds()]
        
        if len(valid_bond_ids) == 0:
            return None
        
        # Create editable molecule
        em = Chem.EditableMol(mol)
        
        # Add deuterium atoms
        deuterium_idx = []
        for _ in valid_bond_ids:
            idx = em.AddAtom(Chem.Atom(1))  # Add hydrogen
            deuterium_idx.append(idx)
        
        # Get bonds to cut (sort in reverse order to maintain indices)
        bonds_to_cut = []
        for bond_id in sorted(valid_bond_ids, reverse=True):
            bond = mol.GetBondWithIdx(bond_id)
            bonds_to_cut.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_id))
        
        # Cut bonds and add deuterium
        cut_info = []
        for i, (begin_idx, end_idx, bond_id) in enumerate(bonds_to_cut):
            # Remove the bond
            em.RemoveBond(begin_idx, end_idx)
            
            # Add deuterium to both atoms
            d1_idx = deuterium_idx[i*2] if i*2 < len(deuterium_idx) else em.AddAtom(Chem.Atom(1))
            d2_idx = deuterium_idx[i*2+1] if i*2+1 < len(deuterium_idx) else em.AddAtom(Chem.Atom(1))
            
            em.AddBond(begin_idx, d1_idx, Chem.rdchem.BondType.SINGLE)
            em.AddBond(end_idx, d2_idx, Chem.rdchem.BondType.SINGLE)
            
            cut_info.append({
                'bond_id': bond_id,
                'begin_atom': begin_idx,
                'end_atom': end_idx,
                'deuterium_1': d1_idx,
                'deuterium_2': d2_idx
            })
        
        # Get the modified molecule
        modified_mol = em.GetMol()
        
        # Try to sanitize
        try:
            Chem.SanitizeMol(modified_mol)
        except:
            # If sanitization fails, try to fix
            modified_mol = Chem.MolFromSmiles(Chem.MolToSmiles(modified_mol))
        
        return modified_mol, cut_info
    
    except Exception as e:
        st.error(f"Error cutting bonds: {str(e)}")
        return None, None

def generate_fragments(smiles, bond_ids):
    """Generate fragments after cutting exactly 2 bonds - PROTAC style"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Require exactly 2 bonds
        if len(bond_ids) != 2:
            return None
        
        # Validate bond IDs
        valid_bond_ids = [bid for bid in bond_ids if 0 <= bid < mol.GetNumBonds()]
        
        if len(valid_bond_ids) != 2:
            return None
        
        # Generate InChI Key for original molecule
        try:
            original_inchi_key = inchi.MolToInchiKey(mol)
        except:
            # Fallback: use SMILES if InChI Key fails
            original_inchi_key = smiles
        
        # Cut both bonds simultaneously
        try:
            fragmented = Chem.FragmentOnBonds(mol, valid_bond_ids, addDummies=True)
            
            # Split into individual fragments
            frags = Chem.GetMolFrags(fragmented, asMols=True)
            
            if len(frags) < 3:
                st.warning("Expected 3 fragments but got fewer. Bond selection may not create proper PROTAC fragments.")
                return None
            
            # Process fragments and identify the linker (fragment with 2 attachment points)
            fragment_data = []
            linker_fragment = None
            end_fragments = []
            
            for frag in frags:
                frag_smiles = Chem.MolToSmiles(frag)
                attachment_count = frag_smiles.count('*')
                
                if attachment_count == 2:
                    # This is the linker (middle piece)
                    clean_smiles = frag_smiles.replace('*', '[2H]')
                    import re
                    clean_smiles = re.sub(r'\[\d+\[2H\]\]', '[2H]', clean_smiles)
                    linker_fragment = clean_smiles
                elif attachment_count == 1:
                    # This is an end piece (warhead or E3 ligase binder)
                    clean_smiles = frag_smiles.replace('*', '[2H]')
                    import re
                    clean_smiles = re.sub(r'\[\d+\[2H\]\]', '[2H]', clean_smiles)
                    end_fragments.append(clean_smiles)
            
            # Ensure we have the expected structure: 2 end pieces + 1 linker
            if linker_fragment is None:
                st.warning("No linker fragment (with 2 attachment points) found. Please check bond selection.")
                return None
            
            if len(end_fragments) != 2:
                st.warning(f"Expected 2 end fragments but got {len(end_fragments)}. Please check bond selection.")
                return None
            
            # Return results: Fragment_1, Fragment_2 (linker), Fragment_3
            results = {
                'original_inchi_key': original_inchi_key,
                'fragments': [
                    end_fragments[0],      # Fragment_1: First end piece (1 [2H])
                    linker_fragment,       # Fragment_2: Linker (2 [2H])
                    end_fragments[1]       # Fragment_3: Second end piece (1 [2H])
                ]
            }
            
            return results
            
        except Exception as e:
            st.warning(f"Could not fragment bonds {valid_bond_ids}: {str(e)}")
            return None
    
    except Exception as e:
        st.error(f"Error generating fragments: {str(e)}")
        return None

def main():
    st.title("🧬 PROTAC Bond Cutting Tool")
    st.markdown("Upload your PROTAC CSV file and cut bonds to generate fragments with deuterium attachment points")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    if 'current_chunk' not in st.session_state:
        st.session_state.current_chunk = 0
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'current_compound_index' not in st.session_state:
        st.session_state.current_compound_index = 0
    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = True  # Default to auto mode
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and process data
        unique_data, total_compounds = load_and_process_data(uploaded_file)
        
        if unique_data is not None:
            st.success(f"Loaded {len(unique_data)} unique SMILES from {total_compounds} total compounds")
            
            # Create chunks of 50 compounds each
            chunk_size = 50
            chunks = [unique_data.iloc[i:i+chunk_size] for i in range(0, len(unique_data), chunk_size)]
            st.session_state.chunks = chunks
            
            # Chunk selection
            st.subheader("📦 Chunk Selection")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                chunk_options = [f"Chunk {i+1} (Compounds {i*chunk_size + 1}-{min((i+1)*chunk_size, len(unique_data))})" 
                               for i in range(len(chunks))]
                selected_chunk_idx = st.selectbox(
                    f"Select chunk to work with ({len(chunks)} total chunks):",
                    range(len(chunks)),
                    index=st.session_state.current_chunk,
                    format_func=lambda x: chunk_options[x]
                )
                st.session_state.current_chunk = selected_chunk_idx
            
            with col2:
                st.metric("Total Chunks", len(chunks))
            
            with col3:
                st.metric("Current Chunk Size", len(chunks[selected_chunk_idx]))
            
            # Work with selected chunk
            current_chunk_data = chunks[selected_chunk_idx]
            
            # Reset compound index when chunk changes
            if st.session_state.current_chunk != selected_chunk_idx:
                st.session_state.current_compound_index = 0
                st.session_state.current_chunk = selected_chunk_idx
            
            # Progress indicator
            st.progress((selected_chunk_idx + 1) / len(chunks))
            st.caption(f"Working on chunk {selected_chunk_idx + 1} of {len(chunks)}")
            
            # Auto mode progress
            auto_mode = True  # Always in auto mode
            compound_progress = min(1.0, (st.session_state.current_compound_index + 1) / len(current_chunk_data))
            st.progress(compound_progress)
            st.caption(f"Compound {st.session_state.current_compound_index + 1} of {len(current_chunk_data)}")
            
            # Auto mode: use current compound index
            if st.session_state.current_compound_index < len(current_chunk_data):
                selected_idx = st.session_state.current_compound_index
                selected_row = current_chunk_data.iloc[selected_idx]
                selected_smiles = selected_row['Smiles']
                
                st.subheader(f"Current Compound ({selected_idx + 1}/{len(current_chunk_data)})")
                st.info(f"**ID:** {selected_row['Compound ID']} | **Target:** {selected_row['Target']}")
                
            else:
                st.success("✅ All compounds in this chunk have been processed!")
                
                # Auto-save completed chunk
                if st.session_state.processed_data:
                    df_results = pd.DataFrame(st.session_state.processed_data)
                    current_chunk_results = df_results[df_results['Chunk'] == selected_chunk_idx + 1]
                    if len(current_chunk_results) > 0:
                        filename = f"protac_fragments_chunk_{selected_chunk_idx + 1}.csv"
                        current_chunk_results.to_csv(filename, index=False)
                        st.info(f"📁 Chunk saved as: {filename}")
                
                # Show navigation to next chunk
                col1, col2 = st.columns(2)
                with col1:
                    if selected_chunk_idx + 1 < len(chunks):
                        if st.button("➡️ Go to Next Chunk", key="next_chunk"):
                            st.session_state.current_chunk = selected_chunk_idx + 1
                            st.session_state.current_compound_index = 0
                            st.rerun()
                
                with col2:
                    # Download current chunk
                    if st.session_state.processed_data:
                        df_results = pd.DataFrame(st.session_state.processed_data)
                        current_chunk_results = df_results[df_results['Chunk'] == selected_chunk_idx + 1]
                        if len(current_chunk_results) > 0:
                            csv_chunk = current_chunk_results.to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download Chunk {selected_chunk_idx + 1}",
                                data=csv_chunk,
                                file_name=f"protac_fragments_chunk_{selected_chunk_idx + 1}.csv",
                                mime="text/csv",
                                key="download_current_chunk"
                            )
                
                st.balloons()
                selected_idx = None
            
            if st.session_state.current_compound_index < len(current_chunk_data):
                
                # Display molecule information
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Molecule Information")
                    st.write(f"**Compound ID:** {selected_row['Compound ID']}")
                    st.write(f"**Target:** {selected_row['Target']}")
                    st.write(f"**Name:** {selected_row['Name'] if pd.notna(selected_row['Name']) else 'N/A'}")
                    st.write(f"**DC50 (nM):** {selected_row['DC50 (nM)'] if pd.notna(selected_row['DC50 (nM)']) else 'N/A'}")
                    st.write(f"**Dmax (%):** {selected_row['Dmax (%)'] if pd.notna(selected_row['Dmax (%)']) else 'N/A'}")
                    
                    # Get molecule info
                    mol = Chem.MolFromSmiles(selected_smiles)
                    if mol:
                        st.write(f"**Number of Bonds:** {mol.GetNumBonds()}")
                        st.write(f"**Number of Atoms:** {mol.GetNumAtoms()}")
                
                with col2:
                    st.subheader("Molecule Structure")
                    img, mol = draw_molecule_with_indices(selected_smiles)
                    if img:
                        st.image(img, caption="Molecule with bond indices", width=600)
                    else:
                        st.error("Could not draw molecule structure")
                
                # Bond cutting interface
                if mol:
                    st.subheader("Bond Cutting")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        bond_id_1_str = st.text_input(
                            "Bond ID 1 to cut:",
                            value="0",
                            help=f"Enter first bond ID (0 to {mol.GetNumBonds()-1})",
                            key=f"bond1_{st.session_state.current_compound_index}"
                        )
                    
                    with col2:
                        bond_id_2_str = st.text_input(
                            "Bond ID 2 to cut:",
                            value="1" if mol.GetNumBonds() > 1 else "0",
                            help=f"Enter second bond ID (0 to {mol.GetNumBonds()-1}) - REQUIRED",
                            key=f"bond2_{st.session_state.current_compound_index}"
                        )
                    
                    # Parse and validate bond IDs
                    try:
                        bond_id_1 = int(bond_id_1_str) if bond_id_1_str.strip().isdigit() else 0
                        bond_id_2 = int(bond_id_2_str) if bond_id_2_str.strip().isdigit() else 1
                        
                        # Clamp values to valid range
                        bond_id_1 = max(0, min(bond_id_1, mol.GetNumBonds()-1))
                        bond_id_2 = max(0, min(bond_id_2, mol.GetNumBonds()-1))
                        
                        # Ensure two different bonds are selected
                        if bond_id_1 == bond_id_2:
                            st.error("⚠️ Bond ID 1 and Bond ID 2 must be different! Please select two different bonds.")
                            bonds_to_cut = None
                        else:
                            bonds_to_cut = [bond_id_1, bond_id_2]
                        
                    except (ValueError, TypeError):
                        bond_id_1 = 0
                        bond_id_2 = 1 if mol.GetNumBonds() > 1 else 0
                        st.warning("Invalid bond ID entered. Using default values.")
                        bonds_to_cut = [bond_id_1, bond_id_2] if bond_id_1 != bond_id_2 else None
                    
                    # Process cutting - only proceed if we have valid bonds
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Only show button if bonds are valid
                        if bonds_to_cut is not None:
                            if st.button("Cut Bonds and Generate Fragments", key=f"cut_{st.session_state.current_compound_index}"):
                                with st.spinner("Processing..."):
                                    fragments = generate_fragments(selected_smiles, bonds_to_cut)
                                    
                                    if fragments and fragments['fragments']:
                                        st.success("✅ PROTAC fragments generated successfully!")
                                        
                                        # Create result row
                                        result_row = {
                                            'Chunk': selected_chunk_idx + 1,
                                            'Compound_ID': selected_row['Compound ID'],
                                            'Target': selected_row['Target'],
                                            'Original_InChI_Key': fragments['original_inchi_key'],
                                            'Original_SMILES': selected_smiles,
                                            'Fragment_1_SMILES': fragments['fragments'][0],  # End piece 1
                                            'Fragment_2_SMILES': fragments['fragments'][1],  # Linker (2 [2H])
                                            'Fragment_3_SMILES': fragments['fragments'][2]   # End piece 2
                                        }
                                        
                                        # Add to session state
                                        st.session_state.processed_data.append(result_row)
                                        
                                        # Display fragments with clear labels
                                        st.write("**Fragment 1 (End piece):** `" + fragments['fragments'][0] + "`")
                                        st.write("**Fragment 2 (Linker):** `" + fragments['fragments'][1] + "`")
                                        st.write("**Fragment 3 (End piece):** `" + fragments['fragments'][2] + "`")
                                        
                                        # Auto-advance to next compound
                                        st.session_state.current_compound_index += 1
                                        
                                        # Auto-save chunk when completed
                                        if st.session_state.current_compound_index >= len(current_chunk_data):
                                            # Save current chunk automatically
                                            df_results = pd.DataFrame(st.session_state.processed_data)
                                            current_chunk_results = df_results[df_results['Chunk'] == selected_chunk_idx + 1]
                                            if len(current_chunk_results) > 0:
                                                filename = f"protac_fragments_chunk_{selected_chunk_idx + 1}.csv"
                                                current_chunk_results.to_csv(filename, index=False)
                                                st.success(f"✅ Chunk {selected_chunk_idx + 1} completed and saved as {filename}")
                                        
                                        st.rerun()
                                    
                                    else:
                                        st.error("❌ Could not generate proper PROTAC fragments. Please check bond selection.")
                        else:
                            st.error("⚠️ Please select two different bonds before cutting.")
                    
                    with col2:
                        if st.session_state.current_compound_index < len(current_chunk_data):
                            if st.button("Skip Current Compound", key=f"skip_{st.session_state.current_compound_index}"):
                                st.session_state.current_compound_index += 1
                                st.rerun()
                
                # Display processed data
                if st.session_state.processed_data:
                    st.subheader("Processed Data Summary")
                    
                    # Convert to DataFrame for display
                    df_results = pd.DataFrame(st.session_state.processed_data)
                    
                    # Filter by current chunk for display
                    current_chunk_results = df_results[df_results['Chunk'] == selected_chunk_idx + 1] if 'Chunk' in df_results.columns else df_results
                    
                    if len(current_chunk_results) > 0:
                        st.write(f"**Results for Chunk {selected_chunk_idx + 1}:** {len(current_chunk_results)} compounds processed")
                        st.dataframe(current_chunk_results)
                        
                        # Always show download button prominently
                        csv_chunk = current_chunk_results.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download Chunk {selected_chunk_idx + 1} Results",
                            data=csv_chunk,
                            file_name=f"protac_fragments_chunk_{selected_chunk_idx + 1}.csv",
                            mime="text/csv",
                            key="download_chunk_results"
                        )
                        
                        # Auto-save info
                        st.info(f"💾 Results are automatically saved as: protac_fragments_chunk_{selected_chunk_idx + 1}.csv")
                    else:
                        st.info("No compounds processed in this chunk yet.")
                    
                    # Show overall progress
                    st.subheader("📊 Overall Progress")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_processed = len(df_results)
                        st.metric("Total Processed", total_processed)
                    
                    with col2:
                        chunks_with_data = df_results['Chunk'].nunique() if 'Chunk' in df_results.columns else 1
                        st.metric("Chunks with Data", chunks_with_data)
                    
                    with col3:
                        completion_rate = (total_processed / len(unique_data)) * 100
                        st.metric("Completion %", f"{completion_rate:.1f}%")
                    
                    with col4:
                        current_chunk_progress = len(current_chunk_results)
                        st.metric(f"Chunk {selected_chunk_idx + 1} Progress", f"{current_chunk_progress}/{len(current_chunk_data)}")
                    
                    # Progress bar for overall completion (capped at 1.0)
                    st.progress(min(1.0, completion_rate / 100))
                    
                    # Download all results
                    if total_processed > 0:
                        csv_all = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Results (All Chunks)",
                            data=csv_all,
                            file_name="protac_fragments_all_chunks.csv",
                            mime="text/csv"
                        )
                    
                    # Chunk management buttons
                    st.subheader("🔧 Chunk Management")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Clear Current Chunk Results"):
                            # Remove results from current chunk
                            if 'Chunk' in df_results.columns:
                                st.session_state.processed_data = [
                                    item for item in st.session_state.processed_data 
                                    if item.get('Chunk') != selected_chunk_idx + 1
                                ]
                            st.rerun()
                    
                    with col2:
                        if st.button("Clear All Results"):
                            st.session_state.processed_data = []
                            st.rerun()
                    
                    with col3:
                        if st.button("Export Chunk Summary"):
                            # Create summary of all chunks
                            summary_data = []
                            for chunk_idx in range(len(chunks)):
                                chunk_results = df_results[df_results['Chunk'] == chunk_idx + 1] if 'Chunk' in df_results.columns else df_results
                                summary_data.append({
                                    'Chunk': chunk_idx + 1,
                                    'Total_Compounds': len(chunks[chunk_idx]),
                                    'Processed_Compounds': len(chunk_results),
                                    'Completion_Rate': f"{(len(chunk_results) / len(chunks[chunk_idx])) * 100:.1f}%"
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            csv_summary = summary_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Progress Summary",
                                data=csv_summary,
                                file_name="protac_chunks_progress_summary.csv",
                                mime="text/csv"
                            )
                
                else:
                    st.info("No compounds processed yet. Start by selecting a molecule and cutting bonds!")
                    
                    # Quick navigation to chunks
                    st.subheader("🗂️ Quick Chunk Navigation")
                    chunk_cols = st.columns(min(5, len(chunks)))
                    for i, col in enumerate(chunk_cols):
                        if i < len(chunks):
                            with col:
                                if st.button(f"Go to Chunk {i+1}"):
                                    st.session_state.current_chunk = i
                                    st.rerun()

if __name__ == "__main__":
    main()
