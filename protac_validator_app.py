import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="PROTAC Fragment Validator",
    page_icon="🧬",
    layout="wide"
)

def draw_molecule(smiles, width=300, height=300, title="", is_original=False):
    """
    Draw a molecule from SMILES string
    
    Args:
        smiles (str): SMILES string
        width (int): Image width
        height (int): Image height
        title (str): Title for the molecule
        is_original (bool): If True, use smaller size and finer lines for original PROTAC
    
    Returns:
        PIL.Image or None: Molecule image
    """
    try:
        if not smiles or pd.isna(smiles):
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Adjust size and line width for original PROTAC
        if is_original:
            # Smaller dimensions for original PROTAC
            width = min(width, 250)
            height = min(height, 250)
            bond_width = 1.0
        else:
            # Normal size for fragments
            bond_width = 1.5
        
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        
        # Set drawing options
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.bondLineWidth = bond_width
        opts.atomLabelFontSize = 12 if is_original else 14
        opts.legendFontSize = 10 if is_original else 12
        
        # Draw molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Convert to image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        return img
    
    except Exception as e:
        st.error(f"Error drawing molecule: {str(e)}")
        return None

def get_fragment_info(row):
    """
    Extract fragment information and organize by type
    
    Args:
        row: DataFrame row
    
    Returns:
        dict: Organized fragment information
    """
    fragments = {}
    
    # Check Fragment_1
    if 'Fragment_1_TYPE' in row and 'Fragment_1_SMILES' in row:
        frag_type = row['Fragment_1_TYPE']
        frag_smiles = row['Fragment_1_SMILES']
        fragments[frag_type] = frag_smiles
    
    # Check Fragment_2 (should be linker)
    if 'Fragment_2_SMILES' in row:
        fragments['Linker'] = row['Fragment_2_SMILES']
    
    # Check Fragment_3
    if 'Fragment_3_TYPE' in row and 'Fragment_3_SMILES' in row:
        frag_type = row['Fragment_3_TYPE']
        frag_smiles = row['Fragment_3_SMILES']
        fragments[frag_type] = frag_smiles
    
    return fragments

def display_molecule_row(molecules, titles, heights=300, is_original_row=False):
    """
    Display a row of molecules with titles
    
    Args:
        molecules (list): List of SMILES strings
        titles (list): List of titles for molecules
        heights (int): Height for molecule images
        is_original_row (bool): If True, this is the original PROTAC row
    """
    if is_original_row:
        # For original PROTAC, center it in a single column
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            mol_smiles, title = molecules[0], titles[0]
            st.markdown(f"**{title}**", unsafe_allow_html=True)
            
            if mol_smiles and not pd.isna(mol_smiles):
                img = draw_molecule(mol_smiles, width=400, height=250, is_original=True)
                if img:
                    st.image(img, use_column_width=True)
                    st.caption(f"SMILES: `{mol_smiles[:60]}{'...' if len(mol_smiles) > 60 else ''}`")
                else:
                    st.error("Could not render molecule")
            else:
                st.warning("No SMILES available")
    else:
        # For fragments, use normal 3-column layout
        cols = st.columns(len(molecules))
        
        for i, (mol_smiles, title) in enumerate(zip(molecules, titles)):
            with cols[i]:
                st.markdown(f"**{title}**")
                
                if mol_smiles and not pd.isna(mol_smiles):
                    img = draw_molecule(mol_smiles, width=300, height=heights, is_original=False)
                    if img:
                        st.image(img, use_column_width=True)
                        st.caption(f"SMILES: `{mol_smiles[:50]}{'...' if len(mol_smiles) > 50 else ''}`")
                    else:
                        st.error("Could not render molecule")
                else:
                    st.warning("No SMILES available")

def main():
    st.title("🧬 PROTAC Fragment Validation App")
    st.markdown("Upload your standardized PROTAC fragments CSV to validate the fragmentation process")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your standardized PROTAC fragments CSV file", 
        type="csv",
        help="Upload the protac_fragments_with_types_*_standardized.csv file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} PROTAC records")
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            
            with col2:
                unique_targets = df['Target'].nunique() if 'Target' in df.columns else 0
                st.metric("Unique Targets", unique_targets)
            
            with col3:
                warhead_count = (df['Fragment_1_TYPE'] == 'Warhead').sum() if 'Fragment_1_TYPE' in df.columns else 0
                st.metric("Warheads in Frag_1", f"{warhead_count}/{len(df)}")
            
            with col4:
                e3_count = (df['Fragment_3_TYPE'] == 'E3 Binder').sum() if 'Fragment_3_TYPE' in df.columns else 0
                st.metric("E3 Binders in Frag_3", f"{e3_count}/{len(df)}")
            
            # Validation summary
            if 'Fragment_1_TYPE' in df.columns and 'Fragment_3_TYPE' in df.columns:
                warhead_pct = (warhead_count / len(df)) * 100
                e3_pct = (e3_count / len(df)) * 100
                
                if warhead_pct >= 95 and e3_pct >= 95:
                    st.success(f"🎉 Excellent! Standardization is {warhead_pct:.1f}% and {e3_pct:.1f}% consistent")
                elif warhead_pct >= 80 and e3_pct >= 80:
                    st.warning(f"⚠️ Good but could be better: {warhead_pct:.1f}% and {e3_pct:.1f}% consistent")
                else:
                    st.error(f"❌ Poor standardization: {warhead_pct:.1f}% and {e3_pct:.1f}% consistent")
            
            st.markdown("---")
            
            # Create selection options
            if 'Compound_ID' in df.columns and 'Target' in df.columns:
                selection_options = [
                    f"ID: {row['Compound_ID']} | Target: {row['Target']}" + 
                    (f" | {row['Name']}" if 'Name' in df.columns and not pd.isna(row['Name']) else "")
                    for idx, row in df.iterrows()
                ]
            else:
                selection_options = [f"Record {i+1}" for i in range(len(df))]
            
            # Initialize session state for selected index
            if 'selected_idx' not in st.session_state:
                st.session_state.selected_idx = 0
            
            # Record selection
            st.subheader("🔍 Select PROTAC to Validate")
            
            # Search and filter options
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_idx = st.selectbox(
                    "Choose a PROTAC record to visualize:",
                    range(len(df)),
                    index=st.session_state.selected_idx,
                    format_func=lambda x: selection_options[x],
                    help="Select any record to see the original PROTAC and its fragments",
                    key="record_selector"
                )
                # Update session state when selectbox changes
                st.session_state.selected_idx = selected_idx
            
            with col2:
                # Quick navigation
                st.markdown("**Quick Jump:**")
                if st.button("🎲 Random Record"):
                    # Generate random index and update session state
                    new_random_idx = np.random.randint(0, len(df))
                    st.session_state.selected_idx = new_random_idx
                    st.rerun()
            
            # Display selected record information
            if selected_idx is not None:
                selected_row = df.iloc[selected_idx]
                
                st.markdown("---")
                st.subheader(f"📋 Record Details (Row {selected_idx + 1})")
                
                # Display record information
                info_cols = st.columns(3)
                
                with info_cols[0]:
                    if 'Compound_ID' in df.columns:
                        st.metric("Compound ID", selected_row['Compound_ID'])
                
                with info_cols[1]:
                    if 'Target' in df.columns:
                        st.metric("Target", selected_row['Target'])
                
                with info_cols[2]:
                    if 'Name' in df.columns and not pd.isna(selected_row['Name']):
                        st.metric("Name", selected_row['Name'])
                
                # Display additional metrics if available
                if any(col in df.columns for col in ['DC50 (nM)', 'Dmax (%)']):
                    metrics_cols = st.columns(2)
                    
                    with metrics_cols[0]:
                        if 'DC50 (nM)' in df.columns and not pd.isna(selected_row['DC50 (nM)']):
                            st.metric("DC50 (nM)", selected_row['DC50 (nM)'])
                    
                    with metrics_cols[1]:
                        if 'Dmax (%)' in df.columns and not pd.isna(selected_row['Dmax (%)']):
                            st.metric("Dmax (%)", selected_row['Dmax (%)'])
                
                st.markdown("---")
                
                # Original PROTAC visualization
                st.subheader("🧪 Original PROTAC")
                
                if 'Original_SMILES' in df.columns:
                    original_smiles = selected_row['Original_SMILES']
                    display_molecule_row([original_smiles], ["Original PROTAC"], heights=250, is_original_row=True)
                else:
                    st.error("Original_SMILES column not found in the dataset")
                
                st.markdown("---")
                
                # Fragment visualization
                st.subheader("🔬 PROTAC Fragments")
                st.markdown("**Fragmentation Result: Warhead → Linker → E3 Binder**")
                
                # Get fragment information
                fragments = get_fragment_info(selected_row)
                
                # Order fragments: Warhead, Linker, E3 Binder
                fragment_order = ['Warhead', 'Linker', 'E3 Binder']
                fragment_smiles = []
                fragment_titles = []
                
                for frag_type in fragment_order:
                    if frag_type in fragments:
                        fragment_smiles.append(fragments[frag_type])
                        fragment_titles.append(frag_type)
                    else:
                        fragment_smiles.append(None)
                        fragment_titles.append(f"{frag_type} (Missing)")
                
                # Display fragments in a row
                display_molecule_row(fragment_smiles, fragment_titles, heights=280, is_original_row=False)
                
                # Display fragment details
                st.markdown("---")
                st.subheader("📊 Fragment Details")
                
                frag_details_cols = st.columns(3)
                
                for i, (frag_type, title) in enumerate(zip(fragment_order, fragment_titles)):
                    with frag_details_cols[i]:
                        st.markdown(f"**{title}**")
                        
                        if frag_type in fragments and fragments[frag_type]:
                            smiles = fragments[frag_type]
                            st.code(smiles, language=None)
                            
                            # Count deuterium
                            import re
                            deuterium_count = len(re.findall(r'\[2H\]', smiles))
                            st.caption(f"Deuterium atoms: {deuterium_count}")
                            
                            # Molecule properties
                            try:
                                mol = Chem.MolFromSmiles(smiles)
                                if mol:
                                    st.caption(f"Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}")
                            except:
                                pass
                        else:
                            st.warning("No SMILES data available")
                
                # Validation checks
                st.markdown("---")
                st.subheader("✅ Validation Checks")
                
                checks_cols = st.columns(3)
                
                with checks_cols[0]:
                    # Check if Fragment_1 is Warhead
                    if 'Fragment_1_TYPE' in df.columns:
                        is_warhead = selected_row['Fragment_1_TYPE'] == 'Warhead'
                        if is_warhead:
                            st.success("✅ Fragment_1 is Warhead")
                        else:
                            st.error(f"❌ Fragment_1 is {selected_row['Fragment_1_TYPE']}")
                
                with checks_cols[1]:
                    # Check if Fragment_2 is Linker (has 2 deuterium)
                    if 'Fragment_2_SMILES' in df.columns:
                        frag2_smiles = selected_row['Fragment_2_SMILES']
                        if frag2_smiles and not pd.isna(frag2_smiles):
                            import re
                            deuterium_count = len(re.findall(r'\[2H\]', frag2_smiles))
                            if deuterium_count == 2:
                                st.success("✅ Fragment_2 is Linker (2 [2H])")
                            else:
                                st.warning(f"⚠️ Fragment_2 has {deuterium_count} [2H]")
                        else:
                            st.error("❌ No Fragment_2 SMILES")
                
                with checks_cols[2]:
                    # Check if Fragment_3 is E3 Binder
                    if 'Fragment_3_TYPE' in df.columns:
                        is_e3 = selected_row['Fragment_3_TYPE'] == 'E3 Binder'
                        if is_e3:
                            st.success("✅ Fragment_3 is E3 Binder")
                        else:
                            st.error(f"❌ Fragment_3 is {selected_row['Fragment_3_TYPE']}")
            
            # Data preview
            st.markdown("---")
            st.subheader("📋 Dataset Preview")
            
            # Show relevant columns
            preview_columns = ['Compound_ID', 'Target', 'Fragment_1_TYPE', 'Fragment_2_SMILES', 'Fragment_3_TYPE']
            available_preview_cols = [col for col in preview_columns if col in df.columns]
            
            if available_preview_cols:
                st.dataframe(df[available_preview_cols].head(10), use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)
            
            # Download section
            st.markdown("---")
            st.subheader("💾 Export Options")
            
            if st.button("📊 Generate Validation Report"):
                # Create a simple validation report
                report_data = []
                
                for idx, row in df.iterrows():
                    checks = {
                        'Compound_ID': row.get('Compound_ID', f'Record_{idx+1}'),
                        'Target': row.get('Target', 'Unknown'),
                        'Fragment_1_is_Warhead': row.get('Fragment_1_TYPE') == 'Warhead' if 'Fragment_1_TYPE' in df.columns else None,
                        'Fragment_3_is_E3Binder': row.get('Fragment_3_TYPE') == 'E3 Binder' if 'Fragment_3_TYPE' in df.columns else None,
                        'Has_Original_SMILES': not pd.isna(row.get('Original_SMILES', None)),
                        'Has_All_Fragments': all(not pd.isna(row.get(col, None)) for col in ['Fragment_1_SMILES', 'Fragment_2_SMILES', 'Fragment_3_SMILES'] if col in df.columns)
                    }
                    report_data.append(checks)
                
                report_df = pd.DataFrame(report_data)
                
                st.success("Validation report generated!")
                st.dataframe(report_df, use_container_width=True)
                
                # Download validation report
                csv_report = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Validation Report",
                    data=csv_report,
                    file_name="protac_validation_report.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            st.info("Please ensure you've uploaded a valid CSV file with the expected columns.")
    
    else:
        st.info("👆 Please upload your standardized PROTAC fragments CSV file to begin validation")
        
        # Show expected format
        st.markdown("### 📋 Expected File Format")
        st.markdown("""
        Your CSV should contain these columns:
        - `Original_SMILES`: Original PROTAC molecule
        - `Fragment_1_SMILES` & `Fragment_1_TYPE`: Should be Warhead
        - `Fragment_2_SMILES`: Linker (with 2 deuterium atoms)  
        - `Fragment_3_SMILES` & `Fragment_3_TYPE`: Should be E3 Binder
        - `Compound_ID`, `Target`: For identification
        """)

if __name__ == "__main__":
    main()
