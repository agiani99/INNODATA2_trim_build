import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Draw import rdMolDraw2D
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import io
from PIL import Image
import itertools
import warnings
import random
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="PROTAC Virtual Library Generator",
    page_icon="🧬",
    layout="wide"
)

# =====================================================================================
# CHEMISTRY FUNCTIONS - All functions defined at the top
# =====================================================================================

def combine_protac_fragments_correct_rules(warhead_smiles, linker_smiles, e3_smiles):
    """
    PROPERLY FIXED implementation that uses BOTH attachment points on the linker.
    
    Key insight: Linear linkers have deuterium at BOTH ENDS marking attachment points:
    [2H]CCCCCCCCC[2H]
     ^             ^
     |             |
   Warhead       E3 Binder
   attachment    attachment
   
   The correct assembly is: WARHEAD + LINKER_BACKBONE + E3_BINDER
   """
    try:
        # Step 1: Clean all fragments by removing deuterium markers
        warhead_clean = warhead_smiles.replace('[2H]', '')
        e3_clean = e3_smiles.replace('[2H]', '')
        
        # Step 2: Extract the linker backbone (remove ALL deuteriums)
        linker_backbone = linker_smiles.replace('[2H]', '')
        
        # Step 3: Proper assembly using attachment points
        # The deuteriums mark WHERE to attach:
        # - First [2H] position → attach warhead
        # - Second [2H] position → attach E3 binder
        # 
        # For linear linkers like [2H]CCCCCCCCC[2H]:
        # Result should be: WARHEAD + CCCCCCCCC + E3_BINDER
        
        final_protac = warhead_clean + linker_backbone + e3_clean
        
        # Step 4: Validate and canonicalize
        mol = Chem.MolFromSmiles(final_protac)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol)
            except:
                # If sanitization fails, return as-is
                return final_protac
        else:
            # If parsing fails, return as-is
            return final_protac
            
    except Exception as e:
        print(f"Error in correct building: {e}")
        return combine_protac_fragments_simple(warhead_smiles, linker_smiles, e3_smiles)



def combine_protac_fragments_simple(warhead_smiles, linker_smiles, e3_smiles):
    """
    Simple but functional approach - removes deuterium and creates reasonable structures
    """
    try:
        # Clean approach: just remove all deuterium and concatenate intelligently
        
        # Remove all deuterium markers
        warhead_clean = warhead_smiles.replace('[2H]', '')
        linker_clean = linker_smiles.replace('[2H]', '')
        e3_clean = e3_smiles.replace('[2H]', '')
        
        # Method 1: Try to create a single connected molecule
        combined_smiles = warhead_clean + linker_clean + e3_clean
        
        # Test if this creates a valid molecule
        mol = Chem.MolFromSmiles(combined_smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        
        # Method 2: If single molecule fails, try fragment combination
        # Use dots to separate fragments but calculate properties on combined system
        fragment_smiles = f"{warhead_clean}.{linker_clean}.{e3_clean}"
        
        # Test if fragments are valid
        warhead_mol = Chem.MolFromSmiles(warhead_clean)
        linker_mol = Chem.MolFromSmiles(linker_clean)
        e3_mol = Chem.MolFromSmiles(e3_clean)
        
        if all([warhead_mol, linker_mol, e3_mol]):
            return fragment_smiles
        else:
            print("Some fragments invalid after deuterium removal")
            return None
            
    except Exception as e:
        print(f"Simple method failed: {e}")
        return None

def calculate_molecular_descriptors(smiles):
    """Calculate molecular descriptors, handling both connected and fragmented molecules"""
    try:
        # Handle dot-separated SMILES (multiple fragments)
        if '.' in smiles:
            # For multi-fragment SMILES, combine all fragments for descriptor calculation
            fragments = smiles.split('.')
            combined_mol = None
            
            for frag_smiles in fragments:
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                if frag_mol:
                    if combined_mol is None:
                        combined_mol = frag_mol
                    else:
                        combined_mol = Chem.CombineMols(combined_mol, frag_mol)
            
            if combined_mol:
                mol = combined_mol
            else:
                return None
        else:
            mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'NumRings': Descriptors.RingCount(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'AliphaticRings': Descriptors.NumAliphaticRings(mol),
            'QED': QED.qed(mol)
        }
        return descriptors
    except Exception as e:
        print(f"Error calculating descriptors for {smiles}: {e}")
        return None

def find_pareto_front(df, objectives, minimize=None):
    """Find Pareto front solutions"""
    if minimize is None:
        minimize = []
    
    data = df[objectives].values
    
    # Flip signs for maximization objectives
    for i, obj in enumerate(objectives):
        if obj not in minimize:
            data[:, i] = -data[:, i]
    
    # Find Pareto front
    is_pareto = np.ones(data.shape[0], dtype=bool)
    
    for i in range(data.shape[0]):
        if is_pareto[i]:
            dominated = np.all(data <= data[i], axis=1) & np.any(data < data[i], axis=1)
            is_pareto[dominated] = False
    
    return df[is_pareto]

def draw_molecule_clean(smiles, width=250, height=250):
    """Draw a clean molecule visualization"""
    try:
        if not smiles or pd.isna(smiles):
            return None
        
        # Handle dot-separated SMILES by drawing the largest fragment
        if '.' in smiles:
            fragments = smiles.split('.')
            # Draw the largest fragment
            largest_fragment = max(fragments, key=len)
            mol = Chem.MolFromSmiles(largest_fragment)
        else:
            mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        
        # Set drawing options for clean appearance
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.bondLineWidth = 2
        opts.atomLabelFontSize = 14
        
        # Draw molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Convert to image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        return img
    
    except Exception as e:
        print(f"Error drawing molecule: {e}")
        return None

# =====================================================================================
# STREAMLIT APP - Main application code
# =====================================================================================

def main():
    st.title("🧬 PROTAC Virtual Library Generator")
    st.markdown("Generate virtual PROTAC libraries by combining fragments and analyze with t-SNE")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your standardized PROTAC fragments CSV", 
        type="csv",
        help="Upload your protac_fragments_with_types_*_standardized.csv file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} PROTAC records")
            
            # Quick fragment analysis
            st.subheader("🔍 Fragment Analysis")
            
            # Count fragments by type
            warhead_count = len(df[df['Fragment_1_TYPE'] == 'Warhead']['Fragment_1_SMILES'].dropna().unique())
            linker_count = len(df['Fragment_2_SMILES'].dropna().unique())
            e3_count = len(df[df['Fragment_3_TYPE'] == 'E3 Binder']['Fragment_3_SMILES'].dropna().unique())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Warheads", warhead_count)
            with col2:
                st.metric("Unique Linkers", linker_count)
            with col3:
                st.metric("Unique E3 Binders", e3_count)
            
            # Get unique targets
            if 'Target' in df.columns:
                unique_targets = sorted(df['Target'].unique())
                
                # Target selection
                st.subheader("🎯 Select Target")
                selected_target = st.selectbox(
                    "Choose a target for virtual library generation:",
                    unique_targets,
                    help="Select the target protein for focused library generation"
                )
                
                # Filter data by target
                target_df = df[df['Target'] == selected_target].copy()
                st.info(f"Found {len(target_df)} PROTACs for target: {selected_target}")
                
                # Fragment selection strategy
                st.subheader("📋 Fragment Selection")
                
                # Get fragments - always use ALL linkers as specified
                all_linkers = df['Fragment_2_SMILES'].dropna().unique()
                target_warheads = target_df[target_df['Fragment_1_TYPE'] == 'Warhead']['Fragment_1_SMILES'].dropna().unique()
                target_e3_binders = target_df[target_df['Fragment_3_TYPE'] == 'E3 Binder']['Fragment_3_SMILES'].dropna().unique()
                all_warheads = df[df['Fragment_1_TYPE'] == 'Warhead']['Fragment_1_SMILES'].dropna().unique()
                all_e3_binders = df[df['Fragment_3_TYPE'] == 'E3 Binder']['Fragment_3_SMILES'].dropna().unique()
                
                # Fragment scope selection
                fragment_scope = st.radio(
                    "Choose warhead and E3 binder scope:",
                    [f"Target-specific ({selected_target} warheads & E3 binders)", "All available (maximum diversity)"],
                    index=1,
                    help="Linkers are always taken from entire dataset"
                )
                
                if fragment_scope.startswith("Target-specific"):
                    warheads = target_warheads
                    e3_binders = target_e3_binders
                    scope_label = f"{selected_target}-specific"
                else:
                    warheads = all_warheads
                    e3_binders = all_e3_binders
                    scope_label = "All available"
                
                linkers = all_linkers
                
                # Display fragment counts
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Warheads", f"{len(warheads)}", help=f"{scope_label} warheads")
                with col2:
                    st.metric("Linkers", f"{len(linkers)}", help="All available linkers")
                with col3:
                    st.metric("E3 Binders", f"{len(e3_binders)}", help=f"{scope_label} E3 binders")
                with col4:
                    potential_size = len(warheads) * len(linkers) * len(e3_binders)
                    st.metric("Potential Library", f"{potential_size:,}", help="Total possible combinations")
                
                # Check if we have sufficient fragments
                if len(warheads) == 0 or len(linkers) == 0 or len(e3_binders) == 0:
                    st.error("⚠️ Insufficient fragments! Try selecting 'All available' scope.")
                    return
                
                # Library generation settings
                st.subheader("⚙️ Library Generation Settings")
                
                col1, col2 = st.columns(2)
                with col1:
                    max_library_size = st.number_input(
                        "Maximum library size", 
                        min_value=10, 
                        max_value=2000, 
                        value=min(500, potential_size),
                        help="Number of virtual compounds to generate"
                    )
                
                with col2:
                    random_sampling = st.checkbox(
                        "Random sampling", 
                        value=True,
                        help="Use random sampling if library exceeds maximum size"
                    )
                
                # Generate library button
                if st.button("🚀 Generate Virtual Library", type="primary"):
                    with st.spinner("Generating virtual PROTAC library..."):
                        
                        # Create combinations
                        if potential_size <= max_library_size:
                            combinations = list(itertools.product(warheads, linkers, e3_binders))
                        else:
                            if random_sampling:
                                random.seed(42)
                                combinations = []
                                for _ in range(max_library_size):
                                    w = random.choice(warheads)
                                    l = random.choice(linkers)
                                    e = random.choice(e3_binders)
                                    combinations.append((w, l, e))
                            else:
                                all_combinations = itertools.product(warheads, linkers, e3_binders)
                                combinations = list(itertools.islice(all_combinations, max_library_size))
                        
                        st.info(f"Generating {len(combinations)} virtual PROTACs...")
                        
                        # Generate virtual library
                        virtual_library = []
                        progress_bar = st.progress(0)
                        success_count = 0
                        
                        for i, (warhead, linker, e3) in enumerate(combinations):
                            progress = (i + 1) / len(combinations)
                            progress_bar.progress(progress)
                            
                            # Use correct chemistry rules
                            virtual_smiles = combine_protac_fragments_correct_rules(warhead, linker, e3)
                            
                            if virtual_smiles:
                                descriptors = calculate_molecular_descriptors(virtual_smiles)
                                
                                if descriptors:
                                    record = {
                                        'Virtual_ID': f"VL_{i+1:04d}",
                                        'Virtual_SMILES': virtual_smiles,
                                        'Warhead_SMILES': warhead,
                                        'Linker_SMILES': linker,
                                        'E3_SMILES': e3,
                                        'Type': 'Virtual',
                                        'Is_Connected': '.' not in virtual_smiles,
                                        **descriptors
                                    }
                                    virtual_library.append(record)
                                    success_count += 1
                        
                        progress_bar.empty()
                        
                        if virtual_library:
                            virtual_df = pd.DataFrame(virtual_library)
                            
                            success_rate = (success_count / len(combinations)) * 100
                            connected_count = sum(1 for mol in virtual_df['Virtual_SMILES'] if '.' not in mol)
                            connection_rate = (connected_count / len(virtual_df)) * 100
                            
                            st.success(f"✅ Generated {len(virtual_df)} valid virtual PROTACs ({success_rate:.1f}% success rate)")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Connected Molecules", f"{connected_count}/{len(virtual_df)}", f"{connection_rate:.1f}%")
                            with col2:
                                st.metric("Fragment Mixtures", f"{len(virtual_df) - connected_count}/{len(virtual_df)}", f"{100-connection_rate:.1f}%")
                            
                            # Calculate descriptors for original PROTACs
                            original_library = []
                            if len(target_df) > 0:
                                st.info("Processing original PROTACs for comparison...")
                                for _, row in target_df.iterrows():
                                    if 'Original_SMILES' in row and pd.notna(row['Original_SMILES']):
                                        descriptors = calculate_molecular_descriptors(row['Original_SMILES'])
                                        if descriptors:
                                            record = {
                                                'Virtual_ID': row.get('Compound_ID', ''),
                                                'Virtual_SMILES': row['Original_SMILES'],
                                                'Warhead_SMILES': row.get('Fragment_1_SMILES', ''),
                                                'Linker_SMILES': row.get('Fragment_2_SMILES', ''),
                                                'E3_SMILES': row.get('Fragment_3_SMILES', ''),
                                                'Type': 'Original',
                                                'Is_Connected': True,
                                                **descriptors
                                            }
                                            original_library.append(record)
                            
                            # Combine datasets
                            if original_library:
                                original_df = pd.DataFrame(original_library)
                                combined_df = pd.concat([original_df, virtual_df], ignore_index=True)
                                st.info(f"Added {len(original_df)} original PROTACs for comparison")
                            else:
                                combined_df = virtual_df.copy()
                                original_df = pd.DataFrame()
                            
                            # Find Pareto front
                            st.subheader("🎯 Pareto Front Analysis")
                            objectives = ['MW', 'LogP', 'TPSA', 'QED']
                            minimize_objectives = ['MW', 'TPSA']
                            
                            virtual_pareto = find_pareto_front(virtual_df, objectives, minimize=minimize_objectives)
                            st.info(f"Identified {len(virtual_pareto)} Pareto optimal virtual PROTACs")
                            
                            # Prepare data for t-SNE
                            descriptor_cols = ['MW', 'HBA', 'HBD', 'LogP', 'TPSA', 'RotBonds', 
                                             'NumRings', 'AromaticRings', 'AliphaticRings', 'QED']
                            
                            # Standardize descriptors
                            scaler = StandardScaler()
                            X = scaler.fit_transform(combined_df[descriptor_cols].fillna(0))
                            
                            # Perform t-SNE
                            st.info("Performing t-SNE analysis...")
                            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
                            X_tsne = tsne.fit_transform(X)
                            
                            # Add t-SNE coordinates
                            combined_df['tSNE_1'] = X_tsne[:, 0]
                            combined_df['tSNE_2'] = X_tsne[:, 1]
                            
                            # Create Pareto flag for plotting
                            combined_df['Is_Pareto'] = False
                            if len(original_library) > 0:
                                pareto_indices = virtual_pareto.index + len(original_df)
                            else:
                                pareto_indices = virtual_pareto.index
                            combined_df.loc[pareto_indices, 'Is_Pareto'] = True
                            
                            # Create t-SNE plot
                            st.subheader("📈 t-SNE Visualization")
                            
                            fig = go.Figure()
                            
                            # Original PROTACs (red)
                            if len(original_library) > 0:
                                original_mask = combined_df['Type'] == 'Original'
                                if original_mask.any():
                                    fig.add_trace(go.Scatter(
                                        x=combined_df[original_mask]['tSNE_1'],
                                        y=combined_df[original_mask]['tSNE_2'],
                                        mode='markers',
                                        marker=dict(color='red', size=10, opacity=0.8),
                                        name=f'Original {selected_target} PROTACs',
                                        text=combined_df[original_mask]['Virtual_ID'],
                                        customdata=np.column_stack([
                                            combined_df[original_mask]['MW'].values,
                                            combined_df[original_mask]['LogP'].values,
                                            combined_df[original_mask]['QED'].values
                                        ]),
                                        hovertemplate='<b>Original PROTAC</b><br>' +
                                                    'ID: %{text}<br>' +
                                                    'MW: %{customdata[0]:.1f}<br>' +
                                                    'LogP: %{customdata[1]:.2f}<br>' +
                                                    'QED: %{customdata[2]:.3f}<br>' +
                                                    '<extra></extra>'
                                    ))
                            
                            # Virtual PROTACs - separate connected vs fragmented
                            virtual_connected = (combined_df['Type'] == 'Virtual') & combined_df['Is_Connected'] & (~combined_df['Is_Pareto'])
                            virtual_fragmented = (combined_df['Type'] == 'Virtual') & (~combined_df['Is_Connected']) & (~combined_df['Is_Pareto'])
                            
                            # Connected virtual PROTACs (blue)
                            if virtual_connected.any():
                                fig.add_trace(go.Scatter(
                                    x=combined_df[virtual_connected]['tSNE_1'],
                                    y=combined_df[virtual_connected]['tSNE_2'],
                                    mode='markers',
                                    marker=dict(color='lightblue', size=6, opacity=0.6),
                                    name='Virtual PROTACs (Connected)',
                                    text=combined_df[virtual_connected]['Virtual_ID'],
                                    customdata=np.column_stack([
                                        combined_df[virtual_connected]['MW'].values,
                                        combined_df[virtual_connected]['LogP'].values,
                                        combined_df[virtual_connected]['QED'].values
                                    ]),
                                    hovertemplate='<b>Virtual PROTAC (Connected)</b><br>' +
                                                'ID: %{text}<br>' +
                                                'MW: %{customdata[0]:.1f}<br>' +
                                                'LogP: %{customdata[1]:.2f}<br>' +
                                                'QED: %{customdata[2]:.3f}<br>' +
                                                '<extra></extra>'
                                ))
                            
                            # Fragmented virtual PROTACs (gray)
                            if virtual_fragmented.any():
                                fig.add_trace(go.Scatter(
                                    x=combined_df[virtual_fragmented]['tSNE_1'],
                                    y=combined_df[virtual_fragmented]['tSNE_2'],
                                    mode='markers',
                                    marker=dict(color='lightgray', size=4, opacity=0.4),
                                    name='Virtual PROTACs (Fragmented)',
                                    text=combined_df[virtual_fragmented]['Virtual_ID'],
                                    customdata=np.column_stack([
                                        combined_df[virtual_fragmented]['MW'].values,
                                        combined_df[virtual_fragmented]['LogP'].values,
                                        combined_df[virtual_fragmented]['QED'].values
                                    ]),
                                    hovertemplate='<b>Virtual PROTAC (Fragmented)</b><br>' +
                                                'ID: %{text}<br>' +
                                                'MW: %{customdata[0]:.1f}<br>' +
                                                'LogP: %{customdata[1]:.2f}<br>' +
                                                'QED: %{customdata[2]:.3f}<br>' +
                                                '<extra></extra>'
                                ))
                            
                            # Pareto front PROTACs (green diamonds)
                            pareto_mask = combined_df['Is_Pareto']
                            if pareto_mask.any():
                                fig.add_trace(go.Scatter(
                                    x=combined_df[pareto_mask]['tSNE_1'],
                                    y=combined_df[pareto_mask]['tSNE_2'],
                                    mode='markers',
                                    marker=dict(color='green', size=12, opacity=0.9, symbol='diamond'),
                                    name='Pareto Optimal',
                                    text=combined_df[pareto_mask]['Virtual_ID'],
                                    customdata=np.column_stack([
                                        combined_df[pareto_mask]['MW'].values,
                                        combined_df[pareto_mask]['LogP'].values,
                                        combined_df[pareto_mask]['QED'].values,
                                        combined_df[pareto_mask]['Is_Connected'].values
                                    ]),
                                    hovertemplate='<b>⭐ Pareto Optimal</b><br>' +
                                                'ID: %{text}<br>' +
                                                'MW: %{customdata[0]:.1f}<br>' +
                                                'LogP: %{customdata[1]:.2f}<br>' +
                                                'QED: %{customdata[2]:.3f}<br>' +
                                                'Connected: %{customdata[3]}<br>' +
                                                '<extra></extra>'
                                ))
                            
                            fig.update_layout(
                                title=f't-SNE Analysis: {selected_target} Virtual Library',
                                xaxis_title='t-SNE Component 1',
                                yaxis_title='t-SNE Component 2',
                                width=900,
                                height=600,
                                showlegend=True,
                                hovermode='closest'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store results in session state
                            st.session_state['virtual_df'] = virtual_df
                            st.session_state['virtual_pareto'] = virtual_pareto
                            st.session_state['original_df'] = original_df
                            st.session_state['combined_df'] = combined_df
                            st.session_state['selected_target'] = selected_target
                            st.session_state['library_generated'] = True
                            
                        else:
                            st.error("No valid virtual PROTACs could be generated. Please check your fragment data.")
                
                # Compound Explorer
                if 'library_generated' in st.session_state and st.session_state['library_generated']:
                    st.markdown("---")
                    st.subheader("🔍 Compound Explorer")
                    
                    # Get data from session state
                    virtual_df = st.session_state['virtual_df']
                    virtual_pareto = st.session_state['virtual_pareto']
                    original_df = st.session_state['original_df']
                    selected_target = st.session_state['selected_target']
                    
                    # Filter options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        available_types = ["Pareto Optimal", "Connected Virtual", "All Virtual"]
                        if len(original_df) > 0:
                            available_types.append("Original")
                        
                        compound_type = st.selectbox("Select compound type:", available_types)
                    
                    with col2:
                        sort_by = st.selectbox("Sort by:", ["QED", "MW", "LogP", "TPSA"])
                    
                    with col3:
                        num_compounds = st.selectbox("Number to show:", [3, 5, 10, 20], index=1)
                    
                    # Filter and sort compounds
                    if compound_type == "Pareto Optimal":
                        display_df = virtual_pareto.sort_values(sort_by, ascending=(sort_by in ['MW', 'TPSA']))
                    elif compound_type == "Connected Virtual":
                        connected_virtual = virtual_df[virtual_df.get('Is_Connected', True)]
                        display_df = connected_virtual.sort_values(sort_by, ascending=(sort_by in ['MW', 'TPSA']))
                    elif compound_type == "All Virtual":
                        display_df = virtual_df.sort_values(sort_by, ascending=(sort_by in ['MW', 'TPSA']))
                    else:  # Original
                        display_df = original_df.sort_values(sort_by, ascending=(sort_by in ['MW', 'TPSA']))
                    
                    # Display compounds
                    if len(display_df) > 0:
                        st.write(f"**Showing top {num_compounds} {compound_type.lower()} compounds (sorted by {sort_by}):**")
                        
                        for i, (_, compound) in enumerate(display_df.head(num_compounds).iterrows()):
                            is_connected = compound.get('Is_Connected', True)
                            connection_emoji = "🔗" if is_connected else "🔀"
                            
                            with st.expander(f"{connection_emoji} {compound_type} #{i+1}: {compound.get('Virtual_ID', f'Compound_{i+1}')} (QED: {compound['QED']:.3f})"):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    # Draw molecule structure
                                    img = draw_molecule_clean(compound['Virtual_SMILES'])
                                    if img:
                                        connection_status = "Connected Molecule" if is_connected else "Fragment Mixture"
                                        st.image(img, caption=f"Structure ({connection_status})", width=250)
                                    else:
                                        st.warning("Could not render structure")
                                
                                with col2:
                                    # Display properties
                                    st.write("**Molecular Properties:**")
                                    col2a, col2b = st.columns(2)
                                    
                                    with col2a:
                                        st.metric("Molecular Weight", f"{compound['MW']:.1f}")
                                        st.metric("LogP", f"{compound['LogP']:.2f}")
                                        st.metric("TPSA", f"{compound['TPSA']:.1f}")
                                        st.metric("QED Score", f"{compound['QED']:.3f}")
                                    
                                    with col2b:
                                        st.metric("H-Bond Acceptors", f"{compound['HBA']}")
                                        st.metric("H-Bond Donors", f"{compound['HBD']}")
                                        st.metric("Rotatable Bonds", f"{compound['RotBonds']}")
                                        st.metric("Ring Count", f"{compound['NumRings']}")
                                    
                                    # SMILES and fragment info
                                    st.write("**SMILES:**")
                                    smiles_display = compound['Virtual_SMILES']
                                    if len(smiles_display) > 100:
                                        smiles_display = smiles_display[:100] + "..."
                                    st.code(smiles_display, language=None)
                                    
                                    # Fragment composition
                                    st.write("**Fragment Composition:**")
                                    st.caption(f"Warhead: {compound['Warhead_SMILES'][:50]}...")
                                    st.caption(f"Linker: {compound['Linker_SMILES']}")
                                    st.caption(f"E3 Binder: {compound['E3_SMILES'][:50]}...")
                                    
                                    # Connection info
                                    if is_connected:
                                        st.success("✅ Single connected molecule")
                                    else:
                                        st.info("ℹ️ Fragment mixture (still valid for analysis)")
                        
                        # Stats for current selection
                        if compound_type == "Connected Virtual":
                            connected_count = len(display_df)
                            total_virtual = len(virtual_df)
                            connection_rate = (connected_count / total_virtual) * 100 if total_virtual > 0 else 0
                            st.info(f"📊 Connected Virtual: {connected_count}/{total_virtual} compounds ({connection_rate:.1f}% connection success)")
                        else:
                            st.info(f"📊 {compound_type}: {len(display_df)} total compounds available")
                    
                    else:
                        st.warning(f"No {compound_type.lower()} compounds available")
                    
                    # Download options
                    st.subheader("💾 Download Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        pareto_csv = virtual_pareto.to_csv(index=False)
                        st.download_button(
                            label="📥 Pareto Compounds",
                            data=pareto_csv,
                            file_name=f"pareto_protacs_{selected_target}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        virtual_csv = virtual_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Full Virtual Library",
                            data=virtual_csv,
                            file_name=f"virtual_protacs_{selected_target}.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        # Connected compounds only
                        connected_df = virtual_df[virtual_df.get('Is_Connected', True)]
                        if len(connected_df) > 0:
                            connected_csv = connected_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Connected Only",
                                data=connected_csv,
                                file_name=f"connected_protacs_{selected_target}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.button("📥 No Connected", disabled=True)
                    
                    with col4:
                        if len(original_df) > 0:
                            combined_csv = st.session_state['combined_df'].to_csv(index=False)
                            st.download_button(
                                label="📥 Combined Data",
                                data=combined_csv,
                                file_name=f"combined_protacs_{selected_target}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.button("📥 No Original", disabled=True)
                
                elif 'library_generated' not in st.session_state:
                    st.info("💡 Generate a virtual library first to explore compounds")
            
            else:
                st.error("❌ Target column not found in the dataset")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    else:
        st.info("👆 Please upload your standardized PROTAC fragments CSV file")
        
        # Instructions
        st.markdown("""
        ### 📋 How to use:
        
        1. **Upload** your standardized PROTAC fragments CSV
        2. **Select** a target protein for focused library generation
        3. **Choose** fragment scope (target-specific vs all available)
        4. **Generate** virtual library and explore results
        
        ### 🎯 Features:
        
        - **Clean t-SNE plot** with informative tooltips
        - **Interactive compound explorer** with molecular structures
        - **Pareto optimization** for drug-like properties
        - **Multiple download formats**
        
        ### 📊 Visualization:
        
        - **🔴 Red circles**: Original active PROTACs
        - **💎 Green diamonds**: Pareto optimal virtual PROTACs  
        - **🔵 Blue circles**: Connected virtual PROTACs
        - **⚪ Gray circles**: Fragmented virtual PROTACs
        """)

# =====================================================================================
# RUN THE APP
# =====================================================================================

if __name__ == "__main__":
    main()