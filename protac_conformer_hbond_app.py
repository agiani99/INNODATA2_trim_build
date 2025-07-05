import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import sys
import pickle
from datetime import datetime
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
warnings.filterwarnings('ignore')

# RDKit imports for molecular analysis
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcNumAromaticRings, CalcFractionCSP3
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Import the molecular conformer analyzer
try:
    from molecular_conformer_analysis_ellipsoid_optimized import MolecularConformerAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="PROTAC Conformer Analysis with H-bonding",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1e40af;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .status-success {
        background-color: #dcfce7;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #22c55e;
    }
    .status-error {
        background-color: #fef2f2;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

def get_optimal_processing_settings():
    """Determine optimal processing settings based on available hardware"""
    
    # Get CPU information
    cpu_count = mp.cpu_count()
    available_cpus = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else cpu_count
    
    # Get memory information (in GB)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Calculate optimal settings
    if available_cpus == 1:
        optimal_workers = 1
        optimal_chunk_size = 1
        use_multiprocessing = False
    else:
        # Use 75% of available CPUs to leave resources for the system
        optimal_workers = max(1, int(available_cpus * 0.75))
        
        # Calculate chunk size based on memory and CPU count
        # Assume each molecule needs ~50MB during processing
        memory_per_worker = memory_gb / optimal_workers
        optimal_chunk_size = max(1, min(10, int(memory_per_worker / 0.05)))
        use_multiprocessing = True
    
    return {
        'cpu_count': cpu_count,
        'available_cpus': available_cpus,
        'memory_gb': memory_gb,
        'optimal_workers': optimal_workers,
        'optimal_chunk_size': optimal_chunk_size,
        'use_multiprocessing': use_multiprocessing
    }

def analyze_molecule_batch(args):
    """
    Analyze a batch of molecules - designed for multiprocessing
    Args: tuple of (smiles_list, analyzer_params, batch_id)
    """
    smiles_list, analyzer_params, batch_id = args
    
    # Recreate analyzer in the worker process
    analyzer = IntramolecularHBondAnalyzer(
        max_conformers=analyzer_params['max_conformers'],
        energy_threshold=analyzer_params['energy_threshold'],
        max_iterations=analyzer_params['max_iterations']
    )
    
    batch_results = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            if pd.isna(smiles) or smiles == '':
                continue
                
            result = analyzer.analyze_molecule(smiles, f"batch_{batch_id}_mol_{i}")
            
            if 'error' not in result:
                batch_results.append(result)
                
        except Exception as e:
            # Skip failed molecules in batch processing
            continue
    
    return batch_results

class IntramolecularHBondAnalyzer:
    """Enhanced molecular analyzer with intramolecular hydrogen bonding analysis"""
    
    def __init__(self, max_conformers=50, energy_threshold=5.0, max_iterations=500):
        self.max_conformers = max_conformers
        self.energy_threshold = energy_threshold
        self.max_iterations = max_iterations
    
    def count_hba_hbd(self, mol):
        """Count hydrogen bond acceptors and donors in a molecule"""
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        return hba, hbd
    
    def detect_intramolecular_hbonds(self, mol):
        """
        Detect intramolecular hydrogen bonds in a molecule
        Uses distance and angle constraints to identify potential H-bonds
        """
        # Get atom indices for potential donors (N-H, O-H) and acceptors (N, O)
        donors = []
        acceptors = []
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' or atom.GetSymbol() == 'O':
                # Check if it's a donor (has H attached)
                is_donor = False
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'H':
                        donors.append((atom.GetIdx(), neighbor.GetIdx()))
                        is_donor = True
                
                # All N and O atoms are potential acceptors
                acceptors.append(atom.GetIdx())
        
        # Detect H-bonds based on distance and angle criteria
        conf = mol.GetConformer()
        hbonds = []
        
        for donor_pair in donors:
            donor_idx, h_idx = donor_pair
            donor_atom = mol.GetAtomWithIdx(donor_idx)
            h_pos = conf.GetAtomPosition(h_idx)
            
            for acceptor_idx in acceptors:
                # Skip self-interaction
                if acceptor_idx == donor_idx:
                    continue
                    
                acceptor_pos = conf.GetAtomPosition(acceptor_idx)
                
                # Calculate H...Acceptor distance
                h_acc_dist = h_pos.Distance(acceptor_pos)
                
                # Distance criterion for H-bond (typically 1.5-2.7 Å)
                if h_acc_dist < 2.7:
                    donor_pos = conf.GetAtomPosition(donor_idx)
                    
                    # Calculate D-H...A angle
                    v1 = [h_pos.x - donor_pos.x, h_pos.y - donor_pos.y, h_pos.z - donor_pos.z]
                    v2 = [acceptor_pos.x - h_pos.x, acceptor_pos.y - h_pos.y, acceptor_pos.z - h_pos.z]
                    
                    # Normalize vectors
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    
                    if v1_norm > 0 and v2_norm > 0:
                        v1 = v1 / v1_norm
                        v2 = v2 / v2_norm
                        
                        # Calculate angle in degrees
                        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
                        
                        # Angle criterion for H-bond (typically >120°)
                        if angle > 110:
                            hbonds.append((donor_idx, h_idx, acceptor_idx, h_acc_dist, angle))
        
        return hbonds
    
    def analyze_molecule(self, smiles, compound_id=None):
        """
        Analyze a single SMILES string with comprehensive conformer and H-bond analysis
        """
        try:
            # Create molecule from SMILES and add hydrogens
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": f"Invalid SMILES: {smiles}"}
            
            mol = Chem.AddHs(mol)
            
            # Count hydrogen bond acceptors and donors
            hba, hbd = self.count_hba_hbd(mol)
            
            # Generate conformers
            confgen = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=self.max_conformers,
                randomSeed=42,
                pruneRmsThresh=-1,
                numThreads=0,
                useRandomCoords=True
            )
            
            if confgen == -1 or mol.GetNumConformers() == 0:
                return {"error": f"Could not generate conformers for: {smiles}"}
            
            num_generated = mol.GetNumConformers()
            
            # Initialize arrays to store results
            conformer_energies = []
            hbond_counts = []
            hbond_conformers = 0
            
            # For each conformer: minimize energy and analyze H-bonds
            for conf_id in range(num_generated):
                try:
                    converged = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=self.max_iterations)
                    # Get the energy of the minimized conformer
                    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                    if mmff_props is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                            conformer_energies.append(energy)
                            
                            # Check for intramolecular H-bonds
                            hbonds = self.detect_intramolecular_hbonds(mol)
                            hbond_counts.append(len(hbonds))
                            
                            if len(hbonds) > 0:
                                hbond_conformers += 1
                        else:
                            conformer_energies.append(1e6)
                            hbond_counts.append(0)
                    else:
                        conformer_energies.append(1e6)
                        hbond_counts.append(0)
                
                except Exception as e:
                    conformer_energies.append(1e6)
                    hbond_counts.append(0)
            
            # Calculate lowest energy and energy differences
            if conformer_energies:
                lowest_energy = min(conformer_energies)
                delta_energies = [e - lowest_energy for e in conformer_energies]
                
                # Count conformers within energy window
                low_energy_conformers = sum(1 for de in delta_energies if de < self.energy_threshold)
                
                # Count conformers with H-bonds within energy window
                hbond_low_energy = sum(1 for i, de in enumerate(delta_energies) if de < self.energy_threshold and hbond_counts[i] > 0)
            else:
                lowest_energy = None
                low_energy_conformers = 0
                hbond_low_energy = 0
            
            # Calculate additional molecular descriptors
            mw = Descriptors.MolWt(mol)
            psa = rdMolDescriptors.CalcTPSA(mol)
            clogp = Descriptors.MolLogP(mol)
            rotatable_bonds = CalcNumRotatableBonds(mol)
            aromatic_rings = CalcNumAromaticRings(mol)
            fsp3 = CalcFractionCSP3(mol)
            
            # Prepare results
            results = {
                "smiles": smiles,
                "compound_id": compound_id if compound_id else "Unknown",
                "HBA": hba,
                "HBD": hbd,
                "conformers_generated": num_generated,
                "percent_conformers_with_h_bonds": (hbond_conformers / num_generated) * 100 if num_generated > 0 else 0,
                "percent_low_energy_conformers": (low_energy_conformers / num_generated) * 100 if num_generated > 0 else 0,
                "percent_low_energy_with_h_bonds": (hbond_low_energy / num_generated) * 100 if num_generated > 0 else 0,
                "molecular_weight": mw,
                "PSA": psa,
                "clogP": clogp,
                "rotatable_bonds": rotatable_bonds,
                "aromatic_rings": aromatic_rings,
                "fsp3": fsp3,
                "lowest_energy": lowest_energy,
                "num_conformers": num_generated,
                "num_low_energy_conformers": low_energy_conformers,
                "num_hbond_conformers": hbond_conformers,
                "num_low_energy_hbond_conformers": hbond_low_energy
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Error analyzing molecule: {str(e)}"}

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 PROTAC Conformer Analysis with H-bonding</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    This application analyzes PROTAC molecular conformers with enhanced features:
    - **Conformer Generation**: Multiple 3D conformations with energy minimization
    - **Intramolecular H-bonding**: Detection and analysis of internal hydrogen bonds
    - **Shape Analysis**: Ellipsoid volume and slice area calculations
    - **Machine Learning**: pDC50 prediction using uploaded models
    
    Upload a CSV file containing a **PROTAC_SMILES** column and optionally a pre-trained model to begin analysis.
    """)
    
    # Check dependencies
    if not RDKIT_AVAILABLE:
        st.error("⚠️ RDKit is not available. Please install RDKit to use this application.")
        return
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Analysis Parameters")
    
    # Hardware optimization section
    st.sidebar.subheader("🖥️ Hardware Optimization")
    
    # Get optimal settings
    hw_settings = get_optimal_processing_settings()
    
    # Display hardware info
    st.sidebar.info(f"""
    **System Info:**
    - CPUs Available: {hw_settings['available_cpus']}/{hw_settings['cpu_count']}
    - Memory: {hw_settings['memory_gb']:.1f} GB
    - Recommended Workers: {hw_settings['optimal_workers']}
    - Recommended Chunk Size: {hw_settings['optimal_chunk_size']}
    """)
    
    # Allow user to override settings
    use_multiprocessing = st.sidebar.checkbox(
        "Enable Multiprocessing",
        value=hw_settings['use_multiprocessing'],
        help="Use multiple CPU cores for faster processing"
    )
    
    if use_multiprocessing and hw_settings['available_cpus'] > 1:
        num_workers = st.sidebar.slider(
            "Number of Workers",
            min_value=1,
            max_value=hw_settings['available_cpus'],
            value=hw_settings['optimal_workers'],
            help="Number of parallel workers (CPU cores to use)"
        )
        
        chunk_size = st.sidebar.slider(
            "Chunk Size",
            min_value=1,
            max_value=20,
            value=hw_settings['optimal_chunk_size'],
            help="Number of molecules per chunk (affects memory usage)"
        )
    else:
        num_workers = 1
        chunk_size = 1
        st.sidebar.info("Single-threaded processing (1 CPU or multiprocessing disabled)")
    
    st.sidebar.subheader("🧪 Chemical Parameters")
    
    max_conformers = st.sidebar.slider(
        "Maximum Conformers",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Maximum number of conformers to generate per molecule"
    )
    
    energy_threshold = st.sidebar.slider(
        "Energy Threshold (kcal/mol)",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Energy threshold for defining low-energy conformers"
    )
    
    max_iterations = st.sidebar.slider(
        "Optimization Iterations",
        min_value=10,
        max_value=500,
        value=200,
        step=10,
        help="Maximum iterations for conformer energy minimization"
    )
    
    # Model prediction section
    st.sidebar.header("🤖 Model Prediction")
    
    uploaded_model = st.sidebar.file_uploader(
        "Upload XGBoost Model (pkl)",
        type=['pkl'],
        help="Upload a pre-trained XGBoost model for pDC50 prediction"
    )
    
    model_loaded = False
    xgb_model = None
    model_features = None
    
    if uploaded_model is not None:
        try:
            # Load the model
            xgb_model = pickle.load(uploaded_model)
            model_loaded = True
            
            # Extract feature names
            if hasattr(xgb_model, 'feature_names_in_'):
                model_features = list(xgb_model.feature_names_in_)
            elif hasattr(xgb_model, 'get_booster'):
                try:
                    model_features = xgb_model.get_booster().feature_names
                except:
                    model_features = None
            
            st.sidebar.success("✅ Model loaded successfully!")
            if model_features:
                st.sidebar.info(f"📊 Model features: {len(model_features)}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            model_loaded = False
    
    # File upload section
    st.markdown('<h2 class="section-header">📁 File Upload</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="File must contain a 'PROTAC_SMILES' column"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Check for SMILES column
            smiles_columns = [col for col in df.columns if 'SMILES' in col.upper()]
            
            if 'PROTAC_SMILES' in df.columns:
                smiles_col = 'PROTAC_SMILES'
                st.success(f"✅ Found PROTAC_SMILES column with {df[smiles_col].notna().sum()} valid entries")
            elif smiles_columns:
                smiles_col = st.selectbox(
                    "Select SMILES column:",
                    options=smiles_columns,
                    help="Choose which column contains the SMILES strings"
                )
                st.info(f"ℹ️ Using {smiles_col} column for analysis")
            else:
                st.error("❌ No SMILES column found. Please ensure your file contains a column with 'SMILES' in the name.")
                return
            
            # Display preview
            st.markdown('<h2 class="section-header">👀 Data Preview</h2>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # Analysis section
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                run_analysis(df, smiles_col, max_conformers, energy_threshold, 
                           chunk_size, max_iterations, xgb_model, model_features, 
                           use_multiprocessing, num_workers)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def run_analysis(df, smiles_col, max_conformers, energy_threshold, chunk_size, max_iterations, 
                xgb_model=None, model_features=None, use_multiprocessing=False, num_workers=1):
    """Run the comprehensive conformer and H-bond analysis with optimized processing"""
    
    st.markdown('<h2 class="section-header">🔬 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    detailed_progress = st.empty()
    
    # Prepare data
    analysis_df = df.copy()
    total_molecules = len(analysis_df)
    valid_smiles = analysis_df[smiles_col].dropna()
    valid_count = len(valid_smiles)
    
    st.info(f"🔍 Starting analysis of {valid_count} valid molecules out of {total_molecules} total...")
    
    # Show processing strategy
    if use_multiprocessing and num_workers > 1:
        st.success(f"🚀 **Multiprocessing Mode**: {num_workers} workers, {chunk_size} molecules per chunk")
        total_chunks = (valid_count + chunk_size - 1) // chunk_size
        st.info(f"📊 Processing {total_chunks} chunks with ~{chunk_size} molecules each")
    else:
        st.info(f"🔄 **Single-threaded Mode**: Processing {valid_count} molecules sequentially")
    
    # Results container
    all_results = []
    successful_molecules = 0
    failed_molecules = 0
    
    # Time tracking
    start_time = time.time()
    
    try:
        if use_multiprocessing and num_workers > 1:
            # Multiprocessing approach
            all_results = run_multiprocessing_analysis(
                valid_smiles.tolist(), max_conformers, energy_threshold, max_iterations,
                chunk_size, num_workers, progress_bar, status_text, detailed_progress
            )
            successful_molecules = len(all_results)
            failed_molecules = valid_count - successful_molecules
            
        else:
            # Single-threaded approach with TQDM-style progress
            analyzer = IntramolecularHBondAnalyzer(
                max_conformers=max_conformers,
                energy_threshold=energy_threshold,
                max_iterations=max_iterations
            )
            
            conformer_analyzer = None
            if ANALYZER_AVAILABLE:
                conformer_analyzer = MolecularConformerAnalyzer(
                    energy_threshold=energy_threshold,
                    max_conformers=max_conformers,
                    max_iterations=max_iterations
                )
            
            for i, (idx, row) in enumerate(analysis_df.iterrows()):
                # Update progress in TQDM style
                progress = (i + 1) / total_molecules
                progress_bar.progress(progress)
                
                # TQDM-style status with ETA
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_mol = elapsed_time / i
                    eta = avg_time_per_mol * (total_molecules - i)
                    eta_str = f"ETA: {eta/60:.1f}min" if eta > 60 else f"ETA: {eta:.1f}s"
                else:
                    eta_str = "ETA: calculating..."
                
                status_text.text(f"Processing molecule {i+1}/{total_molecules} | {eta_str}")
                detailed_progress.text(f"⏱️ Elapsed: {elapsed_time:.1f}s | ✅ Success: {successful_molecules} | ❌ Failed: {failed_molecules}")
                
                try:
                    smiles = row[smiles_col]
                    
                    if pd.isna(smiles) or smiles == '':
                        failed_molecules += 1
                        continue
                    
                    # H-bond analysis
                    hbond_result = analyzer.analyze_molecule(smiles, f"mol_{i}")
                    
                    if 'error' in hbond_result:
                        failed_molecules += 1
                        continue
                    
                    # Conformer analysis if available
                    conformer_result = {}
                    if conformer_analyzer:
                        conformer_result = conformer_analyzer.analyze_molecule(smiles)
                        if 'error' in conformer_result:
                            conformer_result = {}
                    
                    # Combine results
                    combined_result = {**hbond_result, **conformer_result}
                    
                    # Add original data
                    for col in df.columns:
                        if col != smiles_col:
                            combined_result[col] = row[col]
                    
                    all_results.append(combined_result)
                    successful_molecules += 1
                    
                except Exception as e:
                    failed_molecules += 1
                    continue
        
        # Final progress update
        progress_bar.progress(1.0)
        processing_time = time.time() - start_time
        
        status_text.text("✅ Analysis complete!")
        detailed_progress.text(f"🎉 Finished! Total time: {processing_time:.1f}s | Success rate: {successful_molecules/valid_count*100:.1f}%")
        
        # Show final processing summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Successful", successful_molecules)
        with col2:
            st.metric("❌ Failed", failed_molecules)
        with col3:
            st.metric("⏱️ Total Time", f"{processing_time:.1f}s")
        with col4:
            avg_time = processing_time / valid_count if valid_count > 0 else 0
            st.metric("⚡ Avg Time/Mol", f"{avg_time:.1f}s")
        
        # Create results DataFrame
        if all_results:
            st.success(f"🎉 Analysis completed! {len(all_results)} molecules processed successfully.")
            
            results_df = pd.DataFrame(all_results)
            
            # Add model predictions if model is available
            if xgb_model is not None and model_features is not None:
                results_df = add_model_predictions(results_df, xgb_model, model_features)
            
            # Display summary statistics
            display_results_summary(results_df, processing_time)
            
            # Display detailed results
            display_detailed_results(results_df)
            
            # Offer download
            offer_download(results_df)
            
        else:
            st.error("❌ No molecules were successfully processed.")
            st.info("💡 This could be due to:")
            st.write("- Invalid SMILES strings")
            st.write("- RDKit installation issues")
            st.write("- Memory constraints")
            st.write("- MMFF force field issues")
            
    except Exception as e:
        st.error(f"💥 Analysis failed with exception: {str(e)}")
        # Try to save partial results if any
        if all_results:
            st.warning("💾 Attempting to save partial results...")
            try:
                partial_df = pd.DataFrame(all_results)
                offer_download(partial_df, filename_suffix="_partial")
            except Exception as save_error:
                st.error(f"Failed to save partial results: {str(save_error)}")

def run_multiprocessing_analysis(smiles_list, max_conformers, energy_threshold, max_iterations,
                                chunk_size, num_workers, progress_bar, status_text, detailed_progress):
    """Run analysis using multiprocessing for faster execution"""
    
    # Prepare analyzer parameters
    analyzer_params = {
        'max_conformers': max_conformers,
        'energy_threshold': energy_threshold,
        'max_iterations': max_iterations
    }
    
    # Split data into chunks
    chunks = []
    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i + chunk_size]
        chunks.append((chunk, analyzer_params, i // chunk_size))
    
    total_chunks = len(chunks)
    st.info(f"🔧 Created {total_chunks} chunks for parallel processing")
    
    # Process chunks in parallel
    all_results = []
    completed_chunks = 0
    start_time = time.time()
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(analyze_molecule_batch, chunk): chunk for chunk in chunks}
            
            # Process completed chunks
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                completed_chunks += 1
                
                # Update progress
                progress = completed_chunks / total_chunks
                progress_bar.progress(progress)
                
                # Calculate ETA
                elapsed_time = time.time() - start_time
                if completed_chunks > 0:
                    avg_time_per_chunk = elapsed_time / completed_chunks
                    eta = avg_time_per_chunk * (total_chunks - completed_chunks)
                    eta_str = f"ETA: {eta/60:.1f}min" if eta > 60 else f"ETA: {eta:.1f}s"
                else:
                    eta_str = "ETA: calculating..."
                
                status_text.text(f"Completed chunk {completed_chunks}/{total_chunks} | {eta_str}")
                detailed_progress.text(f"⚡ Parallel processing with {num_workers} workers | ⏱️ Elapsed: {elapsed_time:.1f}s")
                
                try:
                    # Get results from completed chunk
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    
                except Exception as exc:
                    st.warning(f"Chunk {chunk[2]} generated an exception: {exc}")
                    continue
    
    except Exception as e:
        st.error(f"Multiprocessing error: {str(e)}")
        st.info("Falling back to single-threaded processing...")
        return []
    
    return all_results

def add_model_predictions(results_df, xgb_model, model_features):
    """Add model predictions using comprehensive feature mapping"""
    
    st.markdown('<h3 class="section-header">🤖 Model Predictions</h3>', unsafe_allow_html=True)
    
    try:
        st.info(f"🎯 Working with {len(model_features)} model features")
        
        # Create comprehensive feature mapping
        feature_mapping = create_comprehensive_feature_mapping(results_df, model_features)
        
        # Display mapping information
        st.markdown("### 🔄 Feature Mapping")
        
        # Create mapping table for display
        mapping_data = []
        missing_features = []
        
        for model_feature in model_features:
            if model_feature in feature_mapping:
                data_feature = feature_mapping[model_feature]
                status = '✅ Mapped'
                mapping_data.append({
                    'Model Feature': model_feature,
                    'Data Feature': data_feature,
                    'Status': status
                })
            else:
                missing_features.append(model_feature)
                mapping_data.append({
                    'Model Feature': model_feature,
                    'Data Feature': 'Default Value',
                    'Status': '⚠️ Default'
                })
        
        # Display mapping table
        mapping_df = pd.DataFrame(mapping_data)
        st.dataframe(mapping_df, use_container_width=True)
        
        if missing_features:
            st.warning(f"⚠️ Using defaults for {len(missing_features)} features")
            with st.expander("View missing features"):
                st.write(missing_features)
        
        # Create feature matrix
        X = create_feature_matrix(results_df, model_features, feature_mapping)
        
        if X is None:
            st.error("❌ Could not create feature matrix")
            return results_df
        
        # Make predictions
        with st.spinner("🔮 Making predictions..."):
            predictions = xgb_model.predict(X)
            results_df['predicted_pDC50'] = predictions
        
        # Display prediction statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean pDC50", f"{predictions.mean():.2f}")
        with col2:
            st.metric("Std pDC50", f"{predictions.std():.2f}")
        with col3:
            st.metric("Min pDC50", f"{predictions.min():.2f}")
        with col4:
            st.metric("Max pDC50", f"{predictions.max():.2f}")
        
        # Create prediction distribution plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predictions,
            nbinsx=20,
            name="pDC50 Predictions",
            marker=dict(color='lightblue', line=dict(color='blue', width=1))
        ))
        
        fig.update_layout(
            title="Distribution of Predicted pDC50 Values",
            xaxis_title="Predicted pDC50",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top predicted compounds
        st.markdown("### 🏆 Top Predicted Compounds")
        display_cols = ['smiles', 'predicted_pDC50']
        
        # Add relevant columns if they exist
        for col in ['Virtual_ID', 'compound_id', 'molecular_weight', 'percent_conformers_with_h_bonds']:
            if col in results_df.columns:
                display_cols.append(col)
        
        top_compounds = results_df.nlargest(10, 'predicted_pDC50')[display_cols]
        st.dataframe(top_compounds, use_container_width=True)
        
        st.success(f"✅ Successfully predicted pDC50 for {len(results_df)} molecules!")
        
    except Exception as e:
        st.error(f"❌ Error making predictions: {str(e)}")
        st.info("💡 Troubleshooting steps:")
        st.write("1. Ensure the model was saved with feature names")
        st.write("2. Check that all required molecular descriptors are calculated")
        st.write("3. Verify the model format is compatible")
        st.write("4. Try with a smaller dataset first")
        
        # Show detailed error info for debugging
        import traceback
        with st.expander("🔍 Detailed Error Information"):
            st.text(traceback.format_exc())
    
    return results_df

def create_comprehensive_feature_mapping(results_df, model_features):
    """Create comprehensive mapping between model features and calculated features"""
    
    # Enhanced mapping dictionary based on the H-bond analyzer output
    mapping_dict = {
        # H-bond and conformer features
        'ConfGenerated': ['conformers_generated', 'num_conformers'],
        'PercentLowEnergy': ['percent_low_energy_conformers'],
        'Percent_confs_with_H_bonds': ['percent_conformers_with_h_bonds'],
        'Percent_low_energy_with_H_bonds': ['percent_low_energy_with_h_bonds'],
        
        # Basic molecular descriptors
        'MW': ['molecular_weight', 'Molecular Weight (Da)'],
        'MolWt': ['molecular_weight', 'Molecular Weight (Da)'],
        'molecular_weight': ['molecular_weight', 'Molecular Weight (Da)'],
        'LogP': ['clogP'],
        'MolLogP': ['clogP'],
        'clogP': ['clogP'],
        'HBA': ['HBA'],
        'NumHBA': ['HBA'],
        'HBD': ['HBD'],
        'NumHBD': ['HBD'],
        'TPSA': ['PSA'],
        'PSA': ['PSA'],
        'RotatableBonds': ['rotatable_bonds'],
        'NumRotatableBonds': ['rotatable_bonds'],
        'rotatable_bonds': ['rotatable_bonds'],
        'AromaticRings': ['aromatic_rings'],
        'NumAromaticRings': ['aromatic_rings'],
        'aromatic_rings': ['aromatic_rings'],
        'Fsp3': ['fsp3'],
        'FractionCsp3': ['fsp3'],
        'fsp3': ['fsp3'],
        
        # Shape and conformer descriptors
        'ellipsoid_volume': ['ellipsoid_volume'],
        'slice_area': ['slice_area'],
        'low_energy_ellipsoid_volume': ['low_energy_ellipsoid_volume'],
        'low_energy_slice_area': ['low_energy_slice_area'],
        'num_conformers': ['num_conformers', 'conformers_generated'],
        'num_low_energy_conformers': ['num_low_energy_conformers'],
        'num_hbond_conformers': ['num_hbond_conformers'],
        'num_low_energy_hbond_conformers': ['num_low_energy_hbond_conformers'],
        
        # Energy descriptors
        'lowest_energy': ['lowest_energy']
    }
    
    # Get all available columns
    available_columns = list(results_df.columns)
    
    feature_mapping = {}
    
    for model_feature in model_features:
        # Try exact match first
        if model_feature in available_columns:
            feature_mapping[model_feature] = model_feature
            continue
        
        # Try case-insensitive exact match
        for col in available_columns:
            if model_feature.lower() == col.lower():
                feature_mapping[model_feature] = col
                break
        
        if model_feature in feature_mapping:
            continue
        
        # Try mapping dictionary
        if model_feature in mapping_dict:
            for possible_name in mapping_dict[model_feature]:
                if possible_name in available_columns:
                    feature_mapping[model_feature] = possible_name
                    break
        
        if model_feature in feature_mapping:
            continue
        
        # Try partial matching for common patterns
        model_feature_lower = model_feature.lower()
        for col in available_columns:
            col_lower = col.lower()
            if (model_feature_lower in col_lower or col_lower in model_feature_lower) and len(model_feature_lower) > 2:
                feature_mapping[model_feature] = col
                break
    
    return feature_mapping

def create_feature_matrix(results_df, model_features, feature_mapping):
    """Create feature matrix for model prediction with comprehensive feature handling"""
    
    try:
        X = pd.DataFrame()
        
        for model_feature in model_features:
            if model_feature in feature_mapping:
                data_feature = feature_mapping[model_feature]
                X[model_feature] = results_df[data_feature].copy()
            else:
                # Handle missing features with enhanced defaults based on PROTAC characteristics
                X[model_feature] = create_enhanced_default_values(model_feature, len(results_df), results_df)
        
        # Handle missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype in ['int64', 'float64']:
                        # Use median for numeric columns
                        median_val = X[col].median()
                        if pd.isna(median_val):
                            # If all values are NaN, use reasonable defaults
                            median_val = get_enhanced_default(col)
                        X[col] = X[col].fillna(median_val)
        
        return X
    
    except Exception as e:
        st.error(f"Error creating feature matrix: {str(e)}")
        return None

def create_enhanced_default_values(feature_name, n_samples, results_df):
    """Create enhanced default values for missing features including derived calculations"""
    
    # Calculate derived features if possible
    if feature_name == 'PercentLowEnergy' and 'num_low_energy_conformers' in results_df.columns and 'num_conformers' in results_df.columns:
        return (results_df['num_low_energy_conformers'] / results_df['num_conformers'].replace(0, 1) * 100).fillna(20)
    
    if feature_name == 'ConfGenerated' and 'num_conformers' in results_df.columns:
        return results_df['num_conformers'].fillna(50)
    
    # Enhanced defaults based on PROTAC characteristics
    enhanced_defaults = {
        # Conformer features
        'ConfGenerated': 50,
        'conformers_generated': 50,
        'PercentLowEnergy': 20,
        'percent_low_energy_conformers': 20,
        'Percent_confs_with_H_bonds': 30,
        'percent_conformers_with_h_bonds': 30,
        'Percent_low_energy_with_H_bonds': 15,
        'percent_low_energy_with_h_bonds': 15,
        
        # Basic molecular properties (PROTAC-appropriate)
        'MW': 800,
        'MolWt': 800,
        'molecular_weight': 800,
        'LogP': 4.0,
        'MolLogP': 4.0,
        'clogP': 4.0,
        'HBA': 12,
        'NumHBA': 12,
        'HBD': 3,
        'NumHBD': 3,
        'TPSA': 150,
        'PSA': 150,
        'RotatableBonds': 15,
        'NumRotatableBonds': 15,
        'rotatable_bonds': 15,
        'AromaticRings': 3,
        'NumAromaticRings': 3,
        'aromatic_rings': 3,
        'Fsp3': 0.4,
        'FractionCsp3': 0.4,
        'fsp3': 0.4,
        
        # Shape descriptors
        'ellipsoid_volume': 2000,
        'slice_area': 200,
        'low_energy_ellipsoid_volume': 1600,
        'low_energy_slice_area': 160,
        'num_conformers': 50,
        'num_low_energy_conformers': 10,
        'num_hbond_conformers': 15,
        'num_low_energy_hbond_conformers': 7,
        'lowest_energy': 50.0
    }
    
    default_value = enhanced_defaults.get(feature_name, 0)
    return pd.Series([default_value] * n_samples)

def get_enhanced_default(feature_name):
    """Get enhanced default values for specific features"""
    
    enhanced_defaults = {
        # Conformer features
        'ConfGenerated': 50,
        'PercentLowEnergy': 20,
        'Percent_confs_with_H_bonds': 30,
        'Percent_low_energy_with_H_bonds': 15,
        
        # Basic molecular properties (PROTAC-appropriate)
        'MW': 800,
        'MolWt': 800,
        'molecular_weight': 800,
        'LogP': 4.0,
        'MolLogP': 4.0,
        'clogP': 4.0,
        'HBA': 12,
        'NumHBA': 12,
        'HBD': 3,
        'NumHBD': 3,
        'TPSA': 150,
        'PSA': 150,
        'RotatableBonds': 15,
        'NumRotatableBonds': 15,
        'AromaticRings': 3,
        'NumAromaticRings': 3,
        'Fsp3': 0.4,
        'FractionCsp3': 0.4,
        'ellipsoid_volume': 2000,
        'slice_area': 200,
        'low_energy_ellipsoid_volume': 1600,
        'low_energy_slice_area': 160
    }
    
    return enhanced_defaults.get(feature_name, 0)

def display_results_summary(results_df, processing_time):
    """Display summary statistics of the analysis"""
    
    st.markdown('<h3 class="section-header">📊 Summary Statistics</h3>', unsafe_allow_html=True)
    
    # Processing summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Molecules Processed",
            len(results_df),
            help="Total number of molecules successfully analyzed"
        )
    
    with col2:
        if 'conformers_generated' in results_df.columns:
            avg_conformers = results_df['conformers_generated'].mean()
        elif 'num_conformers' in results_df.columns:
            avg_conformers = results_df['num_conformers'].mean()
        else:
            avg_conformers = 0
        st.metric(
            "Avg Conformers",
            f"{avg_conformers:.1f}",
            help="Average number of conformers generated per molecule"
        )
    
    with col3:
        if 'percent_conformers_with_h_bonds' in results_df.columns:
            avg_hbonds = results_df['percent_conformers_with_h_bonds'].mean()
        else:
            avg_hbonds = 0
        st.metric(
            "Avg H-bond %",
            f"{avg_hbonds:.1f}%",
            help="Average percentage of conformers with intramolecular H-bonds"
        )
    
    with col4:
        st.metric(
            "Processing Time",
            f"{processing_time:.1f}s",
            help="Total time taken for analysis"
        )
    
    # H-bonding statistics
    st.markdown("### H-bonding Analysis")
    hbond_cols = [col for col in results_df.columns if 'h_bond' in col.lower() or 'hbond' in col.lower()]
    if hbond_cols:
        hbond_stats = results_df[hbond_cols].describe()
        st.dataframe(hbond_stats, use_container_width=True)
    
    # Molecular descriptor statistics
    st.markdown("### Molecular Descriptors")
    descriptor_cols = []
    for col in ['molecular_weight', 'clogP', 'PSA', 'HBA', 'HBD', 'rotatable_bonds', 'aromatic_rings']:
        if col in results_df.columns:
            descriptor_cols.append(col)
    
    if descriptor_cols:
        desc_stats = results_df[descriptor_cols].describe()
        st.dataframe(desc_stats, use_container_width=True)

def display_detailed_results(results_df):
    """Display detailed results with visualizations"""
    
    st.markdown('<h3 class="section-header">📈 Visualizations</h3>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["H-bond Analysis", "Molecular Properties", "Correlation Matrix", "Model Predictions", "Detailed Data"])
    
    with tab1:
        # H-bonding analysis visualizations
        st.markdown("### Intramolecular Hydrogen Bonding Analysis")
        
        # H-bond percentage distribution
        if 'percent_conformers_with_h_bonds' in results_df.columns:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["All Conformers with H-bonds", "Low Energy Conformers with H-bonds"],
                horizontal_spacing=0.1
            )
            
            # All conformers with H-bonds
            fig.add_trace(
                go.Histogram(x=results_df['percent_conformers_with_h_bonds'], name="All H-bonded", nbinsx=20),
                row=1, col=1
            )
            
            # Low energy conformers with H-bonds
            if 'percent_low_energy_with_h_bonds' in results_df.columns:
                fig.add_trace(
                    go.Histogram(x=results_df['percent_low_energy_with_h_bonds'], name="Low-E H-bonded", nbinsx=20),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="H-bond Conformer Distributions",
                height=400,
                showlegend=False
            )
            fig.update_xaxes(title_text="Percentage (%)", row=1, col=1)
            fig.update_xaxes(title_text="Percentage (%)", row=1, col=2)
            fig.update_yaxes(title_text="Count")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # H-bond correlation with molecular properties
        if 'percent_conformers_with_h_bonds' in results_df.columns and 'molecular_weight' in results_df.columns:
            fig = px.scatter(
                results_df, 
                x='molecular_weight', 
                y='percent_conformers_with_h_bonds',
                title="H-bonding vs Molecular Weight",
                labels={'molecular_weight': 'Molecular Weight (Da)', 'percent_conformers_with_h_bonds': 'H-bond Conformers (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Molecular properties distributions
        st.markdown("### Molecular Properties")
        
        # Create property distribution plots
        property_cols = ['molecular_weight', 'clogP', 'PSA', 'rotatable_bonds']
        available_props = [col for col in property_cols if col in results_df.columns]
        
        if len(available_props) >= 2:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=available_props[:4]
            )
            
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for i, prop in enumerate(available_props[:4]):
                row, col = positions[i]
                fig.add_trace(
                    go.Histogram(x=results_df[prop], name=prop, nbinsx=20),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Molecular Property Distributions",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation matrix
        numeric_cols = []
        for col in results_df.columns:
            if results_df[col].dtype in ['int64', 'float64'] and not col.startswith('Unnamed'):
                numeric_cols.append(col)
        
        # Limit to most relevant columns
        relevant_cols = []
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['weight', 'logp', 'psa', 'hba', 'hbd', 'bonds', 'h_bond', 'conformer', 'energy']):
                relevant_cols.append(col)
        
        if len(relevant_cols) > 1:
            # Limit to top 15 most relevant columns to avoid overcrowding
            corr_cols = relevant_cols[:15]
            corr_matrix = results_df[corr_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 8},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Correlation Matrix of Key Molecular Descriptors",
                height=700,
                width=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation analysis")
    
    with tab4:
        # Model predictions visualization
        if 'predicted_pDC50' in results_df.columns:
            st.markdown("### 🤖 Model Prediction Analysis")
            
            # Prediction vs features scatter plots
            numeric_cols = []
            for col in results_df.columns:
                if results_df[col].dtype in ['int64', 'float64'] and col != 'predicted_pDC50':
                    numeric_cols.append(col)
            
            if len(numeric_cols) > 0:
                # Feature selection for scatter plot
                selected_feature = st.selectbox(
                    "Select feature for correlation with pDC50:",
                    options=numeric_cols,
                    index=0
                )
                
                if selected_feature:
                    # Create scatter plot
                    fig = px.scatter(
                        results_df, 
                        x=selected_feature, 
                        y='predicted_pDC50',
                        title=f"Predicted pDC50 vs {selected_feature}",
                        trendline="ols"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate correlation
                    correlation = results_df[selected_feature].corr(results_df['predicted_pDC50'])
                    st.metric("Correlation Coefficient", f"{correlation:.3f}")
            
            # Feature importance for predictions
            st.markdown("### 🔍 Top vs Bottom Predictions Comparison")
            
            top_n = min(5, len(results_df) // 4)
            if top_n > 0:
                top_predictions = results_df.nlargest(top_n, 'predicted_pDC50')
                bottom_predictions = results_df.nsmallest(top_n, 'predicted_pDC50')
                
                col1, col2 = st.columns(2)
                
                comparison_cols = ['predicted_pDC50']
                for col in ['molecular_weight', 'percent_conformers_with_h_bonds', 'clogP']:
                    if col in results_df.columns:
                        comparison_cols.append(col)
                
                with col1:
                    st.markdown("**Top Predictions**")
                    st.dataframe(
                        top_predictions[comparison_cols].round(3),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("**Bottom Predictions**")
                    st.dataframe(
                        bottom_predictions[comparison_cols].round(3),
                        use_container_width=True
                    )
        else:
            st.info("🤖 No model predictions available. Upload a model to see prediction analysis.")
    
    with tab5:
        # Detailed data table
        st.markdown("### Complete Results")
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'molecular_weight' in results_df.columns:
                min_mw = st.number_input(
                    "Minimum Molecular Weight",
                    min_value=0.0,
                    value=0.0,
                    step=50.0
                )
            else:
                min_mw = 0
        
        with col2:
            if 'percent_conformers_with_h_bonds' in results_df.columns:
                min_hbond = st.number_input(
                    "Minimum H-bond %",
                    min_value=0.0,
                    value=0.0,
                    step=5.0
                )
            else:
                min_hbond = 0
        
        with col3:
            if 'conformers_generated' in results_df.columns:
                min_conformers = st.number_input(
                    "Minimum Conformers",
                    min_value=0,
                    value=0,
                    step=1
                )
            else:
                min_conformers = 0
        
        # Filter data
        filtered_df = results_df.copy()
        
        if 'molecular_weight' in results_df.columns:
            filtered_df = filtered_df[filtered_df['molecular_weight'] >= min_mw]
        
        if 'percent_conformers_with_h_bonds' in results_df.columns:
            filtered_df = filtered_df[filtered_df['percent_conformers_with_h_bonds'] >= min_hbond]
        
        if 'conformers_generated' in results_df.columns:
            filtered_df = filtered_df[filtered_df['conformers_generated'] >= min_conformers]
        elif 'num_conformers' in results_df.columns:
            filtered_df = filtered_df[filtered_df['num_conformers'] >= min_conformers]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Show filtering summary
        st.info(f"Showing {len(filtered_df)} of {len(results_df)} molecules")

def offer_download(results_df, filename_suffix=""):
    """Offer download of results"""
    
    st.markdown('<h3 class="section-header">💾 Download Results</h3>', unsafe_allow_html=True)
    
    # Prepare CSV
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"protac_hbond_conformer_analysis{filename_suffix}_{timestamp}.csv"
    
    # Download button
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )
    
    # Display file info
    file_size = len(csv_data.encode('utf-8')) / 1024
    st.info(f"File size: {file_size:.1f} KB | {len(results_df)} rows | {len(results_df.columns)} columns")
    
    # Summary of key features
    key_features = []
    for feature in ['predicted_pDC50', 'percent_conformers_with_h_bonds', 'molecular_weight', 'clogP', 'ellipsoid_volume']:
        if feature in results_df.columns:
            key_features.append(feature)
    
    if key_features:
        st.markdown("### 📋 Key Features Included:")
        for feature in key_features:
            if results_df[feature].dtype in ['int64', 'float64']:
                mean_val = results_df[feature].mean()
                st.write(f"- **{feature}**: Mean = {mean_val:.2f}")
            else:
                st.write(f"- **{feature}**: {results_df[feature].dtype}")

if __name__ == "__main__":
    main()