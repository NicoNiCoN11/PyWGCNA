import PyWGCNA as pywgcna
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
input_path = "/home/jiguo/data/data/anndata/pseudobulk_adata_20replicate.h5ad"
output_dir = "/home/jiguo/data/data/wgcna_negesr/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

print("Loading data...")
# Load anndata object
adata = ad.read_h5ad(input_path)
print(f"Loaded data with {adata.n_obs} samples and {adata.n_vars} genes")
print(f"Data matrix type: {type(adata.X)}")

# Check if matrix is sparse and convert to dense for WGCNA
if hasattr(adata.X, 'todense'):
    print("Converting sparse matrix to dense...")
    adata.X = adata.X.todense()
    # Convert matrix to numpy array
    adata.X = np.asarray(adata.X)
    print("Conversion complete")

# Check if data is already log-transformed
if 'log1p' in adata.uns:
    print("Data appears to be already log-transformed, skipping normalization...")
else:
    print("Normalizing raw counts...")
    # Store raw counts
    adata.layers['raw_counts'] = adata.X.copy()
    
    # Normalize every cell to have the same total count (CPM normalization)
    sc.pp.normalize_total(adata, target_sum=1e6)
    
    # Log transform
    sc.pp.log1p(adata, base=2)  # This does log2(x + 1)
    print("Data normalized to log2(CPM + 1)")

# Create PyWGCNA object
print("Initializing PyWGCNA...")
wgcna = pywgcna.WGCNA(
    name='pseudobulk_analysis',
    species='human',
    anndata=adata,
    TPMcutoff=1,  # Filter genes with low expression
    save=True,
    outputPath=output_dir,
    figureType='pdf',
    networkType="signed hybrid",
    minModuleSize=50
)

# Set metadata colors to avoid plotting errors
print("Setting metadata colors...")
# For condition (categorical)
condition_colors = {
    'control': '#1f77b4',      # blue
    'centrinone': '#ff7f0e',   # orange
    'wo-2h': '#2ca02c',        # green
    'wo-8h': '#d62728'         # red
}
wgcna.setMetadataColor('condition', condition_colors)

# For replicate (you can use a color map or discrete colors)
# Assuming replicates are numbered, we'll use a gradient
import matplotlib.cm as cm
import matplotlib.colors as mcolors
n_replicates = len(adata.obs['replicate'].unique())
replicate_cmap = cm.get_cmap('viridis', n_replicates)
wgcna.setMetadataColor('replicate', replicate_cmap)

# Step 1: Preprocessing
print("\nStep 1: Preprocessing data...")
wgcna.preprocess(show=True)

# Step 2: Find modules
print("\nStep 2: Finding modules...")
wgcna.findModules(kwargs_function={
    'cutreeHybrid': {
        'deepSplit': 2,
        'pamRespectsDendro': False
    }
})

# Step 3: Analyze WGCNA results with modified parameters
print("\nStep 3: Analyzing WGCNA results...")
# Run analysis but handle potential plotting errors
try:
    wgcna.analyseWGCNA(
        order=['condition', 'replicate'],
        show=True
    )
except Exception as e:
    print(f"Full analysis encountered an error: {e}")
    print("Running analysis components separately...")
    
    # Run the essential analyses manually
    # Calculate module-trait relationships
    print("Calculating module-trait relationships...")
    wgcna.module_trait_relationships_heatmap(
        metaData=['condition', 'replicate'],
        alternative='two-sided',
        show=True,
        file_name='module-traitRelationships'
    )
    
    # Calculate signed kME
    print("Calculating module membership...")
    wgcna.CalculateSignedKME()
    
    # Skip the problematic plotting functions
    print("Skipping module eigengene plots due to compatibility issues...")

# Step 4: Additional analyses
print("\nStep 4: Running additional analyses...")

# Get module information
modules = wgcna.getModuleName()
print(f"Found {len(modules)} modules: {modules}")

# Save module assignments
module_assignments = wgcna.datExpr.var[['gene_name', 'ensembl_gene_id', 'moduleColors']]
# Add module size information
module_sizes = module_assignments.groupby('moduleColors').size().to_dict()
module_assignments['module_size'] = module_assignments['moduleColors'].map(module_sizes)
module_assignments.to_csv(os.path.join(output_dir, 'module_assignments.csv'))
print(f"Module assignments saved to {os.path.join(output_dir, 'module_assignments.csv')}")

# Get module eigengenes
if hasattr(wgcna, 'MEs') and wgcna.MEs is not None:
    module_eigengenes = wgcna.MEs
    module_eigengenes.to_csv(os.path.join(output_dir, 'module_eigengenes.csv'))
    print(f"Module eigengenes saved to {os.path.join(output_dir, 'module_eigengenes.csv')}")

# Save module-trait relationships
if hasattr(wgcna, 'moduleTraitCor') and wgcna.moduleTraitCor is not None:
    wgcna.moduleTraitCor.to_csv(os.path.join(output_dir, 'module_trait_correlation.csv'))
    wgcna.moduleTraitPvalue.to_csv(os.path.join(output_dir, 'module_trait_pvalues.csv'))
    print("Module-trait relationships saved")

# Save module membership (kME values)
if hasattr(wgcna, 'signedKME') and wgcna.signedKME is not None:
    wgcna.signedKME.to_csv(os.path.join(output_dir, 'module_membership_kME.csv'))
    print("Module membership (kME) values saved")

# Step 5: Identify hub genes for each module
print("\nStep 5: Identifying hub genes...")
hub_genes_all = {}
for module in modules:
    if module != 'grey':  # Skip unassigned genes
        try:
            hub_genes = wgcna.top_n_hub_genes(module, n=20)
            hub_genes_all[module] = hub_genes
            # Save hub genes for each module
            hub_genes.to_csv(os.path.join(output_dir, f'hub_genes_{module}.csv'))
        except Exception as e:
            print(f"Failed to get hub genes for module {module}: {e}")

print("Hub genes saved for each module")

# Step 6: Functional enrichment analysis
print("\nStep 6: Running functional enrichment analysis...")
# Check if gene_name column exists
if 'gene_name' in wgcna.datExpr.var.columns:
    for module in modules:
        if module != 'grey':
            try:
                # GO enrichment
                wgcna.functional_enrichment_analysis(
                    type="GO",
                    moduleName=module,
                    sets=["GO_Biological_Process_2021", "GO_Molecular_Function_2021"],
                    p_value=0.05,
                    file_name=f"{module}_GO"
                )
                
                # KEGG enrichment
                wgcna.functional_enrichment_analysis(
                    type="KEGG",
                    moduleName=module,
                    sets=["KEGG_2021_Human"],
                    p_value=0.05,
                    file_name=f"{module}_KEGG"
                )
            except Exception as e:
                print(f"Enrichment analysis failed for module {module}: {e}")
else:
    print("Skipping enrichment analysis - gene_name column not found")

# Step 7: Create network visualizations
print("\nStep 7: Creating network visualizations...")
try:
    # Visualize the largest non-grey modules
    module_sizes_sorted = sorted(
        [(m, sum(wgcna.datExpr.var['moduleColors'] == m)) for m in modules if m != 'grey'],
        key=lambda x: x[1],
        reverse=True
    )
    
    if len(module_sizes_sorted) >= 2:
        top_modules = [m[0] for m in module_sizes_sorted[:2]]
        wgcna.CoexpressionModulePlot(
            modules=top_modules,
            numGenes=30,
            numConnections=100,
            minTOM=0.1,
            file_name="top_modules_network"
        )
except Exception as e:
    print(f"Network visualization failed: {e}")

# Step 8: Save the complete WGCNA object
print("\nStep 8: Saving WGCNA object...")
wgcna.saveWGCNA()

# Step 9: Generate comprehensive summary
print("\nStep 9: Generating summary report...")

# Create module summary table
module_summary = []
for module in modules:
    module_info = {
        'Module': module,
        'Size': sum(wgcna.datExpr.var['moduleColors'] == module),
        'Color': module
    }
    
    # Add top hub genes if available
    if module in hub_genes_all and len(hub_genes_all[module]) > 0:
        top_genes = hub_genes_all[module].head(3)
        module_info['Top_Hub_Genes'] = ', '.join(top_genes.index.tolist())
        module_info['Top_Hub_Connectivity'] = round(top_genes['connectivity'].iloc[0], 2)
    
    # Add correlations with conditions if available
    if hasattr(wgcna, 'moduleTraitCor') and wgcna.moduleTraitCor is not None:
        for cond in ['control', 'centrinone', 'wo-2h', 'wo-8h']:
            col_matches = [col for col in wgcna.moduleTraitCor.columns if cond in col]
            if col_matches:
                col = col_matches[0]
                module_info[f'Cor_{cond}'] = round(wgcna.moduleTraitCor.loc[f'ME{module}', col], 3)
                module_info[f'Pval_{cond}'] = f"{wgcna.moduleTraitPvalue.loc[f'ME{module}', col]:.2e}"
    
    module_summary.append(module_info)

module_summary_df = pd.DataFrame(module_summary)
module_summary_df = module_summary_df.sort_values('Size', ascending=False)
module_summary_df.to_csv(os.path.join(output_dir, 'module_summary.csv'), index=False)

# Write text summary
summary = {
    'Total genes analyzed': wgcna.datExpr.shape[1],
    'Total samples': wgcna.datExpr.shape[0],
    'Number of modules': len(modules),
    'Number of assigned genes': sum(wgcna.datExpr.var['moduleColors'] != 'grey'),
    'Number of unassigned genes': sum(wgcna.datExpr.var['moduleColors'] == 'grey'),
    'Soft threshold power': wgcna.power,
    'Network type': wgcna.networkType,
    'Minimum module size': wgcna.minModuleSize
}

with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
    f.write("WGCNA Analysis Summary\n")
    f.write("=" * 50 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\nModule Sizes:\n")
    for _, row in module_summary_df.iterrows():
        f.write(f"  {row['Module']}: {row['Size']} genes\n")

print("\nAnalysis complete! Results saved to:", output_dir)
print("\nKey output files:")
print(f"- Module assignments: module_assignments.csv")
print(f"- Module eigengenes: module_eigengenes.csv")
print(f"- Module-trait correlations: module_trait_correlation.csv")
print(f"- Module summary: module_summary.csv")
print(f"- Analysis summary: analysis_summary.txt")
print(f"- Hub genes: hub_genes_*.csv")
print(f"- WGCNA object: pseudobulk_analysis.p")

print("\nTo reload and explore the results later:")
print(f"import PyWGCNA as pywgcna")
print(f"wgcna = pywgcna.readWGCNA('{os.path.join(output_dir, 'pseudobulk_analysis.p')}')")