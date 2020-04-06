
######IM SORRY FOR THIS MATLABY CODE I JUST WANT TO RUN A SCRIPT ################################################
import sys
import numpy as np
import pandas as pd
from brainsmash.mapgen.base import Base
from sklearn import decomposition

sys.path.append('../')
import echo_utils

make_nulls = False
run_correlations = False

### LOAD DATA
print('Loading data...')
df_tau = pd.read_csv('../data/df_tau_avg.csv', index_col=0)
df_struct = pd.read_csv('../data/df_structural_avg.csv', index_col=0)

# take out genes because there's a lot of operating on them
df_genes = df_struct[df_struct.columns[1:]]

##################################################################
### PARTIALING OUT COVARIATES
# Perform PCA on gene matrix
print('PCA & Regressions...')
n_pcs = 50
gene_pca = decomposition.PCA(n_pcs)
gene_pca.fit(np.array(df_genes))
df_gene_grad = pd.DataFrame(gene_pca.fit_transform(np.array(df_genes)), index = df_tau.index, columns = ['pc%i'%i for i in range(1,n_pcs+1)])
df_pc_weights = pd.DataFrame(gene_pca.components_.T, index=df_struct.columns[1:], columns=['pc%i'%i for i in range(1, n_pcs+1)])

### ----- REGRESS OUT T1T2 from timescale and gene
# regress out T1T2 from timescale features
x = df_struct['T1T2'].values
df_tau_rmvt1t2, _ = echo_utils.get_residuals(x, df_tau)

# remove T1/T2 contribution from all genes
df_gene_rmvt1t2, gene_t1t2_coeffs = echo_utils.get_residuals(x, df_genes)
### -------------------

### ----- REGRESS OUT PC1 from timescale and gene
# regress out PC1 from timescale features
x = df_gene_grad['pc1'].values
df_tau_rmvpc1, _ = echo_utils.get_residuals(x, df_tau)

# remove pc1 contribution from all genes
df_gene_rmvpc1, gene_pc1_coeffs = echo_utils.get_residuals(x, df_genes)
### -------------------

##################################################################
### MAKE NULL MAPS
if make_nulls:
    n_maps = 1000

    print('Making SA-preserving maps')
    # parcel geodesic distance
    df_parcel_geo = pd.read_csv('../data/LeftParcelGeodesicDistmat.txt', header=None, delimiter=' ')

    print('---- Making Surrogate for Raw Tau Maps ----')
    gen = Base(df_tau['tau'].values, df_parcel_geo.values, resample=True)
    tau_surrogates = gen(n=n_maps)
    pd.DataFrame(tau_surrogates.T, index=df_tau.index.values).to_csv('../data/tau_nulls.csv')

    print('---- Making Surrogate for PC1-Removed Tau Maps ----')
    gen = Base(df_tau_rmvpc1['tau'].values, df_parcel_geo.values, resample=True)
    tau_rmvpc1_surrogates = gen(n=n_maps)
    pd.DataFrame(tau_rmvpc1_surrogates.T, index=df_tau.index.values).to_csv('../data/tau_nulls_rmvpc1.csv')

    print('---- Making Surrogate for T1T2-Removed Tau Maps ----')
    gen = Base(df_tau_rmvt1t2['tau'].values, df_parcel_geo.values, resample=True)
    tau_rmvt1t2_surrogates = gen(n=n_maps)
    pd.DataFrame(tau_rmvt1t2_surrogates.T, index=df_tau.index.values).to_csv('../data/tau_nulls_rmvt1t2.csv')


##################################################################
### RUNNING CORRELATIONS
# load empirical and surrogate maps
if run_correlations:
    print('Running Correlations on Tau Maps...')
    map_emp = df_tau['tau'].values
    map_surr = pd.read_csv('../data/tau_nulls.csv', header=0, index_col=0).values
    df_tau_corr = echo_utils.run_emp_surrogate(map_emp, map_surr, df_genes, '../data/tau_all_corr.csv')
    print('-----')

    print('Running Correlations on PC1-removed Tau Maps...')
    map_emp = df_tau_rmvpc1['tau'].values
    map_surr = pd.read_csv('../data/tau_nulls_rmvpc1.csv', header=0, index_col=0).values
    df_tau_corr_rmvpc1 = echo_utils.run_emp_surrogate(map_emp, map_surr, df_gene_rmvpc1, '../data/tau_all_corr_rmvpc1.csv')
    print('-----')

    print('Running Correlations on T1T2-removed Tau Maps...')
    map_emp = df_tau_rmvt1t2['tau'].values
    map_surr = pd.read_csv('../data/tau_nulls_rmvt1t2.csv', header=0, index_col=0).values
    df_tau_corr_rmvt1t2 = echo_utils.run_emp_surrogate(map_emp, map_surr, df_gene_rmvt1t2, '../data/tau_all_corr_rmvt1t2.csv')
    print('-----')
