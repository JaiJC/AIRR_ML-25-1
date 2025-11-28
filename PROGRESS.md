# AIRR-ML-25: Adaptive Immune Profiling Challenge - Progress Log

## 📋 Project Overview
**Competition:** AIRR-ML-25 Kaggle Challenge  
**Objective:** 
1. Predict immune state (disease vs. healthy) from adaptive immune repertoires
2. Identify immune receptor sequences most strongly associated with the target immune state

---

## ✅ Completed Steps

### Phase 1: Exploratory Data Analysis (EDA)

#### 1. Setup & Configuration
- [x] Imported required libraries (pandas, numpy, matplotlib, seaborn, scipy, sklearn)
- [x] Configured data paths for Kaggle environment
- [x] Set up display options and plotting styles

#### 2. Utility Functions
- [x] Created data loading functions:
  - `load_metadata()` - Load metadata.csv from dataset directory
  - `load_all_metadata()` - Combine metadata from all datasets
  - `load_repertoire()` - Load single repertoire TSV file
  - `load_sample_repertoires()` - Load sample files for quick analysis
  - `get_repertoire_summary()` - Get summary statistics for repertoires

#### 3. Metadata Analysis
- [x] **3.1 Overview** - Analyzed all training datasets structure
- [x] **3.2 Class Balance** - Examined label distribution (positive vs negative)
  - Visualized overall and per-dataset class distribution
  - Calculated imbalance ratios
- [x] **3.3 Demographics** - Analyzed age, sex, race, study_group distributions
- [x] **3.4 Correlations** - Statistical tests for feature-label associations
  - Pearson correlation for continuous variables
  - Chi-squared test for categorical variables
  - Cramér's V for effect size

#### 4. Sequence-Level Analysis
- [x] **4.1 Repertoire Structure** - Examined TSV file columns and data types
- [x] **4.2 Repertoire Sizes** - Distribution of sequences per repertoire
- [x] **4.3 Junction AA Analysis**
  - Sequence length distributions
  - Amino acid frequency analysis
- [x] **4.4 V/J/D Gene Calls** - Gene usage distributions and diversity

#### 5. Diversity Metrics
- [x] **5.1 Repertoire Diversity**
  - Shannon entropy
  - Simpson's diversity index
  - Clonality
  - Richness (unique/total sequences)
- [x] **5.2 Shared Sequences (Public Clones)**
  - Identified sequences appearing across multiple individuals
  - Analyzed disease-enriched shared sequences ("star soldiers")

#### 6. Technical Bias Check
- [x] Analyzed batch effects by sequencing_run_id
- [x] Chi-squared test for label distribution across runs
- [x] Visualized potential confounders

#### 7. HLA Gene Analysis
- [x] Identified HLA-related columns in metadata
- [x] Analyzed HLA allele distributions
- [x] Tested HLA-label associations

#### 8. Missing Values Assessment
- [x] Identified missing values in metadata
- [x] Checked for -999.0 placeholder values
- [x] Analyzed repertoire-level missing values

---

### Phase 2: Improved Data Cleaning & Feature Engineering (Revised)

#### Philosophy: Sequences are PRIMARY signal, metadata is secondary!

#### 8.1.1 Handle Missing HLA Values Properly
- [x] Replaced -999.0 placeholders with NaN
- [x] Created explicit `_missing` flags for each HLA locus
- [x] **NO "Unknown" hack** - missing values remain NaN
- [x] All rows kept (no dropping due to missing values)

#### 8.1.2 Out-of-Fold Target Encoding for HLA (With Smoothing)
- [x] Implemented out-of-fold target encoding to prevent leakage
- [x] Applied smoothing toward global mean to regularize rare alleles
- [x] Created `target_encode_oof()` function with configurable smoothing weight
- [x] Stored encoding maps for test set application

#### 8.1.3 HLA Interaction Features
- [x] Created HLA-HLA pairwise interactions (haplotype effects):
  - HLA_A_1 × HLA_B_1, HLA_A_2 × HLA_B_2
  - HLA_A_1 × HLA_C_1, HLA_B_1 × HLA_C_1
  - HLA_DRB1_1 × HLA_DQB1_1, HLA_DRB1_2 × HLA_DQB1_2
- [x] Created HLA × study_group interactions
- [x] Applied target encoding to interaction features (with higher smoothing)

#### 8.1.4 Repertoire-Level Feature Extraction Functions (PRIMARY SIGNAL)
- [x] Created `extract_repertoire_features()` function with:
  - Basic stats: n_sequences, total_templates, clonality
  - CDR3 length: mean, std, median, percentiles, short/long fractions
  - V gene usage: unique count, entropy, Gini, top gene frequencies
  - J gene usage: unique count, entropy, top gene frequency
  - D gene usage: unique count, missing fraction, entropy
  - V-J pairing diversity: unique pairs, entropy
  - Amino acid composition: hydrophobic, charged, aromatic, cysteine fractions
  - Clone size distribution: mean, std, max, Gini, top clone fractions
- [x] Created `gini_coefficient()` helper function

#### 8.1.5 K-mer Features from CDR3 Sequences
- [x] Created `extract_kmers()` for single sequence k-mer extraction
- [x] Created `build_kmer_vocabulary()` for building vocabulary from training data
- [x] Created `extract_kmer_features_with_vocab()` for feature extraction with fixed vocabulary
- [x] Supports abundance-weighted k-mer counting

#### 8.1.6 Final Data Cleaning Summary
- [x] Documented all transformations
- [x] Stored encoding maps in `all_encodings` dictionary
- [x] Ready for feature engineering and modeling

---

### Phase 3: Additional EDA (Post-Cleaning)

#### 9. Train vs Test Comparison
- [x] Compared train/test dataset counts and structures
- [x] Mapped test datasets to training counterparts
- [x] Analyzed column overlap
- [x] Compared repertoire sizes

#### 10. Dimensionality Reduction
- [x] Computed k-mer (3-mer) features for repertoires
- [x] Applied PCA for variance analysis
- [x] Applied t-SNE for visualization
- [x] Checked label separability

#### 11. EDA Summary
- [x] Compiled key findings
- [x] Listed recommendations for modeling

---

## 🔄 In Progress

*Nothing currently in progress*

---

## 📝 Next Steps (TODO)

### Phase 4: Feature Engineering
- [ ] K-mer frequency features (3-mer, 4-mer, 5-mer)
- [ ] V/J gene usage profiles
- [ ] Diversity metrics as features
- [ ] Sequence length distribution features
- [ ] Public clone presence/absence features

### Phase 5: Model Development
- [ ] Baseline model (simple classifier)
- [ ] Traditional ML models (XGBoost, Random Forest)
- [ ] Deep learning models (if applicable)
- [ ] Cross-validation strategy (stratified by dataset)

### Phase 6: Important Sequence Identification
- [ ] Fisher's exact test for sequence-label association
- [ ] Feature importance from tree-based models
- [ ] Ranking top 50,000 sequences per dataset

### Phase 7: Submission
- [ ] Generate predictions for test data
- [ ] Generate ranked important sequences
- [ ] Create submission file
- [ ] Validate submission format

---

## 📊 Key Artifacts

| Artifact | Description | Status |
|----------|-------------|--------|
| `train_metadata` | Original combined metadata | ✅ Created |
| `train_metadata_cleaned` | Cleaned & processed metadata | ✅ Created |
| `all_encodings` | HLA & interaction encoding maps | ✅ Saved |
| `hla_target_encodings` | OOF target encodings for HLA | ✅ Saved |
| `interaction_target_encodings` | OOF target encodings for interactions | ✅ Saved |
| `extract_repertoire_features()` | Primary feature extraction function | ✅ Defined |
| `build_kmer_vocabulary()` | K-mer vocabulary builder | ✅ Defined |
| `extract_kmer_features_with_vocab()` | K-mer feature extraction | ✅ Defined |
| `target_encode_oof()` | Out-of-fold target encoding | ✅ Defined |

---

## 📁 File Structure

```
AIRR_ML-25-1/
├── jc_airr_ml.ipynb      # Main notebook
├── PROGRESS.md           # This file
├── LICENSE
└── .git/
```

---

*Last Updated: November 27, 2025*
