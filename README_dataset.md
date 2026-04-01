---
license: mit
task_categories:
- text-classification
- text2text-generation
tags:
- biology
- medical
- virus
- genomics
- DNA
size_categories:
- 10K<n<100K
---
# Virus-Host-Genomes Dataset

## Latest Update

**Latest Update — v1.0.2 · 2026-04-01**

**+10 sequences** (58,206 total) · 5 human (50%) · 5 non-human · 11 fetched · 1 QC rejected

| Family | Added | Family | Added |
|--------|------:|--------|------:|
| Picornaviridae | 5 | Papillomaviridae | 2 |
| Adenoviridae | 2 | Poxviridae | 1 |
## Dataset Summary

Virus-Host-Genomes is a comprehensive collection of viral genomic sequences paired with host information, containing 58,206 viral sequences. The dataset includes metadata such as viral taxonomy (family, genus), host information, geographic data, isolation sources, and various annotations including zoonotic potential indicators. This dataset was put together to support investigations into genetic determinants of host specificity, zoonotic potential, and genome-based classification models.

**Last Updated:** 2026-04-01. The latest version is always available here on Hugging Face.

## Citation Information

If you use this dataset, please cite:

```
@article{carbajo2026sequence,
  author = {Carbajo, Alan L and Vensko, Taylor A and Pellett, Philip E},
  title = {Sequence Based Virus Host Prediction: A Curated Dataset and Generalizable Framework for Training Artificial Intelligence to Identify Viruses of Humans},
  year = {2026},
  url = {https://doi.org/10.1093/ve/veag009}
}
```

### Supported Tasks

- **Host Prediction**: Using viral sequences to predict potential hosts
- **Zoonotic Potential Assessment**: Identifying viruses with potential to cross between species
- **Taxonomic Classification**: Classifying viruses based on genomic sequences
- **Sequence Analysis**: Extracting sequence features like k-mer frequencies for analyses or preprocessing

### Dataset Structure

#### Data Instances

A typical data instance contains a virus genome sequence with taxonomic classification and host information:

```python
{
  'sequence': 'CCATTCCGGG...', # Viral genomic sequence
  'virus_name': 'Human betaherpesvirus 5', # Common virus name
  'host': 'human',            # Primary host (human or non-human)
  'zoonotic': False,          # Whether virus is known to be zoonotic
  # See Data Fields below for the full schema
}
```

#### Data Fields

The dataset contains the following fields:

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| sequence | string | Genomic sequence of the virus | "CCATTCCGGG..." |
| family | string | Taxonomic family of the virus | "Orthoherpesviridae" |
| accession | string | Database accession number | "AY446894.2" |
| host | string | Primary host (human or non-human) | "human" |
| genus | string | Taxonomic genus of the virus | "Cytomegalovirus" |
| isolation_date | string | Date when virus was isolated | "1999" |
| strain_name | string | Strain or isolate identifier | "Merlin" |
| location | string | Geographic location of isolation | "United Kingdom: Cardiff" |
| virus_name | string | Common name of the virus | "Human betaherpesvirus 5" |
| isolation_source | string | Source material of isolation | "urine from a congenitally infected child" |
| lab_culture | bool | Whether isolated from lab culture | true/false |
| wastewater_sewage | bool | Whether isolated from wastewater | true/false |
| standardized_host | string | Standardized host taxonomy | "Homo sapiens" |
| host_category | string | Category of host organism | "Mammal" |
| standardized_location | string | Standardized location | "United Kingdom" |
| zoonotic | bool | Known to cross species barriers | true/false |
| processing_method | string | How sequence was processed | "NGS" |
| gemini_annotated | bool | Annotated with Gemini AI | true/false |
| is_segmented | bool | Whether virus has segmented genome | true/false |
| segment_label | string | Label for genome segment | "NA" |

#### Data Splits
The dataset contains train and test splits:
| Split Name | Number of Instances |
|------------|---------------------|
| train | 52,079 |
| test | 6,127 |

## Dataset Creation
### Source Data
This dataset compiles virus sequences from multiple public repositories, including:
- NCBI Virus
- GenBank

### Data Processing
The dataset has undergone several processing steps:
- Sequence standardization (using only unambigious IUPAC nucleotide characters)
- Host information standardization
- Geographic location normalization
- Additional annotations including zoonotic potential labeling
- Quality filtering to remove low-quality or incomplete sequences

Host labels were generated through a tier-based approach:
1. Approximately 10,000 sequences were manually labeled by experts
2. First-tier automated labeling used direct string matching against known host names
3. Second-tier labeling employed pattern recognition from a species dictionary
4. For sequences that couldn't be classified by either tier, Google Gemini was used to analyze available metadata and assign host labels

Some sequences were annotated using the Gemini AI system to provide additional metadata where information was incomplete.

## Considerations for Using the Data
### Limitations and Biases
- **Sampling Bias**: The dataset may overrepresent viruses of clinical importance and underrepresent environmental viruses.
- **Temporal Distribution**: More recent viruses (especially those causing outbreaks) may be overrepresented.
- **Geographic Bias**: Samples from regions with stronger research infrastructure may be overrepresented.
- **Host Bias**: Human viruses and viruses from domestic/agricultural animals may be overrepresented.
- **Annotation Quality**: Some metadata fields are incomplete or may contain uncertainties.

## Usage Examples

### Data Preparation and K-mer Vectorization

```python
import numpy as np
from datasets import load_dataset
from itertools import product
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from tqdm import tqdm

# Load dataset
virus_dataset = load_dataset("hiyata/Virus-Host-Genomes")
train_dataset = virus_dataset['train']
test_dataset = virus_dataset['test']


# Generate k-mer dictionary once
def generate_kmer_dict(k):
    return {''.join(kmer): i for i, kmer in enumerate(product('ACGT', repeat=k))}

# Calculate k-mer frequency
def calculate_kmer_freq(seq, k, kmer_dict):
    freq = np.zeros(4**k)
    total_kmers = len(seq) - k + 1
    for i in range(total_kmers):
        kmer = seq[i:i+k]
        if 'N' not in kmer and all(base in 'ACGT' for base in kmer):
            freq[kmer_dict[kmer]] += 1
    return freq / total_kmers if total_kmers > 0 else freq

# Vectorize dataset
def vectorize_dataset(dataset, k=4):
    kmer_dict = generate_kmer_dict(k)
    num_samples = len(dataset['sequence'])
    X = np.zeros((num_samples, 4**k))
    y = np.array(['human' if host.lower() == 'human' else 'non-human' for host in dataset['host']])

    for idx, seq in enumerate(tqdm(dataset['sequence'], desc="Vectorizing sequences")):
        X[idx] = calculate_kmer_freq(seq.upper(), k, kmer_dict)

    return X, y


X_train, y_train = vectorize_dataset(train_dataset)
X_test, y_test = vectorize_dataset(test_dataset)

# Standard Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'standard_scaler.joblib')

# Label encoding
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print("Vectorization complete.")
```

### Neural Network Training for Host Classification

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Define your neural network
class VirusClassifier(nn.Module):
    def __init__(self, input_shape: int):
        super(VirusClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),

            nn.Linear(32, 32),
            nn.GELU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader setup
train_loader = DataLoader(TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train_enc, dtype=torch.long)
), batch_size=64, shuffle=True)

test_loader = DataLoader(TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test_enc, dtype=torch.long)
), batch_size=64, shuffle=False)

# Initialize the model
model = VirusClassifier(input_shape=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 15
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'virus_classifier_model.pth')
```

### Model Evaluation with Matthews Correlation Coefficient

```python
from sklearn.metrics import classification_report, matthews_corrcoef

model.eval()
y_preds = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_preds.extend(preds)
        y_true.extend(y_batch.numpy())

# Classification Report
report = classification_report(y_true, y_preds, target_names=['human', 'non-human'])
print("Classification Report:\n", report)

# MCC Score
mcc = matthews_corrcoef(y_true, y_preds)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
```

## Update History

| Date | Version | Added | Total | Human | Non-human | Note |
|------|---------|-------|-------|-------|-----------|------|
| 2026-04-01 | v1.0.2 | +10 | 58,206 | 5 | 5 |  |
| 2026-03-26 | v1.0.1 | +150 | 58,196 | 89 | 61 |  |
| 2026-03-02 | v1.0.0 | +0 | — | — | — | Initial dataset release |