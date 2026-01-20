# download_hest

Scripts to download HEST-1k dataset files from HuggingFace.

## Files

- **`download_hest.py`** - Main script to download HEST-1k samples from HuggingFace
- **`get_image_metadata.py`** - Extract image metadata (magnification, pixel size) from downloaded images
- **`hest_image_metadata.csv`** - Metadata for all downloaded images
- **`image_info_lung.csv`** - Filtered metadata for lung samples

## Installation

Install required dependencies:

```bash
pip install datasets huggingface_hub tqdm pandas scanpy
```

## Setup HuggingFace Authentication

1. Get your HuggingFace token from https://huggingface.co/settings/tokens

2. **Option 1: Set as environment variable (recommended)**
   ```bash
   # Windows PowerShell
   $env:HF_TOKEN="your_token_here"
   
   # Windows CMD
   set HF_TOKEN=your_token_here
   
   # Linux/Mac
   export HF_TOKEN="your_token_here"
   ```

3. **Option 2: Edit the script directly**
   Edit `download_hest.py` and set your token on line 237:
   ```python
   HF_TOKEN = os.getenv("HF_TOKEN", "your_token_here")
   ```
   
   Replace `"your_token_here"` with your actual token.

**Note:** The script checks the `HF_TOKEN` environment variable first. If not set, it uses the default value in the script. If neither is set, it will prompt you to login interactively.

## Usage

### Download HEST-1k Samples

Run `download_hest.py` to download samples from each provider:

```bash
# Download 2 samples per provider (default)
python download_hest.py

# Download 3 samples per provider
python download_hest.py 3

# Download 1 sample per provider
python download_hest.py 1
```

**What it does:**
- Reads `hest_image_metadata.csv` to find all providers (MEND, MISC, NCBI, TENX, etc.)
- Downloads the first N samples from each provider
- Downloads all data folders: wsis/, st/, patches/, metadata/, etc.
- Saves to `../hest_data/` directory

**Example output:**
```
Found 4 providers: ['MISC', 'NCBI', 'MEND', 'TENX']

MISC: 2 samples
  IDs: ['MISC13', 'MISC130']

NCBI: 2 samples
  IDs: ['NCBI534', 'NCBI535']

MEND: 2 samples
  IDs: ['MEND41', 'MEND45']

TENX: 2 samples
  IDs: ['TENX118', 'TENX141']

Total samples to download: 8
```

### Extract Image Metadata

Run `get_image_metadata.py` to extract metadata from downloaded images:

```bash
python get_image_metadata.py
```

**What it does:**
- Scans `../hest_data/wsis/` for downloaded TIFF files
- Extracts metadata from `hest_image_metadata.csv`
- Combines metadata into a comprehensive CSV
- Saves to `hest_image_metadata.csv`

**Output columns:**
- Sample ID, organ, disease state
- Magnification (20x, 40x)
- Pixel size (µm/pixel)
- Image dimensions
- Technology type, publication info

## Dataset Structure

After downloading, each sample includes:

- **`wsis/`** - H&E-stained whole slide images (TIFF format)
- **`st/`** - Spatial transcriptomics data (scanpy .h5ad format)
- **`patches/`** - 224×224 pixel patches extracted around ST spots (.h5 format)
- **`metadata/`** - Sample metadata (JSON format)
- **`spatial_plots/`** - Overlay visualizations (JPG format)

## Example Workflow

```bash
# 1. Download samples (2 per provider)
python download_hest.py 2

# 2. Extract metadata
python get_image_metadata.py

# 3. Use the data
python
>>> import scanpy as sc
>>> st = sc.read_h5ad('../hest_data/st/MEND41.h5ad')
>>> print(f"Spots: {st.shape[0]}, Genes: {st.shape[1]}")
```

## Requirements

- Python 3.7+
- HuggingFace account and token
- See dependencies in installation section

## Resources

- **HEST GitHub**: https://github.com/mahmoodlab/HEST
- **HuggingFace Dataset**: https://huggingface.co/datasets/MahmoodLab/hest
- **Paper**: NeurIPS 2024 - HEST-1k Dataset

## Citation

If you use HEST-1k in your research, please cite:

```bibtex
@inproceedings{jaume2024hest,
    author = {Guillaume Jaume and Paul Doucet and Andrew H. Song and Ming Y. Lu and Cristina Almagro-Perez and Sophia J. Wagner and Anurag J. Vaidya and Richard J. Chen and Drew F. K. Williamson and Ahrong Kim and Faisal Mahmood},
    title = {HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2024},
    month = dec,
}
```
