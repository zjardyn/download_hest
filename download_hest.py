"""
Download HEST-1k dataset from HuggingFace.
Supports downloading full dataset, specific sample IDs, or filtered samples.
"""

import os
import sys
import time
import zipfile
from huggingface_hub import snapshot_download, login
from tqdm import tqdm


def download_hest(patterns, local_dir, max_workers=4, max_retries=3):
    """
    Download HEST dataset from HuggingFace with retry logic.
    
    Args:
        patterns: List of file patterns to download (e.g., ['*'] for all files)
        local_dir: Local directory to save the dataset
        max_workers: Number of parallel download workers (default: 4, reduced for stability)
        max_retries: Maximum number of retry attempts (default: 3)
    """
    repo_id = 'MahmoodLab/hest'
    
    for attempt in range(max_retries):
        try:
            print(f"\nDownload attempt {attempt + 1}/{max_retries}...")
            snapshot_download(
                repo_id=repo_id, 
                allow_patterns=patterns, 
                repo_type="dataset", 
                local_dir=local_dir,
                max_workers=max_workers,
                resume_download=True  # Resume interrupted downloads
            )
            print("Download completed successfully!")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Error encountered: {str(e)}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Download failed after {max_retries} attempts.")
                print("You can resume the download by running the script again.")
                raise

    # Unzip cellvit segmentation files
    seg_dir = os.path.join(local_dir, 'cellvit_seg')
    if os.path.exists(seg_dir):
        print('Unzipping cell vit segmentation...')
        zip_files = [s for s in os.listdir(seg_dir) if s.endswith('.zip')]
        for filename in tqdm(zip_files):
            path_zip = os.path.join(seg_dir, filename)
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(seg_dir)


def download_full_dataset(local_dir='../hest_data', max_workers=4):
    """
    Download full HEST-1k dataset (~1TB).
    
    Args:
        local_dir: Local directory to save the dataset
        max_workers: Number of parallel download workers (default: 4)
    """
    print("Downloading full HEST-1k dataset (~1TB)...")
    download_hest('*', local_dir, max_workers=max_workers)


def download_by_ids(ids_to_query, local_dir='../hest_data', max_workers=4, folders=None):
    """
    Download HEST-1k samples based on sample IDs.
    
    Args:
        ids_to_query: List of sample IDs (e.g., ['TENX95', 'TENX99'])
        local_dir: Local directory to save the dataset
        max_workers: Number of parallel download workers (default: 4)
        folders: List of specific folders to download (e.g., ['wsis'] for H&E images only)
                 If None, downloads all folders
    """
    print(f"Downloading samples: {ids_to_query}")
    if folders:
        # Download only specific folders (e.g., wsis for H&E images)
        list_patterns = []
        for id in ids_to_query:
            for folder in folders:
                # Match files in folder and subdirectories
                list_patterns.append(f"{folder}/**/*{id}*")
                list_patterns.append(f"{folder}/*{id}*")
    else:
        # Download all folders
        list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    download_hest(list_patterns, local_dir, max_workers=max_workers)


def download_he_slides_only(ids_to_query, local_dir='../hest_data', max_workers=4):
    """
    Download only H&E whole slide images (wsis folder) for given sample IDs.
    
    Args:
        ids_to_query: List of sample IDs (e.g., ['TENX95', 'TENX99'])
        local_dir: Local directory to save the dataset
        max_workers: Number of parallel download workers (default: 4)
    """
    print(f"Downloading H&E slides only for samples: {ids_to_query}")
    download_by_ids(ids_to_query, local_dir, max_workers=max_workers, folders=['wsis'])


def download_by_metadata(organ=None, oncotree_code=None, local_dir='../hest_data', max_workers=4, folders=None):
    """
    Download HEST-1k samples based on metadata filters.
    
    Args:
        organ: Filter by organ (e.g., 'Breast', 'Lung')
        oncotree_code: Filter by oncotree code (e.g., 'IDC')
        local_dir: Local directory to save the dataset
        max_workers: Number of parallel download workers (default: 4)
        folders: List of specific folders to download (e.g., ['wsis'] for H&E images only)
                 If None, downloads all folders
    """
    import pandas as pd
    
    print("Loading metadata...")
    meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_2_0.csv")
    
    # Apply filters
    if oncotree_code:
        meta_df = meta_df[meta_df['oncotree_code'] == oncotree_code]
        print(f"Filtered by oncotree_code: {oncotree_code}")
    
    if organ:
        meta_df = meta_df[meta_df['organ'] == organ]
        print(f"Filtered by organ: {organ}")
    
    ids_to_query = meta_df['id'].values
    print(f"Found {len(ids_to_query)} samples matching criteria")
    
    if folders:
        print(f"Downloading only folders: {folders}")
    
    download_by_ids(ids_to_query, local_dir, max_workers=max_workers, folders=folders)


def get_provider_from_id(sample_id):
    """Extract provider prefix from sample ID."""
    import re
    match = re.match(r'^([A-Z]+)', sample_id)
    if match:
        return match.group(1)
    return None


def download_by_provider(metadata_csv='hest_image_metadata.csv', 
                         samples_per_provider=2,
                         local_dir='../hest_data',
                         max_workers=4,
                         folders=None):
    """
    Download first N samples from each provider using metadata CSV.
    
    Args:
        metadata_csv: Path to metadata CSV file (default: 'hest_image_metadata.csv')
        samples_per_provider: Number of samples to download per provider (default: 2)
        local_dir: Local directory to save the dataset
        max_workers: Number of parallel download workers (default: 4)
        folders: List of specific folders to download (None = all folders)
    
    Returns:
        Tuple of (list of sample IDs, dict of provider -> sample IDs)
    """
    import pandas as pd
    
    # Load metadata
    print(f"Loading metadata from {metadata_csv}...")
    if not os.path.exists(metadata_csv):
        print(f"Warning: {metadata_csv} not found. Trying to load from HuggingFace...")
        df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_2_0.csv")
        df.rename(columns={'id': 'sample_id'}, inplace=True)
    else:
        df = pd.read_csv(metadata_csv)
        # Handle both 'sample_id' and 'id' column names
        if 'sample_id' not in df.columns and 'id' in df.columns:
            df.rename(columns={'id': 'sample_id'}, inplace=True)
    
    # Extract provider for each sample
    df['provider'] = df['sample_id'].apply(get_provider_from_id)
    
    # Get unique providers
    providers = df['provider'].value_counts().index.tolist()
    print(f"\nFound {len(providers)} providers: {providers}")
    
    # Get first N samples from each provider (sorted by sample_id for consistency)
    samples_to_download = []
    provider_samples = {}
    
    for provider in providers:
        provider_df = df[df['provider'] == provider].sort_values('sample_id').head(samples_per_provider)
        sample_ids = provider_df['sample_id'].tolist()
        provider_samples[provider] = sample_ids
        samples_to_download.extend(sample_ids)
        print(f"\n{provider}: {len(sample_ids)} samples (first {samples_per_provider})")
        print(f"  IDs: {sample_ids}")
    
    print(f"\n{'='*60}")
    print(f"Total samples to download: {len(samples_to_download)}")
    print(f"{'='*60}")
    
    if folders:
        print(f"\nDownloading only folders: {folders}")
    else:
        print("\nDownloading all folders (wsis, st, patches, metadata, etc.)")
    
    # Download samples
    download_by_ids(
        ids_to_query=samples_to_download,
        local_dir=local_dir,
        max_workers=max_workers,
        folders=folders
    )
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")
    print(f"\nDownloaded samples by provider:")
    for provider, samples in provider_samples.items():
        print(f"  {provider}: {len(samples)} samples - {samples}")
    
    return samples_to_download, provider_samples


if __name__ == "__main__":
    # Set your HuggingFace token here
    # You can also set it as an environment variable: HF_TOKEN
    HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HUGGING_FACE_TOKEN")
    
    if HF_TOKEN == "YOUR_HUGGING_FACE_TOKEN":
        print("WARNING: Please set your HuggingFace token!")
        print("\nOption 1: Set HF_TOKEN environment variable")
        print("   PowerShell: $env:HF_TOKEN='your_token_here'")
        print("   CMD: set HF_TOKEN=your_token_here")
        print("   Linux/Mac: export HF_TOKEN='your_token_here'")
        print("\nOption 2: Edit this script and set HF_TOKEN variable on line 237")
        print("\nOption 3: Run interactively in a terminal")
        
        # Check if running in interactive mode
        if sys.stdin.isatty():
            try:
                response = input("\nDo you want to login interactively? (y/n): ")
                if response.lower() == 'y':
                    login()
                else:
                    print("Exiting. Please set your token and try again.")
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting. Please set your token and try again.")
                sys.exit(1)
        else:
            print("\nNon-interactive mode detected. Please set HF_TOKEN environment variable or edit the script.")
            sys.exit(1)
    else:
        login(token=HF_TOKEN)
    
    # Get number of samples per provider from command line or prompt
    samples_per_provider = 2  # Default
    
    if len(sys.argv) > 1:
        try:
            samples_per_provider = int(sys.argv[1])
            print(f"Using {samples_per_provider} samples per provider from command line argument")
        except ValueError:
            print(f"Invalid argument '{sys.argv[1]}'. Using default: {samples_per_provider}")
    elif sys.stdin.isatty():
        try:
            response = input(f"\nHow many samples per provider? (default: {samples_per_provider}): ")
            if response.strip():
                samples_per_provider = int(response.strip())
        except (ValueError, EOFError, KeyboardInterrupt):
            print(f"Using default: {samples_per_provider} samples per provider")
    
    # Download first N samples from each provider with all files
    print(f"\n{'='*60}")
    print(f"Downloading {samples_per_provider} samples per provider")
    print(f"{'='*60}")
    download_by_provider(
        metadata_csv='hest_image_metadata.csv',
        samples_per_provider=samples_per_provider,
        local_dir='../hest_data',
        max_workers=4,
        folders=None  # Download all folders
    )
