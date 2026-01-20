"""
Extract image metadata (magnification, um/pixel) from HEST dataset.
Supports reading from metadata CSV and TIFF file headers.
"""

import os
import pandas as pd
from pathlib import Path

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    print("Warning: tifffile not installed. Install with: pip install tifffile")

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Warning: openslide-python not installed. Install with: pip install openslide-python")


def load_metadata_csv():
    """Load the HEST metadata CSV file."""
    try:
        # Try local file first
        if os.path.exists('../assets/HEST_v1_2_0.csv'):
            return pd.read_csv('../assets/HEST_v1_2_0.csv')
        else:
            # Load from HuggingFace
            print("Loading metadata from HuggingFace...")
            return pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_2_0.csv")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def get_tiff_metadata(tiff_path):
    """Extract metadata from TIFF file."""
    metadata = {}
    
    if TIFFFILE_AVAILABLE:
        try:
            with tifffile.TiffFile(tiff_path) as tif:
                # Get basic image info
                metadata['width'] = tif.pages[0].imagewidth
                metadata['height'] = tif.pages[0].imagelength
                
                # Try to get resolution from tags
                if hasattr(tif.pages[0], 'tags'):
                    tags = tif.pages[0].tags
                    # Resolution tags (282, 283 are XResolution and YResolution)
                    if 282 in tags:
                        metadata['x_resolution'] = tags[282].value
                    if 283 in tags:
                        metadata['y_resolution'] = tags[283].value
                    if 296 in tags:  # ResolutionUnit
                        metadata['resolution_unit'] = tags[296].value
        except Exception as e:
            print(f"Error reading TIFF with tifffile: {e}")
    
    if OPENSLIDE_AVAILABLE:
        try:
            slide = openslide.OpenSlide(str(tiff_path))
            metadata['openslide_mpp_x'] = slide.properties.get('openslide.mpp-x')
            metadata['openslide_mpp_y'] = slide.properties.get('openslide.mpp-y')
            metadata['openslide_objective_power'] = slide.properties.get('openslide.objective-power')
            metadata['openslide_vendor'] = slide.properties.get('openslide.vendor')
            # Get all properties
            metadata['all_properties'] = dict(slide.properties)
            slide.close()
        except Exception as e:
            print(f"Error reading TIFF with openslide: {e}")
    
    return metadata


def get_image_metadata(data_dir='../hest_data', sample_ids=None, output_csv='image_metadata.csv'):
    """
    Get metadata for all images in the dataset.
    
    Args:
        data_dir: Directory containing HEST data
        sample_ids: Optional list of sample IDs to process (None = all)
        output_csv: Output CSV file path
    """
    # Load metadata CSV
    print("Loading HEST metadata CSV...")
    meta_df = load_metadata_csv()
    
    if meta_df is None:
        print("Could not load metadata CSV. Trying to extract from TIFF files only...")
        meta_df = pd.DataFrame()
    else:
        print(f"Loaded metadata for {len(meta_df)} samples")
        print(f"Available columns: {list(meta_df.columns)}")
    
    # Find all TIFF files in wsis folder
    wsis_dir = Path(data_dir) / 'wsis'
    if not wsis_dir.exists():
        print(f"WSIs directory not found: {wsis_dir}")
        return None
    
    tiff_files = list(wsis_dir.glob('*.tif')) + list(wsis_dir.glob('*.tiff'))
    print(f"\nFound {len(tiff_files)} TIFF files")
    
    # Extract metadata from each file
    results = []
    for tiff_path in tiff_files:
        sample_id = tiff_path.stem
        if sample_ids and sample_id not in sample_ids:
            continue
        
        print(f"\nProcessing: {sample_id}")
        file_metadata = get_tiff_metadata(tiff_path)
        
        # Combine with CSV metadata if available
        result = {'sample_id': sample_id, 'file_path': str(tiff_path)}
        
        if not meta_df.empty and sample_id in meta_df['id'].values:
            sample_meta = meta_df[meta_df['id'] == sample_id].iloc[0]
            for col in meta_df.columns:
                result[f'meta_{col}'] = sample_meta[col]
        
        # Add TIFF metadata
        result.update(file_metadata)
        results.append(result)
    
    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        print(f"\n{'='*60}")
        print("Summary of extracted metadata:")
        print(f"{'='*60}")
        print(f"Total samples: {len(df_results)}")
        print(f"\nColumns: {list(df_results.columns)}")
        
        # Show samples with openslide metadata
        if 'openslide_mpp_x' in df_results.columns:
            mpp_samples = df_results[df_results['openslide_mpp_x'].notna()]
            print(f"\nSamples with openslide metadata: {len(mpp_samples)}")
            if len(mpp_samples) > 0:
                print("\nSample openslide metadata:")
                print(mpp_samples[['sample_id', 'openslide_mpp_x', 'openslide_mpp_y', 
                                   'openslide_objective_power']].head(10))
        
        # Save to CSV
        df_results.to_csv(output_csv, index=False)
        print(f"\nMetadata saved to: {output_csv}")
        
        return df_results
    else:
        print("No metadata extracted.")
        return None


if __name__ == "__main__":
    import sys
    
    # Check if specific sample IDs provided
    sample_ids = None
    if len(sys.argv) > 1:
        sample_ids = sys.argv[1].split(',')
        print(f"Processing specific samples: {sample_ids}")
    
    # Get metadata for all lung samples (or all if sample_ids not specified)
    metadata_df = get_image_metadata(
        data_dir='../hest_data',
        sample_ids=sample_ids,
        output_csv='hest_image_metadata.csv'
    )
    
    if metadata_df is not None:
        print("\n" + "="*60)
        print("Quick preview:")
        print("="*60)
        print(metadata_df.head())
