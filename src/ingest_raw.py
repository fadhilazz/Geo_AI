#!/usr/bin/env python3
"""
Raw Data Ingestion Module for Geothermal Digital Twin AI

Processes raw subsurface data files to create 3D digital twin models:
1. 3D model files (.dat) → UniformGrid (.vtu) with PyVista
2. Well data (.csv, .xlsx) → structured datasets  
3. Geochemical data (.csv, .xlsx) → analysis summaries
4. Shapefiles (.shp) → spatial context

Dependencies:
- pyvista: 3D visualization and VTU export
- scipy.interpolate: Grid interpolation and smoothing
- pandas: Excel/CSV processing
- numpy: Numerical operations
- geopandas: Shapefile processing

Usage:
    python src/ingest_raw.py [--all] [--models] [--geochem] [--wells] [--shapes]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import hashlib

# Scientific computing
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial import KDTree

# 3D processing and visualization
import pyvista as pv
import vtk

# Progress tracking
from tqdm import tqdm

# Geospatial
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logging.warning("GeoPandas not available - shapefile processing disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
GRID_OUTPUT_DIR = Path("digital_twin/grids")
CACHE_DIR = Path("digital_twin/cache")

# Processing parameters
DEFAULT_GRID_RESOLUTION = (50, 50, 25)  # X, Y, Z resolution for interpolation
MIN_DATA_POINTS = 100  # Minimum points needed for gridding
INTERPOLATION_METHOD = 'linear'  # scipy interpolate method
VTU_COMPRESSION = True  # Compress VTU files


class RawDataIngester:
    """Main class for processing raw geothermal data into digital twin models."""
    
    def __init__(self):
        """Initialize the ingester with required directories and settings."""
        self.setup_directories()
        self.processed_files = self.load_processed_files()
        
        # Statistics tracking
        self.stats = {
            'models_processed': 0,
            'geochem_processed': 0,
            'wells_processed': 0,
            'shapes_processed': 0,
            'grids_created': 0,
            'errors': 0
        }
    
    def setup_directories(self):
        """Create required output directories."""
        for dir_path in [DATA_DIR, GRID_OUTPUT_DIR, CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("Directory structure ready")
    
    def load_processed_files(self) -> Dict[str, str]:
        """Load tracking of previously processed files."""
        processed_file = CACHE_DIR / "processed_raw_files.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load processed files tracking: {e}")
        return {}
    
    def save_processed_files(self):
        """Save tracking of processed files."""
        processed_file = CACHE_DIR / "processed_raw_files.json"
        try:
            with open(processed_file, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save processed files tracking: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to detect changes."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Could not hash file {file_path}: {e}")
            return ""
    
    def should_process_file(self, file_path: Path, force: bool = False) -> bool:
        """Check if file needs processing (new or modified)."""
        if force:
            return True
            
        file_key = str(file_path.relative_to(DATA_DIR))
        current_hash = self.get_file_hash(file_path)
        
        if file_key not in self.processed_files:
            return True
            
        return self.processed_files[file_key] != current_hash
    
    def read_dat_file(self, dat_path: Path) -> Optional[pd.DataFrame]:
        """
        Read .dat file with 3D model data.
        
        Expected format: X Y Z Value (space or tab separated)
        
        Returns:
            DataFrame with columns ['x', 'y', 'z', 'value'] or None if failed
        """
        try:
            logger.info(f"Reading {dat_path.name}...")
            
            # Get file size for progress estimation
            file_size = dat_path.stat().st_size
            
            # Try to read with pandas (with progress indication)
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading file") as pbar:
                df = pd.read_csv(
                    dat_path, 
                    sep=r'\s+',  # Multiple whitespace
                    header=None,
                    names=['x', 'y', 'z', 'value']
                )
                pbar.update(file_size)
            
            # Basic validation
            if len(df) < MIN_DATA_POINTS:
                logger.warning(f"Insufficient data points in {dat_path}: {len(df)}")
                return None
            
            # Check for required columns
            if df.shape[1] != 4:
                logger.error(f"Expected 4 columns (X,Y,Z,Value) in {dat_path}, got {df.shape[1]}")
                return None
            
            # Remove invalid values
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            logger.info(f"Successfully read {len(df)} points from {dat_path.name}")
            logger.info(f"  X range: {df['x'].min():.1f} to {df['x'].max():.1f}")
            logger.info(f"  Y range: {df['y'].min():.1f} to {df['y'].max():.1f}")
            logger.info(f"  Z range: {df['z'].min():.1f} to {df['z'].max():.1f}")
            logger.info(f"  Value range: {df['value'].min():.2e} to {df['value'].max():.2e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read {dat_path}: {e}")
            return None
    
    def create_structured_grid(self, df: pd.DataFrame, property_name: str) -> Optional[pv.StructuredGrid]:
        """
        Create PyVista StructuredGrid from DataFrame with memory optimization.
        
        Args:
            df: DataFrame with x, y, z, value columns
            property_name: Name for the property (e.g., 'resistivity', 'density')
        
        Returns:
            PyVista StructuredGrid or None if failed
        """
        try:
            # Get unique coordinates and sort them
            x_unique = np.sort(df['x'].unique())
            y_unique = np.sort(df['y'].unique())
            z_unique = np.sort(df['z'].unique())
            
            nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
            total_cells = nx * ny * nz
            
            logger.info(f"Original grid dimensions: {nx} x {ny} x {nz} = {total_cells:,} cells")
            
            # Memory check - if grid would be too large, use subsampling
            memory_gb = total_cells * 8 / (1024**3)  # 8 bytes per float64
            max_memory_gb = 4.0  # Maximum 4GB
            
            if memory_gb > max_memory_gb:
                logger.warning(f"Grid would require {memory_gb:.1f} GB memory - applying subsampling")
                
                # Calculate subsampling factor
                subsample_factor = int(np.ceil((memory_gb / max_memory_gb) ** (1/3)))
                logger.info(f"Using subsampling factor: {subsample_factor}")
                
                # Subsample coordinates
                x_sub = x_unique[::subsample_factor]
                y_sub = y_unique[::subsample_factor]
                z_sub = z_unique[::subsample_factor]
                
                nx, ny, nz = len(x_sub), len(y_sub), len(z_sub)
                logger.info(f"Subsampled grid dimensions: {nx} x {ny} x {nz} = {nx*ny*nz:,} cells")
                
                # Use subsampled coordinates
                x_unique, y_unique, z_unique = x_sub, y_sub, z_sub
            
            # Create meshgrid
            logger.info("Creating 3D meshgrid...")
            with tqdm(total=3, desc="Meshgrid creation") as pbar:
                X, Y, Z = np.meshgrid(x_unique, y_unique, z_unique, indexing='ij')
                pbar.update(3)
            
            # Initialize value array
            values = np.full((nx, ny, nz), np.nan, dtype=np.float32)  # Use float32 to save memory
            
            # Fill values from DataFrame using efficient indexing
            logger.info("Filling grid values...")
            
            # Create KDTree for efficient nearest neighbor search
            logger.info("Preparing spatial data...")
            with tqdm(total=2, desc="Spatial preparation") as pbar:
                grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
                pbar.update(1)
                data_points = df[['x', 'y', 'z']].values
                pbar.update(1)
            
            # Build KDTree from data points
            logger.info("Building spatial index (KDTree)...")
            with tqdm(total=len(data_points), desc="Building KDTree") as pbar:
                tree = KDTree(data_points)
                pbar.update(len(data_points))
            
            # Find nearest data point for each grid point (this is the slowest step)
            logger.info(f"Finding nearest neighbors for {len(grid_points):,} grid points...")
            with tqdm(total=len(grid_points), desc="Spatial interpolation") as pbar:
                # Process in chunks to show progress
                chunk_size = min(10000, len(grid_points))
                distances = np.zeros(len(grid_points))
                indices = np.zeros(len(grid_points), dtype=int)
                
                for i in range(0, len(grid_points), chunk_size):
                    end_idx = min(i + chunk_size, len(grid_points))
                    chunk_points = grid_points[i:end_idx]
                    
                    chunk_distances, chunk_indices = tree.query(chunk_points, k=1)
                    distances[i:end_idx] = chunk_distances
                    indices[i:end_idx] = chunk_indices
                    
                    pbar.update(len(chunk_points))
            
            # Set tolerance for interpolation (use mean spacing)
            x_spacing = np.mean(np.diff(x_unique)) if len(x_unique) > 1 else 1000
            y_spacing = np.mean(np.diff(y_unique)) if len(y_unique) > 1 else 1000
            z_spacing = np.mean(np.diff(z_unique)) if len(z_unique) > 1 else 25
            max_distance = np.sqrt(x_spacing**2 + y_spacing**2 + z_spacing**2)
            
            # Only use values within reasonable distance
            valid_mask = distances <= max_distance
            valid_indices = indices[valid_mask]
            
            # Fill values
            logger.info("Filling grid with interpolated values...")
            with tqdm(total=len(valid_indices), desc="Filling grid") as pbar:
                values_flat = values.ravel()
                values_flat[valid_mask] = df.iloc[valid_indices]['value'].values
                values = values_flat.reshape((nx, ny, nz))
                pbar.update(len(valid_indices))
            
            filled_count = np.sum(~np.isnan(values))
            fill_percentage = 100 * filled_count / values.size
            logger.info(f"Filled {filled_count:,} / {values.size:,} grid cells ({fill_percentage:.1f}%)")
            
            # Create structured grid
            grid = pv.StructuredGrid(X, Y, Z)
            
            # Add property data
            grid[property_name] = values.flatten(order='F')  # Fortran order for VTK
            
            # Add metadata
            grid.field_data[f'{property_name}_min'] = np.array([np.nanmin(values)])
            grid.field_data[f'{property_name}_max'] = np.array([np.nanmax(values)])
            grid.field_data[f'{property_name}_mean'] = np.array([np.nanmean(values)])
            grid.field_data[f'{property_name}_filled_ratio'] = np.array([np.sum(~np.isnan(values)) / values.size])
            
            logger.info(f"Created structured grid with {grid.n_points:,} points, {grid.n_cells:,} cells")
            
            return grid
            
        except Exception as e:
            logger.error(f"Failed to create structured grid: {e}")
            return None
    
    def interpolate_missing_values(self, grid: pv.StructuredGrid, property_name: str) -> pv.StructuredGrid:
        """
        Interpolate missing values in the grid using scipy.
        
        Args:
            grid: PyVista StructuredGrid with missing values
            property_name: Name of the property to interpolate
        
        Returns:
            Grid with interpolated values
        """
        try:
            # Get points and values
            points = grid.points
            values = grid[property_name]
            
            # Find valid (non-NaN) points
            valid_mask = ~np.isnan(values)
            if np.sum(valid_mask) < MIN_DATA_POINTS:
                logger.warning(f"Too few valid points for interpolation: {np.sum(valid_mask)}")
                return grid
            
            valid_points = points[valid_mask]
            valid_values = values[valid_mask]
            
            # Create interpolator
            logger.info(f"Interpolating {np.sum(~valid_mask)} missing values...")
            interpolator = interpolate.LinearNDInterpolator(
                valid_points, 
                valid_values, 
                fill_value=np.nan
            )
            
            # Interpolate missing values
            missing_points = points[~valid_mask]
            if len(missing_points) > 0:
                interpolated_values = interpolator(missing_points)
                values[~valid_mask] = interpolated_values
            
            # Update grid
            grid[property_name] = values
            
            # Update statistics
            grid.field_data[f'{property_name}_interpolated'] = np.array([np.sum(~valid_mask)])
            
            logger.info(f"Interpolation completed. {np.sum(~np.isnan(values))} valid values")
            
            return grid
            
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            return grid
    
    def process_3d_model_file(self, dat_path: Path, force: bool = False) -> bool:
        """
        Process a single 3D model .dat file into VTU format.
        
        Args:
            dat_path: Path to .dat file
            force: Force reprocessing even if cached
            
        Returns:
            True if processing succeeded
        """
        if not self.should_process_file(dat_path, force):
            logger.info(f"Skipping {dat_path.name} (already processed)")
            return True
        
        logger.info(f"Processing 3D model: {dat_path.name}")
        
        try:
            # Read data
            df = self.read_dat_file(dat_path)
            if df is None:
                return False
            
            # Determine property name from filename
            if 'res' in dat_path.name.lower():
                property_name = 'resistivity'
                unit = 'ohm_m'
            elif 'dens' in dat_path.name.lower():
                property_name = 'density'
                unit = 'g_cm3'
            else:
                property_name = 'value'
                unit = 'unknown'
            
            # Create structured grid
            grid = self.create_structured_grid(df, property_name)
            if grid is None:
                return False
            
            # Interpolate missing values
            grid = self.interpolate_missing_values(grid, property_name)
            
            # Add metadata
            grid.field_data['source_file'] = [dat_path.name]
            grid.field_data['processing_date'] = [time.time()]
            grid.field_data['property_unit'] = [unit]
            
            # Generate output filename
            output_name = f"{dat_path.stem}.vtu"
            output_path = GRID_OUTPUT_DIR / output_name
            
            # Save VTU file
            logger.info("Saving VTU grid file...")
            with tqdm(total=1, desc=f"Saving {output_name}") as pbar:
                if VTU_COMPRESSION:
                    grid.save(output_path, binary=True)
                else:
                    grid.save(output_path)
                pbar.update(1)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Saved grid to {output_path} ({file_size_mb:.1f} MB)")
            
            # Create summary metadata
            metadata = {
                'source_file': dat_path.name,
                'output_file': output_name,
                'property_name': property_name,
                'property_unit': unit,
                'grid_dimensions': [grid.dimensions[0], grid.dimensions[1], grid.dimensions[2]],
                'n_points': int(grid.n_points),
                'n_cells': int(grid.n_cells),
                'bounds': {
                    'x_min': float(grid.bounds[0]),
                    'x_max': float(grid.bounds[1]),
                    'y_min': float(grid.bounds[2]),
                    'y_max': float(grid.bounds[3]),
                    'z_min': float(grid.bounds[4]),
                    'z_max': float(grid.bounds[5])
                },
                'value_stats': {
                    'min': float(np.nanmin(grid[property_name])),
                    'max': float(np.nanmax(grid[property_name])),
                    'mean': float(np.nanmean(grid[property_name])),
                    'std': float(np.nanstd(grid[property_name]))
                },
                'processing_timestamp': time.time()
            }
            
            # Save metadata
            metadata_path = CACHE_DIR / f"{dat_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Mark as processed
            file_key = str(dat_path.relative_to(DATA_DIR))
            self.processed_files[file_key] = self.get_file_hash(dat_path)
            
            self.stats['models_processed'] += 1
            self.stats['grids_created'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {dat_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_geochemical_file(self, file_path: Path, force: bool = False) -> bool:
        """
        Process geochemical data file (Excel or CSV).
        
        Args:
            file_path: Path to geochemical file
            force: Force reprocessing
            
        Returns:
            True if processing succeeded
        """
        if not self.should_process_file(file_path, force):
            logger.info(f"Skipping {file_path.name} (already processed)")
            return True
        
        logger.info(f"Processing geochemical data: {file_path.name}")
        
        try:
            # Read file based on extension
            if file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.txt':
                # For text files, create a simple summary
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                summary = {
                    'source_file': file_path.name,
                    'file_type': 'text_report',
                    'content_length': len(content),
                    'processing_timestamp': time.time()
                }
                
                # Save summary
                summary_path = CACHE_DIR / f"{file_path.stem}_geochem_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # Mark as processed
                file_key = str(file_path.relative_to(DATA_DIR))
                self.processed_files[file_key] = self.get_file_hash(file_path)
                self.stats['geochem_processed'] += 1
                
                return True
            else:
                logger.warning(f"Unsupported geochemical file format: {file_path.suffix}")
                return False
            
            # Process DataFrame
            logger.info(f"Loaded geochemical data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Basic statistics
            summary = {
                'source_file': file_path.name,
                'file_type': 'spreadsheet',
                'n_rows': int(df.shape[0]),
                'n_columns': int(df.shape[1]),
                'columns': list(df.columns),
                'processing_timestamp': time.time()
            }
            
            # Add numeric column statistics with error handling
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary['numeric_stats'] = {}
                for col in numeric_cols:
                    try:
                        # Convert to numeric, coercing errors to NaN
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        summary['numeric_stats'][col] = {
                            'min': float(numeric_series.min()) if not numeric_series.isna().all() else None,
                            'max': float(numeric_series.max()) if not numeric_series.isna().all() else None,
                            'mean': float(numeric_series.mean()) if not numeric_series.isna().all() else None,
                            'std': float(numeric_series.std()) if not numeric_series.isna().all() else None,
                            'count': int(numeric_series.count())
                        }
                    except Exception as e:
                        logger.warning(f"Could not compute statistics for column {col}: {e}")
                        summary['numeric_stats'][col] = {'error': str(e)}
            
            # Look for coordinate columns with improved detection
            coord_columns = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                
                # X coordinate (Easting)
                if (col_lower in ['x', 'easting', 'longitude', 'lon', 'utm_x'] or
                    (col_lower.startswith('x') and len(col_lower) <= 3) or
                    ('east' in col_lower and 'southeast' not in col_lower)):
                    if 'x' not in coord_columns:  # Take first match
                        coord_columns['x'] = col
                
                # Y coordinate (Northing) - be more specific to avoid false matches
                elif (col_lower in ['y', 'northing', 'latitude', 'lat', 'utm_y'] or
                      (col_lower.startswith('y') and len(col_lower) <= 3) or
                      ('north' in col_lower and 'northeast' not in col_lower and 'northwest' not in col_lower)):
                    if 'y' not in coord_columns:  # Take first match
                        coord_columns['y'] = col
                
                # Z coordinate (Elevation/Depth)
                elif (col_lower in ['z', 'elevation', 'depth', 'altitude', 'elev'] or
                      (col_lower.startswith('z') and len(col_lower) <= 3)):
                    if 'z' not in coord_columns:  # Take first match
                        coord_columns['z'] = col
            
            if coord_columns:
                summary['coordinate_columns'] = coord_columns
                
                # Calculate spatial bounds
                if 'x' in coord_columns and 'y' in coord_columns:
                    summary['spatial_bounds'] = {
                        'x_min': float(df[coord_columns['x']].min()),
                        'x_max': float(df[coord_columns['x']].max()),
                        'y_min': float(df[coord_columns['y']].min()),
                        'y_max': float(df[coord_columns['y']].max())
                    }
                    
                    if 'z' in coord_columns:
                        summary['spatial_bounds'].update({
                            'z_min': float(df[coord_columns['z']].min()),
                            'z_max': float(df[coord_columns['z']].max())
                        })
            
            # Save processed data as CSV for easy access
            processed_path = CACHE_DIR / f"{file_path.stem}_processed.csv"
            df.to_csv(processed_path, index=False)
            summary['processed_file'] = str(processed_path)
            
            # Save summary
            summary_path = CACHE_DIR / f"{file_path.stem}_geochem_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Processed geochemical data saved to {processed_path}")
            
            # Mark as processed
            file_key = str(file_path.relative_to(DATA_DIR))
            self.processed_files[file_key] = self.get_file_hash(file_path)
            self.stats['geochem_processed'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process geochemical file {file_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_all_data(self, process_models: bool = True, process_geochem: bool = True, 
                        process_wells: bool = True, process_shapes: bool = True, 
                        force: bool = False) -> Dict[str, int]:
        """
        Process all raw data files.
        
        Args:
            process_models: Process 3D model files
            process_geochem: Process geochemical files
            process_wells: Process well data files
            process_shapes: Process shapefiles
            force: Force reprocessing of all files
            
        Returns:
            Processing statistics
        """
        logger.info("=== Starting Raw Data Ingestion ===")
        
        # Process 3D models
        if process_models:
            logger.info("Processing 3D model files...")
            model_files = list((DATA_DIR / "3d_models").rglob("*.dat"))
            
            if model_files:
                with tqdm(model_files, desc="Processing 3D models") as pbar:
                    for model_file in pbar:
                        pbar.set_description(f"Processing {model_file.name}")
                        self.process_3d_model_file(model_file, force)
                        self.save_processed_files()  # Save progress
            else:
                logger.info("No 3D model files found")
        
        # Process geochemical data
        if process_geochem:
            logger.info("Processing geochemical files...")
            geochem_files = list((DATA_DIR / "geochem").rglob("*.xlsx")) + \
                           list((DATA_DIR / "geochem").rglob("*.csv")) + \
                           list((DATA_DIR / "geochem").rglob("*.txt"))
            
            if geochem_files:
                with tqdm(geochem_files, desc="Processing geochemical data") as pbar:
                    for geochem_file in pbar:
                        pbar.set_description(f"Processing {geochem_file.name}")
                        self.process_geochemical_file(geochem_file, force)
                        self.save_processed_files()  # Save progress
            else:
                logger.info("No geochemical files found")
        
        # Process well data (similar to geochemical)
        if process_wells:
            logger.info("Processing well data files...")
            well_files = list((DATA_DIR / "wells").rglob("*.xlsx")) + \
                        list((DATA_DIR / "wells").rglob("*.csv"))
            
            if well_files:
                with tqdm(well_files, desc="Processing well data") as pbar:
                    for well_file in pbar:
                        pbar.set_description(f"Processing {well_file.name}")
                        # Use same processing as geochemical data
                        if self.process_geochemical_file(well_file, force):
                            self.stats['wells_processed'] += 1
                            self.stats['geochem_processed'] -= 1  # Adjust counter
                        self.save_processed_files()  # Save progress
            else:
                logger.info("No well data files found")
        
        # Process shapefiles
        if process_shapes and GEOPANDAS_AVAILABLE:
            logger.info("Processing shapefiles...")
            shape_files = list((DATA_DIR / "shapefiles").rglob("*.shp"))
            
            if shape_files:
                with tqdm(shape_files, desc="Processing shapefiles") as pbar:
                    for shape_file in pbar:
                        pbar.set_description(f"Processing {shape_file.name}")
                        if self.process_shapefile(shape_file, force):
                            self.stats['shapes_processed'] += 1
                        self.save_processed_files()  # Save progress
            else:
                logger.info("No shapefile files found")
        elif process_shapes and not GEOPANDAS_AVAILABLE:
            logger.warning("Shapefile processing requested but GeoPandas not available")
        
        # Save final state
        self.save_processed_files()
        
        logger.info(f"=== Raw Data Ingestion Complete ===")
        logger.info(f"Statistics: {self.stats}")
        
        return self.stats
    
    def process_shapefile(self, shp_path: Path, force: bool = False) -> bool:
        """
        Process shapefile data.
        
        Args:
            shp_path: Path to shapefile
            force: Force reprocessing
            
        Returns:
            True if processing succeeded
        """
        if not GEOPANDAS_AVAILABLE:
            logger.warning("GeoPandas not available - skipping shapefile")
            return False
        
        try:
            # Read shapefile
            gdf = gpd.read_file(shp_path)
            
            # Create summary
            summary = {
                'source_file': shp_path.name,
                'file_type': 'shapefile',
                'n_features': len(gdf),
                'geometry_type': str(gdf.geometry.type.iloc[0]) if len(gdf) > 0 else 'unknown',
                'crs': str(gdf.crs) if gdf.crs else 'unknown',
                'columns': list(gdf.columns),
                'bounds': list(gdf.bounds.iloc[0]) if len(gdf) > 0 else None,
                'processing_timestamp': time.time()
            }
            
            # Save processed data as GeoJSON for easy access
            processed_path = CACHE_DIR / f"{shp_path.stem}_processed.geojson"
            gdf.to_file(processed_path, driver='GeoJSON')
            summary['processed_file'] = str(processed_path)
            
            # Save summary
            summary_path = CACHE_DIR / f"{shp_path.stem}_shape_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Processed shapefile: {len(gdf)} features")
            
            # Mark as processed
            file_key = str(shp_path.relative_to(DATA_DIR))
            self.processed_files[file_key] = self.get_file_hash(shp_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process shapefile {shp_path}: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest raw geothermal data into digital twin")
    parser.add_argument("--all", action="store_true", help="Process all data types")
    parser.add_argument("--models", action="store_true", help="Process 3D model files only")
    parser.add_argument("--geochem", action="store_true", help="Process geochemical files only")
    parser.add_argument("--wells", action="store_true", help="Process well data files only")
    parser.add_argument("--shapes", action="store_true", help="Process shapefiles only")  
    parser.add_argument("--force", action="store_true", help="Force reprocessing (ignore cache)")
    
    args = parser.parse_args()
    
    # Default to all if no specific type selected
    if not any([args.models, args.geochem, args.wells, args.shapes]):
        args.all = True
    
    try:
        ingester = RawDataIngester()
        
        stats = ingester.process_all_data(
            process_models=args.all or args.models,
            process_geochem=args.all or args.geochem, 
            process_wells=args.all or args.wells,
            process_shapes=args.all or args.shapes,
            force=args.force
        )
        
        # Print summary
        print(f"\n=== Raw Data Ingestion Summary ===")
        print(f"3D models processed: {stats['models_processed']}")
        print(f"Geochemical files processed: {stats['geochem_processed']}")
        print(f"Well files processed: {stats['wells_processed']}")
        print(f"Shapefiles processed: {stats['shapes_processed']}")
        print(f"Grids created: {stats['grids_created']}")
        print(f"Errors: {stats['errors']}")
        
        if stats['errors'] > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()