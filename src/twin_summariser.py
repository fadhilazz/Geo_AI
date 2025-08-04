#!/usr/bin/env python3
"""
Digital Twin Summariser Module for Geothermal Systems

Analyzes processed 3D grids and applies geological interpretation rules to extract
engineering insights about geothermal reservoirs, caprocks, and heat sources.

Key Features:
1. Load and analyze VTU grid files from digital_twin/grids/
2. Apply geothermal interpretation rules based on resistivity, density, depth
3. Identify key geological zones (caprock, reservoir, heat source, fractures)
4. Generate YAML summaries with quantitative interpretations
5. Create engineering recommendations and drilling targets

Dependencies:
- pyvista: Load and analyze VTU grids
- ruamel.yaml: Generate structured YAML outputs
- numpy, scipy: Numerical analysis and statistics
- scikit-learn: Clustering and anomaly detection

Usage:
    python src/twin_summariser.py [--all] [--update] [--models MODEL_NAME]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from datetime import datetime

# Scientific computing
import numpy as np
from scipy import ndimage, stats
from scipy.spatial.distance import cdist

# Machine learning for pattern recognition
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available - advanced analysis disabled")

# 3D data handling
import pyvista as pv

# YAML output
from ruamel.yaml import YAML

# Progress tracking
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
GRIDS_DIR = Path("digital_twin/grids")
CACHE_DIR = Path("digital_twin/cache")
SUMMARIES_FILE = CACHE_DIR / "twin_summaries.yaml"

# Geothermal interpretation thresholds (based on international standards)
GEOTHERMAL_RULES = {
    'caprock': {
        'resistivity_max': 10.0,      # Ω·m - Clay caprock typically <10
        'depth_max': 500.0,           # m - Shallow caprock
        'description': 'Low-permeability seal preventing fluid escape'
    },
    'reservoir': {
        'resistivity_min': 10.0,      # Ω·m - Above caprock values
        'resistivity_max': 200.0,     # Ω·m - Below basement values
        'depth_min': 200.0,           # m - Below caprock
        'depth_max': 3000.0,          # m - Economic drilling depth
        'temperature_min': 150.0,     # °C - Minimum for power generation
        'description': 'Permeable zone with geothermal fluids'
    },
    'basement': {
        'resistivity_min': 200.0,     # Ω·m - High resistivity rocks
        'depth_min': 1000.0,          # m - Deep crystalline rocks
        'description': 'Crystalline basement with potential heat source'
    },
    'fracture_zones': {
        'resistivity_contrast_min': 50.0,  # Ω·m - Sharp resistivity contrasts
        'density_contrast_min': 0.2,       # g/cm³ - Density variations
        'description': 'Fractured zones enhancing permeability'
    },
    'alteration_zones': {
        'density_variation_threshold': 0.15,  # g/cm³ - Hydrothermal alteration
        'resistivity_variation_threshold': 30.0,  # Ω·m - Alteration effects
        'description': 'Hydrothermally altered rocks indicating fluid flow'
    }
}

# Temperature estimation from depth (simplified geothermal gradient)
GEOTHERMAL_GRADIENT = 30.0  # °C/km - typical geothermal gradient
SURFACE_TEMPERATURE = 25.0  # °C - surface temperature


class GeothermalTwinSummariser:
    """Main class for analyzing geothermal digital twin models."""
    
    def __init__(self):
        """Initialize the summariser with required directories."""
        self.setup_directories()
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False
        
        # Analysis results storage
        self.grid_summaries = {}
        self.integrated_summary = {}
        
    def setup_directories(self):
        """Create required directories."""
        for dir_path in [GRIDS_DIR, CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("Directory structure ready")
    
    def load_vtu_grid(self, vtu_path: Path) -> Optional[pv.StructuredGrid]:
        """
        Load VTU grid file and validate structure.
        
        Args:
            vtu_path: Path to VTU file
            
        Returns:
            PyVista StructuredGrid or None if failed
        """
        try:
            logger.info(f"Loading grid: {vtu_path.name}")
            grid = pv.read(vtu_path)
            
            # Ensure it's a structured grid
            if not isinstance(grid, pv.StructuredGrid):
                logger.warning(f"Grid {vtu_path.name} is not structured - attempting conversion")
                # Try to convert to structured grid if possible
                return None
            
            logger.info(f"Loaded grid: {grid.n_points:,} points, {grid.n_cells:,} cells")
            logger.info(f"Properties: {list(grid.array_names)}")
            
            return grid
            
        except Exception as e:
            logger.error(f"Failed to load grid {vtu_path}: {e}")
            return None
    
    def estimate_temperature_from_depth(self, depths: np.ndarray) -> np.ndarray:
        """
        Estimate temperature from depth using geothermal gradient.
        
        Args:
            depths: Depth values in meters (positive down)
            
        Returns:
            Estimated temperatures in Celsius
        """
        # Convert to positive depths (below surface)
        depths_positive = np.abs(depths)
        temperatures = SURFACE_TEMPERATURE + (depths_positive / 1000.0) * GEOTHERMAL_GRADIENT
        return temperatures
    
    def identify_geological_zones(self, grid: pv.StructuredGrid, property_name: str) -> Dict[str, Any]:
        """
        Apply geothermal interpretation rules to identify geological zones.
        
        Args:
            grid: PyVista grid with geophysical properties
            property_name: Name of the primary property (resistivity/density)
            
        Returns:
            Dictionary with zone interpretations
        """
        try:
            # Get coordinates and property values
            points = grid.points
            values = grid[property_name]
            
            # Remove NaN values
            valid_mask = ~np.isnan(values)
            if np.sum(valid_mask) == 0:
                logger.warning("No valid data points for analysis")
                return {}
            
            valid_points = points[valid_mask]
            valid_values = values[valid_mask]
            
            # Extract spatial coordinates
            x_coords = valid_points[:, 0]
            y_coords = valid_points[:, 1]
            z_coords = valid_points[:, 2]  # Elevation (positive up)
            depths = -z_coords  # Convert to depth (positive down)
            
            # Estimate temperatures
            temperatures = self.estimate_temperature_from_depth(depths)
            
            zones = {}
            
            # Identify Caprock
            if property_name.lower() == 'resistivity':
                caprock_mask = (
                    (valid_values <= GEOTHERMAL_RULES['caprock']['resistivity_max']) &
                    (depths <= GEOTHERMAL_RULES['caprock']['depth_max']) &
                    (depths >= 0)  # Above surface
                )
                
                if np.sum(caprock_mask) > 0:
                    zones['caprock'] = {
                        'volume_km3': self.calculate_zone_volume(valid_points[caprock_mask]),
                        'depth_range_m': [float(depths[caprock_mask].min()), float(depths[caprock_mask].max())],
                        'resistivity_range_ohm_m': [float(valid_values[caprock_mask].min()), float(valid_values[caprock_mask].max())],
                        'area_km2': self.calculate_zone_area(valid_points[caprock_mask]),
                        'thickness_m': float(depths[caprock_mask].max() - depths[caprock_mask].min()),
                        'confidence': self.calculate_zone_confidence(caprock_mask, 'caprock'),
                        'description': GEOTHERMAL_RULES['caprock']['description']
                    }
            
            # Identify Reservoir Zone
            if property_name.lower() == 'resistivity':
                reservoir_mask = (
                    (valid_values >= GEOTHERMAL_RULES['reservoir']['resistivity_min']) &
                    (valid_values <= GEOTHERMAL_RULES['reservoir']['resistivity_max']) &
                    (depths >= GEOTHERMAL_RULES['reservoir']['depth_min']) &
                    (depths <= GEOTHERMAL_RULES['reservoir']['depth_max']) &
                    (temperatures >= GEOTHERMAL_RULES['reservoir']['temperature_min'])
                )
                
                if np.sum(reservoir_mask) > 0:
                    zones['reservoir'] = {
                        'volume_km3': self.calculate_zone_volume(valid_points[reservoir_mask]),
                        'depth_range_m': [float(depths[reservoir_mask].min()), float(depths[reservoir_mask].max())],
                        'resistivity_range_ohm_m': [float(valid_values[reservoir_mask].min()), float(valid_values[reservoir_mask].max())],
                        'temperature_range_c': [float(temperatures[reservoir_mask].min()), float(temperatures[reservoir_mask].max())],
                        'area_km2': self.calculate_zone_area(valid_points[reservoir_mask]),
                        'average_depth_m': float(depths[reservoir_mask].mean()),
                        'estimated_capacity_mw': self.estimate_power_capacity(zones.get('reservoir', {})),
                        'confidence': self.calculate_zone_confidence(reservoir_mask, 'reservoir'),
                        'description': GEOTHERMAL_RULES['reservoir']['description']
                    }
            
            # Identify Basement
            if property_name.lower() == 'resistivity':
                basement_mask = (
                    (valid_values >= GEOTHERMAL_RULES['basement']['resistivity_min']) &
                    (depths >= GEOTHERMAL_RULES['basement']['depth_min'])
                )
                
                if np.sum(basement_mask) > 0:
                    zones['basement'] = {
                        'volume_km3': self.calculate_zone_volume(valid_points[basement_mask]),
                        'depth_range_m': [float(depths[basement_mask].min()), float(depths[basement_mask].max())],
                        'resistivity_range_ohm_m': [float(valid_values[basement_mask].min()), float(valid_values[basement_mask].max())],
                        'temperature_range_c': [float(temperatures[basement_mask].min()), float(temperatures[basement_mask].max())],
                        'heat_source_potential': self.assess_heat_source_potential(depths[basement_mask], temperatures[basement_mask]),
                        'confidence': self.calculate_zone_confidence(basement_mask, 'basement'),
                        'description': GEOTHERMAL_RULES['basement']['description']
                    }
            
            # Identify Fracture Zones (requires spatial analysis)
            fracture_zones = self.identify_fracture_zones(valid_points, valid_values, property_name)
            if fracture_zones:
                zones['fracture_zones'] = fracture_zones
            
            # Identify Alteration Zones
            alteration_zones = self.identify_alteration_zones(valid_points, valid_values, depths, property_name)
            if alteration_zones:
                zones['alteration_zones'] = alteration_zones
            
            return zones
            
        except Exception as e:
            logger.error(f"Failed to identify geological zones: {e}")
            return {}
    
    def calculate_zone_volume(self, points: np.ndarray) -> float:
        """Calculate approximate volume of a zone in km³."""
        if len(points) < 4:
            return 0.0
        
        try:
            # Use convex hull volume approximation
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            volume_m3 = hull.volume
            volume_km3 = volume_m3 / (1000**3)
            return float(volume_km3)
        except:
            # Fallback: bounding box volume
            x_range = points[:, 0].max() - points[:, 0].min()
            y_range = points[:, 1].max() - points[:, 1].min()
            z_range = points[:, 2].max() - points[:, 2].min()
            volume_km3 = (x_range * y_range * z_range) / (1000**3)
            return float(volume_km3)
    
    def calculate_zone_area(self, points: np.ndarray) -> float:
        """Calculate approximate surface area of a zone in km²."""
        if len(points) < 3:
            return 0.0
        
        # Project to horizontal plane and calculate area
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        area_km2 = (x_range * y_range) / (1000**2)
        return float(area_km2)
    
    def calculate_zone_confidence(self, zone_mask: np.ndarray, zone_type: str) -> float:
        """Calculate confidence score for zone identification (0-1)."""
        # Simple confidence based on data coverage
        data_points = np.sum(zone_mask)
        total_points = len(zone_mask)
        
        if total_points == 0:
            return 0.0
        
        coverage_score = min(data_points / max(total_points * 0.1, 100), 1.0)  # Normalize
        
        # Add zone-specific confidence factors
        zone_confidence_factors = {
            'caprock': 0.8,  # Usually well-defined
            'reservoir': 0.7,  # More complex
            'basement': 0.6,  # Deep, less certain
            'fracture_zones': 0.5,  # Requires more analysis
            'alteration_zones': 0.6  # Moderate confidence
        }
        
        base_confidence = zone_confidence_factors.get(zone_type, 0.5)
        final_confidence = coverage_score * base_confidence
        
        return float(min(final_confidence, 1.0))
    
    def estimate_power_capacity(self, reservoir_info: Dict) -> float:
        """Estimate geothermal power capacity in MW."""
        if not reservoir_info:
            return 0.0
        
        try:
            # Simplified capacity estimation based on volume and temperature
            volume_km3 = reservoir_info.get('volume_km3', 0)
            avg_temp = reservoir_info.get('temperature_range_c', [150, 200])
            avg_temp = (avg_temp[0] + avg_temp[1]) / 2 if len(avg_temp) >= 2 else 175
            
            # Empirical formula: ~1-3 MW per km³ depending on temperature
            temp_factor = max((avg_temp - 100) / 100, 0.5)  # Temperature effectiveness
            capacity_mw = volume_km3 * temp_factor * 2.0  # Base 2 MW/km³
            
            return float(min(capacity_mw, 1000))  # Cap at 1000 MW
            
        except:
            return 0.0
    
    def assess_heat_source_potential(self, depths: np.ndarray, temperatures: np.ndarray) -> str:
        """Assess heat source potential based on depth and temperature."""
        if len(temperatures) == 0:
            return "Unknown"
        
        max_temp = temperatures.max()
        avg_depth = depths.mean()
        
        if max_temp > 300 and avg_depth > 2000:
            return "High - Deep high-temperature source"
        elif max_temp > 200 and avg_depth > 1500:
            return "Moderate - Adequate temperature and depth"
        elif max_temp > 150:
            return "Low - Marginal temperature"
        else:
            return "Poor - Insufficient temperature"
    
    def identify_fracture_zones(self, points: np.ndarray, values: np.ndarray, property_name: str) -> Optional[Dict]:
        """Identify fracture zones based on property contrasts."""
        if not SKLEARN_AVAILABLE or len(points) < 100:
            return None
        
        try:
            # Use spatial clustering to find anomalous zones
            features = np.column_stack([points, values.reshape(-1, 1)])
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use DBSCAN to find clusters and outliers
            dbscan = DBSCAN(eps=0.3, min_samples=20)
            clusters = dbscan.fit_predict(features_scaled)
            
            # Identify outlier zones (potential fractures)
            outlier_mask = clusters == -1
            
            if np.sum(outlier_mask) > 0:
                fracture_points = points[outlier_mask]
                fracture_values = values[outlier_mask]
                
                return {
                    'volume_km3': self.calculate_zone_volume(fracture_points),
                    'area_km2': self.calculate_zone_area(fracture_points),
                    'property_range': [float(fracture_values.min()), float(fracture_values.max())],
                    'n_anomalies': int(np.sum(outlier_mask)),
                    'confidence': 0.5,  # Moderate confidence for fracture detection
                    'description': GEOTHERMAL_RULES['fracture_zones']['description']
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Fracture zone identification failed: {e}")
            return None
    
    def identify_alteration_zones(self, points: np.ndarray, values: np.ndarray, depths: np.ndarray, property_name: str) -> Optional[Dict]:
        """Identify hydrothermal alteration zones."""
        try:
            # Calculate local property variations
            if len(values) < 50:
                return None
            
            # Use rolling statistics to find high-variation zones
            window_size = min(50, len(values) // 10)
            if window_size < 10:
                return None
            
            # Calculate moving standard deviation
            rolling_std = np.array([
                np.std(values[max(0, i-window_size//2):min(len(values), i+window_size//2)])
                for i in range(len(values))
            ])
            
            # Identify high-variation zones
            variation_threshold = np.percentile(rolling_std, 90)  # Top 10% variation
            alteration_mask = rolling_std > variation_threshold
            
            if np.sum(alteration_mask) > 0:
                alteration_points = points[alteration_mask]
                alteration_values = values[alteration_mask]
                alteration_depths = depths[alteration_mask]
                
                return {
                    'volume_km3': self.calculate_zone_volume(alteration_points),
                    'area_km2': self.calculate_zone_area(alteration_points),
                    'depth_range_m': [float(alteration_depths.min()), float(alteration_depths.max())],
                    'property_variation': float(variation_threshold),
                    'n_zones': int(np.sum(alteration_mask)),
                    'confidence': 0.6,
                    'description': GEOTHERMAL_RULES['alteration_zones']['description']
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Alteration zone identification failed: {e}")
            return None
    
    def generate_drilling_recommendations(self, zones: Dict[str, Any]) -> List[Dict]:
        """Generate drilling target recommendations based on zone analysis."""
        recommendations = []
        
        # Priority 1: Reservoir zones with good characteristics
        if 'reservoir' in zones:
            reservoir = zones['reservoir']
            if reservoir['confidence'] > 0.5 and reservoir.get('volume_km3', 0) > 0.1:
                recommendations.append({
                    'priority': 1,
                    'target_type': 'Production Well',
                    'target_zone': 'reservoir',
                    'depth_range_m': reservoir['depth_range_m'],
                    'estimated_capacity_mw': reservoir.get('estimated_capacity_mw', 0),
                    'confidence': reservoir['confidence'],
                    'rationale': 'Primary reservoir zone with favorable temperature and resistivity'
                })
        
        # Priority 2: Intersection of reservoir and fracture zones
        if 'reservoir' in zones and 'fracture_zones' in zones:
            recommendations.append({
                'priority': 2,
                'target_type': 'Enhanced Production Well',
                'target_zone': 'reservoir + fractures',
                'depth_range_m': zones['reservoir']['depth_range_m'],
                'estimated_capacity_mw': zones['reservoir'].get('estimated_capacity_mw', 0) * 1.2,  # 20% boost
                'confidence': min(zones['reservoir']['confidence'], zones['fracture_zones']['confidence']),
                'rationale': 'Reservoir zone enhanced by natural fracture permeability'
            })
        
        # Priority 3: Injection wells in caprock margins
        if 'caprock' in zones and 'reservoir' in zones:
            recommendations.append({
                'priority': 3,
                'target_type': 'Injection Well',
                'target_zone': 'caprock margin',
                'depth_range_m': [zones['caprock']['depth_range_m'][1], zones['reservoir']['depth_range_m'][0]],
                'estimated_capacity_mw': 0,  # Injection, no production
                'confidence': zones['caprock']['confidence'],
                'rationale': 'Injection at caprock-reservoir boundary for sustainable operation'
            })
        
        # Priority 4: Exploration targets in basement
        if 'basement' in zones:
            basement = zones['basement']
            if basement.get('heat_source_potential', '').startswith(('High', 'Moderate')):
                recommendations.append({
                    'priority': 4,
                    'target_type': 'Exploration Well',
                    'target_zone': 'basement',
                    'depth_range_m': basement['depth_range_m'],
                    'estimated_capacity_mw': 0,  # Exploration
                    'confidence': basement['confidence'] * 0.7,  # Reduced for exploration
                    'rationale': f"Deep heat source exploration - {basement.get('heat_source_potential', 'Unknown')}"
                })
        
        return sorted(recommendations, key=lambda x: x['priority'])
    
    def analyze_single_grid(self, vtu_path: Path) -> Dict[str, Any]:
        """
        Analyze a single VTU grid file.
        
        Args:
            vtu_path: Path to VTU file
            
        Returns:
            Analysis summary dictionary
        """
        logger.info(f"Analyzing grid: {vtu_path.name}")
        
        # Load grid
        grid = self.load_vtu_grid(vtu_path)
        if grid is None:
            return {'error': f'Failed to load grid {vtu_path.name}'}
        
        # Get metadata
        metadata_path = CACHE_DIR / f"{vtu_path.stem}_metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Determine primary property
        property_names = grid.array_names
        primary_property = None
        
        for prop in ['resistivity', 'density']:
            if prop in property_names:
                primary_property = prop
                break
        
        if primary_property is None and property_names:
            primary_property = property_names[0]
            logger.warning(f"Using {primary_property} as primary property")
        
        if primary_property is None:
            return {'error': 'No suitable properties found in grid'}
        
        # Analyze zones
        zones = self.identify_geological_zones(grid, primary_property)
        
        # Generate recommendations
        recommendations = self.generate_drilling_recommendations(zones)
        
        # Create summary
        summary = {
            'source_file': vtu_path.name,
            'analysis_timestamp': datetime.now().isoformat(),
            'grid_info': {
                'dimensions': list(grid.dimensions),
                'n_points': int(grid.n_points),
                'n_cells': int(grid.n_cells),
                'bounds': {
                    'x_range_m': [float(grid.bounds[0]), float(grid.bounds[1])],
                    'y_range_m': [float(grid.bounds[2]), float(grid.bounds[3])],
                    'z_range_m': [float(grid.bounds[4]), float(grid.bounds[5])]
                },
                'properties': list(property_names)
            },
            'primary_property': primary_property,
            'geological_zones': zones,
            'drilling_recommendations': recommendations,
            'engineering_summary': self.create_engineering_summary(zones, recommendations),
            'metadata': metadata
        }
        
        return summary
    
    def create_engineering_summary(self, zones: Dict, recommendations: List[Dict]) -> Dict[str, Any]:
        """Create high-level engineering summary."""
        summary = {
            'geothermal_potential': 'Unknown',
            'development_phase': 'Exploration',
            'key_findings': [],
            'risks_and_challenges': [],
            'estimated_total_capacity_mw': 0.0,
            'recommended_next_steps': []
        }
        
        # Assess overall potential
        if 'reservoir' in zones:
            reservoir = zones['reservoir']
            capacity = reservoir.get('estimated_capacity_mw', 0)
            confidence = reservoir.get('confidence', 0)
            
            if capacity > 50 and confidence > 0.7:
                summary['geothermal_potential'] = 'High'
                summary['development_phase'] = 'Development'
            elif capacity > 20 and confidence > 0.5:
                summary['geothermal_potential'] = 'Moderate'
                summary['development_phase'] = 'Pre-development'
            elif capacity > 5:
                summary['geothermal_potential'] = 'Low'
            
            summary['estimated_total_capacity_mw'] = float(capacity)
        
        # Key findings
        for zone_name, zone_data in zones.items():
            if zone_data.get('confidence', 0) > 0.5:
                volume = zone_data.get('volume_km3', 0)
                if volume > 0.1:
                    summary['key_findings'].append(
                        f"{zone_name.title()}: {volume:.2f} km³ identified with {zone_data.get('confidence', 0):.1%} confidence"
                    )
        
        # Risks and challenges
        if 'caprock' not in zones:
            summary['risks_and_challenges'].append("No clear caprock identified - potential fluid loss risk")
        
        if zones.get('reservoir', {}).get('confidence', 0) < 0.6:
            summary['risks_and_challenges'].append("Reservoir characterization uncertain - requires additional data")
        
        if summary['estimated_total_capacity_mw'] < 10:
            summary['risks_and_challenges'].append("Low estimated capacity - economic viability uncertain")
        
        # Next steps
        if recommendations:
            summary['recommended_next_steps'] = [
                f"{rec['target_type']} at {rec['depth_range_m'][0]:.0f}-{rec['depth_range_m'][1]:.0f}m depth"
                for rec in recommendations[:3]  # Top 3 recommendations
            ]
        
        return summary
    
    def analyze_all_grids(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Analyze all VTU grid files in the grids directory.
        
        Args:
            force_update: Force re-analysis even if cached
            
        Returns:
            Integrated analysis summary
        """
        logger.info("=== Starting Digital Twin Analysis ===")
        
        # Find all VTU files
        vtu_files = list(GRIDS_DIR.glob("*.vtu"))
        if not vtu_files:
            logger.warning("No VTU grid files found")
            return {'error': 'No grid files to analyze'}
        
        logger.info(f"Found {len(vtu_files)} grid files to analyze")
        
        # Analyze each grid
        grid_analyses = {}
        
        with tqdm(vtu_files, desc="Analyzing grids") as pbar:
            for vtu_file in pbar:
                pbar.set_description(f"Analyzing {vtu_file.name}")
                
                analysis = self.analyze_single_grid(vtu_file)
                grid_analyses[vtu_file.stem] = analysis
        
        # Create integrated summary
        integrated_summary = self.create_integrated_summary(grid_analyses)
        
        # Save summaries
        self.save_summaries({
            'individual_grids': grid_analyses,
            'integrated_summary': integrated_summary,
            'analysis_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'n_grids_analyzed': len(vtu_files),
                'grids_processed': [f.name for f in vtu_files]
            }
        })
        
        logger.info("=== Digital Twin Analysis Complete ===")
        return integrated_summary
    
    def create_integrated_summary(self, grid_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Create integrated summary from multiple grid analyses."""
        integrated = {
            'project_name': 'Semurup Geothermal Field',
            'analysis_summary': {},
            'combined_zones': {},
            'integrated_recommendations': [],
            'field_development_strategy': {},
            'economic_assessment': {}
        }
        
        # Combine zone information from all grids
        all_zones = {}
        total_capacity = 0.0
        
        for grid_name, analysis in grid_analyses.items():
            if 'error' in analysis:
                continue
            
            zones = analysis.get('geological_zones', {})
            for zone_name, zone_data in zones.items():
                if zone_name not in all_zones:
                    all_zones[zone_name] = []
                all_zones[zone_name].append(zone_data)
            
            # Sum capacities
            capacity = analysis.get('engineering_summary', {}).get('estimated_total_capacity_mw', 0)
            total_capacity += capacity
        
        # Create combined zone summaries
        for zone_name, zone_list in all_zones.items():
            if zone_list:
                combined_volume = sum(z.get('volume_km3', 0) for z in zone_list)
                combined_area = sum(z.get('area_km2', 0) for z in zone_list)
                avg_confidence = np.mean([z.get('confidence', 0) for z in zone_list])
                
                integrated['combined_zones'][zone_name] = {
                    'total_volume_km3': float(combined_volume),
                    'total_area_km2': float(combined_area),
                    'average_confidence': float(avg_confidence),
                    'n_occurrences': len(zone_list),
                    'description': zone_list[0].get('description', '')
                }
        
        # Overall field assessment
        integrated['analysis_summary'] = {
            'total_estimated_capacity_mw': float(total_capacity),
            'development_status': self.assess_development_status(total_capacity, all_zones),
            'geothermal_potential_rating': self.rate_geothermal_potential(total_capacity, all_zones),
            'confidence_level': self.calculate_overall_confidence(all_zones)
        }
        
        return integrated
    
    def assess_development_status(self, total_capacity: float, zones: Dict) -> str:
        """Assess overall field development status."""
        if total_capacity > 100:
            return "Ready for Commercial Development"
        elif total_capacity > 50:
            return "Pre-commercial Development Phase"
        elif total_capacity > 20:
            return "Advanced Exploration"
        elif total_capacity > 5:
            return "Early Exploration"
        else:
            return "Reconnaissance Phase"
    
    def rate_geothermal_potential(self, total_capacity: float, zones: Dict) -> str:
        """Rate overall geothermal potential."""
        has_reservoir = 'reservoir' in zones
        has_caprock = 'caprock' in zones
        has_heat_source = 'basement' in zones
        
        if total_capacity > 50 and has_reservoir and has_caprock:
            return "Excellent"
        elif total_capacity > 20 and has_reservoir:
            return "Good"
        elif total_capacity > 10:
            return "Fair"
        elif has_reservoir or has_heat_source:
            return "Poor"
        else:
            return "Very Poor"
    
    def calculate_overall_confidence(self, zones: Dict) -> float:
        """Calculate overall confidence in the analysis."""
        if not zones:
            return 0.0
        
        all_confidences = []
        for zone_list in zones.values():
            all_confidences.extend([z.get('confidence', 0) for z in zone_list])
        
        if all_confidences:
            return float(np.mean(all_confidences))
        else:
            return 0.0
    
    def save_summaries(self, summaries: Dict[str, Any]):
        """Save analysis summaries to YAML file."""
        try:
            with open(SUMMARIES_FILE, 'w') as f:
                self.yaml.dump(summaries, f)
            
            logger.info(f"Summaries saved to {SUMMARIES_FILE}")
            
        except Exception as e:
            logger.error(f"Failed to save summaries: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze geothermal digital twin models")
    parser.add_argument("--all", action="store_true", help="Analyze all grid files")
    parser.add_argument("--update", action="store_true", help="Force update existing summaries")
    parser.add_argument("--models", nargs="+", help="Specific model names to analyze")
    
    args = parser.parse_args()
    
    try:
        summariser = GeothermalTwinSummariser()
        
        if args.models:
            # Analyze specific models
            for model_name in args.models:
                vtu_path = GRIDS_DIR / f"{model_name}.vtu"
                if vtu_path.exists():
                    analysis = summariser.analyze_single_grid(vtu_path)
                    print(f"\n=== Analysis: {model_name} ===")
                    print(f"Geothermal Potential: {analysis.get('engineering_summary', {}).get('geothermal_potential', 'Unknown')}")
                    print(f"Estimated Capacity: {analysis.get('engineering_summary', {}).get('estimated_total_capacity_mw', 0):.1f} MW")
                else:
                    logger.error(f"Model {model_name} not found")
        else:
            # Analyze all models
            integrated_summary = summariser.analyze_all_grids(force_update=args.update)
            
            # Print summary
            print(f"\n=== Integrated Geothermal Field Analysis ===")
            analysis = integrated_summary.get('analysis_summary', {})
            print(f"Total Estimated Capacity: {analysis.get('total_estimated_capacity_mw', 0):.1f} MW")
            print(f"Development Status: {analysis.get('development_status', 'Unknown')}")
            print(f"Geothermal Potential: {analysis.get('geothermal_potential_rating', 'Unknown')}")
            print(f"Overall Confidence: {analysis.get('confidence_level', 0):.1%}")
            
            # Print zones
            zones = integrated_summary.get('combined_zones', {})
            if zones:
                print(f"\nIdentified Geological Zones:")
                for zone_name, zone_data in zones.items():
                    print(f"  {zone_name.title()}: {zone_data.get('total_volume_km3', 0):.2f} km³")
            
            print(f"\nDetailed summaries saved to: {SUMMARIES_FILE}")
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()