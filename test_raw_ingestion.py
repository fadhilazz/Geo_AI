#!/usr/bin/env python3
"""
Test script for Raw Data Ingestion Module

Quick test to verify the raw data processing pipeline works correctly.
Processes 3D models and geochemical data from your Semurup dataset.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest_raw import RawDataIngester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_3d_models():
    """Test 3D model file processing."""
    print("=== Testing 3D Model Processing ===\n")
    
    model_dir = Path("data/3d_models")
    if not model_dir.exists():
        print("⚠️  data/3d_models/ directory not found")
        return False
    
    dat_files = list(model_dir.glob("*.dat"))
    if not dat_files:
        print("⚠️  No .dat files found in data/3d_models/")
        return False
    
    print(f"📊 Found {len(dat_files)} 3D model files:")
    for dat_file in dat_files:
        print(f"  - {dat_file.name} ({dat_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    try:
        ingester = RawDataIngester()
        
        # Test processing first model file
        test_file = dat_files[0]
        print(f"\n🧪 Testing with: {test_file.name}")
        
        success = ingester.process_3d_model_file(test_file, force=True)
        
        if success:
            print("✅ 3D model processing test passed!")
            
            # Check output files
            vtu_file = Path("digital_twin/grids") / f"{test_file.stem}.vtu"
            metadata_file = Path("digital_twin/cache") / f"{test_file.stem}_metadata.json"
            
            if vtu_file.exists():
                print(f"  ✅ VTU grid created: {vtu_file} ({vtu_file.stat().st_size / 1024:.1f} KB)")
            else:
                print("  ❌ VTU grid file not found")
                
            if metadata_file.exists():
                print(f"  ✅ Metadata created: {metadata_file}")
                # Read and display metadata
                import json
                with open(metadata_file) as f:
                    metadata = json.load(f)
                print(f"    Grid: {metadata['grid_dimensions']} cells")
                print(f"    Property: {metadata['property_name']} ({metadata['property_unit']})")
                print(f"    Value range: {metadata['value_stats']['min']:.2e} to {metadata['value_stats']['max']:.2e}")
            else:
                print("  ❌ Metadata file not found")
                
            return True
        else:
            print("❌ 3D model processing test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_geochemical_data():
    """Test geochemical data processing."""
    print("\n=== Testing Geochemical Data Processing ===\n")
    
    geochem_dir = Path("data/geochem")
    if not geochem_dir.exists():
        print("⚠️  data/geochem/ directory not found")
        return False
    
    geochem_files = list(geochem_dir.glob("*.xlsx")) + list(geochem_dir.glob("*.csv")) + list(geochem_dir.glob("*.txt"))
    if not geochem_files:
        print("⚠️  No geochemical files found")
        return False
    
    print(f"🧪 Found {len(geochem_files)} geochemical files:")
    for file in geochem_files:
        print(f"  - {file.name} ({file.suffix})")
    
    try:
        ingester = RawDataIngester()
        
        # Test each file type
        success_count = 0
        for file in geochem_files:
            print(f"\n🧪 Testing: {file.name}")
            success = ingester.process_geochemical_file(file, force=True)
            if success:
                success_count += 1
                print(f"  ✅ Processed successfully")
                
                # Check for output files
                summary_file = Path("digital_twin/cache") / f"{file.stem}_geochem_summary.json"
                if summary_file.exists():
                    import json
                    with open(summary_file) as f:
                        summary = json.load(f)
                    print(f"    Type: {summary['file_type']}")
                    if 'n_rows' in summary:
                        print(f"    Data: {summary['n_rows']} rows, {summary['n_columns']} columns")
                    if 'coordinate_columns' in summary:
                        print(f"    Coordinates: {summary['coordinate_columns']}")
            else:
                print(f"  ❌ Processing failed")
        
        if success_count == len(geochem_files):
            print(f"\n✅ All {len(geochem_files)} geochemical files processed successfully!")
            return True
        else:
            print(f"\n⚠️  {success_count}/{len(geochem_files)} files processed successfully")
            return success_count > 0
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_verification():
    """Verify that output files are created correctly."""
    print("\n=== Verifying Output Files ===\n")
    
    grids_dir = Path("digital_twin/grids")
    cache_dir = Path("digital_twin/cache")
    
    # Check directories exist
    if not grids_dir.exists():
        print("❌ Grids directory not found")
        return False
    if not cache_dir.exists():
        print("❌ Cache directory not found")
        return False
    
    # Check VTU files
    vtu_files = list(grids_dir.glob("*.vtu"))
    print(f"📊 VTU grid files: {len(vtu_files)}")
    for vtu_file in vtu_files:
        print(f"  - {vtu_file.name} ({vtu_file.stat().st_size / 1024:.1f} KB)")
    
    # Check metadata files
    json_files = list(cache_dir.glob("*_metadata.json"))
    print(f"📋 Metadata files: {len(json_files)}")
    
    summary_files = list(cache_dir.glob("*_summary.json"))
    print(f"📋 Summary files: {len(summary_files)}")
    
    # Check processed data files
    csv_files = list(cache_dir.glob("*_processed.csv"))
    print(f"📊 Processed CSV files: {len(csv_files)}")
    
    total_files = len(vtu_files) + len(json_files) + len(summary_files) + len(csv_files)
    if total_files > 0:
        print(f"\n✅ Output verification passed! {total_files} files created")
        return True
    else:
        print("\n❌ No output files found")
        return False

def main():
    """Run all tests."""
    print("🧪 Geothermal Digital Twin - Raw Data Ingestion Tests\n")
    
    # Check if we're in the right directory
    if not Path("data").exists():
        print("❌ Run this script from the geo_twin_ai root directory")
        sys.exit(1)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_3d_models():
        tests_passed += 1
    
    if test_geochemical_data():
        tests_passed += 1
    
    if test_output_verification():
        tests_passed += 1
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        print("\nReady for full pipeline:")
        print("   python src/file_watcher.py  # Auto-detect and process changes")
        print("   python src/ingest_raw.py --all  # Process all raw data")
    else:
        print("⚠️  Some tests failed - check the logs above")

if __name__ == "__main__":
    main()