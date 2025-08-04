#!/usr/bin/env python3
"""
Test script for Digital Twin Summariser Module

Tests the geological interpretation and engineering analysis capabilities
of the twin summariser with processed VTU grid files.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from twin_summariser import GeothermalTwinSummariser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_grid_loading():
    """Test VTU grid file loading."""
    print("=== Testing VTU Grid Loading ===\n")
    
    grids_dir = Path("digital_twin/grids")
    if not grids_dir.exists():
        print("⚠️  digital_twin/grids/ directory not found")
        return False
    
    vtu_files = list(grids_dir.glob("*.vtu"))
    if not vtu_files:
        print("⚠️  No VTU grid files found")
        print("   Run raw data ingestion first: python src/ingest_raw.py --models")
        return False
    
    print(f"📊 Found {len(vtu_files)} VTU grid files:")
    for vtu_file in vtu_files:
        size_mb = vtu_file.stat().st_size / (1024 * 1024)
        print(f"  - {vtu_file.name} ({size_mb:.1f} MB)")
    
    try:
        summariser = GeothermalTwinSummariser()
        
        # Test loading first grid
        test_file = vtu_files[0]
        print(f"\n🧪 Testing with: {test_file.name}")
        
        grid = summariser.load_vtu_grid(test_file)
        if grid is not None:
            print(f"  ✅ Grid loaded successfully")
            print(f"    Points: {grid.n_points:,}")
            print(f"    Cells: {grid.n_cells:,}")
            print(f"    Properties: {list(grid.array_names)}")
            print(f"    Bounds: X({grid.bounds[0]:.0f}-{grid.bounds[1]:.0f}), Y({grid.bounds[2]:.0f}-{grid.bounds[3]:.0f}), Z({grid.bounds[4]:.0f}-{grid.bounds[5]:.0f})")
            return True
        else:
            print("  ❌ Failed to load grid")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_geological_analysis():
    """Test geological zone identification."""
    print("\n=== Testing Geological Analysis ===\n")
    
    try:
        summariser = GeothermalTwinSummariser()
        
        # Find VTU files
        vtu_files = list(Path("digital_twin/grids").glob("*.vtu"))
        if not vtu_files:
            print("⚠️  No VTU files available for analysis")
            return False
        
        test_file = vtu_files[0]
        print(f"🧪 Analyzing: {test_file.name}")
        
        # Run single grid analysis
        analysis = summariser.analyze_single_grid(test_file)
        
        if 'error' in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return False
        
        print("✅ Geological analysis completed!")
        
        # Display results
        zones = analysis.get('geological_zones', {})
        if zones:
            print(f"\n📍 Identified Geological Zones ({len(zones)}):")
            for zone_name, zone_data in zones.items():
                confidence = zone_data.get('confidence', 0)
                volume = zone_data.get('volume_km3', 0)
                print(f"  {zone_name.title()}: {volume:.3f} km³ (confidence: {confidence:.1%})")
                
                if 'depth_range_m' in zone_data:
                    depth_range = zone_data['depth_range_m']
                    print(f"    Depth: {depth_range[0]:.0f}-{depth_range[1]:.0f} m")
                
                if 'temperature_range_c' in zone_data:
                    temp_range = zone_data['temperature_range_c']
                    print(f"    Temperature: {temp_range[0]:.0f}-{temp_range[1]:.0f} °C")
        else:
            print("  No distinct geological zones identified")
        
        # Engineering summary
        eng_summary = analysis.get('engineering_summary', {})
        if eng_summary:
            print(f"\n⚡ Engineering Assessment:")
            print(f"  Geothermal Potential: {eng_summary.get('geothermal_potential', 'Unknown')}")
            print(f"  Development Phase: {eng_summary.get('development_phase', 'Unknown')}")
            print(f"  Estimated Capacity: {eng_summary.get('estimated_total_capacity_mw', 0):.1f} MW")
            
            findings = eng_summary.get('key_findings', [])
            if findings:
                print(f"  Key Findings:")
                for finding in findings[:3]:  # Show top 3
                    print(f"    • {finding}")
        
        # Drilling recommendations
        recommendations = analysis.get('drilling_recommendations', [])
        if recommendations:
            print(f"\n🎯 Drilling Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"  {i}. {rec['target_type']}")
                print(f"     Zone: {rec['target_zone']}")
                print(f"     Depth: {rec['depth_range_m'][0]:.0f}-{rec['depth_range_m'][1]:.0f} m")
                print(f"     Confidence: {rec['confidence']:.1%}")
                if rec.get('estimated_capacity_mw', 0) > 0:
                    print(f"     Capacity: {rec['estimated_capacity_mw']:.1f} MW")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_analysis():
    """Test integrated analysis of all grids."""
    print("\n=== Testing Integrated Analysis ===\n")
    
    try:
        summariser = GeothermalTwinSummariser()
        
        # Run integrated analysis
        print("🔄 Running integrated field analysis...")
        integrated_summary = summariser.analyze_all_grids()
        
        if 'error' in integrated_summary:
            print(f"❌ Integrated analysis failed: {integrated_summary['error']}")
            return False
        
        print("✅ Integrated analysis completed!")
        
        # Display field-level results
        analysis = integrated_summary.get('analysis_summary', {})
        if analysis:
            print(f"\n🏭 Semurup Geothermal Field Assessment:")
            print(f"  Total Capacity: {analysis.get('total_estimated_capacity_mw', 0):.1f} MW")
            print(f"  Development Status: {analysis.get('development_status', 'Unknown')}")
            print(f"  Geothermal Potential: {analysis.get('geothermal_potential_rating', 'Unknown')}")
            print(f"  Overall Confidence: {analysis.get('confidence_level', 0):.1%}")
        
        # Combined zones
        zones = integrated_summary.get('combined_zones', {})
        if zones:
            print(f"\n🗺️  Field-Wide Geological Zones:")
            for zone_name, zone_data in zones.items():
                volume = zone_data.get('total_volume_km3', 0)
                area = zone_data.get('total_area_km2', 0)
                confidence = zone_data.get('average_confidence', 0)
                occurrences = zone_data.get('n_occurrences', 0)
                
                print(f"  {zone_name.title()}:")
                print(f"    Volume: {volume:.2f} km³")
                print(f"    Area: {area:.2f} km²")
                print(f"    Confidence: {confidence:.1%}")
                print(f"    Occurrences: {occurrences}")
        
        # Check YAML output
        yaml_file = Path("digital_twin/cache/twin_summaries.yaml")
        if yaml_file.exists():
            print(f"\n📄 Detailed analysis saved to: {yaml_file}")
            size_kb = yaml_file.stat().st_size / 1024
            print(f"     File size: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yaml_output():
    """Test YAML summary output."""
    print("\n=== Testing YAML Output ===\n")
    
    yaml_file = Path("digital_twin/cache/twin_summaries.yaml")
    if not yaml_file.exists():
        print("⚠️  No YAML summary file found - run integrated analysis first")
        return False
    
    try:
        # Read and validate YAML
        from ruamel.yaml import YAML
        yaml = YAML(typ='safe')
        
        with open(yaml_file, 'r') as f:
            data = yaml.load(f)
        
        print(f"✅ YAML file loaded successfully")
        print(f"  Size: {yaml_file.stat().st_size / 1024:.1f} KB")
        
        # Check structure
        expected_keys = ['individual_grids', 'integrated_summary', 'analysis_metadata']
        for key in expected_keys:
            if key in data:
                print(f"  ✅ {key}: Present")
            else:
                print(f"  ❌ {key}: Missing")
        
        # Check metadata
        metadata = data.get('analysis_metadata', {})
        if metadata:
            print(f"\n📊 Analysis Metadata:")
            print(f"  Timestamp: {metadata.get('analysis_timestamp', 'Unknown')}")
            print(f"  Grids analyzed: {metadata.get('n_grids_analyzed', 0)}")
            
            grids = metadata.get('grids_processed', [])
            if grids:
                print(f"  Processed files:")
                for grid_name in grids:
                    print(f"    • {grid_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ YAML test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Geothermal Digital Twin - Summariser Tests\n")
    
    # Check if we're in the right directory
    if not Path("digital_twin").exists():
        print("❌ Run this script from the geo_twin_ai root directory") 
        sys.exit(1)
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    if test_grid_loading():
        tests_passed += 1
    
    if test_geological_analysis():
        tests_passed += 1
    
    if test_integrated_analysis():
        tests_passed += 1
    
    if test_yaml_output():
        tests_passed += 1
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        print("\nReady for production use:")
        print("   python src/twin_summariser.py --all  # Analyze all grids")
        print("   python src/file_watcher.py           # Auto-run when data changes")
    else:
        print("⚠️  Some tests failed - check the logs above")
        if tests_passed == 0:
            print("💡 Make sure to run raw data ingestion first:")
            print("   python src/ingest_raw.py --models")

if __name__ == "__main__":
    main()