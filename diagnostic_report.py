import h3
import json
import pandas as pd
import random
import numpy as np
from services.safety_service import SeverityService

print("=== SAFEZONE DIAGNOSTIC REPORT ===")
print()

# Check crime data
df = pd.read_csv('data/crime_data.tsv', sep='\t')
print(f"Crime data records: {len(df)}")
print(f"Sample crime hex (from data): {df.iloc[0]['h3_index']}")
print(f"Crime hex resolution: {h3.get_resolution(df.iloc[0]['h3_index'])}")
print()

# Check severity data
with open('data/severity_data.json') as f:
    severity_data = json.load(f)

print(f"Severity data hexes: {len(severity_data)}")
first_severity_hex = list(severity_data.keys())[0]
print(f"Sample severity hex: {first_severity_hex}")
print(f"Severity hex resolution: {h3.get_resolution(first_severity_hex)}")
print()

# Test resolution mismatch
test_lat, test_lon = 23.8103, 90.4125  # Dhaka center
test_hex_res6 = h3.latlng_to_cell(test_lat, test_lon, 6)
test_hex_res8 = h3.latlng_to_cell(test_lat, test_lon, 8)

print(f"Test location (Dhaka): {test_lat}, {test_lon}")
print(f"Hex at resolution 6: {test_hex_res6}")
print(f"Hex at resolution 8: {test_hex_res8}")
print(f"Resolution 6 hex in severity data: {test_hex_res6 in severity_data}")
print(f"Resolution 8 hex in severity data: {test_hex_res8 in severity_data}")
print()

# Real coverage analysis - test 100 random locations across Bangladesh
print("=== BANGLADESH MAP COVERAGE ANALYSIS ===")
print("Testing 100 random locations across Bangladesh...")

# Bangladesh bounds (excluding edge areas)
BD_BOUNDS = {
    "min_lat": 20.8,  # Slightly inside edges
    "max_lat": 26.4,
    "min_lon": 88.2,
    "max_lon": 92.5
}

# Generate 100 random test locations
test_locations = []
for _ in range(100):
    lat = random.uniform(BD_BOUNDS["min_lat"], BD_BOUNDS["max_lat"])
    lon = random.uniform(BD_BOUNDS["min_lon"], BD_BOUNDS["max_lon"])
    test_locations.append((lat, lon))

print(f"Generated 100 test locations within bounds:")
print(f"  Latitude range: {BD_BOUNDS['min_lat']} to {BD_BOUNDS['max_lat']}")
print(f"  Longitude range: {BD_BOUNDS['min_lon']} to {BD_BOUNDS['max_lon']}")
print()

# Test each location using the actual API
locations_with_scores = 0
locations_with_zero_scores = 0
sample_results = []

print("Testing locations...")
for i, (lat, lon) in enumerate(test_locations):
    try:
        # Use the actual API service to get scores
        result = SeverityService.calculate_stats(lat, lon)
        score = result.crime_score

        if score > 0:
            locations_with_scores += 1
            sample_results.append((lat, lon, score, "HAS_SCORE"))
        else:
            locations_with_zero_scores += 1
            sample_results.append((lat, lon, score, "NO_SCORE"))

        # Show progress every 20 locations
        if (i + 1) % 20 == 0:
            print(f"  Tested {i + 1}/100 locations...")

    except Exception as e:
        locations_with_zero_scores += 1
        sample_results.append((lat, lon, 0.0, "ERROR"))
        print(f"  Error testing location {lat:.6f}, {lon:.6f}: {e}")

print("\n=== COVERAGE RESULTS ===")
total_tested = len(test_locations)
coverage_percentage = (locations_with_scores / total_tested) * 100

print(f"Total locations tested: {total_tested}")
print(f"Locations with scores > 0: {locations_with_scores}")
print(f"Locations with zero scores: {locations_with_zero_scores}")
print(f"Bangladesh map coverage: {coverage_percentage:.1f}%")
print()

# Show sample results
print("Sample results (first 10 locations):")
for i, (lat, lon, score, status) in enumerate(sample_results[:10]):
    print(f"  {i+1}. ({lat:.6f}, {lon:.6f}) -> Score: {score:.2f} [{status}]")

if locations_with_scores > 0:
    scores_only = [r[2] for r in sample_results if r[2] > 0]
    print(f"\nScore statistics for locations with scores > 0:")
    print(f"  Minimum score: {min(scores_only):.2f}")
    print(f"  Maximum score: {max(scores_only):.2f}")
    print(f"  Average score: {sum(scores_only)/len(scores_only):.2f}")

print()
if coverage_percentage < 50:
    print("⚠️  LOW COVERAGE: Less than 50% of Bangladesh has crime data")
elif coverage_percentage < 80:
    print("⚠️  MEDIUM COVERAGE: 50-80% of Bangladesh has crime data")
else:
    print("✅ GOOD COVERAGE: 80%+ of Bangladesh has crime data")

print()
print("=== ISSUE ANALYSIS ===")
current_resolution = h3.get_resolution(first_severity_hex)
if current_resolution != 6:
    print(f"ERROR: Severity data is still at resolution {current_resolution}, not 6!")
    print("Need to regenerate crime data with resolution 6 or fix the conversion logic.")
else:
    print("Severity data is correctly at resolution 6.")
    print(f"Map coverage: {coverage_percentage:.1f}% of Bangladesh returns scores > 0")
