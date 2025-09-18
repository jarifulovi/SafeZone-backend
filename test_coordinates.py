import h3
import json

# Load severity data
with open('data/severity_data.json', 'r') as f:
    severity_data = json.load(f)

print("Test coordinates with expected crime scores:")
print("=" * 50)

# Get coordinates for the top 5 hexes with highest scores
sorted_hexes = sorted(severity_data.items(), key=lambda x: x[1]['crime_score'], reverse=True)

for i, (hex_id, data) in enumerate(sorted_hexes[:5]):
    lat, lon = h3.cell_to_latlng(hex_id)
    score = data['crime_score']
    print(f"\n{i+1}. High crime area (score: {score}):")
    print(f'   {{"latitude": {lat:.6f}, "longitude": {lon:.6f}}}')
    print(f"   Hex ID: {hex_id}")
