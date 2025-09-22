import json

# Load the original ISTAT cities JSON file
with open('istat-cities.json', 'r', encoding='utf-8') as f:
    cities_data = json.load(f)

# Extract the city names from the "Denominazione in italiano" field
city_names = [entry["Denominazione in italiano"] for entry in cities_data if "Denominazione in italiano" in entry]

# Write the city names to a new JSON file
with open('city_names.json', 'w', encoding='utf-8') as f:
    json.dump(city_names, f, ensure_ascii=False, indent=4)

print(f"Extracted {len(city_names)} city names and saved to city_names.json.")
