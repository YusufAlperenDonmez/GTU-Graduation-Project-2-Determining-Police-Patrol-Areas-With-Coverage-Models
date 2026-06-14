import os
import csv

def determine_crime_weight(row):
    """
    Assigns a priority weight (1-5) based on FBI UCR Part 1/2 classification 
    and the specific crime description.
    """
    part_type = str(row.get('Part 1-2', '')).strip()
    desc = str(row.get('Crm Cd Desc', '')).upper()
    
    # Part 1 crimes are high-priority/serious offenses
    if part_type == '1':
        # Highest priority for violent Part 1 crimes
        if 'MURDER' in desc or 'ROBBERY' in desc or 'AGGRAVATED ASSAULT' in desc:
            return 5
        # Standard priority for other Part 1 crimes (Burglary, Larceny, etc.)
        return 4 
        
    # Part 2 crimes are lower-priority/less severe offenses
    elif part_type == '2':
        # Lowest priority for non-violent administrative/property damage
        if 'VANDALISM' in desc or 'THEFT OF IDENTITY' in desc:
            return 1
        # Standard priority for other Part 2 crimes
        return 2 
        
    # Fallback if 'Part 1-2' data is missing for some reason
    else:
        if 'MURDER' in desc or 'ROBBERY' in desc or 'AGGRAVATED ASSAULT' in desc:
            return 5
        elif 'BURGLARY' in desc or 'GRAND THEFT' in desc:
            return 3
        elif 'THEFT OF IDENTITY' in desc or 'VANDALISM' in desc:
            return 1
        return 2

def clean_police_data(input_file, output_file):
    # Added AREA, AREA NAME, Rpt Dist No, and Part 1-2 to the retained columns
    columns_to_keep = [
        'DR_NO', 'DATE OCC', 'TIME OCC', 'AREA', 'AREA NAME', 
        'Rpt Dist No', 'Part 1-2', 'Crm Cd', 'Crm Cd Desc', 
        'LOCATION', 'LAT', 'LON'
    ]
    
    output_headers = columns_to_keep + ['crime_weight']

    try:
        with open(input_file, mode='r', encoding='utf-8') as infile, \
             open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=output_headers)
            
            writer.writeheader()
            
            for row in reader:
                # Create a new dictionary containing only the fields we want
                cleaned_row = {col: row[col] for col in columns_to_keep if col in row}
                
                # Pass the whole row to the weight function so it can check Part 1-2 and Description
                cleaned_row['crime_weight'] = determine_crime_weight(cleaned_row)
                
                # Write the formatted row to the new file
                writer.writerow(cleaned_row)
                
        print(f"Success! Cleaned data saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{input_file}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Get the exact directory where clean_dataset.py lives (taha_src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the path: go up one level, then into 'resources'
    resources_dir = os.path.join(script_dir, '..', 'resources')
    
    # Define the exact input and output file paths
    INPUT_CSV = os.path.join(resources_dir, 'Crime_Data_from_2020_to_2024.csv')
    OUTPUT_CSV = os.path.join(resources_dir, 'cleaned_data.csv')
    
    clean_police_data(INPUT_CSV, OUTPUT_CSV)