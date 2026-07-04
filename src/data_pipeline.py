import pandas as pd
import numpy as np
from datetime import datetime

# Exact feature list from your train_test_split cell
EXPECTED_COLUMNS = [
    'Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min', 
    'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins', 
    'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways', 
    'Airline_Jet Airways Business', 'Airline_Multiple carriers', 
    'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet', 'Airline_Trujet', 
    'Airline_Vistara', 'Airline_Vistara Premium economy', 'Source_Chennai', 
    'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai', 'Dest_Cochin', 
    'Dest_Delhi', 'Dest_Hyderabad', 'Dest_Kolkata', 'Dest_New Delhi'
]

def preprocess_flight_data(
    airline: str, 
    source: str, 
    destination: str, 
    departure_datetime: datetime, 
    arrival_datetime: datetime, 
    stops: int
) -> pd.DataFrame:
    """
    Transforms UI inputs into a DataFrame matching the model's exact expected columns.
    """
    input_data = {col: 0 for col in EXPECTED_COLUMNS}
    
    input_data['Journey_day'] = departure_datetime.day
    input_data['Journey_month'] = departure_datetime.month
    input_data['Dep_hour'] = departure_datetime.hour
    input_data['Dep_min'] = departure_datetime.minute
    input_data['Arrival_hour'] = arrival_datetime.hour
    input_data['Arrival_min'] = arrival_datetime.minute
    
    duration_delta = arrival_datetime - departure_datetime
    total_seconds = duration_delta.total_seconds()
    if total_seconds < 0:
        total_seconds += 24 * 3600  
        
    input_data['Duration_hours'] = int(total_seconds // 3600)
    input_data['Duration_mins'] = int((total_seconds % 3600) // 60)
    
    input_data['Total_Stops'] = stops
    
    airline_key = f"Airline_{airline}"
    if airline_key in input_data:
        input_data[airline_key] = 1
        
    source_key = f"Source_{source}"
    if source_key in input_data:
        input_data[source_key] = 1
        
    dest_key = f"Dest_{destination}"
    if dest_key in input_data:
        input_data[dest_key] = 1
        
    return pd.DataFrame([input_data], columns=EXPECTED_COLUMNS)