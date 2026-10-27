from datetime import datetime
import pandas as pd
import pytest

from src.config import EXPECTED_COLUMNS
from src.feature_engineering import FeatureEngineer


def test_parse_duration_string():
    """Tests duration string parsing logic."""
    fe = FeatureEngineer()
    
    # Hours and minutes
    h, m = fe.parse_duration_string("2h 50m")
    assert h == 2 and m == 50

    # Hours only
    h, m = fe.parse_duration_string("19h")
    assert h == 19 and m == 0

    # Minutes only
    h, m = fe.parse_duration_string("45m")
    assert h == 0 and m == 45

    # Invalid or empty
    h, m = fe.parse_duration_string("")
    assert h == 0 and m == 0


def test_transform_single_flight():
    """Tests single flight query transformation into 29 aligned model features."""
    dep = datetime(2026, 11, 20, 10, 30)
    arr = datetime(2026, 11, 20, 14, 45)

    df = FeatureEngineer.transform_single_flight(
        airline="IndiGo",
        source="Delhi",
        destination="Cochin",
        departure_datetime=dep,
        arrival_datetime=arr,
        stops=1
    )

    # Must match exact 29 columns
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 1

    # Check temporal values
    assert df.loc[0, "Journey_day"] == 20
    assert df.loc[0, "Journey_month"] == 11
    assert df.loc[0, "Dep_hour"] == 10
    assert df.loc[0, "Dep_min"] == 30
    assert df.loc[0, "Arrival_hour"] == 14
    assert df.loc[0, "Arrival_min"] == 45
    assert df.loc[0, "Duration_hours"] == 4
    assert df.loc[0, "Duration_mins"] == 15
    assert df.loc[0, "Total_Stops"] == 1

    # Check one-hot indicators
    assert df.loc[0, "Airline_IndiGo"] == 1
    assert df.loc[0, "Airline_Air India"] == 0
    assert df.loc[0, "Source_Delhi"] == 1
    assert df.loc[0, "Source_Mumbai"] == 0
    assert df.loc[0, "Dest_Cochin"] == 1
    assert df.loc[0, "Dest_Delhi"] == 0
