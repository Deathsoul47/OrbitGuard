import logging
from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta
import pandas as pd
from sgp4.api import Satrec

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)


def extract_epoch_datetime(satrec: Satrec) -> datetime:
    """
    Converts the Satrec epoch (year and days) into a standard Python datetime object.

    Args:
        satrec (Satrec): An initialized sgp4 Satrec object.

    Returns:
        datetime: The exact epoch time as a datetime object.
    """
    year = satrec.epochyr
    # SGP4 year is usually 2 digits. 
    # Convention: 57-99 -> 1900s, 00-56 -> 2000s
    full_year = 1900 + year if year >= 57 else 2000 + year
    
    # Start of the year
    epoch_start = datetime(year=full_year, month=1, day=1)
    # Add the fractional days (subtracting 1 because Jan 1 is day 1)
    return epoch_start + timedelta(days=satrec.epochdays - 1)


def parse_tle_file(filepath: str) -> pd.DataFrame:
    """
    Parses a raw TLE (Two-Line Element) text file into a structured pandas DataFrame.
    Uses the SGP4 orbital propagator to extract true Earth-centered inertial (ECI) 
    spatial coordinates at the epoch time.

    Args:
        filepath (str): Absolute or relative path to the TLE text file.

    Returns:
        pd.DataFrame: A structured DataFrame containing satellite IDs, epoch times, 
                      and spatial coordinates (x, y, z) in kilometers.
                      
    Raises:
        FileNotFoundError: If the specified TLE file does not exist.
        ValueError: If the file is empty or improperly formatted.
    """
    logger.info(f"Initiating TLE parsing for file: {filepath}")
    records: List[Dict[str, Union[str, float, datetime]]] = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as e:
        logger.error(f"TLE file not found: {filepath}")
        raise e
    except Exception as e:
        logger.error(f"Failed to read TLE file: {filepath}. Error: {str(e)}")
        raise
        
    if not lines:
        raise ValueError(f"The provided TLE file is empty: {filepath}")

    # Iterate through lines to find consecutive TLE Line 1 and Line 2
    success_count = 0
    error_count = 0
    
    for i in range(len(lines)):
        # Identify Line 1 and ensure Line 2 follows
        if lines[i].startswith('1 ') and i + 1 < len(lines) and lines[i+1].startswith('2 '):
            l1 = lines[i]
            l2 = lines[i+1]
            
            # Extract satellite name if it exists on the line prior
            sat_name = lines[i-1] if i > 0 and not lines[i-1].startswith('2 ') else "UNKNOWN"
            sat_id = l1[2:7].strip()
            
            try:
                # Initialize SGP4 satellite record
                sat = Satrec.twoline2rv(l1, l2)
                
                # Propagate at exactly t=0 (the epoch time) to get initial coordinates
                e, r, v = sat.sgp4_tsince(0.0)
                
                if e != 0:
                    logger.debug(f"SGP4 propagation error code {e} for satellite {sat_id}. Skipping.")
                    error_count += 1
                    continue
                    
                epoch_dt = extract_epoch_datetime(sat)
                
                records.append({
                    'sat_name': sat_name,
                    'sat_id': sat_id,
                    'epoch': epoch_dt,
                    'x_km': r[0],
                    'y_km': r[1],
                    'z_km': r[2],
                    'vx_kms': v[0],
                    'vy_kms': v[1],
                    'vz_kms': v[2],
                    'bstar': sat.bstar,
                    'inclination_rad': sat.inclo,
                    'eccentricity': sat.ecco
                })
                success_count += 1
                
            except Exception as e:
                logger.debug(f"Failed to parse TLE for sat_id {sat_id}: {str(e)}")
                error_count += 1

    logger.info(f"Parsing complete. Successfully extracted {success_count} records. Failed/Skipped: {error_count}.")
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    if not df.empty:
        # Sort chronologically by epoch
        df.sort_values(by=['sat_id', 'epoch'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    return df
