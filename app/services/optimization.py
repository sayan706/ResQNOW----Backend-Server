import os
import json
import math
import time
import concurrent.futures
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pulp
from geopy.geocoders import Nominatim
import requests as req_lib
from app.services.cache import cache_service

# Constants
COVERAGE_THRESHOLD_MINUTES = 15
FALLBACK_SPEED_KMH = 30
TIME_PERIOD_ORDER = ['Morning', 'Afternoon', 'Evening', 'Night']
TIME_RANGES = {
    'Morning':   '06:00 – 11:59',
    'Afternoon': '12:00 – 17:59',
    'Evening':   '18:00 – 21:59',
    'Night':     '22:00 – 05:59',
}
ZONE_PRIORITY = {'Red': 0, 'Orange': 1, 'Green': 2}

MAP_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

def get_distance_matrix_cached(origins, destinations):
    """
    Fetches travel times with Redis caching to avoid redundant API calls.
    Origins/Destinations are lists of (lat, lon).
    """
    matrix = np.zeros((len(origins), len(destinations)))
    to_query = [] # List of (i, j, origin, destination)
    
    for i, o in enumerate(origins):
        for j, d in enumerate(destinations):
            cache_key = f"dist:{o[0]:.5f}:{o[1]:.5f}:{d[0]:.5f}:{d[1]:.5f}"
            cached_val = cache_service.get(cache_key)
            if cached_val is not None:
                matrix[i][j] = float(cached_val)
            else:
                to_query.append((i, j, o, d))
                
    if not to_query:
        return matrix

    # Logic for Google Maps API or Haversine fallback
    if MAP_API_KEY and MAP_API_KEY != "YOUR_GOOGLE_MAPS_API_KEY_HERE":
        # Realistic implementation would batch these, but for simplicity here:
        for i, j, o, d in to_query:
            # Simulated API call result
            dist = haversine_km(o[0], o[1], d[0], d[1])
            t_time = dist * 1.4 / FALLBACK_SPEED_KMH * 60
            
            cache_key = f"dist:{o[0]:.5f}:{o[1]:.5f}:{d[0]:.5f}:{d[1]:.5f}"
            cache_service.set(cache_key, t_time, expire=86400) # cache for 24h
            matrix[i][j] = t_time
    else:
        # Fallback to Haversine
        for i, j, o, d in to_query:
            dist = haversine_km(o[0], o[1], d[0], d[1])
            t_time = dist * 1.4 / FALLBACK_SPEED_KMH * 60
            matrix[i][j] = t_time
            # We still cache the fallback to avoid re-calculating Haversine frequently
            cache_key = f"dist:{o[0]:.5f}:{o[1]:.5f}:{d[0]:.5f}:{d[1]:.5f}"
            cache_service.set(cache_key, t_time, expire=86400)

    return matrix

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

# --- Existing logic remains similar but uses the new cached matrix function ---

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    for col in ['LATITUDE', 'LONGITUDE']:
        if col not in df.columns: raise ValueError(f"Missing required column: {col}")
    
    fatal_col = [c for c in df.columns if 'FATAL' in c.upper()]
    grievous_col = [c for c in df.columns if 'GRIEV' in c.upper()]
    minor_col = [c for c in df.columns if 'MINOR' in c.upper()]
    
    fatal_col = fatal_col[0] if fatal_col else 'TEMP_FATAL'
    grievous_col = grievous_col[0] if grievous_col else 'TEMP_GRIEVOUS'
    minor_col = minor_col[0] if minor_col else 'TEMP_MINOR'

    for c in [fatal_col, grievous_col, minor_col]:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

    time_col = [c for c in df.columns if 'TIME' in c.upper()]
    time_col = time_col[0] if time_col else None
    if time_col:
        df['HOUR'] = df[time_col].apply(lambda x: pd.to_datetime(str(x), errors='coerce').hour)
        df['TIME_PERIOD'] = df['HOUR'].apply(lambda h: 'Morning' if 6<=h<12 else 'Afternoon' if 12<=h<18 else 'Evening' if 18<=h<22 else 'Night')
    else:
        df['TIME_PERIOD'] = 'Morning'

    df['SEVERITY_SCORE'] = 4 * df[fatal_col] + 2 * df[grievous_col] + 1 * df[minor_col]
    return df

def auto_tune_dbscan(X_scaled):
    best_score = -9999
    best_params = (0.5, 5)
    best_labels = None
    for eps in np.arange(0.2, 1.0, 0.2):
        for ms in [3, 5]:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_scaled)
            if len(set(labels) - {-1}) < 1: continue
            score = 0
            if len(set(labels[labels!=-1])) >= 2:
                try: score = silhouette_score(X_scaled[labels!=-1], labels[labels!=-1])
                except: score = -1
            if score > best_score:
                best_score = score; best_params = (eps, ms); best_labels = labels
    return best_labels

def generate_boundary_points(clustered_df, cluster_id, zone_color):
    pts = clustered_df[clustered_df['CLUSTER'] == cluster_id]
    clat, clon = pts['LATITUDE'].mean(), pts['LONGITUDE'].mean()
    lat_off = max(pts['LATITUDE'].std() * 2, 0.002)
    lon_off = max(pts['LONGITUDE'].std() * 2, 0.002)
    diag = 0.707
    dirs = {'N':(clat+lat_off, clon), 'S':(clat-lat_off, clon), 'E':(clat, clon+lon_off), 'W':(clat, clon-lon_off),
            'NE':(clat+lat_off*diag, clon+lon_off*diag), 'NW':(clat+lat_off*diag, clon-lon_off*diag),
            'SE':(clat-lat_off*diag, clon+lon_off*diag), 'SW':(clat-lat_off*diag, clon-lon_off*diag)}
    return [{'CLUSTER': cluster_id, 'ZONE': zone_color, 'DIRECTION': d, 'LATITUDE': la, 'LONGITUDE': lo} for d, (la, lo) in dirs.items()]

def solve_lscp(demand_pts, cand_locs, tt_matrix, threshold):
    nc, nd = len(cand_locs), len(demand_pts)
    cov = (tt_matrix <= threshold).astype(int)
    prob = pulp.LpProblem("LSCP", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{j}", cat='Binary') for j in range(nc)]
    prob += pulp.lpSum(x)
    for i in range(nd):
        covers = [j for j in range(nc) if cov[j][i]]
        if covers: prob += pulp.lpSum(x[j] for j in covers) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] == 'Optimal':
        sel = [j for j in range(nc) if x[j].varValue > 0.5]
        return {'selected_locations': [cand_locs[j] for j in sel], 'ambulance_count': len(sel)}
    return {'ambulance_count': 0, 'status': 'Infeasible'}

def process_ambulance_optimization(csv_path):
    df = load_and_clean_data(csv_path)
    X_scaled = StandardScaler().fit_transform(df[['LATITUDE', 'LONGITUDE']])
    df['CLUSTER'] = auto_tune_dbscan(X_scaled)
    cdf = df[df['CLUSTER'] != -1].copy()
    cc = cdf.groupby('CLUSTER').agg(CENTER_LATITUDE=('LATITUDE', 'mean'), CENTER_LONGITUDE=('LONGITUDE', 'mean')).reset_index()

    all_results = {}
    for period in TIME_PERIOD_ORDER:
        period_df = df[df['TIME_PERIOD'] == period]
        if period_df.empty: continue
        pc = period_df[period_df['CLUSTER'] != -1].copy()
        if pc.empty: continue
        
        s = pc.groupby('CLUSTER').agg(AVG_SEVERITY=('SEVERITY_SCORE', 'mean'), TOTAL_SEVERITY=('SEVERITY_SCORE', 'sum'), ACCIDENT_COUNT=('CLUSTER', 'count')).reset_index()
        s['RISK_SCORE'] = 0.4 * s['ACCIDENT_COUNT'] + 0.3 * s['AVG_SEVERITY'] + 0.3 * s['TOTAL_SEVERITY']
        s = s.sort_values('RISK_SCORE', ascending=False)
        s['ZONE'] = 'Green'
        if len(s) > 0: s.iloc[0, s.columns.get_loc('ZONE')] = 'Red'
        if len(s) > 1: s.iloc[1, s.columns.get_loc('ZONE')] = 'Orange'
        
        pz = s.merge(cc, on='CLUSTER')
        bpts = []
        for _, r in pz.iterrows():
            bpts.extend(generate_boundary_points(cdf, int(r['CLUSTER']), r['ZONE']))
        
        dem = [(p['LATITUDE'], p['LONGITUDE']) for p in bpts]
        origins = [(r['CENTER_LATITUDE'], r['CENTER_LONGITUDE']) for _, r in cc.iterrows()]
        
        # USE THE NEW CACHED MATRIX FUNCTION
        tt = get_distance_matrix_cached(origins, dem)
        
        lscp = solve_lscp(dem, origins, tt, COVERAGE_THRESHOLD_MINUTES)
        
        all_results[period] = {
            "metadata": {"time_period_name": period, "coverage_threshold_minutes": COVERAGE_THRESHOLD_MINUTES, "total_zones": len(pz)},
            "zones": [{"zone_id": f"zone_{int(r['CLUSTER'])}", "risk_level": r['ZONE'], "centroid": {"lat": r['CENTER_LATITUDE'], "lng": r['CENTER_LONGITUDE']}} for _, r in pz.iterrows()],
            "selected_ambulances": [{"ambulance_id": f"amb_{idx}", "lat": loc[0], "lng": loc[1]} for idx, loc in enumerate(lscp.get('selected_locations', []))]
        }

    return {"time_periods": all_results}
