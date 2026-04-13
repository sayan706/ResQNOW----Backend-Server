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
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371 # Earth radius in km
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

def travel_time_haversine(lat1, lon1, lat2, lon2, speed=FALLBACK_SPEED_KMH):
    """Estimated travel time (minutes) via Haversine + road detour factor 1.4x."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1.4 / speed * 60

def get_travel_time(lat1, lon1, lat2, lon2):
    """
    Get travel time (minutes) between points.
    Prioritizes Google Maps API, falls back to Haversine.
    """
    # Try Cache first
    cache_key = f"dist:{lat1:.5f}:{lon1:.5f}:{lat2:.5f}:{lon2:.5f}"
    cached_val = cache_service.get(cache_key)
    if cached_val is not None:
        return float(cached_val)

    # Try Google Maps
    if MAP_API_KEY and MAP_API_KEY != "YOUR_GOOGLE_MAPS_API_KEY_HERE":
        url = (f"https://maps.googleapis.com/maps/api/distancematrix/json"
               f"?origins={lat1},{lon1}&destinations={lat2},{lon2}&key={MAP_API_KEY}&mode=driving")
        try:
            res = req_lib.get(url, timeout=10).json()
            if res['status'] == 'OK' and res['rows'][0]['elements'][0]['status'] == 'OK':
                t_time = res['rows'][0]['elements'][0]['duration']['value'] / 60
                cache_service.set(cache_key, t_time, expire=86400)
                return t_time
        except Exception:
            pass

    # Fallback to Haversine
    t_time = travel_time_haversine(lat1, lon1, lat2, lon2)
    cache_service.set(cache_key, t_time, expire=86400)
    return t_time

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

def reconstruct_state_from_dict(data: dict):
    """
    Reconstructs the internal system state from the standardized JSON structure.
    Used for loading the source of truth before real-time updates.
    """
    all_results = {}
    for period, p_data in data.get("time_periods", {}).items():
        sel_ambs = p_data.get("selected_ambulances", [])
        
        # Reconstruct Area summary
        zones_data = []
        for z in p_data.get("zones", []):
            zones_data.append({
                "CLUSTER": z["cluster_label"],
                "ZONE": z["risk_level"],
                "CENTER_LATITUDE": z["centroid"]["lat"],
                "CENTER_LONGITUDE": z["centroid"]["lng"],
                "ACCIDENT_COUNT": z.get("total_original_points", 0)
            })
            
        # Reconstruct Boundary Points
        bpts = []
        rev_map = {'north': 'N', 'south': 'S', 'east': 'E', 'west': 'W',
                   'north_east': 'NE', 'north_west': 'NW', 'south_east': 'SE', 'south_west': 'SW'}
        for z in p_data.get("zones", []):
            for dp_key, dp_val in z.get("directional_points", {}).items():
                bpts.append({
                    "CLUSTER": z["cluster_label"],
                    "ZONE": z["risk_level"],
                    "DIRECTION": rev_map.get(dp_key, dp_key.upper()),
                    "LATITUDE": dp_val["lat"],
                    "LONGITUDE": dp_val["lng"]
                })

        all_results[period] = {
            'lscp': {
                'selected_locations': [(a["lat"], a["lng"]) for a in sel_ambs],
                'selected_labels': [a.get("label", f"amb_{i}") for i, a in enumerate(sel_ambs)],
                'selected_indices': [int(a["ambulance_id"].split("_")[1]) if "_" in a.get("ambulance_id", "") else i for i, a in enumerate(sel_ambs)],
                'coverage_map': {} # regenerated on update
            },
            'zs': pd.DataFrame(zones_data),
            'bpts': bpts,
            'cands': [(c["lat"], c["lng"]) for c in p_data.get("candidate_ambulance_points", [])],
            'clabels': [c.get("label", f"cand_{i}") for i, c in enumerate(p_data.get("candidate_ambulance_points", []))]
        }
    return all_results

def add_real_time_point(all_results, period, lat, lon, risk_level):
    """
    Calculates coverage and updates the system state for a real-time point.
    """
    if period not in all_results:
        # If the requested period is missing, we create a fresh state for it.
        # we isolate incident zones (zs) and active deployments (lscp) 
        # but WE MUST carry over the resource pool (cands/clabels) so we know where bases are.
        import copy
        existing = next(iter(all_results.values())) if all_results else None
        all_results[period] = {
            'lscp': {
                'selected_locations': [], 
                'selected_labels': [], 
                'selected_indices': [],
                'coverage_map': {}
            },
            'zs': pd.DataFrame(columns=["CLUSTER", "ZONE", "CENTER_LATITUDE", "CENTER_LONGITUDE", "ACCIDENT_COUNT", "TIME_PERIOD"]),
            'bpts': [],
            'cands': copy.deepcopy(existing.get('cands', [])) if existing else [],
            'clabels': copy.deepcopy(existing.get('clabels', [])) if existing else []
        }
        
    res = all_results[period]
    
    # Check current coverage
    amb_locs = res['lscp']['selected_locations']
    nearest_time = 999
    nearest_idx = -1
    
    for i, loc in enumerate(amb_locs):
        t = get_travel_time(loc[0], loc[1], lat, lon)
        if t < nearest_time:
            nearest_time = t
            nearest_idx = i
            
    is_covered = (nearest_time <= COVERAGE_THRESHOLD_MINUTES)
    
    # 1. Create New Cluster
    max_c = res['zs']['CLUSTER'].max() if not res['zs'].empty else -1
    new_cluster_id = int(max_c + 1)
    
    # 2. Add Zone
    new_zone = pd.DataFrame([{
        "CLUSTER": new_cluster_id,
        "ZONE": risk_level,
        "CENTER_LATITUDE": lat,
        "CENTER_LONGITUDE": lon,
        "ACCIDENT_COUNT": 1,
        "TIME_PERIOD": period
    }])
    res['zs'] = pd.concat([res['zs'], new_zone], ignore_index=True)
    
    # 3. Add Boundary Points
    offset = 0.001
    diag = 0.707
    dirs = {'N':(lat+offset, lon), 'S':(lat-offset, lon), 'E':(lat, lon+offset), 'W':(lat, lon-offset),
            'NE':(lat+offset*diag, lon+offset*diag), 'NW':(lat+offset*diag, lon-offset*diag),
            'SE':(lat-offset*diag, lon+offset*diag), 'SW':(lat-offset*diag, lon-offset*diag)}
    
    for d, (la, lo) in dirs.items():
        res['bpts'].append({'CLUSTER': new_cluster_id, 'ZONE': risk_level, 'DIRECTION': d, 'LATITUDE': la, 'LONGITUDE': lo})

    # 4. Handle Deployment
    if not is_covered:
        new_label = f"RT-Ambulance-C{new_cluster_id}"
        if 'cands' not in res: res['cands'] = []
        if 'clabels' not in res: res['clabels'] = []
        res['cands'].append((lat, lon))
        res['clabels'].append(new_label)
        new_idx = len(res['cands']) - 1
        
        res['lscp']['selected_locations'].append((lat, lon))
        res['lscp']['selected_labels'].append(new_label)
        res['lscp']['selected_indices'].append(new_idx)
        
    return all_results

def serialize_state_to_json(all_results):
    """
    Serializes the internal state back into the standardized JSON format.
    """
    output = {"time_periods": {}}
    for period, res in all_results.items():
        p_data = {
            "metadata": {"time_period_name": period, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "total_selected_ambulances": len(res['lscp']['selected_locations'])},
            "zones": [],
            "candidate_ambulance_points": [],
            "selected_ambulances": []
        }
        
        # Zones
        for _, row in res['zs'].iterrows():
            cid = int(row['CLUSTER'])
            dp = {}
            for p in [x for x in res['bpts'] if x['CLUSTER'] == cid]:
                key = {'N':'north','S':'south','E':'east','W':'west','NE':'north_east','NW':'north_west','SE':'south_east','SW':'south_west'}.get(p['DIRECTION'], p['DIRECTION'].lower())
                dp[key] = {"lat": float(p['LATITUDE']), "lng": float(p['LONGITUDE'])}
            
            p_data["zones"].append({
                "zone_id": f"zone_{cid}",
                "cluster_label": cid,
                "risk_level": row['ZONE'],
                "centroid": {"lat": float(row['CENTER_LATITUDE']), "lng": float(row['CENTER_LONGITUDE'])},
                "directional_points": dp
            })

        # Candidates
        for j, (loc, lab) in enumerate(zip(res.get('cands', []), res.get('clabels', []))):
            p_data["candidate_ambulance_points"].append({"ambulance_id": f"amb_{j}", "label": lab, "lat": float(loc[0]), "lng": float(loc[1])})

        # Selected
        for loc, lab, idx in zip(res['lscp']['selected_locations'], res['lscp']['selected_labels'], res['lscp']['selected_indices']):
            p_data["selected_ambulances"].append({"ambulance_id": f"amb_{idx}", "label": lab, "lat": float(loc[0]), "lng": float(loc[1])})

        output["time_periods"][period] = p_data
    return json.dumps(output, indent=2)

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
        
        tt = get_distance_matrix_cached(origins, dem)
        lscp = solve_lscp(dem, origins, tt, COVERAGE_THRESHOLD_MINUTES)
        
        all_results[period] = {
            "metadata": {"time_period_name": period, "coverage_threshold_minutes": COVERAGE_THRESHOLD_MINUTES, "total_zones": len(pz)},
            "zones": [],
            "candidate_ambulance_points": [],
            "selected_ambulances": []
        }

        # Align with structured JSON format
        for _, r in pz.iterrows():
            cid = int(r['CLUSTER'])
            b_pts = [p for p in bpts if p['CLUSTER'] == cid]
            dp = {}
            for p in b_pts:
                key = {'N':'north','S':'south','E':'east','W':'west','NE':'north_east','NW':'north_west','SE':'south_east','SW':'south_west'}.get(p['DIRECTION'], p['DIRECTION'].lower())
                dp[key] = {"lat": float(p['LATITUDE']), "lng": float(p['LONGITUDE'])}
            
            all_results[period]["zones"].append({
                "zone_id": f"zone_{cid}",
                "cluster_label": cid,
                "risk_level": r['ZONE'],
                "centroid": {"lat": float(r['CENTER_LATITUDE']), "lng": float(r['CENTER_LONGITUDE'])},
                "directional_points": dp
            })

        # Candidate Points (use origins as base)
        for idx, loc in enumerate(origins):
             all_results[period]["candidate_ambulance_points"].append({
                 "ambulance_id": f"amb_{idx}", "label": f"Amb-Base-{idx}", "lat": loc[0], "lng": loc[1]
             })

        # Selected Ambulances
        sel_locs = lscp.get('selected_locations', [])
        for idx, loc in enumerate(sel_locs):
            all_results[period]["selected_ambulances"].append({
                "ambulance_id": f"amb_{idx}", "label": f"Ambulance-{idx}", "lat": loc[0], "lng": loc[1]
            })

    return {"time_periods": all_results}
