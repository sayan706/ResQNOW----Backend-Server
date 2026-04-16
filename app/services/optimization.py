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
    # Use Google API if configured, otherwise fallback to Haversine directly
    use_google = bool(MAP_API_KEY and MAP_API_KEY != "YOUR_GOOGLE_MAPS_API_KEY_HERE")
    
    if not use_google:
        # 1. OPTIMIZATION: Do not hit cloud Redis 40,000+ times for Haversine. 
        # Haversine is a pure math operation in python that completes in ~0.001 ms, 
        # while an external cloud network request takes 50-100 ms!
        for i, o in enumerate(origins):
            for j, d in enumerate(destinations):
                dist = haversine_km(o[0], o[1], d[0], d[1])
                matrix[i][j] = dist * 1.4 / FALLBACK_SPEED_KMH * 60
        return matrix
                
    # 2. Redis Batching Optimization for API fetched results
    keys = []
    for i, o in enumerate(origins):
        for j, d in enumerate(destinations):
            keys.append((i, j, f"dist:{o[0]:.5f}:{o[1]:.5f}:{d[0]:.5f}:{d[1]:.5f}"))

    # Batch MGET to avoid N*M roundtrips
    key_strings = [k[2] for k in keys]
    
    # Process in chunks of 1000 to prevent overwhelming Upstash free tier memory handling
    CHUNK_SIZE = 1000
    to_query = []
    
    for chunk_start in range(0, len(keys), CHUNK_SIZE):
        chunk_keys = keys[chunk_start:chunk_start+CHUNK_SIZE]
        chunk_key_strings = key_strings[chunk_start:chunk_start+CHUNK_SIZE]
        try:
            cached_vals = cache_service.client.mget(chunk_key_strings)
            for (i, j, k), val in zip(chunk_keys, cached_vals):
                if val is not None:
                    try:
                        matrix[i][j] = float(json.loads(val) if isinstance(val, str) and val.startswith('{') else val)
                    except:
                        matrix[i][j] = float(val)
                else:
                    to_query.append((i, j, origins[i], destinations[j]))
        except Exception:
            # Redis failed (memory limit/connection), fallback all chunk members to query list
            for (i, j, k) in chunk_keys:
                to_query.append((i, j, origins[i], destinations[j]))

    if not to_query:
        return matrix

    print(f"Fetching {len(to_query)} missing distance matrix points from Google API...")

    # Group the missing queries by origin for slightly better batching
    origins_to_fetch = {}
    for i, j, o, d in to_query:
        if i not in origins_to_fetch:
            origins_to_fetch[i] = {'orig': o, 'dests': []}
        origins_to_fetch[i]['dests'].append((j, d))

    BS = 10  # Max destinations per Google Matrix request

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for i, origin_data in origins_to_fetch.items():
            o = origin_data['orig']
            dests_list = origin_data['dests']
            
            for d_idx in range(0, len(dests_list), BS):
                chunk = dests_list[d_idx:d_idx+BS]
                tasks.append(executor.submit(_fetch_google_chunk, i, o, chunk, MAP_API_KEY))

        cache_updates = {}
        for future in concurrent.futures.as_completed(tasks):
            success, result = future.result()
            if success:
                i, res_chunk = result
                for j, dest_coords, t_time in res_chunk:
                    matrix[i][j] = t_time
                    k = f"dist:{origins[i][0]:.5f}:{origins[i][1]:.5f}:{destinations[j][0]:.5f}:{destinations[j][1]:.5f}"
                    cache_updates[k] = json.dumps(t_time)
            else:
                # If API fails for a chunk, fallback to Haversine
                i, failed_chunk, _ = result
                for j, dest_coords in failed_chunk:
                    t_time = haversine_km(origins[i][0], origins[i][1], dest_coords[0], dest_coords[1]) * 1.4 / FALLBACK_SPEED_KMH * 60
                    matrix[i][j] = t_time
                    k = f"dist:{origins[i][0]:.5f}:{origins[i][1]:.5f}:{destinations[j][0]:.5f}:{destinations[j][1]:.5f}"
                    cache_updates[k] = json.dumps(t_time)

    # Save newly fetched Google results (or their fallbacks) to Redis en masse
    if cache_updates:
        # Pipelined MSET for best performance
        try:
            pipe = cache_service.client.pipeline()
            # MSET takes a dict of keys/values. Set expiry individually if via pipeline or just MSET directly
            # Free tier friendly: Chunk MSET into 1000s
            items = list(cache_updates.items())
            for chunk_start in range(0, len(items), 1000):
                chunk_items = dict(items[chunk_start:chunk_start+1000])
                pipe.mset(chunk_items)
            pipe.execute()
        except Exception as e:
            print(f"Failed to bulk cache Google Matrix results: {e}")

    return matrix

def _fetch_google_chunk(i, origin, dest_chunk, api_key):
    """Worker function to fetch a batch from Google API."""
    try:
        o_str = f"{origin[0]},{origin[1]}"
        d_str = "|".join(f"{d[0]},{d[1]}" for j, d in dest_chunk)
        url = (f"https://maps.googleapis.com/maps/api/distancematrix/json"
               f"?origins={o_str}&destinations={d_str}&key={api_key}&mode=driving")
        
        data = req_lib.get(url, timeout=20).json()
        if data['status'] != 'OK':
            return False, (i, dest_chunk, data['status'])
            
        row_data = []
        for d_idx, element in enumerate(data['rows'][0]['elements']):
            if element['status'] == 'OK':
                t_time = element['duration']['value'] / 60
            else:
                d = dest_chunk[d_idx][1]
                t_time = haversine_km(origin[0], origin[1], d[0], d[1]) * 1.4 / FALLBACK_SPEED_KMH * 60
            row_data.append((dest_chunk[d_idx][0], dest_chunk[d_idx][1], t_time))
            
        return True, (i, row_data)
    except Exception as e:
        return False, (i, dest_chunk, str(e))
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

def auto_tune_dbscan(X_scaled, n_rows):
    """Auto-tune DBSCAN and return best labels."""
    best_score = -9999
    best_params = None
    best_labels = None

    for eps in np.arange(0.2, 1.6, 0.1):
        for ms in [2, 3, 4, 5]:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_scaled)
            uq = set(labels)
            nc = len(uq - {-1})
            nn = list(labels).count(-1)
            nr = nn / len(labels)
            if nc < 1:
                continue

            if nc >= 2:
                mask = labels != -1
                if len(set(labels[mask])) >= 2 and np.sum(mask) > 2:
                    try:
                        sil = silhouette_score(X_scaled[mask], labels[mask])
                    except:
                        sil = -1
                else:
                    sil = -1
            else:
                sil = 0.1

            cs = pd.Series(labels[labels != -1]).value_counts()
            avg_cs = cs.mean() if len(cs) > 0 else 0
            cp = 1.5 if nc > max(6, n_rows // 5) else 0
            np_ = 1.5 if nr > 0.5 else (0.7 if nr > 0.3 else 0)
            sr = min(avg_cs / 5, 2.0)
            fs = 2.0 * sil + 1.2 * sr - np_ - cp

            if fs > best_score:
                best_score = fs
                best_params = (eps, ms)
                best_labels = labels

    if best_labels is None:
         return DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)
    return best_labels

def assign_zones_for_period(period_df, period_name, cluster_centers):
    """Compute per-cluster risk & assign Red/Orange/Green for one time period."""
    if len(period_df) == 0:
        return pd.DataFrame()
    pc = period_df[period_df['CLUSTER'] != -1].copy()
    if len(pc) == 0:
        return pd.DataFrame()

    s = pc.groupby('CLUSTER').agg(
        AVG_SEVERITY=('SEVERITY_SCORE', 'mean'),
        TOTAL_SEVERITY=('SEVERITY_SCORE', 'sum'),
        ACCIDENT_COUNT=('CLUSTER', 'count'),
    ).reset_index()

    s['RISK_SCORE'] = 0.4 * s['ACCIDENT_COUNT'] + 0.3 * s['AVG_SEVERITY'] + 0.3 * s['TOTAL_SEVERITY']

    if len(s) == 1:
        s['ZONE'] = 'Red'
    elif len(s) == 2:
        s = s.sort_values('RISK_SCORE', ascending=False).reset_index(drop=True)
        s['ZONE'] = ['Red', 'Orange']
    else:
        rt = s['RISK_SCORE'].quantile(0.66)
        ot = s['RISK_SCORE'].quantile(0.33)
        s['ZONE'] = s['RISK_SCORE'].apply(
            lambda x: 'Red' if x >= rt else ('Orange' if x >= ot else 'Green'))

    if len(s) > 1:
        mn, mx = s['RISK_SCORE'].min(), s['RISK_SCORE'].max()
        if mx > 0 and mn / mx >= 0.50:
            s.loc[s['ZONE'] == 'Green', 'ZONE'] = 'Orange'

    s['TIME_PERIOD'] = period_name
    s = s.merge(cluster_centers, on='CLUSTER', how='left')
    return s

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

def generate_candidates(cc_df, boundary_pts):
    """
    Candidate ambulance locations =
      cluster centres + boundary points + midpoints between centres.
    """
    locs, labels = [], []
    for _, r in cc_df.iterrows():
        locs.append((r['CENTER_LATITUDE'], r['CENTER_LONGITUDE']))
        labels.append(f"Center-C{int(r['CLUSTER'])}")
    for p in boundary_pts:
        locs.append((p['LATITUDE'], p['LONGITUDE']))
        labels.append(f"Bnd-C{int(p['CLUSTER'])}-{p['DIRECTION']}")
    centres = cc_df[['CENTER_LATITUDE', 'CENTER_LONGITUDE']].values
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            locs.append(((centres[i][0] + centres[j][0]) / 2,
                         (centres[i][1] + centres[j][1]) / 2))
            labels.append(f"Mid-C{int(cc_df.iloc[i]['CLUSTER'])}-C{int(cc_df.iloc[j]['CLUSTER'])}")
    return locs, labels

def solve_lscp(demand_pts, demand_zones, cand_locs, tt_matrix, threshold):
    """
    Solve Location Set Covering Problem with zone priorities.
    """
    nc, nd = len(cand_locs), len(demand_pts)
    cov = (tt_matrix <= threshold).astype(int)

    uncoverable = [i for i in range(nd) if cov[:, i].sum() == 0]
    
    red_i   = [i for i in range(nd) if demand_zones[i] == 'Red'    and i not in uncoverable]
    orange_i = [i for i in range(nd) if demand_zones[i] == 'Orange' and i not in uncoverable]
    green_i  = [i for i in range(nd) if demand_zones[i] == 'Green'  and i not in uncoverable]

    for label, indices in [("All", red_i + orange_i + green_i),
                           ("Red+Orange", red_i + orange_i),
                           ("Red only", red_i)]:
        if not indices:
            continue
        prob = pulp.LpProblem("LSCP", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x{j}", cat='Binary') for j in range(nc)]
        prob += pulp.lpSum(x)
        for i in indices:
            covers = [j for j in range(nc) if cov[j][i]]
            if covers:
                prob += pulp.lpSum(x[j] for j in covers) >= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] == 'Optimal':
            sel = [j for j in range(nc) if x[j].varValue is not None and x[j].varValue > 0.5]
            cov_map = {i: [j for j in sel if cov[j][i]] for i in range(nd)}
            return {'selected_locations': [cand_locs[j] for j in sel], 'ambulance_count': len(sel), 'selected_indices': sel, 'coverage_map': cov_map}
            
    return {'ambulance_count': 0, 'status': 'Infeasible', 'selected_indices': [], 'selected_locations': [], 'coverage_map': {}}

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

        coverage_map = {}
        for sel_amb in sel_ambs:
            a_idx_str = sel_amb.get("ambulance_id", "0")
            a_idx = int(a_idx_str.split("_")[1]) if "_" in a_idx_str else 0
            
            for z_id in sel_amb.get("covered_zone_ids", []):
                if "_" in z_id:
                    cid = int(z_id.split("_")[1])
                    for i, p in enumerate(bpts):
                        if p["CLUSTER"] == cid:
                            if i not in coverage_map:
                                coverage_map[i] = []
                            if a_idx not in coverage_map[i]:
                                coverage_map[i].append(a_idx)

        all_results[period] = {
            'lscp': {
                'selected_locations': [(a["lat"], a["lng"]) for a in sel_ambs],
                'selected_labels': [a.get("label", f"amb_{i}") for i, a in enumerate(sel_ambs)],
                'selected_indices': [int(a["ambulance_id"].split("_")[1]) if "_" in a.get("ambulance_id", "") else i for i, a in enumerate(sel_ambs)],
                'coverage_map': coverage_map
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
            boundary_points = []
            for p in [x for x in res['bpts'] if x['CLUSTER'] == cid]:
                key = {'N':'north','S':'south','E':'east','W':'west','NE':'north_east','NW':'north_west','SE':'south_east','SW':'south_west'}.get(p['DIRECTION'], p['DIRECTION'].lower())
                dp[key] = {"lat": float(p['LATITUDE']), "lng": float(p['LONGITUDE'])}
                boundary_points.append({"lat": float(p['LATITUDE']), "lng": float(p['LONGITUDE'])})
            
            selected_ambs_for_zone = []
            for i, ambs in res['lscp'].get('coverage_map', {}).items():
                if i < len(res['bpts']) and res['bpts'][i]['CLUSTER'] == cid:
                    for a_idx in ambs:
                        a_id = f"amb_{a_idx}"
                        if a_id not in selected_ambs_for_zone:
                            selected_ambs_for_zone.append(a_id)
            
            p_data["zones"].append({
                "zone_id": f"zone_{cid}",
                "cluster_label": cid,
                "risk_level": row['ZONE'],
                "centroid": {"lat": float(row['CENTER_LATITUDE']), "lng": float(row['CENTER_LONGITUDE'])},
                "boundary_points": boundary_points,
                "directional_points": dp,
                "total_original_points": int(row.get('ACCIDENT_COUNT', 1)),
                "selected_ambulance_ids_covering_zone": selected_ambs_for_zone
            })

        # Candidates
        for j, (loc, lab) in enumerate(zip(res.get('cands', []), res.get('clabels', []))):
            p_data["candidate_ambulance_points"].append({"ambulance_id": f"amb_{j}", "label": lab, "lat": float(loc[0]), "lng": float(loc[1])})

        # Selected
        for loc, lab, idx in zip(res['lscp']['selected_locations'], res['lscp']['selected_labels'], res['lscp']['selected_indices']):
            covered_zones = []
            for i, ambs in res['lscp'].get('coverage_map', {}).items():
                if idx in ambs and i < len(res['bpts']):
                    zid = f"zone_{int(res['bpts'][i]['CLUSTER'])}"
                    if zid not in covered_zones:
                        covered_zones.append(zid)
                        
            p_data["selected_ambulances"].append({
                "ambulance_id": f"amb_{idx}", 
                "label": lab, 
                "lat": float(loc[0]), 
                "lng": float(loc[1]),
                "covered_zone_ids": covered_zones
            })

        output["time_periods"][period] = p_data
    return json.dumps(output, indent=2)

def process_ambulance_optimization(csv_path):
    df = load_and_clean_data(csv_path)
    X_scaled = StandardScaler().fit_transform(df[['LATITUDE', 'LONGITUDE']])
    df['CLUSTER'] = auto_tune_dbscan(X_scaled, len(df))
    cdf = df[df['CLUSTER'] != -1].copy()
    cc = cdf.groupby('CLUSTER').agg(CENTER_LATITUDE=('LATITUDE', 'mean'), CENTER_LONGITUDE=('LONGITUDE', 'mean')).reset_index()

    all_results = {}
    for period in TIME_PERIOD_ORDER:
        period_df = df[df['TIME_PERIOD'] == period]
        if period_df.empty: continue
        
        pz = assign_zones_for_period(period_df, period, cc)
        if pz.empty: continue
        
        bpts = []
        for _, r in pz.iterrows():
            bpts.extend(generate_boundary_points(cdf, int(r['CLUSTER']), r['ZONE']))
        
        dem = [(p['LATITUDE'], p['LONGITUDE']) for p in bpts]
        dzones = [p['ZONE'] for p in bpts]
        cands, clabels = generate_candidates(pz[['CLUSTER', 'CENTER_LATITUDE', 'CENTER_LONGITUDE']], bpts)
        
        tt = get_distance_matrix_cached(cands, dem)
        lscp = solve_lscp(dem, dzones, cands, tt, COVERAGE_THRESHOLD_MINUTES)
        
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
            boundary_points = []
            for p in b_pts:
                key = {'N':'north','S':'south','E':'east','W':'west','NE':'north_east','NW':'north_west','SE':'south_east','SW':'south_west'}.get(p['DIRECTION'], p['DIRECTION'].lower())
                dp[key] = {"lat": float(p['LATITUDE']), "lng": float(p['LONGITUDE'])}
                boundary_points.append({"lat": float(p['LATITUDE']), "lng": float(p['LONGITUDE'])})
                
            selected_ambs_for_zone = []
            for i, ambs in lscp.get('coverage_map', {}).items():
                if i < len(bpts) and bpts[i]['CLUSTER'] == cid:
                    for a_idx in ambs:
                        a_id = f"amb_{a_idx}"
                        if a_id not in selected_ambs_for_zone:
                            selected_ambs_for_zone.append(a_id)

            zone_pts_df = cdf[cdf['CLUSTER'] == cid]
            all_zone_pts = zone_pts_df[['LATITUDE', 'LONGITUDE']].rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lng'}).to_dict('records')
            # Sample max 200 points for frontend like desktop script does
            sample_count = len(all_zone_pts)
            if sample_count > 200:
                all_zone_pts = all_zone_pts[:200]
            
            all_results[period]["zones"].append({
                "zone_id": f"zone_{cid}",
                "cluster_label": cid,
                "risk_level": r['ZONE'],
                "centroid": {"lat": float(r['CENTER_LATITUDE']), "lng": float(r['CENTER_LONGITUDE'])},
                "boundary_points": boundary_points,
                "directional_points": dp,
                "all_zone_points": all_zone_pts,
                "total_original_points": int(r.get('ACCIDENT_COUNT', 1)),
                "sampled_points_count": sample_count,
                "selected_ambulance_ids_covering_zone": selected_ambs_for_zone
            })

        # Candidate Points
        for idx, (loc, lab) in enumerate(zip(cands, clabels)):
             all_results[period]["candidate_ambulance_points"].append({
                 "ambulance_id": f"amb_{idx}", "label": lab, "lat": float(loc[0]), "lng": float(loc[1])
             })

        # Selected Ambulances
        sel_locs = lscp.get('selected_locations', [])
        sel_indices = lscp.get('selected_indices', [])
        for loc, c_idx in zip(sel_locs, sel_indices):
            covered_zones = []
            for i, ambs in lscp.get('coverage_map', {}).items():
                if c_idx in ambs and i < len(bpts):
                    zid = f"zone_{int(bpts[i]['CLUSTER'])}"
                    if zid not in covered_zones:
                        covered_zones.append(zid)
            
            lab = clabels[c_idx] if c_idx < len(clabels) else f"Ambulance-{c_idx}"
            all_results[period]["selected_ambulances"].append({
                "ambulance_id": f"amb_{c_idx}", "label": lab, "lat": loc[0], "lng": loc[1], "covered_zone_ids": covered_zones
            })

    return {"time_periods": all_results}
