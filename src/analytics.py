from typing import Optional
from src.database import get_all_snapshots, get_db_connection

def compute_occupancy_analytics(space_type: Optional[str] = None):
    """
    Compute statistical analytics from stored SQLite occupancy snapshots and predictions.
    Supports filtering by specific space (e.g. 'computer_lab', 'lecture_room', 'auditorium')
    or computing aggregate metrics across all spaces.
    """
    snapshots = get_all_snapshots(space_type=space_type)
    if not snapshots:
        return {
            'has_data': False,
            'space_type': space_type or 'all',
            'message': 'Historical data will appear after sufficient real observations are recorded for this space.'
        }
        
    occupancy_rates = [s['occupancy_percentage'] for s in snapshots]
    occupied_counts = [s['occupied'] for s in snapshots]
    
    avg_occ = sum(occupancy_rates) / len(occupancy_rates)
    peak_occ = max(occupancy_rates)
    min_occ = min(occupancy_rates)
    latest_snapshot = snapshots[-1]
    
    # Calculate peak seat utilization for this space
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if space_type and space_type != 'all':
        cursor.execute("""
        SELECT workstation_id, COUNT(*) as occ_count 
        FROM predictions 
        WHERE prediction = 'OCCUPIED' AND space_type = ?
        GROUP BY workstation_id 
        ORDER BY occ_count DESC 
        LIMIT 1
        """, (space_type,))
        most_used_row = cursor.fetchone()
        
        cursor.execute("""
        SELECT workstation_id, COUNT(*) as empty_count 
        FROM predictions 
        WHERE prediction = 'EMPTY' AND space_type = ?
        GROUP BY workstation_id 
        ORDER BY empty_count DESC 
        LIMIT 1
        """, (space_type,))
        most_avail_row = cursor.fetchone()
    else:
        cursor.execute("""
        SELECT workstation_id, COUNT(*) as occ_count 
        FROM predictions 
        WHERE prediction = 'OCCUPIED' 
        GROUP BY workstation_id 
        ORDER BY occ_count DESC 
        LIMIT 1
        """)
        most_used_row = cursor.fetchone()
        
        cursor.execute("""
        SELECT workstation_id, COUNT(*) as empty_count 
        FROM predictions 
        WHERE prediction = 'EMPTY' 
        GROUP BY workstation_id 
        ORDER BY empty_count DESC 
        LIMIT 1
        """)
        most_avail_row = cursor.fetchone()
        
    conn.close()
    
    most_used_pc = dict(most_used_row)['workstation_id'] if most_used_row else "N/A"
    most_avail_pc = dict(most_avail_row)['workstation_id'] if most_avail_row else "N/A"
    
    return {
        'has_data': True,
        'space_type': space_type or 'all',
        'total_observations': len(snapshots),
        'latest_occupancy_pct': latest_snapshot['occupancy_percentage'],
        'average_occupancy_pct': round(avg_occ, 1),
        'peak_occupancy_pct': round(peak_occ, 1),
        'min_occupancy_pct': round(min_occ, 1),
        'most_used_workstation': most_used_pc,
        'most_available_workstation': most_avail_pc,
        'snapshots': snapshots
    }
