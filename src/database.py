import os
import sqlite3
from datetime import datetime
from src.config import DATABASE_PATH
from src.logging_config import logger

def get_db_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table 1: Individual Workstation / Seat Predictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        space_type TEXT DEFAULT 'computer_lab',
        workstation_id TEXT NOT NULL,
        prediction TEXT NOT NULL,
        probability REAL NOT NULL,
        confidence REAL NOT NULL,
        confidence_level TEXT NOT NULL,
        review_required INTEGER NOT NULL,
        model_version TEXT NOT NULL,
        image_source TEXT
    )
    """)
    
    # Table 2: Space Occupancy Snapshots
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS occupancy_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        space_type TEXT DEFAULT 'computer_lab',
        total_workstations INTEGER NOT NULL,
        occupied INTEGER NOT NULL,
        available INTEGER NOT NULL,
        occupancy_percentage REAL NOT NULL,
        availability_percentage REAL NOT NULL,
        occupancy_level TEXT NOT NULL
    )
    """)
    
    # Migration: check if space_type column exists in existing tables
    try:
        cursor.execute("PRAGMA table_info(predictions)")
        cols = [row['name'] for row in cursor.fetchall()]
        if 'space_type' not in cols:
            cursor.execute("ALTER TABLE predictions ADD COLUMN space_type TEXT DEFAULT 'computer_lab'")
            
        cursor.execute("PRAGMA table_info(occupancy_snapshots)")
        cols_snap = [row['name'] for row in cursor.fetchall()]
        if 'space_type' not in cols_snap:
            cursor.execute("ALTER TABLE occupancy_snapshots ADD COLUMN space_type TEXT DEFAULT 'computer_lab'")
    except Exception as e:
        logger.warning(f"Database column migration note: {e}")
        
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully at: %s", DATABASE_PATH)

def save_predictions_batch(prediction_records, space_type: str = "computer_lab"):
    """Save batch of prediction records into SQLite DB."""
    if not prediction_records:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for rec in prediction_records:
        s_type = rec.get('space_type', space_type)
        cursor.execute("""
        INSERT INTO predictions 
        (timestamp, space_type, workstation_id, prediction, probability, confidence, confidence_level, review_required, model_version, image_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec.get('timestamp', now_str),
            s_type,
            rec['workstation_id'],
            rec['prediction'],
            rec['probability'],
            rec['confidence'],
            rec['confidence_level'],
            1 if rec['review_required'] else 0,
            rec.get('model_version', 'seat_occupancy_cnn_v1'),
            rec.get('image_source', 'uploaded_image')
        ))
        
    conn.commit()
    conn.close()

def save_occupancy_snapshot(stats, space_type: str = "computer_lab"):
    """Save space-wide occupancy snapshot into SQLite DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
    INSERT INTO occupancy_snapshots
    (timestamp, space_type, total_workstations, occupied, available, occupancy_percentage, availability_percentage, occupancy_level)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        space_type,
        stats['total'],
        stats['occupied'],
        stats['available'],
        stats['occupancy_pct'],
        stats['availability_pct'],
        stats['level']
    ))
    
    conn.commit()
    conn.close()

def get_latest_occupancy(space_type: str = None):
    """Fetch most recent occupancy snapshot, optionally filtered by space."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if space_type and space_type != 'all':
        cursor.execute("SELECT * FROM occupancy_snapshots WHERE space_type = ? ORDER BY id DESC LIMIT 1", (space_type,))
    else:
        cursor.execute("SELECT * FROM occupancy_snapshots ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_all_snapshots(space_type: str = None, limit: int = 500):
    """Fetch historical occupancy snapshots, optionally filtered by space."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if space_type and space_type != 'all':
        cursor.execute("SELECT * FROM occupancy_snapshots WHERE space_type = ? ORDER BY id ASC LIMIT ?", (space_type, limit))
    else:
        cursor.execute("SELECT * FROM occupancy_snapshots ORDER BY id ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Initialize DB on module import
init_db()
