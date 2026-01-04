"""
FedEx DCA Management Platform
Centralized, AI-assisted Debt Collection Agency Management System
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from functools import wraps

# AI/ML imports
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)
app.secret_key = 'fedex_dca_hackathon_2024_secret_key'

# Custom Jinja2 filters
@app.template_filter('format_currency')
def format_currency_filter(value):
    """Format a number as Indian currency"""
    try:
        return "{:,.0f}".format(float(value) if value else 0)
    except:
        return "0"

@app.template_filter('format_percent')
def format_percent_filter(value):
    """Format a number as percentage"""
    try:
        return "{:.1f}".format(float(value) if value else 0)
    except:
        return "0.0"

@app.template_filter('to_int')
def to_int_filter(value):
    """Convert to int safely"""
    try:
        return int(float(value) if value else 0)
    except:
        return 0

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

DATABASE = 'fedex_dca.db'

# ============================================
# SQLITE ADAPTERS FOR PYTHON 3.12
# ============================================

def adapt_datetime(dt):
    """Adapt datetime object to string"""
    return dt.isoformat() if dt else None

def adapt_date(d):
    """Adapt date object to string"""
    return d.isoformat() if d else None

def convert_datetime(val):
    """Convert string to datetime"""
    try:
        return datetime.fromisoformat(val.decode() if isinstance(val, bytes) else val)
    except:
        return val

def convert_date(val):
    """Convert string to date"""
    try:
        return datetime.fromisoformat(val.decode() if isinstance(val, bytes) else val).date()
    except:
        return val

# Register adapters
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_adapter(timedelta, lambda td: str(td.total_seconds()))

# Register converters to override defaults - these handle ISO format
def my_convert_timestamp(val):
    """Custom timestamp converter that handles ISO format"""
    try:
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        # Just return as string, we'll handle in template
        return val
    except:
        return val

def my_convert_date(val):
    """Custom date converter that handles ISO format"""
    try:
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        return val
    except:
        return val

# Override the default converters
sqlite3.register_converter("TIMESTAMP", my_convert_timestamp)
sqlite3.register_converter("DATE", my_convert_date)
sqlite3.register_converter("timestamp", my_convert_timestamp)
sqlite3.register_converter("date", my_convert_date)

def dict_factory(cursor, row):
    """Convert sqlite Row to dictionary with proper Python types"""
    d = {}
    for idx, col in enumerate(cursor.description):
        val = row[idx]
        # Convert numpy types to Python types first
        if hasattr(val, 'item'):
            val = val.item()
        # Convert bytes to appropriate type
        if isinstance(val, bytes):
            try:
                decoded = val.decode('utf-8')
                try:
                    val = float(decoded)
                except ValueError:
                    val = decoded
            except UnicodeDecodeError:
                # If we can't decode as UTF-8, try latin-1 or just keep as string repr
                try:
                    val = val.decode('latin-1')
                except:
                    val = str(val)
        d[col[0]] = val
    return d

# ============================================
# DATABASE FUNCTIONS
# ============================================

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = dict_factory
    return conn

def init_db():
    """Initialize the database with all tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table (Admin and DCA users)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'dca',
            dca_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dca_id) REFERENCES dcas(id)
        )
    ''')
    
    # DCAs table (Debt Collection Agencies)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dcas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            region TEXT,
            performance_score REAL DEFAULT 50.0,
            total_cases INTEGER DEFAULT 0,
            recovered_cases INTEGER DEFAULT 0,
            total_recovered_amount REAL DEFAULT 0,
            sla_compliance_rate REAL DEFAULT 100.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Cases table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_number TEXT UNIQUE NOT NULL,
            customer_id TEXT NOT NULL,
            customer_name TEXT,
            invoice_number TEXT,
            amount REAL NOT NULL,
            overdue_days INTEGER NOT NULL,
            region TEXT,
            priority TEXT DEFAULT 'MEDIUM',
            recovery_probability REAL DEFAULT 0.5,
            status TEXT DEFAULT 'NEW',
            dca_id INTEGER,
            assigned_at TIMESTAMP,
            sla_due_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP,
            recovery_amount REAL DEFAULT 0,
            notes TEXT,
            FOREIGN KEY (dca_id) REFERENCES dcas(id)
        )
    ''')
    
    # Case Assignments History
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS case_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            dca_id INTEGER NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'ACTIVE',
            FOREIGN KEY (case_id) REFERENCES cases(id),
            FOREIGN KEY (dca_id) REFERENCES dcas(id)
        )
    ''')
    
    # Recovery Logs (Audit Trail)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recovery_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            dca_id INTEGER,
            user_id INTEGER,
            action TEXT NOT NULL,
            old_status TEXT,
            new_status TEXT,
            amount REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases(id),
            FOREIGN KEY (dca_id) REFERENCES dcas(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # SLA Escalations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            dca_id INTEGER,
            reason TEXT,
            escalated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved BOOLEAN DEFAULT 0,
            resolved_at TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases(id),
            FOREIGN KEY (dca_id) REFERENCES dcas(id)
        )
    ''')
    
    conn.commit()
    
    # Create default admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        ''', ('admin', 'admin@fedex.com', generate_password_hash('admin123'), 'admin'))
        conn.commit()
    
    # Create sample DCAs if not exists
    cursor.execute("SELECT COUNT(*) as cnt FROM dcas")
    if cursor.fetchone()['cnt'] == 0:
        sample_dcas = [
            ('Recovery Solutions Inc.', 'North', 85.0),
            ('FastTrack Collections', 'South', 78.5),
            ('DebtBusters Ltd.', 'East', 72.0),
            ('SecureRecovery Co.', 'West', 68.5),
            ('PrimeCollect Agency', 'Central', 80.0)
        ]
        for dca in sample_dcas:
            cursor.execute('''
                INSERT INTO dcas (name, region, performance_score)
                VALUES (?, ?, ?)
            ''', dca)
        conn.commit()
        
        # Create DCA users
        cursor.execute("SELECT id, name FROM dcas")
        dcas = cursor.fetchall()
        for dca in dcas:
            username = dca['name'].lower().replace(' ', '_').replace('.', '')[:20]
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role, dca_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, f'{username}@dca.com', generate_password_hash('dca123'), 'dca', dca['id']))
        conn.commit()
    
    conn.close()

# ============================================
# USER CLASS FOR FLASK-LOGIN
# ============================================

class User(UserMixin):
    def __init__(self, id, username, email, role, dca_id=None):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.dca_id = dca_id

@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user['id'], user['username'], user['email'], user['role'], user['dca_id'])
    return None

# ============================================
# AI / ML FUNCTIONS
# ============================================

def calculate_priority(overdue_days, amount):
    """Rule-based priority calculation"""
    if overdue_days > 90 and amount > 50000:
        return "HIGH"
    elif overdue_days > 90 or amount > 50000:
        return "HIGH"
    elif overdue_days > 45 or amount > 20000:
        return "MEDIUM"
    else:
        return "LOW"

def calculate_recovery_probability(overdue_days, amount, region=None):
    """
    Simple ML-based recovery probability calculation
    Uses logistic regression principles
    """
    # Normalize factors (0-1 scale)
    days_factor = max(0, 1 - (overdue_days / 365))  # More days = lower probability
    amount_factor = max(0, 1 - (amount / 100000))    # Higher amount = slightly lower
    
    # Weighted combination
    probability = (days_factor * 0.6) + (amount_factor * 0.3) + 0.1
    
    # Add some randomness for realism
    probability = min(0.95, max(0.05, probability + np.random.uniform(-0.1, 0.1)))
    
    return round(probability, 2)

def select_best_dca(case_priority, region=None):
    """Select the best DCA based on performance and region"""
    conn = get_db()
    cursor = conn.cursor()
    
    if case_priority == "HIGH":
        # High priority cases go to best performing DCAs
        cursor.execute('''
            SELECT id, name, performance_score FROM dcas 
            ORDER BY performance_score DESC LIMIT 1
        ''')
    elif region:
        # Try to match region first
        cursor.execute('''
            SELECT id, name, performance_score FROM dcas 
            WHERE region = ? ORDER BY performance_score DESC LIMIT 1
        ''', (region,))
        result = cursor.fetchone()
        if result:
            conn.close()
            return result['id'], result['name']
        # Fallback to any DCA
        cursor.execute('''
            SELECT id, name, performance_score FROM dcas 
            ORDER BY performance_score DESC LIMIT 1
        ''')
    else:
        # Medium/Low priority - round robin with performance weight
        cursor.execute('''
            SELECT id, name, performance_score, total_cases FROM dcas 
            ORDER BY (total_cases * 0.3 - performance_score * 0.7) ASC LIMIT 1
        ''')
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result['id'], result['name']
    return None, None

def check_sla_breaches():
    """Check for SLA breaches and create escalations"""
    conn = get_db()
    cursor = conn.cursor()
    
    today = datetime.now().date()
    
    # Find cases with breached SLA
    cursor.execute('''
        SELECT c.id, c.case_number, c.dca_id, c.sla_due_date 
        FROM cases c
        WHERE c.sla_due_date < ? 
        AND c.status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
        AND c.id NOT IN (SELECT case_id FROM escalations WHERE resolved = 0)
    ''', (today,))
    
    breached_cases = cursor.fetchall()
    
    for case in breached_cases:
        # Create escalation
        cursor.execute('''
            INSERT INTO escalations (case_id, dca_id, reason)
            VALUES (?, ?, ?)
        ''', (case['id'], case['dca_id'], 'SLA Breach - Due date exceeded'))
        
        # Update DCA performance
        if case['dca_id']:
            cursor.execute('''
                UPDATE dcas SET sla_compliance_rate = sla_compliance_rate - 1
                WHERE id = ?
            ''', (case['dca_id'],))
        
        # Log the escalation
        cursor.execute('''
            INSERT INTO recovery_logs (case_id, dca_id, action, notes)
            VALUES (?, ?, ?, ?)
        ''', (case['id'], case['dca_id'], 'ESCALATED', 'SLA breach detected by system'))
    
    conn.commit()
    conn.close()
    
    return len(breached_cases)

# ============================================
# DECORATORS
# ============================================

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Admin access required!', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# ROUTES - AUTHENTICATION
# ============================================

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('dca_dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            user_obj = User(user['id'], user['username'], user['email'], user['role'], user['dca_id'])
            login_user(user_obj)
            flash('Login successful!', 'success')
            
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dca_dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

# ============================================
# ROUTES - ADMIN DASHBOARD
# ============================================

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    conn = get_db()
    cursor = conn.cursor()
    
    # Check for SLA breaches
    breaches = check_sla_breaches()
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) as total FROM cases")
    total_cases = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) as count FROM cases WHERE status = 'NEW'")
    new_cases = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM cases WHERE status = 'RECOVERED'")
    recovered_cases = cursor.fetchone()['count']
    
    cursor.execute("SELECT SUM(amount) as total FROM cases")
    total_amount = cursor.fetchone()['total'] or 0
    
    cursor.execute("SELECT SUM(recovery_amount) as total FROM cases WHERE status = 'RECOVERED'")
    recovered_amount = cursor.fetchone()['total'] or 0
    
    recovery_rate = (recovered_amount / total_amount * 100) if total_amount > 0 else 0
    
    # Get cases by priority
    cursor.execute('''
        SELECT priority, COUNT(*) as count FROM cases 
        WHERE status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
        GROUP BY priority
    ''')
    priority_data = cursor.fetchall()
    
    # Get aging buckets
    cursor.execute('''
        SELECT 
            CASE 
                WHEN overdue_days <= 30 THEN '0-30 days'
                WHEN overdue_days <= 60 THEN '31-60 days'
                WHEN overdue_days <= 90 THEN '61-90 days'
                ELSE '90+ days'
            END as bucket,
            COUNT(*) as count,
            SUM(amount) as total_amount
        FROM cases 
        WHERE status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
        GROUP BY bucket
        ORDER BY MIN(overdue_days)
    ''')
    aging_data = cursor.fetchall()
    
    # Get DCA rankings
    cursor.execute('''
        SELECT id, name, performance_score, total_cases, recovered_cases, 
               sla_compliance_rate, total_recovered_amount
        FROM dcas ORDER BY performance_score DESC
    ''')
    dca_rankings = cursor.fetchall()
    
    # Get active escalations
    cursor.execute('''
        SELECT e.*, c.case_number, c.amount, d.name as dca_name
        FROM escalations e
        JOIN cases c ON e.case_id = c.id
        LEFT JOIN dcas d ON e.dca_id = d.id
        WHERE e.resolved = 0
        ORDER BY e.escalated_at DESC
    ''')
    escalations = cursor.fetchall()
    
    # Get recent cases
    cursor.execute('''
        SELECT c.*, d.name as dca_name FROM cases c
        LEFT JOIN dcas d ON c.dca_id = d.id
        ORDER BY c.created_at DESC LIMIT 10
    ''')
    recent_cases = cursor.fetchall()
    
    conn.close()
    
    return render_template('admin_dashboard.html',
        total_cases=total_cases,
        new_cases=new_cases,
        recovered_cases=recovered_cases,
        total_amount=total_amount,
        recovered_amount=recovered_amount,
        recovery_rate=recovery_rate,
        priority_data=priority_data,
        aging_data=aging_data,
        dca_rankings=dca_rankings,
        escalations=escalations,
        recent_cases=recent_cases,
        breaches=breaches
    )

@app.route('/admin/cases')
@login_required
@admin_required
def admin_cases():
    conn = get_db()
    cursor = conn.cursor()
    
    # Get filter parameters
    status_filter = request.args.get('status', '')
    priority_filter = request.args.get('priority', '')
    dca_filter = request.args.get('dca', '')
    
    query = '''
        SELECT c.*, d.name as dca_name FROM cases c
        LEFT JOIN dcas d ON c.dca_id = d.id
        WHERE 1=1
    '''
    params = []
    
    if status_filter:
        query += " AND c.status = ?"
        params.append(status_filter)
    if priority_filter:
        query += " AND c.priority = ?"
        params.append(priority_filter)
    if dca_filter:
        query += " AND c.dca_id = ?"
        params.append(dca_filter)
    
    query += " ORDER BY c.created_at DESC"
    
    cursor.execute(query, params)
    cases = cursor.fetchall()
    
    # Get DCAs for filter
    cursor.execute("SELECT id, name FROM dcas ORDER BY name")
    dcas = cursor.fetchall()
    
    conn.close()
    
    return render_template('admin_cases.html', cases=cases, dcas=dcas,
                          status_filter=status_filter, priority_filter=priority_filter,
                          dca_filter=dca_filter)

@app.route('/admin/case/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_case():
    if request.method == 'POST':
        customer_id = request.form.get('customer_id')
        customer_name = request.form.get('customer_name')
        invoice_number = request.form.get('invoice_number')
        amount = float(request.form.get('amount'))
        overdue_days = int(request.form.get('overdue_days'))
        region = request.form.get('region')
        notes = request.form.get('notes')
        auto_assign = request.form.get('auto_assign') == 'on'
        
        # Calculate priority and recovery probability
        priority = calculate_priority(overdue_days, amount)
        recovery_prob = calculate_recovery_probability(overdue_days, amount, region)
        
        # Generate case number
        case_number = f"FDX-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
        
        conn = get_db()
        cursor = conn.cursor()
        
        dca_id = None
        dca_name = None
        assigned_at = None
        sla_due_date = None
        status = 'NEW'
        
        if auto_assign:
            dca_id, dca_name = select_best_dca(priority, region)
            if dca_id:
                assigned_at = datetime.now()
                sla_due_date = (datetime.now() + timedelta(days=7 if priority == 'HIGH' else 14)).date()
                status = 'ASSIGNED'
        
        cursor.execute('''
            INSERT INTO cases (case_number, customer_id, customer_name, invoice_number,
                             amount, overdue_days, region, priority, recovery_probability,
                             status, dca_id, assigned_at, sla_due_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (case_number, customer_id, customer_name, invoice_number, amount,
              overdue_days, region, priority, recovery_prob, status, dca_id,
              assigned_at, sla_due_date, notes))
        
        case_id = cursor.lastrowid
        
        # Create assignment record
        if dca_id:
            cursor.execute('''
                INSERT INTO case_assignments (case_id, dca_id)
                VALUES (?, ?)
            ''', (case_id, dca_id))
            
            # Update DCA stats
            cursor.execute('''
                UPDATE dcas SET total_cases = total_cases + 1 WHERE id = ?
            ''', (dca_id,))
        
        # Log the creation
        cursor.execute('''
            INSERT INTO recovery_logs (case_id, dca_id, user_id, action, new_status, amount, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (case_id, dca_id, current_user.id, 'CREATED', status, amount,
              f'Case created. Auto-assigned to {dca_name}' if dca_id else 'Case created'))
        
        conn.commit()
        conn.close()
        
        flash(f'Case {case_number} created successfully!' + 
              (f' Assigned to {dca_name}' if dca_name else ''), 'success')
        return redirect(url_for('admin_cases'))
    
    return render_template('add_case.html')

@app.route('/admin/case/<int:case_id>')
@login_required
@admin_required
def view_case(case_id):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT c.*, d.name as dca_name FROM cases c
        LEFT JOIN dcas d ON c.dca_id = d.id
        WHERE c.id = ?
    ''', (case_id,))
    case = cursor.fetchone()
    
    if not case:
        flash('Case not found!', 'error')
        return redirect(url_for('admin_cases'))
    
    # Get case history
    cursor.execute('''
        SELECT rl.*, u.username FROM recovery_logs rl
        LEFT JOIN users u ON rl.user_id = u.id
        WHERE rl.case_id = ?
        ORDER BY rl.created_at DESC
    ''', (case_id,))
    history = cursor.fetchall()
    
    # Get DCAs for reassignment
    cursor.execute("SELECT id, name, performance_score FROM dcas ORDER BY performance_score DESC")
    dcas = cursor.fetchall()
    
    conn.close()
    
    return render_template('view_case.html', case=case, history=history, dcas=dcas)

@app.route('/admin/case/<int:case_id>/assign', methods=['POST'])
@login_required
@admin_required
def assign_case(case_id):
    dca_id = int(request.form.get('dca_id'))
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get case details
    cursor.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
    case = cursor.fetchone()
    
    if not case:
        flash('Case not found!', 'error')
        return redirect(url_for('admin_cases'))
    
    # Get DCA name
    cursor.execute("SELECT name FROM dcas WHERE id = ?", (dca_id,))
    dca = cursor.fetchone()
    
    old_dca_id = case['dca_id']
    sla_due_date = (datetime.now() + timedelta(days=7 if case['priority'] == 'HIGH' else 14)).date()
    
    # Update case
    cursor.execute('''
        UPDATE cases SET dca_id = ?, status = 'ASSIGNED', assigned_at = ?, 
                        sla_due_date = ?, updated_at = ?
        WHERE id = ?
    ''', (dca_id, datetime.now(), sla_due_date, datetime.now(), case_id))
    
    # Create assignment record
    cursor.execute('''
        INSERT INTO case_assignments (case_id, dca_id)
        VALUES (?, ?)
    ''', (case_id, dca_id))
    
    # Update DCA stats
    cursor.execute('''
        UPDATE dcas SET total_cases = total_cases + 1 WHERE id = ?
    ''', (dca_id,))
    
    # Log
    cursor.execute('''
        INSERT INTO recovery_logs (case_id, dca_id, user_id, action, old_status, new_status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (case_id, dca_id, current_user.id, 'ASSIGNED', case['status'], 'ASSIGNED',
          f'Assigned to {dca["name"]}'))
    
    conn.commit()
    conn.close()
    
    flash(f'Case assigned to {dca["name"]}!', 'success')
    return redirect(url_for('view_case', case_id=case_id))

@app.route('/admin/dcas')
@login_required
@admin_required
def admin_dcas():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT d.*, 
               (SELECT COUNT(*) FROM cases WHERE dca_id = d.id AND status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')) as active_cases
        FROM dcas d
        ORDER BY performance_score DESC
    ''')
    dcas = cursor.fetchall()
    
    conn.close()
    
    return render_template('admin_dcas.html', dcas=dcas)

@app.route('/admin/dca/add', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_add_dca():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        region = request.form.get('region', '').strip()
        contact_email = request.form.get('contact_email', '').strip()
        contact_phone = request.form.get('contact_phone', '').strip()
        
        # Validation
        if not name or not username or not password:
            flash('Agency Name, Username, and Password are required!', 'error')
            return redirect(request.url)
        
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'error')
            return redirect(request.url)
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            flash('Username already exists! Choose a different one.', 'error')
            conn.close()
            return redirect(request.url)
        
        # Create DCA agency record
        cursor.execute('''
            INSERT INTO dcas (name, region, performance_score, total_cases, recovered_cases, total_recovered_amount, sla_compliance_rate)
            VALUES (?, ?, 75.0, 0, 0, 0, 100.0)
        ''', (name, region))
        dca_id = cursor.lastrowid
        
        # Create user account for DCA
        password_hash = generate_password_hash(password)
        email = contact_email if contact_email else f'{username}@dca.com'
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role, dca_id)
            VALUES (?, ?, ?, 'dca', ?)
        ''', (username, email, password_hash, dca_id))
        
        conn.commit()
        conn.close()
        
        flash(f'DCA "{name}" created successfully! Login: {username} / {password}', 'success')
        return redirect(url_for('admin_dcas'))
    
    return render_template('admin_add_dca.html')

@app.route('/admin/dca/<int:dca_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_dca(dca_id):
    conn = get_db()
    cursor = conn.cursor()
    
    # Get DCA details
    cursor.execute("SELECT * FROM dcas WHERE id = ?", (dca_id,))
    dca = cursor.fetchone()
    
    if not dca:
        conn.close()
        flash('DCA not found!', 'error')
        return redirect(url_for('admin_dcas'))
    
    # Get associated user
    cursor.execute("SELECT * FROM users WHERE dca_id = ?", (dca_id,))
    user = cursor.fetchone()
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        region = request.form.get('region', '').strip()
        contact_email = request.form.get('contact_email', '').strip()
        contact_phone = request.form.get('contact_phone', '').strip()
        new_password = request.form.get('new_password', '').strip()
        
        if not name:
            flash('Agency Name is required!', 'error')
            conn.close()
            return redirect(request.url)
        
        # Update DCA record
        cursor.execute('''
            UPDATE dcas SET name = ?, region = ?
            WHERE id = ?
        ''', (name, region, dca_id))
        
        # Update user email if provided
        if user and contact_email:
            cursor.execute("UPDATE users SET email = ? WHERE id = ?", (contact_email, user['id']))
        
        # Update password if provided
        if user and new_password:
            if len(new_password) < 6:
                flash('Password must be at least 6 characters!', 'error')
                conn.close()
                return redirect(request.url)
            password_hash = generate_password_hash(new_password)
            cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user['id']))
            flash(f'Password updated for {user["username"]}!', 'success')
        
        conn.commit()
        conn.close()
        
        flash(f'DCA "{name}" updated successfully!', 'success')
        return redirect(url_for('admin_dcas'))
    
    conn.close()
    return render_template('admin_edit_dca.html', dca=dca, user=user)

@app.route('/admin/dca/<int:dca_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_dca(dca_id):
    conn = get_db()
    cursor = conn.cursor()
    
    # Get DCA details
    cursor.execute("SELECT name FROM dcas WHERE id = ?", (dca_id,))
    dca = cursor.fetchone()
    
    if not dca:
        conn.close()
        flash('DCA not found!', 'error')
        return redirect(url_for('admin_dcas'))
    
    # Check if DCA has active cases
    cursor.execute("SELECT COUNT(*) as cnt FROM cases WHERE dca_id = ? AND status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')", (dca_id,))
    active_cases = cursor.fetchone()['cnt']
    
    if active_cases > 0:
        conn.close()
        flash(f'Cannot delete DCA with {active_cases} active cases! Reassign or close them first.', 'error')
        return redirect(url_for('admin_dcas'))
    
    # Delete associated user
    cursor.execute("DELETE FROM users WHERE dca_id = ?", (dca_id,))
    
    # Delete DCA record
    cursor.execute("DELETE FROM dcas WHERE id = ?", (dca_id,))
    
    conn.commit()
    conn.close()
    
    flash(f'DCA "{dca["name"]}" deleted successfully!', 'success')
    return redirect(url_for('admin_dcas'))

@app.route('/admin/upload', methods=['GET', 'POST'])
@login_required
@admin_required
def upload_cases():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                
                required_cols = ['customer_id', 'amount', 'overdue_days']
                if not all(col in df.columns for col in required_cols):
                    flash('CSV must contain: customer_id, amount, overdue_days', 'error')
                    return redirect(request.url)
                
                conn = get_db()
                cursor = conn.cursor()
                
                added = 0
                auto_assign = request.form.get('auto_assign') == 'on'
                
                for _, row in df.iterrows():
                    customer_id = str(row['customer_id'])
                    amount = float(row['amount'])
                    overdue_days = int(row['overdue_days'])
                    customer_name = row.get('customer_name', '')
                    invoice_number = row.get('invoice_number', '')
                    region = row.get('region', '')
                    
                    priority = calculate_priority(overdue_days, amount)
                    recovery_prob = calculate_recovery_probability(overdue_days, amount, region)
                    case_number = f"FDX-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
                    
                    dca_id = None
                    status = 'NEW'
                    assigned_at = None
                    sla_due_date = None
                    
                    if auto_assign:
                        dca_id, _ = select_best_dca(priority, region)
                        if dca_id:
                            assigned_at = datetime.now()
                            sla_due_date = (datetime.now() + timedelta(days=7 if priority == 'HIGH' else 14)).date()
                            status = 'ASSIGNED'
                    
                    cursor.execute('''
                        INSERT INTO cases (case_number, customer_id, customer_name, invoice_number,
                                         amount, overdue_days, region, priority, recovery_probability,
                                         status, dca_id, assigned_at, sla_due_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (case_number, customer_id, customer_name, invoice_number, amount,
                          overdue_days, region, priority, recovery_prob, status, dca_id,
                          assigned_at, sla_due_date))
                    
                    if dca_id:
                        cursor.execute('''
                            UPDATE dcas SET total_cases = total_cases + 1 WHERE id = ?
                        ''', (dca_id,))
                    
                    added += 1
                
                conn.commit()
                conn.close()
                
                flash(f'Successfully imported {added} cases!', 'success')
                return redirect(url_for('admin_cases'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Please upload a CSV file!', 'error')
    
    return render_template('upload_cases.html')

# ============================================
# ROUTES - DCA DASHBOARD
# ============================================

@app.route('/dca/dashboard')
@login_required
def dca_dashboard():
    if current_user.role != 'dca':
        return redirect(url_for('admin_dashboard'))
    
    conn = get_db()
    cursor = conn.cursor()
    
    dca_id = current_user.dca_id
    
    # Get DCA info
    cursor.execute("SELECT * FROM dcas WHERE id = ?", (dca_id,))
    dca_info = cursor.fetchone()
    
    # Get statistics
    cursor.execute('''
        SELECT COUNT(*) as total FROM cases WHERE dca_id = ?
    ''', (dca_id,))
    total_cases = cursor.fetchone()['total']
    
    cursor.execute('''
        SELECT COUNT(*) as count FROM cases 
        WHERE dca_id = ? AND status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
    ''', (dca_id,))
    active_cases = cursor.fetchone()['count']
    
    cursor.execute('''
        SELECT COUNT(*) as count FROM cases 
        WHERE dca_id = ? AND status = 'RECOVERED'
    ''', (dca_id,))
    recovered_cases = cursor.fetchone()['count']
    
    cursor.execute('''
        SELECT SUM(recovery_amount) as total FROM cases 
        WHERE dca_id = ? AND status = 'RECOVERED'
    ''', (dca_id,))
    recovered_amount = cursor.fetchone()['total'] or 0
    
    # Get pending cases by priority
    cursor.execute('''
        SELECT priority, COUNT(*) as count FROM cases 
        WHERE dca_id = ? AND status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
        GROUP BY priority
    ''', (dca_id,))
    priority_data = cursor.fetchall()
    
    # Get cases nearing SLA
    today = datetime.now().date()
    cursor.execute('''
        SELECT * FROM cases 
        WHERE dca_id = ? AND status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
        AND sla_due_date IS NOT NULL
        ORDER BY sla_due_date ASC
        LIMIT 10
    ''', (dca_id,))
    urgent_cases = cursor.fetchall()
    
    conn.close()
    
    return render_template('dca_dashboard.html',
        dca_info=dca_info,
        total_cases=total_cases,
        active_cases=active_cases,
        recovered_cases=recovered_cases,
        recovered_amount=recovered_amount,
        priority_data=priority_data,
        urgent_cases=urgent_cases
    )

@app.route('/dca/cases')
@login_required
def dca_cases():
    if current_user.role != 'dca':
        return redirect(url_for('admin_cases'))
    
    conn = get_db()
    cursor = conn.cursor()
    
    status_filter = request.args.get('status', '')
    priority_filter = request.args.get('priority', '')
    
    query = '''
        SELECT * FROM cases WHERE dca_id = ?
    '''
    params = [current_user.dca_id]
    
    if status_filter:
        query += " AND status = ?"
        params.append(status_filter)
    if priority_filter:
        query += " AND priority = ?"
        params.append(priority_filter)
    
    query += " ORDER BY sla_due_date ASC, priority DESC"
    
    cursor.execute(query, params)
    cases = cursor.fetchall()
    
    conn.close()
    
    return render_template('dca_cases.html', cases=cases,
                          status_filter=status_filter, priority_filter=priority_filter)

@app.route('/dca/case/<int:case_id>', methods=['GET', 'POST'])
@login_required
def dca_view_case(case_id):
    if current_user.role != 'dca':
        return redirect(url_for('view_case', case_id=case_id))
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM cases WHERE id = ? AND dca_id = ?
    ''', (case_id, current_user.dca_id))
    case = cursor.fetchone()
    
    if not case:
        flash('Case not found or access denied!', 'error')
        return redirect(url_for('dca_cases'))
    
    if request.method == 'POST':
        new_status = request.form.get('status')
        recovery_amount = request.form.get('recovery_amount', 0)
        notes = request.form.get('notes', '')
        
        old_status = case['status']
        
        update_fields = ['status = ?', 'updated_at = ?', 'notes = ?']
        update_params = [new_status, datetime.now(), notes]
        
        if new_status == 'RECOVERED' and recovery_amount:
            recovery_amount = float(recovery_amount)
            update_fields.append('recovery_amount = ?')
            update_fields.append('closed_at = ?')
            update_params.extend([recovery_amount, datetime.now()])
            
            # Update DCA stats
            cursor.execute('''
                UPDATE dcas SET 
                    recovered_cases = recovered_cases + 1,
                    total_recovered_amount = total_recovered_amount + ?,
                    performance_score = MIN(100, performance_score + 2)
                WHERE id = ?
            ''', (recovery_amount, current_user.dca_id))
        
        update_params.append(case_id)
        
        cursor.execute(f'''
            UPDATE cases SET {', '.join(update_fields)} WHERE id = ?
        ''', update_params)
        
        # Log the update
        cursor.execute('''
            INSERT INTO recovery_logs (case_id, dca_id, user_id, action, old_status, new_status, amount, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (case_id, current_user.dca_id, current_user.id, 'STATUS_UPDATE', 
              old_status, new_status, recovery_amount if new_status == 'RECOVERED' else None, notes))
        
        conn.commit()
        
        flash('Case updated successfully!', 'success')
        return redirect(url_for('dca_view_case', case_id=case_id))
    
    # Get case history
    cursor.execute('''
        SELECT rl.*, u.username FROM recovery_logs rl
        LEFT JOIN users u ON rl.user_id = u.id
        WHERE rl.case_id = ?
        ORDER BY rl.created_at DESC
    ''', (case_id,))
    history = cursor.fetchall()
    
    conn.close()
    
    return render_template('dca_view_case.html', case=case, history=history)

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/dashboard/stats')
@login_required
def api_dashboard_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    if current_user.role == 'admin':
        cursor.execute("SELECT COUNT(*) as total FROM cases")
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as count FROM cases WHERE status = 'RECOVERED'")
        recovered = cursor.fetchone()['count']
        
        cursor.execute("SELECT SUM(amount) as total FROM cases")
        total_amount = cursor.fetchone()['total'] or 0
        
        cursor.execute("SELECT SUM(recovery_amount) as total FROM cases WHERE status = 'RECOVERED'")
        recovered_amount = cursor.fetchone()['total'] or 0
    else:
        cursor.execute("SELECT COUNT(*) as total FROM cases WHERE dca_id = ?", (current_user.dca_id,))
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as count FROM cases WHERE dca_id = ? AND status = 'RECOVERED'", (current_user.dca_id,))
        recovered = cursor.fetchone()['count']
        
        cursor.execute("SELECT SUM(amount) as total FROM cases WHERE dca_id = ?", (current_user.dca_id,))
        total_amount = cursor.fetchone()['total'] or 0
        
        cursor.execute("SELECT SUM(recovery_amount) as total FROM cases WHERE dca_id = ? AND status = 'RECOVERED'", (current_user.dca_id,))
        recovered_amount = cursor.fetchone()['total'] or 0
    
    conn.close()
    
    return jsonify({
        'total_cases': total,
        'recovered_cases': recovered,
        'total_amount': total_amount,
        'recovered_amount': recovered_amount,
        'recovery_rate': round((recovered_amount / total_amount * 100), 2) if total_amount > 0 else 0
    })

@app.route('/api/chart/priority')
@login_required
def api_chart_priority():
    conn = get_db()
    cursor = conn.cursor()
    
    if current_user.role == 'admin':
        cursor.execute('''
            SELECT priority, COUNT(*) as count FROM cases 
            WHERE status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
            GROUP BY priority
        ''')
    else:
        cursor.execute('''
            SELECT priority, COUNT(*) as count FROM cases 
            WHERE dca_id = ? AND status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
            GROUP BY priority
        ''', (current_user.dca_id,))
    
    data = cursor.fetchall()
    conn.close()
    
    return jsonify([{'priority': row['priority'], 'count': row['count']} for row in data])

@app.route('/api/chart/aging')
@login_required
def api_chart_aging():
    conn = get_db()
    cursor = conn.cursor()
    
    query = '''
        SELECT 
            CASE 
                WHEN overdue_days <= 30 THEN '0-30'
                WHEN overdue_days <= 60 THEN '31-60'
                WHEN overdue_days <= 90 THEN '61-90'
                ELSE '90+'
            END as bucket,
            COUNT(*) as count,
            SUM(amount) as total_amount
        FROM cases 
        WHERE status NOT IN ('RECOVERED', 'CLOSED', 'WRITTEN_OFF')
    '''
    
    if current_user.role != 'admin':
        query = query.replace('WHERE', f'WHERE dca_id = {current_user.dca_id} AND')
    
    query += ' GROUP BY bucket ORDER BY MIN(overdue_days)'
    
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    
    return jsonify([{'bucket': row['bucket'], 'count': row['count'], 'amount': row['total_amount']} for row in data])

@app.route('/api/case/<int:case_id>')
@login_required
def api_get_case(case_id):
    """API endpoint to get case details for the drawer/modal"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get case details with DCA name
    cursor.execute('''
        SELECT c.*, d.name as dca_name FROM cases c
        LEFT JOIN dcas d ON c.dca_id = d.id
        WHERE c.id = ?
    ''', (case_id,))
    case = cursor.fetchone()
    
    if not case:
        conn.close()
        return jsonify({'error': 'Case not found'}), 404
    
    # Check access permissions for DCA users
    if current_user.role == 'dca' and case['dca_id'] != current_user.dca_id:
        conn.close()
        return jsonify({'error': 'Access denied'}), 403
    
    # Get case history
    cursor.execute('''
        SELECT rl.*, u.username FROM recovery_logs rl
        LEFT JOIN users u ON rl.user_id = u.id
        WHERE rl.case_id = ?
        ORDER BY rl.created_at DESC
    ''', (case_id,))
    history = cursor.fetchall()
    
    conn.close()
    
    return jsonify({
        'case': dict(case),
        'history': [dict(h) for h in history]
    })

# ============================================
# GENERATE SAMPLE DATA
# ============================================

@app.route('/admin/generate-sample-data')
@login_required
@admin_required
def generate_sample_data():
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if data already exists
    cursor.execute("SELECT COUNT(*) as cnt FROM cases")
    if cursor.fetchone()['cnt'] > 0:
        flash('Sample data already exists!', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    # Generate 50 sample cases
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    cursor.execute("SELECT id FROM dcas")
    dca_ids = [row['id'] for row in cursor.fetchall()]
    
    import random
    for i in range(50):
        customer_id = f"CUST-{1000 + i}"
        customer_name = f"Customer {i+1}"
        invoice_number = f"INV-{2024000 + i}"
        amount = float(random.choice([5000, 10000, 25000, 50000, 75000, 100000, 150000]))
        overdue_days = int(random.choice([15, 30, 45, 60, 75, 90, 120, 150]))
        region = random.choice(regions)
        
        priority = calculate_priority(overdue_days, amount)
        recovery_prob = float(calculate_recovery_probability(overdue_days, amount, region))
        case_number = f"FDX-{datetime.now().strftime('%Y%m%d')}-{1000 + i}"
        
        # Randomly assign some cases
        if random.random() > 0.3:
            dca_id = int(random.choice(dca_ids))
            status = random.choice(['ASSIGNED', 'IN_PROGRESS', 'CONTACTED', 'PROMISE_TO_PAY'])
            days_offset = random.randint(1, 30)
            assigned_at = (datetime.now() - timedelta(days=days_offset)).isoformat()
            sla_due_date = (datetime.now() - timedelta(days=days_offset) + timedelta(days=7 if priority == 'HIGH' else 14)).date().isoformat()
        else:
            dca_id = None
            status = 'NEW'
            assigned_at = None
            sla_due_date = None
        
        # Some recovered cases
        recovery_amount = 0.0
        closed_at = None
        if random.random() > 0.8:
            status = 'RECOVERED'
            recovery_amount = float(amount * random.uniform(0.5, 1.0))
            closed_at = (datetime.now() - timedelta(days=random.randint(1, 10))).isoformat()
        
        cursor.execute('''
            INSERT INTO cases (case_number, customer_id, customer_name, invoice_number,
                             amount, overdue_days, region, priority, recovery_probability,
                             status, dca_id, assigned_at, sla_due_date, recovery_amount, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (case_number, customer_id, customer_name, invoice_number, amount,
              overdue_days, region, priority, recovery_prob, status, dca_id,
              assigned_at, sla_due_date, recovery_amount, closed_at))
        
        if dca_id:
            cursor.execute('''
                UPDATE dcas SET total_cases = total_cases + 1 WHERE id = ?
            ''', (dca_id,))
            if status == 'RECOVERED':
                cursor.execute('''
                    UPDATE dcas SET 
                        recovered_cases = recovered_cases + 1,
                        total_recovered_amount = total_recovered_amount + ?
                    WHERE id = ?
                ''', (recovery_amount, dca_id))
    
    conn.commit()
    conn.close()
    
    flash('Sample data generated successfully! 50 cases added.', 'success')
    return redirect(url_for('admin_dashboard'))

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
