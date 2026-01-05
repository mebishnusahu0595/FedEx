# 🚚 FedEx DCA Recovery Platform

> An AI-powered Debt Collection Agency management system designed to transform how FedEx handles overdue customer accounts.

Built for the **FedEx SMART Hackathon 2025** @ IIT Madras

---

## � Quick Note for Evaluators

This repository contains:
- ✅ A **fully working prototype** with UI
- ✅ **AI-based prioritization** and DCA assignment logic
- ✅ **End-to-end data pipeline** implementation

> 📌 For a high-level overview, please refer to the **PPT submission**.  
> This README provides detailed technical documentation for reference.

---
## 🎥 Demo Video

Check out the platform in action:

<video src="VN20260105_100551.mp4" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

*(Note: If the video doesn't play, you can find it in the root directory as `VN20260105_100551.mp4`)*

---
## �📌 The Problem We're Solving

FedEx currently manages thousands of overdue customer accounts through external Debt Collection Agencies. But here's the catch — the entire process runs on spreadsheets, emails, and endless follow-ups. As the volume grows, so do the problems:

- **No centralized system** — Everything scattered across Excel files
- **Slow feedback loops** — Days pass before teams know what's happening
- **Zero accountability** — Who owns what? Nobody really knows
- **Blind spots everywhere** — No real visibility into DCA performance

We built this platform to fix all of that.

---

## 💡 Our Solution

A complete digital platform that brings intelligence to debt recovery:

### What It Does

| Feature | How It Helps |
|---------|--------------|
| **Smart Case Assignment** | AI picks the best DCA based on performance, workload & region |
| **Real-time Dashboards** | No more waiting for weekly reports — see everything live |
| **Performance Tracking** | Know exactly which agencies deliver and which don't |
| **SLA Monitoring** | Automatic alerts before deadlines are missed |
| **Secure DCA Portals** | Each agency gets their own login to manage assigned cases |

---

## 🏗️ How We Built It

### Tech Stack

```
Frontend    →  HTML5, CSS3, Vanilla JavaScript, Chart.js
Backend     →  Python Flask
Database    →  SQLite
Auth        →  Flask-Login with password hashing
AI/ML       →  Custom scoring algorithms for priority & assignment
```

### Project Structure

```
FedEx/
├── app.py                 # Main application (Flask routes, AI logic)
├── requirements.txt       # Python dependencies
├── fedex_dca.db          # SQLite database
├── static/
│   └── css/
│       └── style.css     # All styling
├── templates/
│   ├── index.html        # Landing page
│   ├── login.html        # Authentication
│   ├── admin_dashboard.html
│   ├── admin_cases.html
│   ├── admin_dcas.html
│   ├── admin_add_dca.html
│   ├── admin_edit_dca.html
│   ├── add_case.html
│   ├── upload_cases.html
│   ├── view_case.html
│   ├── dca_dashboard.html
│   ├── dca_cases.html
│   └── dca_view_case.html
└── data/
    └── sample_cases.csv  # Sample data for testing
```

---

## 🤖 AI/ML Model

This is where the intelligence lives. We've built **three core AI models** that power the entire platform:

### Model 1: Case Priority Scoring Engine

**Purpose:** Automatically determine which cases need immediate attention.

**How it works:**
```python
def calculate_priority_score(case):
    """
    Multi-factor weighted scoring algorithm
    Returns: Priority level (HIGH / MEDIUM / LOW) + numeric score
    """
    
    # Factor 1: Amount Impact (40% weight)
    amount = case['amount']
    if amount > 100000:
        amount_score = 100
    elif amount > 50000:
        amount_score = 75
    elif amount > 20000:
        amount_score = 50
    else:
        amount_score = 25
    
    # Factor 2: Ageing Impact (30% weight)
    days_overdue = case['days_overdue']
    age_score = min(100, days_overdue * 0.8)
    
    # Factor 3: Customer Segment (20% weight)
    segment_weights = {
        'enterprise': 100,
        'sme': 70,
        'individual': 40
    }
    segment_score = segment_weights.get(case['segment'], 50)
    
    # Factor 4: Payment History (10% weight)
    history_score = 100 if case['missed_payments'] > 2 else 50
    
    # Final weighted calculation
    final_score = (
        amount_score * 0.40 +
        age_score * 0.30 +
        segment_score * 0.20 +
        history_score * 0.10
    )
    
    # Classification
    if final_score >= 70:
        return 'HIGH', final_score
    elif final_score >= 40:
        return 'MEDIUM', final_score
    else:
        return 'LOW', final_score
```

**Input Features:**
| Feature | Type | Description |
|---------|------|-------------|
| `amount` | Float | Outstanding debt amount in INR |
| `days_overdue` | Integer | Days since payment was due |
| `customer_segment` | Category | enterprise / sme / individual |
| `missed_payments` | Integer | Historical missed payment count |

**Output:** Priority label (HIGH/MEDIUM/LOW) + Score (0-100)

---

### Model 2: Intelligent DCA Assignment Algorithm

**Purpose:** Match each case to the optimal Debt Collection Agency.

**How it works:**
```python
def find_best_dca(case, available_dcas):
    """
    Multi-criteria DCA selection algorithm
    Considers: Performance, Capacity, Region Expertise
    """
    
    scores = []
    
    for dca in available_dcas:
        # Performance Score (50% weight)
        # Based on historical recovery rate and SLA compliance
        perf_score = dca['performance_score']  # 0-100
        
        # Capacity Score (30% weight)
        # DCAs with fewer active cases get higher scores
        active_ratio = dca['active_cases'] / dca['max_capacity']
        capacity_score = (1 - active_ratio) * 100
        
        # Region Match Score (20% weight)
        # Bonus if DCA specializes in case's region
        region_score = 100 if dca['region'] == case['region'] else 50
        
        # Final composite score
        total = (
            perf_score * 0.50 +
            capacity_score * 0.30 +
            region_score * 0.20
        )
        
        scores.append((dca, total))
    
    # Return DCA with highest score
    return max(scores, key=lambda x: x[1])[0]
```

**Selection Criteria:**
| Factor | Weight | Why It Matters |
|--------|--------|----------------|
| Performance Score | 50% | Better agencies = higher recovery |
| Available Capacity | 30% | Don't overload high performers |
| Region Expertise | 20% | Local knowledge improves results |

---

### Model 3: Recovery Probability Predictor

**Purpose:** Estimate likelihood of successful debt recovery.

**Algorithm:**
```python
def predict_recovery_probability(case):
    """
    Logistic-regression inspired probability estimation
    """
    
    # Base probability
    base_prob = 0.50
    
    # Adjust based on factors
    adjustments = 0
    
    # Amount factor (smaller = easier to recover)
    if case['amount'] < 25000:
        adjustments += 0.15
    elif case['amount'] > 100000:
        adjustments -= 0.10
    
    # Age factor (older = harder)
    if case['days_overdue'] < 30:
        adjustments += 0.20
    elif case['days_overdue'] > 90:
        adjustments -= 0.20
    
    # Customer type
    if case['segment'] == 'enterprise':
        adjustments += 0.10  # More predictable
    
    # Contact availability
    if case['has_valid_contact']:
        adjustments += 0.15
    
    probability = min(0.95, max(0.10, base_prob + adjustments))
    
    return round(probability * 100, 1)  # Return as percentage
```

---

## 🔄 Data Pipeline

Here's how data flows through our system — from case creation to recovery:

### End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │   INPUT      │     │  PROCESSING  │     │   AI/ML      │     │   OUTPUT     │
  │   LAYER      │────▶│   LAYER      │────▶│   LAYER      │────▶│   LAYER      │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
        │                     │                    │                    │
        ▼                     ▼                    ▼                    ▼
   • Manual Entry        • Validation         • Priority Score     • Dashboard
   • CSV Upload          • Normalization      • DCA Assignment     • Reports
   • API (future)        • Deduplication      • Recovery Pred.     • Alerts
```

### Pipeline Stages Explained

#### Stage 1: Data Ingestion
```python
# Single case entry via form
@app.route('/admin/case/add', methods=['POST'])
def add_case():
    case_data = {
        'customer_name': request.form['customer_name'],
        'amount': float(request.form['amount']),
        'due_date': request.form['due_date'],
        'region': request.form['region'],
        # ... more fields
    }
    return process_case(case_data)

# Bulk upload via CSV
@app.route('/admin/cases/upload', methods=['POST'])
def upload_cases():
    csv_file = request.files['file']
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        process_case(row.to_dict())
```

#### Stage 2: Data Validation & Cleaning
```python
def validate_case(case):
    """Ensure data quality before processing"""
    
    # Required fields check
    required = ['customer_name', 'amount', 'due_date']
    for field in required:
        if not case.get(field):
            raise ValidationError(f"Missing: {field}")
    
    # Data type validation
    case['amount'] = float(case['amount'])
    case['due_date'] = parse_date(case['due_date'])
    
    # Normalize region names
    case['region'] = case['region'].strip().title()
    
    return case
```

#### Stage 3: AI Processing
```python
def process_case(case_data):
    """Main processing pipeline"""
    
    # Step 1: Validate
    case = validate_case(case_data)
    
    # Step 2: Calculate priority (Model 1)
    priority, score = calculate_priority_score(case)
    case['priority'] = priority
    case['priority_score'] = score
    
    # Step 3: Find best DCA (Model 2)
    available_dcas = get_active_dcas()
    best_dca = find_best_dca(case, available_dcas)
    case['assigned_dca_id'] = best_dca['id']
    
    # Step 4: Predict recovery (Model 3)
    case['recovery_probability'] = predict_recovery_probability(case)
    
    # Step 5: Save to database
    save_to_database(case)
    
    # Step 6: Trigger notifications
    notify_dca(best_dca, case)
    
    return case
```

#### Stage 4: Real-time Updates
```python
# Status update flow
def update_case_status(case_id, new_status, notes):
    """Track case lifecycle"""
    
    case = get_case(case_id)
    old_status = case['status']
    
    # Update case
    case['status'] = new_status
    case['last_updated'] = datetime.now()
    case['history'].append({
        'from': old_status,
        'to': new_status,
        'notes': notes,
        'timestamp': datetime.now()
    })
    
    # Recalculate metrics
    update_dca_metrics(case['dca_id'])
    
    # Check SLA compliance
    check_sla_breach(case)
    
    save_case(case)
```

### Data Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Admin     │    │   Flask     │    │   SQLite    │    │    DCA      │
│   Portal    │    │   Backend   │    │   Database  │    │   Portal    │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       │  Create Case     │                  │                  │
       │─────────────────▶│                  │                  │
       │                  │  Validate        │                  │
       │                  │  + AI Scoring    │                  │
       │                  │─────────────────▶│                  │
       │                  │                  │  Store           │
       │                  │◀─────────────────│                  │
       │                  │                  │                  │
       │                  │  Notify DCA      │                  │
       │                  │─────────────────────────────────────▶
       │                  │                  │                  │
       │                  │                  │    Update Status │
       │                  │◀─────────────────────────────────────
       │                  │                  │                  │
       │  Dashboard       │   Refresh        │                  │
       │◀─────────────────│◀─────────────────│                  │
       │                  │                  │                  │
```

---

## 🖥️ Working UI (User Interface)

Our platform has a **fully functional web interface** — not just mockups!

### How to Access

```bash
# Start the server
python app.py

# Open in browser
http://127.0.0.1:5000
```

### UI Components

#### 1. Landing Page (`/`)
- Modern hero section with platform overview
- Quick navigation to login
- Feature highlights with animations
- Responsive design

#### 2. Login Page (`/login`)
- Secure authentication form
- Role-based routing (Admin vs DCA)
- Password visibility toggle
- Flash messages for errors

#### 3. Admin Dashboard (`/admin/dashboard`)
| Component | Description |
|-----------|-------------|
| Stats Cards | Total cases, recovered amount, recovery rate |
| Recovery Trend Chart | Line chart showing monthly performance |
| Ageing Buckets | Bar chart of cases by overdue days |
| Recent Cases | Quick view of latest entries |
| Top Performers | DCA leaderboard |

#### 4. Case Management (`/admin/cases`)
- Searchable, filterable case list
- Status badges with colors
- Priority indicators
- Quick actions (View, Edit, Assign)
- Bulk CSV upload option

#### 5. DCA Management (`/admin/dcas`)
- Agency cards with performance scores
- Add new DCA with portal credentials
- Edit agency details
- View assigned cases per agency

#### 6. DCA Portal (`/dca/dashboard`)
- Agency-specific view (only their cases)
- Performance score gauge
- Active/Recovered/Pending stats
- Urgent SLA cases highlighted
- Case update functionality

### UI Technology Stack

| Layer | Technology |
|-------|------------|
| Structure | HTML5, Jinja2 Templates |
| Styling | Custom CSS (no frameworks) |
| Interactivity | Vanilla JavaScript |
| Charts | Chart.js |
| Icons | Font Awesome 6 |
| Design | Glassmorphism, Dark mode, Gradients |

### Demo Credentials

| Role | Username | Password | What You'll See |
|------|----------|----------|-----------------|
| Admin | `admin` | `admin123` | Full platform access |
| DCA Agent | `recovery_solutions_inc` | `dca123` | Agency-specific portal |

### UI Screenshots Preview

The UI includes:
- ✅ Modern dark theme with FedEx purple/orange branding
- ✅ Glassmorphism card effects
- ✅ Animated stat counters
- ✅ Interactive charts with hover effects
- ✅ Responsive sidebar navigation
- ✅ Flash messages for feedback
- ✅ Form validation with live preview

---

## 🎯 Key Features Walkthrough

### For Admins

1. **Dashboard** — Bird's eye view of all metrics
2. **Case Management** — Add single cases or bulk upload via CSV
3. **DCA Management** — Full CRUD operations for agencies
4. **Analytics** — Recovery trends, ageing buckets, performance comparisons

### For DCAs

1. **Dedicated Portal** — Separate login, separate interface
2. **Assigned Cases** — Only see what's assigned to them
3. **Status Updates** — Update case progress in real-time
4. **Performance Stats** — Track their own metrics

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repo
git clone https://github.com/mebishnusahu0595/FedEx.git
cd FedEx

# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Access the Platform

Once running, open your browser:

| URL | Purpose |
|-----|---------|
| `http://127.0.0.1:5000` | Landing page |
| `http://127.0.0.1:5000/login` | Login page |

### Demo Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| DCA Agent | `recovery_solutions_inc` | `dca123` |

---

## 📊 KPIs We Track

| Metric | What It Measures |
|--------|------------------|
| **Recovery Rate** | Percentage of cases successfully closed |
| **SLA Compliance** | Cases resolved within target timeline |
| **Performance Score** | Composite score for each DCA (0-100) |
| **Average Resolution Time** | Days taken to close cases |
| **Ageing Distribution** | Cases in 30/60/90/120+ day buckets |

---

## 📸 Screenshots

### Landing Page
The entry point with clear navigation and modern design.

### Admin Dashboard
Complete overview with charts, metrics, and quick actions.

### DCA Portal
Focused interface for agency-specific case management.

---

## 🔐 Security Measures

- Password hashing using Werkzeug
- Session-based authentication
- Role-based access control (Admin vs DCA)
- Input validation and sanitization

---

## 🎯 Impact & Value Proposition

| Before | After |
|--------|-------|
| Manual Excel tracking | Centralized digital platform |
| Email-based coordination | Real-time status updates |
| Monthly performance reviews | Live dashboards |
| Reactive escalations | Proactive SLA alerts |
| Subjective DCA selection | AI-driven assignment |

### Expected Outcomes

- **30-40% reduction** in average case resolution time
- **Higher recovery rates** through intelligent prioritization
- **Complete audit trail** for compliance and governance
- **Scalable architecture** ready for production

---

## 🛠️ Future Enhancements

Things we'd love to add with more time:

- [ ] SMS/Email notifications for DCA agents
- [ ] Advanced ML model with historical data training
- [ ] Integration with payment gateways
- [ ] Mobile-responsive PWA
- [ ] Multi-language support
- [ ] RPA bots for legacy system sync

---

## 👥 Team

**Team Name:** [Your Team Name]

| Name | Role | Contact |
|------|------|---------|
| [Member 1] | Lead Developer | email@example.com |
| [Member 2] | UI/UX Design | email@example.com |
| [Member 3] | ML/Backend | email@example.com |

---

## 📝 License

This project was built for the FedEx SMART Hackathon 2025. All rights reserved.

---

## 🙏 Acknowledgments

- FedEx for presenting this challenge
- IIT Madras for hosting the event
- The open-source community for the amazing tools

---

*Built with ☕ and determination*

