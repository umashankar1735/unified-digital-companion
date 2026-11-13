# UDC System Frontend Documentation

## Overview
Academic/Government-style frontend for the Universal Digital Campus (UDC) system, following strict design specifications for function over aesthetics.

## Design Principles
- **Function over aesthetics**: Predictable layouts, clear hierarchy
- **No decorative elements**: No gradients, glass effects, or unnecessary animations
- **Typography**: Roboto font family for clean, professional appearance
- **Color Palette**:
  - Primary: Dark Blue (#0A3D62)
  - Secondary: Navy (#1B2A49)
  - Background: Light Gray (#F5F6FA)
  - Border: #DADCE0
  - Success: #28A745
  - Warning: #FFC107
  - Error: #DC3545

## Pages Structure

### 1. Dashboard (index.html)
- **Purpose**: Main overview page with system metrics
- **Components**:
  - Header with project title and logout
  - Horizontal navigation (7 tabs)
  - 4 metric summary cards
  - Module status table
  - Performance comparison table
  - System information grid
  - Quick action buttons

### 2. Legal Awareness Module (legal-module.html)
- **Purpose**: Legal QA interface
- **Components**:
  - Query form (textarea + optional category)
  - Criminal keyword warning system
  - Results table (rank, question, answer, category, confidence)
  - Module statistics (9,763 pairs, EM=99.6%, F1=99.7%)
  - Dataset breakdown (IPC, IndicLegalQA, Ace Attorney)

### 3. University Support Module (university-module.html)
- **Purpose**: Campus support QA interface
- **Components**:
  - Query form with 6 category options
  - Hybrid retrieval results display
  - Color-coded relevance badges
  - Module statistics (503 pairs, EM=66.5%, F1=77.6%)
  - Architecture details (BM25, FAISS, CrossEncoder)
  - Dataset distribution table

### 4. Analysis (analysis.html)
- **Purpose**: Performance metrics and evaluation results
- **Components**:
  - Table 1: Comparison with baselines (TF-IDF+BERT, Dense+Extractive, LLM vs UDC)
  - Table 2: Intent routing performance
  - Module-specific metrics tables
  - Technical implementation details
  - Key findings alerts
  - Export functionality (JSON download)

### 5. Reports (reports.html)
- **Purpose**: Generate and download evaluation reports
- **Components**:
  - Quick download buttons (PDF, CSV, ZIP)
  - Legal module report section
  - University module report section
  - Intent routing metrics
  - Published research papers table
  - Report generation history
  - Viva presentation materials

### 6. Settings (settings.html)
- **Purpose**: System configuration and technical specifications
- **Components**:
  - System information table
  - Technology stack grid
  - Legal module configuration form (Top-K, confidence, max length)
  - University module settings (hybrid weights, thresholds)
  - API endpoints status table
  - Performance settings (batch size, timeout, caching)
  - Dataset statistics
  - About project section
  - Maintenance actions

### 7. Help (help.html)
- **Purpose**: User guides and documentation
- **Components**:
  - Getting started guide
  - User manual (step-by-step instructions)
  - FAQ (8 common questions)
  - Troubleshooting table
  - System architecture diagram (ASCII art)
  - Contact information
  - Additional resources (paper, docs, presentation)

## Shared Resources

### styles.css (476 lines)
Complete stylesheet following academic design system:
- CSS variables for color palette
- Roboto font family
- Header (70px fixed height)
- Horizontal navigation tabs with active state
- Data tables with alternating row hover
- Forms with labels above fields
- Rectangular solid buttons
- Status badges (success/warning/error)
- Alerts with icons
- Modals for confirmations
- Footer
- Responsive design (@media max-width: 768px)

### script.js (125 lines)
Utility functions for all pages:
- `logout()`: Confirmation modal
- `validateForm()`: Required field validation
- `showAlert()`: Dynamic alert creation
- `openModal()` / `closeModal()`: Modal management
- `exportTableToCSV()`: CSV download
- `showLoading()` / `hideLoading()`: Loading overlay
- `formatDate()`: Indian locale formatting
- `printPage()`: Print functionality

## Current Status

### ✅ Completed
- All 7 pages created with consistent design
- Academic/government styling applied
- Navigation between pages functional
- Metric cards, tables, forms, buttons implemented
- Color palette and typography consistent
- Responsive design for mobile
- JavaScript utilities functional

### ⏳ Pending (Backend Integration)
- Connect legal-module.html to Flask API (http://localhost:5000)
- Connect university-module.html to FastAPI (http://localhost:8000)
- Replace setTimeout simulations with actual fetch() calls
- Implement real PDF/CSV downloads
- Add error handling for API failures
- Enable real-time result streaming

## API Integration Guide

### Legal Module API Call Example
```javascript
async function submitLegalQuery() {
    const query = document.getElementById('legalQuery').value;
    const category = document.getElementById('legalCategory').value;
    
    showLoading();
    
    try {
        const response = await fetch('http://localhost:5000/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, category })
        });
        
        const data = await response.json();
        displayResults(data.results);
    } catch (error) {
        showAlert('API Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}
```

### University Module API Call Example
```javascript
async function submitCampusQuery() {
    const query = document.getElementById('campusQuery').value;
    const category = document.getElementById('campusCategory').value;
    
    showLoading();
    
    try {
        const response = await fetch('http://localhost:8000/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, category })
        });
        
        const data = await response.json();
        displayCampusResults(data.results);
    } catch (error) {
        showAlert('API Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}
```

## Running the Frontend

### Option 1: Simple HTTP Server (Python)
```bash
cd frontend
python3 -m http.server 8080
# Open http://localhost:8080 in browser
```

### Option 2: Node.js HTTP Server
```bash
cd frontend
npx http-server -p 8080
# Open http://localhost:8080 in browser
```

### Option 3: VS Code Live Server Extension
1. Install "Live Server" extension in VS Code
2. Right-click index.html
3. Select "Open with Live Server"

## File Structure
```
frontend/
├── index.html              (Dashboard - 185 lines)
├── legal-module.html       (Legal QA - 195 lines)
├── university-module.html  (Campus Support - 238 lines)
├── analysis.html           (Performance Metrics - 320 lines)
├── reports.html            (Report Downloads - 380 lines)
├── settings.html           (Configuration - 415 lines)
├── help.html               (Documentation - 440 lines)
├── styles.css              (Shared Styles - 476 lines)
├── script.js               (Utilities - 125 lines)
└── README.md               (This file)
```

## Key Features

### Component Consistency
- All pages use same header/navigation/footer
- Consistent table styling across pages
- Uniform button appearance (primary/secondary)
- Standardized form layouts
- Identical alert/modal styling

### Accessibility
- Clear labels for all form fields
- Required field indicators (*)
- Keyboard navigation support
- High contrast color scheme
- Readable font sizes (16px base)

### Performance Considerations
- No heavy JavaScript libraries
- Minimal CSS animations
- Efficient DOM manipulation
- Lazy loading where applicable
- Optimized image assets

## Browser Compatibility
- Chrome/Edge: ✅ Fully supported
- Firefox: ✅ Fully supported
- Safari: ✅ Fully supported
- Mobile browsers: ✅ Responsive design

## Future Enhancements
1. Real-time API integration
2. WebSocket support for streaming responses
3. Advanced data visualizations (Chart.js)
4. PDF report generation (jsPDF)
5. User authentication system
6. Query history tracking
7. Export to Excel functionality
8. Dark mode toggle

## Notes for Viva Presentation
- Highlight academic design approach (function over form)
- Demonstrate both modules working
- Show evaluation metrics matching paper results
- Explain hybrid retrieval architecture
- Discuss zero hallucination achievement
- Present performance comparison with baselines

## Contact
Developer: Umashankar  
Project: Final Year B.Tech (CSE)  
System: Universal Digital Campus (UDC)  
Conference Paper: CRC_587.pdf
