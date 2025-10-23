# Visualization Guide

This guide explains how to use the new `main_with_viz.py` file to run the intersection optimization with your choice of visualization.

## Quick Start

### 1. Choose Your Visualization Method

Open `main_with_viz.py` and modify the `VISUALIZATION_METHOD` variable (around line 21):

```python
VISUALIZATION_METHOD = 'web'  # Options: 'matplotlib', 'web', or 'none'
```

### 2. Run the Program

```bash
python main_with_viz.py
```

## Visualization Options

### Option 1: Matplotlib Animation (`'matplotlib'`)

**Description:** Traditional matplotlib-based animation with interactive plot window

**Features:**
- Frame-by-frame animation of vehicles moving through the intersection
- Shows conflict points and vehicle trajectories
- Can pause/play animation
- Good for detailed analysis and recording

**Requirements:**
- matplotlib library
- `visualization.py` file

**Usage:**
```python
VISUALIZATION_METHOD = 'matplotlib'
```

**Pros:**
- Self-contained visualization
- Can save animations as video files
- Good for presentations and analysis

**Cons:**
- Blocks execution until window is closed
- Less interactive
- May not work well in headless environments

---

### Option 2: Web-Based Visualization (`'web'`)

**Description:** Modern web-based D3.js visualization with interactive browser interface

**Features:**
- Real-time animation in your web browser
- Shows permutation order panel
- Highlights currently moving vehicles
- Interactive SVG-based rendering
- Conflict points clearly labeled

**Requirements:**
- Flask web server
- `visualization_server.py` file
- `visualization_utils.py` file
- `templates/intersection.html` file

**Usage:**
```python
VISUALIZATION_METHOD = 'web'
```

**How to use:**
1. Run the program
2. Wait for message: "Open your browser to: http://localhost:5000"
3. Open your browser and navigate to that URL
4. Watch the visualization in real-time
5. Press Ctrl+C in terminal to stop server

**Pros:**
- Modern, interactive interface
- Can access from any device on local network
- Better for real-time monitoring
- Visual permutation order panel
- Color-coded emergency vehicles

**Cons:**
- Requires web server
- Needs browser to view

---

### Option 3: No Visualization (`'none'`)

**Description:** Run optimization without any visualization (headless mode)

**Usage:**
```python
VISUALIZATION_METHOD = 'none'
```

**Pros:**
- Fastest execution
- Works in any environment
- Good for batch processing
- Ideal for server/cluster computing

**Cons:**
- No visual feedback
- Results shown only as text output

## Complete Example Workflows

### Workflow 1: Development with Web Visualization

```python
# In main_with_viz.py
VISUALIZATION_METHOD = 'web'

# Run
python main_with_viz.py

# Output:
# ======================================================================
# INTERSECTION TRAFFIC OPTIMIZATION
# ======================================================================
# Visualization Method: WEB
# ======================================================================
# 
# ✓ Web-based visualization enabled
# 
# [Running SA optimization...]
# 
# ======================================================================
# ✓ Web visualization server running!
#   Open your browser to: http://localhost:5000
#   Press Ctrl+C to stop the server
# ======================================================================
```

### Workflow 2: Analysis with Matplotlib

```python
# In main_with_viz.py
VISUALIZATION_METHOD = 'matplotlib'

# Run
python main_with_viz.py

# The program will:
# 1. Run SA optimization
# 2. Open matplotlib window with animation
# 3. Continue when you close the window
```

### Workflow 3: Batch Processing (No Visualization)

```python
# In main_with_viz.py
VISUALIZATION_METHOD = 'none'

# Run
python main_with_viz.py > results.txt

# Perfect for:
# - Running multiple experiments
# - Automated testing
# - Performance benchmarking
```

## Configuration Details

### SA Parameters

You can modify these at the top of `main_with_viz.py`:

```python
T_INITIAL = 1000.0              # Initial temperature
T_MIN = 1.0                     # Final temperature
COOLING_RATE = 0.99             # Cooling rate (0.95-0.99)
MAX_ITER_PER_TEMP = 20          # Iterations per temperature
MAX_TOTAL_ITERATIONS = 100000   # Maximum total iterations
```

### Vehicle Configuration

Edit `config.py` to change:
- Vehicle list (`pi`)
- Velocity ranges
- Safety distances
- Headway time (tau)

## Troubleshooting

### "Could not import matplotlib visualization"
- Install matplotlib: `pip install matplotlib`
- Or switch to `VISUALIZATION_METHOD = 'web'`

### "Could not import web visualization"
- Install Flask: `pip install flask`
- Ensure all web files exist:
  - `visualization_server.py`
  - `visualization_utils.py`
  - `templates/intersection.html`

### Web server won't start
- Check if port 5000 is already in use
- Try killing any existing Python processes
- Change port in `visualization_utils.py` if needed

### Animation too fast/slow (web version)
- Edit `templates/intersection.html`
- Modify `visualScale` values (lines ~388, ~458)
- Modify stagger delay (line ~527)

## Comparison Table

| Feature | Matplotlib | Web | None |
|---------|-----------|-----|------|
| Visual Quality | High | Very High | N/A |
| Interactivity | Low | High | N/A |
| Performance | Medium | High | Highest |
| Setup Complexity | Low | Medium | Lowest |
| Remote Access | No | Yes | N/A |
| Headless Support | No | Partial | Yes |
| Best For | Analysis | Demos | Batch Jobs |

## Advanced Usage

### Custom Visualization Selection at Runtime

You can modify the code to accept command-line arguments:

```python
# Add at top of main_with_viz.py
import sys

if len(sys.argv) > 1:
    VISUALIZATION_METHOD = sys.argv[1]

# Then run:
python main_with_viz.py web
python main_with_viz.py matplotlib
python main_with_viz.py none
```

### Running Multiple Experiments

```python
# Create a batch script
experiments = ['web', 'matplotlib', 'none']
for viz_method in experiments:
    VISUALIZATION_METHOD = viz_method
    main()
```

## Support

For issues or questions:
1. Check this guide
2. Review error messages in console
3. Verify all dependencies are installed
4. Check that required files exist

---

**Last Updated:** October 23, 2025
