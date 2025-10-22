from flask import Flask, render_template, jsonify, request
import os
import json

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Ensure templates and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Store current state
current_vehicles = []
current_permutation = []
is_running = False

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    global is_running
    is_running = True
    return jsonify({"status": "running"})


@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    global is_running
    is_running = False
    return jsonify({"status": "stopped"})

@app.route('/')
def index():
    return render_template('intersection.html')

@app.route('/api/update_vehicles', methods=['POST'])
def update_vehicles():
    """Receive vehicle data from Python script"""
    global current_vehicles, current_permutation
    data = request.json
    current_vehicles = data.get('vehicles', [])
    current_permutation = data.get('permutation', [])
    return jsonify({"status": "success"})

@app.route('/api/get_state')
def get_state():
    """Return current visualization state"""
    return jsonify({
        "vehicles": current_vehicles,
        "permutation": current_permutation,
        "running": is_running
    })

def start_server():
    """Function to start the server programmatically"""
    app.run(debug=True, port=5000, use_reloader=False)

if __name__ == '__main__':
    app.run(debug=True, port=5000)