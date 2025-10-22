import requests
import threading
from visualization_server import start_server
import config

class IntersectionVisualizer:
    def __init__(self, host='http://localhost:5000'):
        self.host = host
        self.server_thread = None
        
    def start(self):
        """Start the visualization server in a separate thread"""
        if not self.server_thread:
            self.server_thread = threading.Thread(target=start_server)
            self.server_thread.daemon = True
            self.server_thread.start()
    
    def update_vehicles(self, vehicles, permutation=None):
        """
        Update the visualization with new vehicle positions
        
        Parameters:
        -----------
        vehicles : list
            List of Vehicle objects with properties:
            - id: vehicle identifier
            - approach: 'N', 'S', 'E', or 'W'
            - maneuver: 'L', 'R', or 'S'
            - priority_status: boolean
        permutation : list, optional
            List of vehicle IDs in their processing order
        """
        # Convert vehicle objects to dictionaries
        vehicle_data = []
        for v in vehicles:
            vehicle_data.append({
                'id': v.id,
                'approach': v.approach,
                'maneuver': v.maneuver,
                'priority_status': v.priority_status,
                'velocity': getattr(v, 'velocity', None),
                'path': getattr(v, 'path', []),
                'delay': getattr(v, 'delay', 0.0)
            })
            
        data = {
            'vehicles': vehicle_data,
            'permutation': permutation or []
        }
        print(data)
        
        try:
            response = requests.post(f"{self.host}/api/update_vehicles", json=data)
            return response.ok
        except requests.exceptions.RequestException as e:
            print(f"Failed to update visualization: {e}")
            return False

    def start_simulation(self):
        try:
            response = requests.post(f"{self.host}/api/start_simulation")
            return response.ok
        except requests.exceptions.RequestException as e:
            print(f"Failed to start simulation: {e}")
            return False

    def stop_simulation(self):
        try:
            response = requests.post(f"{self.host}/api/stop_simulation")
            return response.ok
        except requests.exceptions.RequestException as e:
            print(f"Failed to stop simulation: {e}")
            return False

# Example usage:
if __name__ == '__main__':
    visualizer = IntersectionVisualizer()
    visualizer.start()  # Start the visualization server

    # Test the visualization with a single update of the vehicles
    visualizer.update_vehicles(vehicles=config.pi, permutation=[v.id for v in config.pi])
    visualizer.start_simulation()
    
    # Keep the main thread alive to maintain the server
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down visualization server...")
