import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Path, PathPatch
from matplotlib.path import Path as MPath
import matplotlib.colors as mcolors

class IntersectionVisualization:
    def __init__(self):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        
        # Road dimensions
        self.road_width = 2
        self.vehicles = []  # List to store vehicle objects
        
        # Define entry and exit points for each direction
        self.entry_points = {
            'N': (0, -2),  # North entry (coming from South)
            'S': (0, 2),   # South entry (coming from North)
            'E': (-2, 0),  # East entry (coming from West)
            'W': (2, 0),   # West entry (coming from East)
        }
        
        self.exit_points = {
            'N': (0, 2),   # North exit
            'S': (0, -2),  # South exit
            'E': (2, 0),   # East exit
            'W': (-2, 0),  # West exit
        }
        
        # Define trajectories and conflict points
        # Keep trajectory data (used for placing conflict/merge points) but don't draw arcs by default
        self.show_trajectories = False
        self.setup_trajectories()
        
        # Define conflict points based on trajectory intersections
        self.crossing_points = {
            # North-South traffic crossing East-West traffic
            'C1': (-0.5, 0.5),   'C2': (0.5, 0.5),
            'C3': (-0.5, -0.5),  'C4': (0.5, -0.5),
            # Left turns crossing straight traffic
            'C5': (-0.5, 0),     'C6': (0.5, 0),
            'C7': (0, 0.5),      'C8': (0, -0.5),
            # Left turns crossing opposing left turns
            'C9': (0, 0),
            # Left turns crossing opposing straight traffic
            'C10': (-0.25, 0.25), 'C11': (0.25, 0.25),
            'C12': (-0.25, -0.25), 'C13': (0.25, -0.25),
            # Right turns crossing straight traffic
            'C14': (-0.75, 0.75), 'C15': (0.75, 0.75),
            'C16': (-0.75, -0.75)
        }
        
        # Merge points are now placed on the exit lanes where turning vehicles join straight traffic
        self.merge_points = {
            'Mn': (0, 2),    # North merge (right turns from East and left turns from West merge)
            'Ms': (0, -2),   # South merge (right turns from West and left turns from East merge)
            'Me': (2, 0),    # East merge (right turns from North and left turns from South merge)
            'Mw': (-2, 0)    # West merge (right turns from South and left turns from North merge)
        }
        
        # Initialize the visualization
        self.setup_intersection()
        
    def create_turn_trajectory(self, start, end, control_points):
        """Create a curved trajectory using Bézier curve"""
        t = np.linspace(0, 1, 100)
        trajectory = []
        for t_i in t:
            x = (1-t_i)**3 * start[0] + \
                3*(1-t_i)**2 * t_i * control_points[0][0] + \
                3*(1-t_i) * t_i**2 * control_points[1][0] + \
                t_i**3 * end[0]
            y = (1-t_i)**3 * start[1] + \
                3*(1-t_i)**2 * t_i * control_points[0][1] + \
                3*(1-t_i) * t_i**2 * control_points[1][1] + \
                t_i**3 * end[1]
            trajectory.append([x, y])
        return np.array(trajectory)
    
    def setup_trajectories(self):
        """Setup all possible trajectories for vehicles"""
        self.trajectories = {}
        
        # Define the directions
        directions = ['N', 'S', 'E', 'W']
        
        for start_dir in directions:
            start_point = self.entry_points[start_dir]
            
            for end_dir in directions:
                if start_dir == end_dir:
                    continue
                    
                end_point = self.exit_points[end_dir]
                path_key = f"{start_dir}-{end_dir}"
                
                # Straight movement
                if (start_dir == 'N' and end_dir == 'S') or \
                   (start_dir == 'S' and end_dir == 'N') or \
                   (start_dir == 'E' and end_dir == 'W') or \
                   (start_dir == 'W' and end_dir == 'E'):
                    self.trajectories[path_key] = np.array([start_point, end_point])
                
                # Right turn
                elif ((start_dir == 'N' and end_dir == 'W') or 
                      (start_dir == 'E' and end_dir == 'N') or
                      (start_dir == 'S' and end_dir == 'E') or
                      (start_dir == 'W' and end_dir == 'S')):
                    if start_dir == 'N':
                        control_points = [(0, -1), (-1, -1)]
                    elif start_dir == 'E':
                        control_points = [(-1, 0), (-1, 1)]
                    elif start_dir == 'S':
                        control_points = [(0, 1), (1, 1)]
                    else:  # West
                        control_points = [(1, 0), (1, -1)]
                    self.trajectories[path_key] = self.create_turn_trajectory(start_point, end_point, control_points)
                
                # Left turn
                else:
                    if start_dir == 'N':
                        control_points = [(0, -0.5), (0.5, 0)]
                    elif start_dir == 'E':
                        control_points = [(-0.5, 0), (0, -0.5)]
                    elif start_dir == 'S':
                        control_points = [(0, 0.5), (-0.5, 0)]
                    else:  # West
                        control_points = [(0.5, 0), (0, 0.5)]
                    self.trajectories[path_key] = self.create_turn_trajectory(start_point, end_point, control_points)
        
    def setup_intersection(self):
        """Draw the intersection with four roads, trajectories, and conflict points"""
        # Draw horizontal road
        self.ax.add_patch(Rectangle((-6, -1), 12, 2, color='gray'))
        # Draw vertical road
        self.ax.add_patch(Rectangle((-1, -6), 2, 12, color='gray'))
        
        # Draw road markings (center lines)
        self.ax.plot([-6, 6], [0, 0], '--', color='white', linewidth=1)
        self.ax.plot([0, 0], [-6, 6], '--', color='white', linewidth=1)
        
        # Add direction labels
        self.ax.text(5.5, 1.5, 'East', fontsize=10)
        self.ax.text(-5.5, 1.5, 'West', fontsize=10)
        self.ax.text(1.5, 5.5, 'North', fontsize=10)
        self.ax.text(1.5, -5.5, 'South', fontsize=10)
        
        # Optionally draw trajectories (disabled by default to keep visualization clean)
        colors = {
            'straight': 'yellow',
            'right': 'cyan',
            'left': 'magenta'
        }
        if self.show_trajectories:
            for path_key, trajectory in self.trajectories.items():
                start_dir, end_dir = path_key.split('-')
                # Determine movement type
                if (start_dir == 'N' and end_dir == 'S') or \
                   (start_dir == 'S' and end_dir == 'N') or \
                   (start_dir == 'E' and end_dir == 'W') or \
                   (start_dir == 'W' and end_dir == 'E'):
                    color = colors['straight']
                elif ((start_dir == 'N' and end_dir == 'W') or 
                      (start_dir == 'E' and end_dir == 'N') or
                      (start_dir == 'S' and end_dir == 'E') or
                      (start_dir == 'W' and end_dir == 'S')):
                    color = colors['right']
                else:
                    color = colors['left']
                # Plot trajectory
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], '-', 
                             color=color, alpha=0.5, linewidth=1)
        
        # Draw crossing conflict points
        for name, (x, y) in self.crossing_points.items():
            self.ax.plot(x, y, 'ro', markersize=5)
            self.ax.text(x + 0.1, y + 0.1, name, fontsize=8, color='red')
        
        # Draw merge conflict points
        for name, (x, y) in self.merge_points.items():
            self.ax.plot(x, y, 'bo', markersize=5)
            self.ax.text(x + 0.1, y + 0.1, name, fontsize=8, color='blue')
        
        # Add legend (only for conflicts and merges; trajectories optional)
        self.ax.plot([], [], 'ro', label='Crossing Conflicts')
        self.ax.plot([], [], 'bo', label='Merge Points')
        self.ax.legend(loc='upper right')
        
        # Set background color
        self.ax.set_facecolor('green')
        
        # Remove axis
        self.ax.axis('off')
        
    def add_vehicle(self, x, y, is_emergency=False):
        """Add a vehicle to the intersection"""
        color = 'red' if is_emergency else 'blue'
        vehicle = Circle((x, y), 0.4, color=color)
        self.vehicles.append(vehicle)
        self.ax.add_patch(vehicle)
        return vehicle
    
    def update(self, frame):
        """Update function for animation"""
        # This will be implemented later for vehicle movement
        pass
    
    def show(self):
        """Display the intersection"""
        plt.show()

# Test the visualization
if __name__ == "__main__":
    viz = IntersectionVisualization()
    
    # Add some test vehicles
    # Regular vehicle coming from west
    viz.add_vehicle(-4, -0.5)
    # Emergency vehicle coming from south
    viz.add_vehicle(0.5, -4, is_emergency=True)
    # Regular vehicle coming from east
    viz.add_vehicle(4, 0.5)
    
    viz.show()