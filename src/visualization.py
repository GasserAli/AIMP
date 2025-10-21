import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

class IntersectionVisualization:
    def __init__(self):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        
        # Road dimensions
        self.road_width = 2
        self.vehicles = []  # List to store vehicle objects
        
        # Define conflict points
        self.crossing_points = {
            # North-South traffic crossing East-West traffic (and vice versa)
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
        
        self.merge_points = {
            'Mn': (0, 1),    # North merge
            'Ms': (0, -1),   # South merge
            'Me': (1, 0),    # East merge
            'Mw': (-1, 0)    # West merge
        }
        
        # Initialize the visualization
        self.setup_intersection()
        
    def setup_intersection(self):
        """Draw the intersection with four roads and conflict points"""
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
        
        # Draw crossing conflict points
        for name, (x, y) in self.crossing_points.items():
            self.ax.plot(x, y, 'ro', markersize=5)
            self.ax.text(x + 0.1, y + 0.1, name, fontsize=8, color='red')
        
        # Draw merge conflict points
        for name, (x, y) in self.merge_points.items():
            self.ax.plot(x, y, 'bo', markersize=5)
            self.ax.text(x + 0.1, y + 0.1, name, fontsize=8, color='blue')
        
        # Add legend
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