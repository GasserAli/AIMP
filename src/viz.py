import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from geometry import Geometry
import config

class IntersectionViz:
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.safety_distance = config.safety_distance
        
        # Define intersection center
        self.center = grid_size // 2
        
        # Define approach points (entry points for N,S,E,W)
        self.approaches = {
            "N": (self.center, grid_size-10),  # North approach
            "S": (self.center, 10),            # South approach
            "E": (grid_size-10, self.center),  # East approach
            "W": (10, self.center)             # West approach
        }
        
        # Define conflict points with fixed coordinates
        self.conflict_points = {
            # North-South straight path conflicts
            "NS1": (self.center, self.center + 10),
            "NS2": (self.center, self.center - 10),
            
            # East-West straight path conflicts
            "EW1": (self.center + 10, self.center),
            "EW2": (self.center - 10, self.center),
            
            # Left turn conflicts
            "NL": (self.center - 10, self.center + 10),
            "SL": (self.center + 10, self.center - 10),
            "EL": (self.center + 10, self.center + 10),
            "WL": (self.center - 10, self.center - 10),
            
            # Right turn conflicts
            "NR": (self.center + 10, self.center + 10),
            "SR": (self.center - 10, self.center - 10),
            "ER": (self.center + 10, self.center - 10),
            "WR": (self.center - 10, self.center + 10)
        }
        
        # Define path templates (sequence of points for each maneuver)
        self.path_templates = {
            # Straight paths
            ("N", "S"): [(self.center, y) for y in range(grid_size-10, 10, -5)],
            ("S", "N"): [(self.center, y) for y in range(10, grid_size-10, 5)],
            ("E", "W"): [(x, self.center) for x in range(grid_size-10, 10, -5)],
            ("W", "E"): [(x, self.center) for x in range(10, grid_size-10, 5)],
            
            # Left turn paths (approximated with a few points)
            ("N", "E"): self._generate_turn_path("N", "E", "L"),
            ("S", "W"): self._generate_turn_path("S", "W", "L"),
            ("E", "N"): self._generate_turn_path("E", "N", "L"),
            ("W", "S"): self._generate_turn_path("W", "S", "L"),
            
            # Right turn paths
            ("N", "W"): self._generate_turn_path("N", "W", "R"),
            ("S", "E"): self._generate_turn_path("S", "E", "R"),
            ("E", "S"): self._generate_turn_path("E", "S", "R"),
            ("W", "N"): self._generate_turn_path("W", "N", "R")
        }

    def _generate_turn_path(self, from_dir, to_dir, turn_type):
        """Generate a smooth turning path between two directions"""
        start = self.approaches[from_dir]
        
        # Define control points for the turn
        if turn_type == "L":  # Left turn
            if from_dir == "N":
                control = (self.center - 10, self.center + 10)
                end = (self.grid_size-10, self.center)
            elif from_dir == "S":
                control = (self.center + 10, self.center - 10)
                end = (10, self.center)
            elif from_dir == "E":
                control = (self.center + 10, self.center + 10)
                end = (self.center, self.grid_size-10)
            else:  # from_dir == "W"
                control = (self.center - 10, self.center - 10)
                end = (self.center, 10)
        else:  # Right turn
            if from_dir == "N":
                control = (self.center + 10, self.center + 10)
                end = (10, self.center)
            elif from_dir == "S":
                control = (self.center - 10, self.center - 10)
                end = (self.grid_size-10, self.center)
            elif from_dir == "E":
                control = (self.center + 10, self.center - 10)
                end = (self.center, 10)
            else:  # from_dir == "W"
                control = (self.center - 10, self.center + 10)
                end = (self.center, self.grid_size-10)
        
        # Generate points along the curve
        t = np.linspace(0, 1, 10)
        points = []
        for ti in t:
            x = (1-ti)**2 * start[0] + 2*(1-ti)*ti * control[0] + ti**2 * end[0]
            y = (1-ti)**2 * start[1] + 2*(1-ti)*ti * control[1] + ti**2 * end[1]
            points.append((int(x), int(y)))
        return points

    def _draw_road_lanes(self, ax):
        """Draw the road lanes for the intersection"""
        lane_width = 5
        road_length = self.grid_size - 20
        center = self.center
        
        # Road colors and style
        road_color = '#404040'  # Dark gray
        lane_marker_color = 'yellow'
        
        # Draw horizontal road (E-W)
        ax.add_patch(plt.Rectangle((10, center-lane_width), road_length, 2*lane_width, 
                                 facecolor=road_color))
        # Draw vertical road (N-S)
        ax.add_patch(plt.Rectangle((center-lane_width, 10), 2*lane_width, road_length, 
                                 facecolor=road_color))
        
        # Draw lane markers
        # Horizontal lanes
        plt.plot([10, center-lane_width-5], [center, center], '--', color=lane_marker_color)
        plt.plot([center+lane_width+5, self.grid_size-10], [center, center], '--', color=lane_marker_color)
        
        # Vertical lanes
        plt.plot([center, center], [10, center-lane_width-5], '--', color=lane_marker_color)
        plt.plot([center, center], [center+lane_width+5, self.grid_size-10], '--', color=lane_marker_color)
        
        # Add direction arrows
        arrow_props = dict(head_width=3, head_length=4, fc='white', ec='white', alpha=0.8)
        
        # East-West arrows
        plt.arrow(15, center-2, 10, 0, **arrow_props)
        plt.arrow(self.grid_size-25, center+2, 10, 0, **arrow_props)
        
        # North-South arrows
        plt.arrow(center-2, 15, 0, 10, **arrow_props)
        plt.arrow(center+2, self.grid_size-25, 0, 10, **arrow_props)

    def plot_static_intersection(self):
        """Plot the static intersection layout with conflict points and roads"""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor('#98FB98')  # Light green background for grass
        
        # Draw the roads first
        self._draw_road_lanes(ax)
        
        # Plot conflict points
        for point_name, (x, y) in self.conflict_points.items():
            ax.plot(x, y, 'rx', markersize=8, label=point_name)
        
        # Plot approach points
        for dir_name, (x, y) in self.approaches.items():
            ax.plot(x, y, 'go', markersize=10, label=f"{dir_name} Approach")
        
        # Plot safety distance circles around conflict points
        for (x, y) in self.conflict_points.values():
            circle = plt.Circle((x, y), self.safety_distance, 
                              fill=False, linestyle='--', color='white', alpha=0.5)
            ax.add_patch(circle)
        
        # Set plot limits and labels
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Intersection Layout with Roads and Conflict Points')
        
        # Add legend
        handles = [
            plt.Line2D([0], [0], marker='x', color='r', label='Conflict Point', markersize=8, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='g', label='Approach Point', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], color='yellow', label='Lane Marker', linestyle='--'),
            plt.Circle((0, 0), 1, fill=False, color='white', alpha=0.5, label='Safety Distance')
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def animate_vehicles(self, vehicles, vehicle_paths, speeds, total_time=20, fps=30):
        """
        Animate vehicles moving through their paths
        
        Parameters:
        - vehicles: list of Vehicle objects
        - vehicle_paths: dict mapping vehicle IDs to their paths
        - speeds: dict mapping vehicle IDs to their speeds
        - total_time: total animation time in seconds
        - fps: frames per second
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor('#98FB98')  # Light green background
        
        # Initialize vehicle positions
        positions = {v.id: 0 for v in vehicles}
        vehicle_dots = {}
        
        def init():
            ax.clear()
            ax.set_facecolor('#98FB98')
            ax.set_xlim(0, self.grid_size)
            ax.set_ylim(0, self.grid_size)
            
            # Draw the roads
            self._draw_road_lanes(ax)
            
            # Plot conflict points and their safety circles
            for (x, y) in self.conflict_points.values():
                ax.plot(x, y, 'rx', markersize=8)
                circle = plt.Circle((x, y), self.safety_distance, 
                                 fill=False, linestyle='--', color='white', alpha=0.5)
                ax.add_patch(circle)
            
            # Initialize vehicle dots with labels
            for v in vehicles:
                color = 'red' if v.priority_status else 'blue'
                dot = ax.plot([], [], 'o', color=color, 
                            markersize=8 if v.priority_status else 6,
                            label=f'Vehicle {v.id}{"(E)" if v.priority_status else ""}'
                            )[0]
                vehicle_dots[v.id] = dot
            
            # Add legend
            handles = [
                plt.Line2D([0], [0], marker='o', color='red', label='Emergency Vehicle', 
                          markersize=8, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='blue', label='Normal Vehicle', 
                          markersize=6, linestyle='None'),
                plt.Line2D([0], [0], marker='x', color='red', label='Conflict Point', 
                          markersize=8, linestyle='None'),
                plt.Line2D([0], [0], color='yellow', label='Lane Marker', 
                          linestyle='--')
            ]
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.grid(True, alpha=0.3)
            return list(vehicle_dots.values())
        
        def update(frame):
            # Update position of each vehicle
            for v in vehicles:
                path = vehicle_paths[v.id]
                speed = speeds[v.id]
                
                # Update position index based on speed and time
                pos_idx = int((frame / fps) * speed) % len(path)
                if pos_idx < len(path):
                    x, y = path[pos_idx]
                    vehicle_dots[v.id].set_data([x], [y])
            
            return list(vehicle_dots.values())
        
        anim = animation.FuncAnimation(fig, update, init_func=init,
                                     frames=total_time*fps, interval=1000/fps,
                                     blit=True)
        plt.tight_layout()
        plt.show()

def test_visualization():
    """Test function to demonstrate the visualization"""
    viz = IntersectionViz(grid_size=100)
    
    # 1. Show static intersection layout
    viz.plot_static_intersection()
    
    # 2. Create some test vehicles with paths
    vehicles = config.pi[:5]  # Take first 5 vehicles from config
    
    # Generate paths for each vehicle
    vehicle_paths = {}
    speeds = {}
    
    for v in vehicles:
        if v.maneuver == "S":  # Straight
            to_dir = {"N": "S", "S": "N", "E": "W", "W": "E"}[v.approach]
            vehicle_paths[v.id] = viz.path_templates[(v.approach, to_dir)]
        elif v.maneuver == "L":  # Left
            to_dir = {"N": "E", "S": "W", "E": "N", "W": "S"}[v.approach]
            vehicle_paths[v.id] = viz.path_templates[(v.approach, to_dir)]
        elif v.maneuver == "R":  # Right
            to_dir = {"N": "W", "S": "E", "E": "S", "W": "N"}[v.approach]
            vehicle_paths[v.id] = viz.path_templates[(v.approach, to_dir)]
            
        # Assign random speed within bounds
        v_min, v_max = config.velocity_range
        speeds[v.id] = (v_max - v_min) * 0.5 + v_min
    
    # 3. Animate vehicles
    viz.animate_vehicles(vehicles, vehicle_paths, speeds, total_time=20, fps=30)

if __name__ == "__main__":
    test_visualization()
