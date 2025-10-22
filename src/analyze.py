import matplotlib.pyplot as plt
import numpy as np
from sa import run_sa
from viz import IntersectionViz
import config
from geometry import Geometry
from decoder import run_decoder

class SolutionAnalyzer:
    def __init__(self):
        self.viz = IntersectionViz()
        self.geom = Geometry()
        
    def analyze_solution(self, perm, speeds, obj_value):
        """Analyze a specific solution in detail"""
        print("\n=== Solution Analysis ===")
        print(f"Objective Value: {obj_value:.2f}")
        
        # Analyze vehicle ordering
        emergency_positions = [i for i, v in enumerate(perm) if v.priority_status]
        print("\nEmergency Vehicle Positions:", emergency_positions)
        
        # Analyze speeds
        speed_stats = {
            "min": min(speeds),
            "max": max(speeds),
            "avg": sum(speeds) / len(speeds),
            "std": np.std(speeds)
        }
        print("\nSpeed Statistics:")
        for key, value in speed_stats.items():
            print(f"{key.capitalize()}: {value:.2f}")
        
        # Analyze delays
        tau_p_dict = {p: config.tau for p in self.geom.get_all_points()}
        decoder_results = run_decoder(perm, speeds, self.geom, tau_p_dict)
        
        delays = [r["delay"] for r in decoder_results]
        emergency_delays = [r["delay"] for r in decoder_results if r["is_emergency"]]
        
        print("\nDelay Statistics:")
        print(f"Average Delay: {np.mean(delays):.2f} seconds")
        print(f"Max Delay: {max(delays):.2f} seconds")
        if emergency_delays:
            print(f"Average Emergency Vehicle Delay: {np.mean(emergency_delays):.2f} seconds")
        
        return decoder_results

    def plot_delay_distribution(self, decoder_results):
        """Plot the distribution of delays"""
        delays = [r["delay"] for r in decoder_results]
        emergency_delays = [r["delay"] for r in decoder_results if r["is_emergency"]]
        normal_delays = [r["delay"] for r in decoder_results if not r["is_emergency"]]
        
        plt.figure(figsize=(10, 6))
        plt.hist([normal_delays, emergency_delays], label=['Normal', 'Emergency'],
                bins=15, alpha=0.7)
        plt.xlabel('Delay (seconds)')
        plt.ylabel('Number of Vehicles')
        plt.title('Distribution of Vehicle Delays')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_speed_distribution(self, speeds, perm):
        """Plot the distribution of assigned speeds"""
        emergency_speeds = [s for v, s in zip(perm, speeds) if v.priority_status]
        normal_speeds = [s for v, s in zip(perm, speeds) if not v.priority_status]
        
        plt.figure(figsize=(10, 6))
        plt.hist([normal_speeds, emergency_speeds], label=['Normal', 'Emergency'],
                bins=15, alpha=0.7)
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Number of Vehicles')
        plt.title('Distribution of Vehicle Speeds')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_approach_analysis(self, perm, speeds):
        """Analyze and plot statistics per approach"""
        approaches = {"N": [], "S": [], "E": [], "W": []}
        for v, s in zip(perm, speeds):
            approaches[v.approach].append(s)
        
        # Plot average speeds per approach
        avg_speeds = {k: np.mean(v) if v else 0 for k, v in approaches.items()}
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(avg_speeds.keys(), avg_speeds.values())
        plt.xlabel('Approach Direction')
        plt.ylabel('Average Speed (m/s)')
        plt.title('Average Vehicle Speed by Approach')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.grid(True, axis='y')
        plt.show()

def run_analysis():
    """Run complete analysis of SA solution"""
    print("=== Starting Analysis ===")
    print("\nRunning Simulated Annealing...")
    
    # 1. Run SA algorithm
    best_perm, best_speeds, best_obj = run_sa()
    
    # 2. Create analyzer
    analyzer = SolutionAnalyzer()
    
    # 3. Analyze the solution
    decoder_results = analyzer.analyze_solution(best_perm, best_speeds, best_obj)
    
    # 4. Plot various distributions
    print("\nGenerating analysis plots...")
    analyzer.plot_delay_distribution(decoder_results)
    analyzer.plot_speed_distribution(best_speeds, best_perm)
    analyzer.plot_approach_analysis(best_perm, best_speeds)
    
    # 5. Show animation of the solution
    print("\nGenerating solution animation...")
    viz = IntersectionViz()
    
    # Create paths dictionary for visualization
    vehicle_paths = {}
    for v in best_perm:
        if v.maneuver == "S":  # Straight
            to_dir = {"N": "S", "S": "N", "E": "W", "W": "E"}[v.approach]
        elif v.maneuver == "L":  # Left
            to_dir = {"N": "E", "S": "W", "E": "N", "W": "S"}[v.approach]
        else:  # Right
            to_dir = {"N": "W", "S": "E", "E": "S", "W": "N"}[v.approach]
        vehicle_paths[v.id] = viz.path_templates[(v.approach, to_dir)]
    
    # Create speeds dictionary
    speeds_dict = {v.id: s for v, s in zip(best_perm, best_speeds)}
    
    # Show the static intersection first
    viz.plot_static_intersection()
    
    # Then animate the solution
    viz.animate_vehicles(best_perm, vehicle_paths, speeds_dict, total_time=30, fps=30)

if __name__ == "__main__":
    run_analysis()