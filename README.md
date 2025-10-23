# Automated Intersection Management Project (AIMP)

This project implements an Automated Intersection Management system using Simulated Annealing (SA) optimization algorithm. The system optimizes vehicle scheduling through an intersection while considering safety constraints and traffic flow efficiency.

## Project Structure

```
AIMP/
├── src/
│   ├── config.py      # Configuration parameters
│   ├── decoder.py     # Solution decoder
│   ├── geometry.py    # Intersection geometry
│   ├── main.py        # Main application
│   ├── objective.py   # Objective function
│   ├── sa.py         # Simulated Annealing implementation
│   ├── utils.py      # Utility functions
│   ├── vehicle.py    # Vehicle class
│   └── visualization.py # Visualization tools
└── requirements.txt   # Project dependencies
```

## Getting Started

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the system parameters:

### Configuring SA Parameters (sa.py)
The Simulated Annealing algorithm can be tuned by adjusting these parameters in `sa.py`:

```python
# Key SA Parameters
initial_temperature = 1000  # Starting temperature
final_temperature = 1      # Ending temperature
alpha = 0.95              # Cooling rate
iterations_per_temp = 100  # Number of iterations at each temperature
```

Tuning Guidelines:
- Higher `initial_temperature` allows for more exploration
- Lower `final_temperature` ensures convergence
- `alpha` closer to 1 means slower cooling (more exploration)
- Increase `iterations_per_temp` for more thorough search at each temperature

### Configuring System Parameters (config.py)
Adjust the intersection and vehicle parameters in `config.py`:

```python
# Vehicle Parameters
velocity_range = (10, 20)  # Vehicle velocity range
safety_distance = 5        # Minimum distance between vehicles
tau = 0.5                 # Processing time at conflict points

# Intersection Parameters
pi = [...]                # Vehicle configuration list
```

Tuning Guidelines:
- `velocity_range`: Set based on road speed limits
- `safety_distance`: Adjust based on safety requirements
- `tau`: Processing time at intersection points
- Modify `pi` list to change the vehicle configuration scenario

## Running the Project

### Running with Dynamic Visualization
To run the algorithm with real-time visualization:

```bash
python src/sa.py
```

This will:
1. Run the Simulated Annealing algorithm
2. Show the optimization progress
3. Display the dynamic visualization of vehicles moving through the intersection

The visualization shows:
- Vehicles color-coded by approach direction
- Smooth vehicle movements through the intersection
- Real-time position updates
- Queue management and conflict point handling

### Alternative Visualization
(Note: Instructions for the alternative visualization method will be provided soon)

## Visualization Features

The current visualization includes:
- Color-coded vehicles based on approach direction:
  - North Approach: Blue
  - East Approach: Yellow
  - South Approach: Green
  - West Approach: Pink
  - Emergency Vehicles: Red
- Smooth vehicle movements with proper acceleration/deceleration
- Clear trajectory paths
- Queue visualization
- Conflict point markers
- Real-time animation with status display

## Tips for Best Results

1. Start with default parameters and adjust based on your specific needs
2. Monitor the objective function value during optimization
3. Increase iterations if solutions are not satisfactory
4. Adjust safety parameters based on traffic density
5. Use visualization to verify the solution quality

## Notes

- The visualization speed can be adjusted in the visualization settings
- Emergency vehicles are given priority in the scheduling
- The system handles multiple vehicle types and turning patterns
- Configuration files can be modified to test different scenarios

(Further documentation and alternative visualization instructions will be added soon)