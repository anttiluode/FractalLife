# Fractal Life Simulator

A neural evolution simulation featuring organisms with fractal neural networks that learn and evolve through interaction.

## Overview

The Fractal Life Simulator creates a virtual environment where organisms with fractal neural networks interact, evolve, and learn. Each organism possesses:
- A fractal neural network brain consisting of self-similar neural patterns
- Physical properties that affect movement and interaction
- Visual patterns that emerge from their neural properties
- Energy systems that govern survival and reproduction

## Installation

1. Clone the repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Running the Simulation

Launch the simulation:
```bash
python app.py
```

The simulation will start a Gradio web interface accessible at `http://localhost:7860`

## Features

### Organisms
- Fractal neural networks that process visual input and control movement
- Energy systems that govern survival and reproduction
- Physical properties affecting movement and collisions
- Visual patterns that emerge from neural properties
- Ability to interact and learn from other organisms

### Neural Networks
- Self-similar neural patterns that form a fractal structure
- Visual cortex for processing environmental input
- Thought processor for decision making
- Action generator for movement control
- Memory systems for learning from interactions

### Physics
- Realistic collision detection and response
- Momentum-based movement
- Force-based interactions
- Energy transfer during collisions

### Visualization
- Real-time visualization of organisms and their neural states
- Pattern visualization based on neural properties
- Energy level indicators
- Population statistics and neural activity plots

## Controls

Through the Gradio interface:
- Start/Stop/Pause simulation
- Adjust population parameters
- Modify mutation rates
- Control energy decay
- Adjust interaction strengths
- Monitor population statistics
- View neural activity
- Select and interact with individual organisms

## Configuration

Key parameters can be adjusted through the interface:
- Initial Population: 5-100 organisms
- Maximum Population: 50-500 organisms
- Mutation Rate: 0-1
- Brain Update Rate: 1-30 Hz
- Maximum Neurons: 100-5000
- Energy Decay Rate: 0-1
- Interaction Strength: 0-1
- Field Resolution: 10-100

## Technical Details

Built using:
- PyTorch for neural networks
- Pygame for visualization
- Pymunk for physics
- Gradio for user interface
- NumPy for numerical computations
- Plotly for statistics visualization

## Requirements

See `requirements.txt` for full dependencies. Key requirements:
- Python 3.8+
- torch>=2.0.0
- pygame>=2.4.0
- gradio==3.50.2
- numpy>=1.21.0
- pymunk>=6.4.0
- plotly>=5.13.0
- Pillow>=9.4.0

## Notes

- The simulation requires significant computational resources for larger populations
- Neural network complexity increases with depth of fractal patterns
- Energy systems prevent unlimited growth and enforce natural selection
- Visual patterns emerge from neural properties rather than being predefined

## Troubleshooting

Common issues:
1. Graphics issues: Update Pygame or graphics drivers
2. Performance issues: Reduce population size or neural network complexity
3. Memory issues: Reduce maximum neurons or population cap
4. Interface issues: Ensure Gradio 3.50.2 is installed

## License

MIT License - Feel free to use and modify for your own projects.
