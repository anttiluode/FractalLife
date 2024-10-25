import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for activation functions
import numpy as np
from dataclasses import dataclass, field
import pygame
import gradio as gr
from typing import List, Tuple, Dict, Optional, Set
import random
import colorsys
import pymunk
import time
import threading
from queue import Queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import logging  # Added for logging

# ==============================
# Logging Configuration
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)

# ==============================
# Configuration Dataclasses
# ==============================

@dataclass
class SimulationConfig:
    WIDTH: int = 1024
    HEIGHT: int = 768
    TARGET_FPS: int = 60
    MIN_ORGANISMS: int = 5
    MAX_ORGANISMS: int = 50  # Population cap set to 50
    MUTATION_RATE: float = 0.1
    REPRODUCTION_ENERGY: float = 150.0
    INITIAL_ENERGY: float = 100.0
    BRAIN_UPDATE_RATE: int = 10  # Hz
    MAX_NEURONS: int = 1000
    ENERGY_DECAY: float = 0.1

@dataclass
class NeuronState:
    activation: float = 0.0
    connections: int = 0
    energy: float = 100.0
    memory: List[float] = field(default_factory=lambda: [0.0] * 8)

@dataclass
class VisualizationConfig:
    BACKGROUND_COLOR: Tuple[int, int, int] = (10, 10, 30)
    NEURON_COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'active': (255, 255, 0),
        'inactive': (100, 100, 100),
        'connected': (0, 255, 255)
    })
    CONNECTION_COLOR: Tuple[int, int, int, int] = (50, 50, 200, 100)
    ENERGY_COLOR: Tuple[int, int, int] = (0, 255, 0)
    MAX_NEURAL_CONNECTIONS: int = 50

@dataclass
class PhysicsConfig:
    COLLISION_TYPE_ORGANISM: int = 1
    ELASTICITY: float = 0.7
    FRICTION: float = 0.5
    DAMPING: float = 0.9
    INTERACTION_RADIUS: float = 50.0
    FORCE_SCALE: float = 100.0

# ==============================
# Neural Processing System
# ==============================

class FractalNeuron(nn.Module):
    def __init__(self, input_dim=16, output_dim=16, depth=0, max_depth=2):
            super().__init__()
            self.depth = depth
            self.max_depth = max_depth

            # Store dimensions
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = max(input_dim // 2, 8)  # Add explicit hidden_dim

            # Enhanced neural processing layers with LeakyReLU
            self.synapse = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),  # First layer: input_dim to hidden_dim
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Linear(self.hidden_dim, output_dim),  # Second layer: hidden_dim to output_dim
                nn.Tanh()
            )

            # Initialize weights using Xavier uniform initialization
            for layer in self.synapse:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

            # Set to eval mode to prevent BatchNorm issues
            self.eval()

            # State maintenance with bounded values
            self.state = NeuronState()
            self.state.activation = 0.0
            self.state.energy = min(100.0, max(0.0, self.state.energy))

            # Memory processing with correct dimensions
            self.memory_gate = nn.Sequential(
                nn.Linear(output_dim + 8, 8),
                nn.Sigmoid()
            )

            # Initialize memory_gate weights
            for layer in self.memory_gate:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

            # Child neurons with matching dimensions
            self.sub_neurons = nn.ModuleList([])
            if depth < max_depth:
                branching_factor = max(1, 2 - depth)
                for _ in range(branching_factor):
                    child = FractalNeuron(
                        input_dim=output_dim,  # Child's input_dim matches parent's output_dim
                        output_dim=output_dim,  # Keep output_dim consistent
                        depth=depth + 1,
                        max_depth=max_depth
                    )
                    self.sub_neurons.append(child)

    def forward(self, x):
        """Forward pass for PyTorch module compatibility"""
        return self.process_signal(x)

    def process_signal(self, x, external_input=None):
        """Process input signal through the neuron"""
        try:
            with torch.no_grad():
                # Ensure we're in eval mode
                self.eval()

                # Reshape input for processing
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)  # Add batch dimension

                # Check for NaNs in input
                if torch.isnan(x).any():
                    logging.warning("NaN detected in input tensor. Returning zero tensor.")
                    return torch.zeros(self.output_dim)

                # Add external input if provided
                if external_input is not None:
                    if len(external_input.shape) == 1:
                        external_input = external_input.unsqueeze(0)
                    x = torch.cat([x, external_input], dim=-1)

                # Process through synapse with proper shapes
                x = x.to(torch.float32)  # Ensure float32 dtype
                
                try:
                    x = self.synapse(x)
                except RuntimeError as e:
                    logging.error(f"Error in synapse processing: {e}")
                    return torch.zeros(self.output_dim)
                
                # Update memory with bounds checking
                try:
                    memory_tensor = torch.tensor(self.state.memory).to(torch.float32)
                    if len(x.shape) == 1:
                        x_for_memory = x.unsqueeze(0)
                    else:
                        x_for_memory = x
                    memory_input = torch.cat([x_for_memory, memory_tensor.unsqueeze(0)], dim=-1)
                    new_memory = self.memory_gate(memory_input)
                    new_memory = torch.clamp(new_memory, 0.0, 1.0)
                    if not torch.isnan(new_memory).any():
                        self.state.memory = new_memory[0].tolist()
                except Exception as e:
                    logging.error(f"Error updating memory: {e}")

                # Update activation with bounded value
                activation = float(torch.clamp(x.mean(), -1.0, 1.0))
                
                if not np.isnan(activation):
                    self.state.activation = activation
                
                # Process through children with error handling
                if self.sub_neurons:
                    child_outputs = []
                    for child in self.sub_neurons:
                        try:
                            # Ensure x has correct shape before passing to child
                            child_input = x.squeeze(0) if len(x.shape) == 2 else x
                            # Ensure input matches child's expected input dimension
                            if child_input.shape[-1] != child.input_dim:
                                child_input = child_input[:child.input_dim]
                            child_out = child.process_signal(child_input)
                            if not torch.isnan(child_out).any():
                                # Ensure child output has correct shape for stacking
                                if len(child_out.shape) == 1:
                                    child_out = child_out.unsqueeze(0)
                                child_outputs.append(child_out)
                        except Exception as e:
                            logging.error(f"Error in child neuron processing: {e}")
                            continue
                    
                    if child_outputs:
                        child_outputs = torch.stack(child_outputs)
                        x = torch.mean(child_outputs, dim=0)
                        x = torch.clamp(x, -1.0, 1.0)

                # Update energy with bounds
                energy_cost = 0.1 * self.depth
                self.state.energy = max(0.0, min(100.0, self.state.energy - energy_cost))

                # Remove batch dimension if it was added
                if len(x.shape) == 2:
                    x = x.squeeze(0)

                return x

        except Exception as e:
            logging.error(f"Error in process_signal: {e}")
            return torch.zeros(self.output_dim)
        
    def interact_with(self, other_neuron, strength=0.5):
        """Interact with another neuron"""
        try:
            # Bound strength value
            strength = max(0.0, min(1.0, strength))

            # Share neural states with bounds
            shared_activation = (self.state.activation + other_neuron.state.activation) / 2
            shared_activation = float(shared_activation)

            if np.isnan(shared_activation):
                logging.warning("NaN detected in shared activation. Using default value.")
                shared_activation = 0.0
            
            self.state.activation = shared_activation
            other_neuron.state.activation = shared_activation

            # Share memories with bounds checking
            shared_memory = []
            for a, b in zip(self.state.memory, other_neuron.state.memory):
                shared_value = (float(a) + float(b)) / 2
                shared_value = max(0.0, min(1.0, shared_value))
                shared_memory.append(shared_value)

            self.state.memory = shared_memory
            other_neuron.state.memory = shared_memory

            # Update connections with bounds
            max_connections = 100
            self.state.connections = min(self.state.connections + 1, max_connections)
            other_neuron.state.connections = min(other_neuron.state.connections + 1, max_connections)

            return shared_activation
        except Exception as e:
            logging.error(f"Error in interact_with: {e}")
            return 0.0

    def save_state(self):
        """Save the current state of the neuron"""
        return {
            'activation': self.state.activation,
            'connections': self.state.connections,
            'energy': self.state.energy,
            'memory': self.state.memory.copy()
        }

    def load_state(self, state_dict):
        """Load a previously saved state"""
        try:
            self.state.activation = state_dict['activation']
            self.state.connections = state_dict['connections']
            self.state.energy = state_dict['energy']
            self.state.memory = state_dict['memory'].copy()
        except Exception as e:
            logging.error(f"Error loading neuron state: {e}")

    def clone(self):
        """Create a deep copy of the neuron"""
        try:
            new_neuron = FractalNeuron(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                depth=self.depth,
                max_depth=self.max_depth
            )
            new_neuron.load_state(self.save_state())
            return new_neuron
        except Exception as e:
            logging.error(f"Error cloning neuron: {e}")
            return None

    def mutate(self, mutation_rate=0.1):
        """Apply random mutations to the neuron"""
        try:
            with torch.no_grad():
                # Mutate weights
                for layer in self.synapse:
                    if isinstance(layer, nn.Linear):
                        mask = torch.rand_like(layer.weight) < mutation_rate
                        mutations = torch.randn_like(layer.weight) * 0.1
                        layer.weight.data[mask] += mutations[mask]
                        
                        if layer.bias is not None:
                            mask = torch.rand_like(layer.bias) < mutation_rate
                            mutations = torch.randn_like(layer.bias) * 0.1
                            layer.bias.data[mask] += mutations[mask]

                # Mutate memory gate
                for layer in self.memory_gate:
                    if isinstance(layer, nn.Linear):
                        mask = torch.rand_like(layer.weight) < mutation_rate
                        mutations = torch.randn_like(layer.weight) * 0.1
                        layer.weight.data[mask] += mutations[mask]
                        
                        if layer.bias is not None:
                            mask = torch.rand_like(layer.bias) < mutation_rate
                            mutations = torch.randn_like(layer.bias) * 0.1
                            layer.bias.data[mask] += mutations[mask]

                # Recursively mutate child neurons
                for child in self.sub_neurons:
                    child.mutate(mutation_rate)

        except Exception as e:
            logging.error(f"Error mutating neuron: {e}")

class FractalBrain:
    def __init__(self, input_dim=32, hidden_dim=64, max_neurons=1000):
        self.input_dim = min(input_dim, 32)  # Limit input dimension
        self.hidden_dim = min(hidden_dim, 64)  # Limit hidden dimension
        self.max_neurons = min(max_neurons, 1000)  # Limit maximum neurons

        # Core neural network components with reduced complexity
        self.visual_cortex = FractalNeuron(self.input_dim, self.hidden_dim, max_depth=2)
        self.thought_processor = FractalNeuron(self.hidden_dim, self.hidden_dim, max_depth=2)
        self.action_generator = FractalNeuron(self.hidden_dim, self.input_dim, max_depth=2)

        # State tracking with bounds
        self.total_neurons = self.count_neurons()
        self.total_energy = 100.0  # Reduced initial energy
        self.memories = []
        self.current_vision = None

    def get_vitals(self):
        """Get vital statistics of the brain with safety checks"""
        try:
            # Calculate average activation safely
            activations = []
            for neuron in [self.visual_cortex, self.thought_processor, self.action_generator]:
                try:
                    activation = float(neuron.state.activation)
                    if not np.isnan(activation) and not np.isinf(activation):
                        activations.append(activation)
                except (AttributeError, ValueError, TypeError):
                    activations.append(0.0)
            
            avg_activation = sum(activations) / max(len(activations), 1)
            avg_activation = max(-1.0, min(1.0, avg_activation))

            # Get connection counts safely
            connections = []
            for neuron in [self.visual_cortex, self.thought_processor, self.action_generator]:
                try:
                    conn_count = int(neuron.state.connections)
                    if not np.isnan(conn_count) and not np.isinf(conn_count):
                        connections.append(conn_count)
                except (AttributeError, ValueError, TypeError):
                    connections.append(0)
            
            total_connections = sum(connections)

            return {
                'neurons': min(self.total_neurons, self.max_neurons),
                'energy': max(0.0, min(1000.0, float(self.total_energy))),
                'connections': max(0, min(1000, total_connections)),
                'activation': avg_activation
            }
        except Exception as e:
            logging.error(f"Exception in get_vitals: {e}. Returning default vitals.")
            # Return safe default values if anything goes wrong
            return {
                'neurons': 1,
                'energy': 0.0,
                'connections': 0,
                'activation': 0.0
            }

    def process_vision(self, visual_input):
        try:
            with torch.no_grad():
                # Ensure input is valid and properly shaped
                visual_input = visual_input.clone().detach()
                if len(visual_input.shape) == 1:
                    visual_input = visual_input.unsqueeze(0)  # Add batch dimension

                if torch.isnan(visual_input).any():
                    logging.warning("NaN detected in visual_input. Replacing with zeros.")
                    visual_input = torch.zeros_like(visual_input)
                
                visual_input = torch.clamp(visual_input, -10.0, 10.0)

                # Process through neural components with shape handling
                try:
                    visual_features = self.visual_cortex.process_signal(visual_input)
                    if len(visual_features.shape) == 1:
                        visual_features = visual_features.unsqueeze(0)
                except Exception as e:
                    logging.error(f"Exception in visual_cortex.process_signal: {e}. Using zero tensor.")
                    visual_features = torch.zeros((1, self.hidden_dim))

                try:
                    thoughts = self.thought_processor.process_signal(visual_features)
                    if len(thoughts.shape) == 1:
                        thoughts = thoughts.unsqueeze(0)
                except Exception as e:
                    logging.error(f"Exception in thought_processor.process_signal: {e}. Using zero tensor.")
                    thoughts = torch.zeros((1, self.hidden_dim))

                try:
                    actions = self.action_generator.process_signal(thoughts)
                except Exception as e:
                    logging.error(f"Exception in action_generator.process_signal: {e}. Using zero tensor.")
                    actions = torch.zeros(self.input_dim)

                # Remove batch dimension from final output if present
                if len(actions.shape) > 1:
                    actions = actions.squeeze(0)

                # Ensure outputs are bounded
                actions = torch.clamp(actions, -1.0, 1.0)

                # Energy consumption with bounds
                self.total_energy = max(0.0, min(1000.0, self.total_energy - 0.1))

                return actions
            
        except Exception as e:
            logging.error(f"Exception in process_vision: {e}. Returning zero actions.")
            return torch.zeros(self.input_dim)
        
    def interact_with(self, other_brain, strength=0.5):
        try:
            # Bound strength value
            strength = max(0.0, min(1.0, strength))

            # Neural interactions with error handling
            shared_visual = self.visual_cortex.interact_with(other_brain.visual_cortex, strength)
            shared_thoughts = self.thought_processor.interact_with(other_brain.thought_processor, strength)
            shared_actions = self.action_generator.interact_with(other_brain.action_generator, strength)

            # Energy transfer with bounds
            energy_diff = self.total_energy - other_brain.total_energy
            transfer = max(-10.0, min(10.0, energy_diff * 0.1))
            
            self.total_energy = max(0.0, min(1000.0, self.total_energy - transfer))
            other_brain.total_energy = max(0.0, min(1000.0, other_brain.total_energy + transfer))

            return shared_visual, shared_thoughts, shared_actions
        except Exception as e:
            logging.error(f"Exception in interact_with: {e}. Returning zeros.")
            return 0.0, 0.0, 0.0

    def count_neurons(self):
        """Safely count neurons with error handling"""
        try:
            def count_recursive(module):
                count = 1
                if hasattr(module, 'sub_neurons'):
                    for child in module.sub_neurons:
                        count += count_recursive(child)
                return min(count, self.max_neurons)  # Limit total count

            total = sum(count_recursive(x) for x in [
                self.visual_cortex,
                self.thought_processor,
                self.action_generator
            ])
            return min(total, self.max_neurons)
        except Exception as e:
            logging.error(f"Exception in count_neurons: {e}. Returning 1.")
            return 1  # Return minimum count if counting fails

    def can_grow(self):
        """Check if brain can grow new neurons"""
        return (self.total_neurons < self.max_neurons and
                self.total_energy > 100.0)

# ==============================
# Organism Definition and Behavior
# ==============================

class FractalOrganism:
    def __init__(self, x, y, size=20, feature_dim=32, max_neurons=1000):
        # Physical properties
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.size = size
        self.mass = size * 0.1

        # Neural system
        self.brain = FractalBrain(input_dim=feature_dim, hidden_dim=feature_dim*2, max_neurons=max_neurons)
        self.feature_dim = feature_dim
        self.features = torch.randn(feature_dim)

        # Visual properties with validation
        self.color = self._features_to_color()
        self.pattern_type = self._determine_pattern_type()
        self.pattern_intensity = self._determine_pattern_intensity()
        self.shape_points = self._generate_shape()

        # Life properties
        self.alive = True
        self.age = 0
    def _validate_color_component(self, value):
        """Ensure color component is a valid integer between 0 and 255"""
        try:
            value = int(value)
            return max(0, min(255, value))
        except (ValueError, TypeError):
            return 0

    def update(self, screen_width, screen_height, organisms):
            """Update organism state"""
            if not self.alive:
                return

            try:
                # Physics integration
                # Update velocity with acceleration
                self.vel += self.acc
                
                # Apply friction/damping
                self.vel *= 0.98  # Slight damping to prevent infinite movement
                
                # Update position with velocity
                self.pos += self.vel
                
                # Clear acceleration for next frame
                self.acc.x = 0
                self.acc.y = 0

                # Get visual input and process through brain
                visual_input = self._get_visual_input(organisms)
                actions = self.brain.process_vision(visual_input)

                # Apply neural network outputs as forces if valid
                if isinstance(actions, torch.Tensor) and not torch.isnan(actions).any():
                    self._apply_action_forces(actions)

                # Wrap around screen edges
                self.pos.x = self.pos.x % screen_width
                self.pos.y = self.pos.y % screen_height

                # Update life properties
                self.age += 1
                vitals = self.brain.get_vitals()

                # Death conditions
                if vitals['energy'] <= 0 or self.age > 1000:
                    self.alive = False

            except Exception as e:
                logging.error(f"Error updating organism {id(self)}: {e}")
                logging.debug(f"Organism state - Age: {self.age}, Alive: {self.alive}")

    def _get_visual_input(self, organisms):
        """Create visual input tensor from surroundings"""
        try:
            visual_input = torch.zeros(self.feature_dim)

            # Add self-perception (first 3 features are color)
            color_tensor = torch.tensor([c/255.0 for c in self.color])
            visual_input[:3] = color_tensor[:3]

            # Add velocity perception (helps with movement learning)
            if hasattr(self, 'vel'):
                velocity_magnitude = np.sqrt(self.vel.x**2 + self.vel.y**2)
                velocity_direction = np.arctan2(self.vel.y, self.vel.x) / np.pi
                if 3 < len(visual_input):
                    visual_input[3] = float(velocity_magnitude) / 10.0  # Normalize velocity
                if 4 < len(visual_input):
                    visual_input[4] = float(velocity_direction)

            # Add perception of nearby organisms
            for other in organisms:
                if other != self and other.alive:
                    distance = self.pos.distance_to(other.pos)
                    if distance < 100:  # Visual range
                        direction = (other.pos - self.pos)
                        if direction.length() > 0:
                            direction = direction.normalize()
                            angle = np.arctan2(direction.y, direction.x)
                            
                            # Map angle to feature index
                            idx = int((angle + np.pi) / (2 * np.pi) * (self.feature_dim - 5)) + 5
                            idx = min(max(5, idx), self.feature_dim - 1)
                            
                            # Set feature value based on distance and target's properties
                            intensity = 1.0 - min(1.0, distance / 100)
                            visual_input[idx] = intensity

                            # Add information about target's energy level if visible
                            if idx + 1 < self.feature_dim:
                                target_energy = float(other.brain.total_energy) / 1000.0
                                if not np.isnan(target_energy):
                                    visual_input[idx + 1] = target_energy


            return visual_input

        except Exception as e:
            logging.error(f"Error in _get_visual_input: {e}")
            return torch.zeros(self.feature_dim)


    def _apply_action_forces(self, actions):
            """Convert neural actions to physical forces with better control"""
            try:
                if not isinstance(actions, torch.Tensor):
                    return

                # Get first two action dimensions for movement control
                if len(actions) >= 2:
                    # Scale force based on neural network activation
                    activation = float(self.brain.visual_cortex.state.activation)
                    force_scale = 20.0  # Increased for more visible movement

                    # Convert actions to directional movement
                    force_x = float(actions[0].item()) * force_scale * (1 + abs(activation))
                    force_y = float(actions[1].item()) * force_scale * (1 + abs(activation))
                    
                    # Add some randomness for exploration when activation is low
                    if abs(activation) < 0.2:
                        force_x += random.uniform(-2.0, 2.0)  # Increased random movement
                        force_y += random.uniform(-2.0, 2.0)

                    # Clamp forces but allow for stronger movement
                    max_force = 40.0  # Increased maximum force
                    force_x = max(-max_force, min(max_force, force_x))
                    force_y = max(-max_force, min(max_force, force_y))

                    # Apply the forces
                    self.apply_force((force_x, force_y))

                    # Additional actions for other behaviors
                    if len(actions) >= 4:
                        try:
                            # Action 3: Energy usage control
                            energy_control = float(actions[2].item())
                            if energy_control > 0.8:
                                self.brain.total_energy += energy_control * 0.1
                            
                            # Action 4: Interaction strength
                            interaction_strength = max(0, min(1, float(actions[3].item())))
                            if not hasattr(self, 'interaction_strength'):
                                self.__dict__['interaction_strength'] = interaction_strength
                            else:
                                self.interaction_strength = interaction_strength

                        except Exception as e:
                            logging.error(f"Error processing additional actions: {e}")

            except Exception as e:
                logging.error(f"Error in _apply_action_forces: {e}")

    def apply_force(self, force):
        """Apply physics force with validation"""
        try:
            if isinstance(force, (tuple, list)) and len(force) >= 2:
                fx = float(force[0])
                fy = float(force[1])
                
                # Check for NaN
                if np.isnan(fx) or np.isnan(fy):
                    return
                    
                # Limit maximum force
                max_force = 10.0
                fx = max(-max_force, min(max_force, fx))
                fy = max(-max_force, min(max_force, fy))
                
                force = pygame.math.Vector2(fx, fy)
                
                # Validate acceleration before applying
                new_acc = force / self.mass
                if not (np.isnan(new_acc.x) or np.isnan(new_acc.y)):
                    self.acc.update(new_acc.x, new_acc.y)
                    
                    # Clamp acceleration
                    max_acc = 5.0
                    self.acc.x = max(-max_acc, min(max_acc, self.acc.x))
                    self.acc.y = max(-max_acc, min(max_acc, self.acc.y))
        except Exception as e:
            logging.error(f"Error in apply_force: {e}")


    def _features_to_color(self):
        """Convert feature vector to RGB color with validation"""
        try:
            r = self._validate_color_component((self.features[0].item() + 1) / 2 * 255)
            g = self._validate_color_component((self.features[1].item() + 1) / 2 * 255)
            b = self._validate_color_component((self.features[2].item() + 1) / 2 * 255)
            return (r, g, b)
        except (IndexError, AttributeError) as e:
            logging.error(f"Error in _features_to_color: {e}. Defaulting to (100, 100, 100).")
            return (100, 100, 100)

    def _determine_pattern_type(self):
        """Determine pattern type based on specific features"""
        try:
            # Use features 3 and 4 to determine pattern type safely
            if len(self.features) >= 5:
                feature_sum = float(self.features[3].item() + self.features[4].item())
                if feature_sum > 1:
                    return 'stripes'
                elif feature_sum < -1:
                    return 'spots'
                else:
                    return 'gradient'
            return 'gradient'  # Default pattern
        except (IndexError, AttributeError, ValueError) as e:
            logging.error(f"Error in _determine_pattern_type: {e}. Defaulting to 'gradient'.")
            return 'gradient'  # Fallback pattern

    def _determine_pattern_intensity(self):
        """Determine pattern intensity based on specific features"""
        try:
            if len(self.features) >= 6:
                intensity = (float(self.features[5].item()) + 1) / 2
                return max(0.0, min(1.0, intensity))
            return 0.5  # Default intensity
        except (IndexError, AttributeError, ValueError) as e:
            logging.error(f"Error in _determine_pattern_intensity: {e}. Defaulting to 0.5.")
            return 0.5  # Fallback intensity

    def _generate_shape(self):
        """Generate a polygon shape based on the pattern type"""
        try:
            points = []
            if self.pattern_type == 'stripes':
                # Generate a star-like shape with protrusions
                for angle in range(0, 360, 30):
                    rad = np.radians(angle)
                    x = self.size * np.cos(rad)
                    y = self.size * np.sin(rad)
                    # Alternate between outer and inner points for stripes
                    if (angle // 30) % 2 == 0:
                        points.append((x * 1.2, y * 1.2))
                    else:
                        points.append((x * 0.8, y * 0.8))
            elif self.pattern_type == 'spots':
                # Generate a more circular, smooth shape with bulges
                for angle in range(0, 360, 45):
                    rad = np.radians(angle)
                    x = self.size * (1 + 0.3 * np.sin(4 * rad)) * np.cos(rad)
                    y = self.size * (1 + 0.3 * np.sin(4 * rad)) * np.sin(rad)
                    points.append((x, y))
            else:  # 'gradient' or other patterns
                # Simple regular polygon
                for angle in range(0, 360, 60):
                    rad = np.radians(angle)
                    x = self.size * np.cos(rad)
                    y = self.size * np.sin(rad)
                    points.append((x, y))

            # Validate points and ensure we have at least a triangle
            if len(points) < 3:
                # Fallback to basic triangle
                points = [
                    (-self.size, -self.size),
                    (self.size, -self.size),
                    (0, self.size)
                ]
            return points
        except Exception as e:
            logging.error(f"Error in _generate_shape: {e}. Defaulting to basic triangle.")
            # Fallback to basic triangle if anything goes wrong
            return [
                (-self.size, -self.size),
                (self.size, -self.size),
                (0, self.size)
            ]

    def reproduce(self, mate, mutation_rate=0.1):
        """Reproduce with another organism to create a child organism with possible mutations"""
        try:
            # Check reproduction energy requirements
            if not hasattr(self.brain, 'REPRODUCTION_ENERGY'):
                self.brain.REPRODUCTION_ENERGY = 150.0  # Default value if not set
                
            if self.brain.total_energy < self.brain.REPRODUCTION_ENERGY or mate.brain.total_energy < mate.brain.REPRODUCTION_ENERGY:
                return None

            # Deduct energy for reproduction
            self.brain.total_energy -= 50.0
            mate.brain.total_energy -= 50.0

            # Blend features
            child_features = (self.features + mate.features) / 2

            # Apply mutations
            for i in range(len(child_features)):
                if random.random() < mutation_rate:
                    child_features[i] += random.uniform(-0.1, 0.1)

            # Clamp mutated features to prevent extreme values
            child_features = torch.clamp(child_features, -1.0, 1.0)

            # Create child organism
            child = FractalOrganism(
                x=(self.pos.x + mate.pos.x) / 2 + random.uniform(-10, 10),
                y=(self.pos.y + mate.pos.y) / 2 + random.uniform(-10, 10),
                size=self.size,
                feature_dim=self.feature_dim,
                max_neurons=self.brain.max_neurons
            )
            child.features = child_features
            child.color = child._features_to_color()
            child.pattern_type = child._determine_pattern_type()
            child.pattern_intensity = child._determine_pattern_intensity()
            child.shape_points = child._generate_shape()
            child.brain = self._mutate_brain(mate.brain, mutation_rate)

            return child
        except Exception as e:
            logging.error(f"Error in reproduction: {e}")
            return None

    def _mutate_brain(self, brain, mutation_rate):
        """Mutate the brain's neurons"""
        try:
            # For simplicity, we can randomly add connections or adjust activation
            # Here, we'll randomly adjust activation levels
            brain.visual_cortex.state.activation += random.uniform(-0.1, 0.1)
            brain.thought_processor.state.activation += random.uniform(-0.1, 0.1)
            brain.action_generator.state.activation += random.uniform(-0.1, 0.1)
            
            # Ensure activations stay in valid range
            brain.visual_cortex.state.activation = max(-1.0, min(1.0, brain.visual_cortex.state.activation))
            brain.thought_processor.state.activation = max(-1.0, min(1.0, brain.thought_processor.state.activation))
            brain.action_generator.state.activation = max(-1.0, min(1.0, brain.action_generator.state.activation))
            
            return brain
        except Exception as e:
            logging.error(f"Error in brain mutation: {e}. Returning unmutated brain.")
            return brain

    def interact_with(self, other):
        """Interact with another organism"""
        try:
            distance = self.pos.distance_to(other.pos)
            if distance < self.size + other.size:
                # Neural interaction
                interaction_strength = 1.0 - distance / (self.size + other.size)
                self.brain.interact_with(other.brain, interaction_strength)

                # Physical interaction (simple collision)
                direction = (self.pos - other.pos).normalize()
                force = direction * interaction_strength * 5
                self.apply_force(force)
                other.apply_force(-force)

                return True
            return False
        except Exception as e:
            logging.error(f"Error in organism interaction: {e}")
            return False

    def _blend_patterns(self, pattern1: str, pattern2: str) -> str:
        """Blend two pattern types to create a new pattern type"""
        try:
            if pattern1 == pattern2:
                return pattern1
            else:
                # Simple blending logic: randomly choose one of the parent patterns or a new pattern
                return random.choice([pattern1, pattern2, 'stripes', 'spots', 'gradient'])
        except Exception as e:
            logging.error(f"Error in _blend_patterns: {e}. Defaulting to 'gradient'.")
            return 'gradient'  # Default pattern if anything goes wrong

# ==============================
# Physics and Interaction Handling
# ==============================

class PhysicsEngine:
    def __init__(self, width: int, height: int, config: PhysicsConfig):
        self.config = config
        # Initialize pymunk space
        self.space = pymunk.Space()
        self.space.damping = self.config.DAMPING

        # Create boundaries
        self.create_boundaries(width, height)

        # Collision handler for organisms
        handler = self.space.add_collision_handler(
            self.config.COLLISION_TYPE_ORGANISM,
            self.config.COLLISION_TYPE_ORGANISM
        )
        handler.begin = self.handle_collision

        # Track interactions
        self.current_interactions: Set[tuple] = set()

        # Store dimensions
        self.width = width
        self.height = height

    def update(self, dt: float):
        """Update physics simulation"""
        try:
            # Pymunk works best with a fixed time step
            fixed_dt = 1.0 / 60.0
            steps = max(1, min(4, int(dt / fixed_dt)))  # Limit max steps to prevent spiral

            for _ in range(steps):
                self.space.step(fixed_dt)

            # Update organism positions from physics bodies
            for body in self.space.bodies:
                if hasattr(body, 'organism'):
                    try:
                        organism = body.organism
                        
                        # Validate positions
                        if not (np.isnan(body.position.x) or np.isnan(body.position.y)):
                            new_x = float(body.position.x % self.width)
                            new_y = float(body.position.y % self.height)
                            # Update pygame Vector2 position
                            organism.pos.update(new_x, new_y)
                        else:
                            # Reset to center if NaN
                            body.position = self.width/2, self.height/2
                            organism.pos.update(self.width/2, self.height/2)
                        
                        # Validate velocities
                        if not (np.isnan(body.velocity.x) or np.isnan(body.velocity.y)):
                            max_velocity = 200.0
                            vx = max(-max_velocity, min(max_velocity, body.velocity.x))
                            vy = max(-max_velocity, min(max_velocity, body.velocity.y))
                            # Update pygame Vector2 velocity
                            organism.vel.update(vx, vy)
                        else:
                            body.velocity = (0, 0)
                            organism.vel.update(0, 0)

                    except Exception as e:
                        logging.error(f"Error updating organism physics state: {e}")
                        # Reset body to safe state
                        body.position = self.width/2, self.height/2
                        body.velocity = (0, 0)
                        try:
                            organism.pos.update(self.width/2, self.height/2)
                            organism.vel.update(0, 0)
                        except:
                            pass

        except Exception as e:
            logging.error(f"Error updating physics: {e}")

    def create_boundaries(self, width: int, height: int):
        """Create screen boundaries"""
        try:
            walls = [
                [(0, 0), (width, 0)],      # Top
                [(width, 0), (width, height)],  # Right
                [(width, height), (0, height)],  # Bottom
                [(0, height), (0, 0)]       # Left
            ]

            for wall in walls:
                shape = pymunk.Segment(self.space.static_body, wall[0], wall[1], 0)
                shape.elasticity = self.config.ELASTICITY
                shape.friction = self.config.FRICTION
                self.space.add(shape)
        except Exception as e:
            logging.error(f"Error creating boundaries: {e}")

    def add_organism(self, organism: FractalOrganism) -> pymunk.Body:
        """Add organism to physics space"""
        try:
            # Validate mass
            mass = max(0.1, organism.mass)  # Ensure positive mass
            moment = pymunk.moment_for_circle(mass, 0, organism.size)
            
            body = pymunk.Body(mass, moment)
            
            # Validate initial position and velocity
            body.position = (
                float(organism.pos.x % self.width),
                float(organism.pos.y % self.height)
            )
            
            # Clamp initial velocity
            max_initial_velocity = 50.0
            vel_x = max(-max_initial_velocity, min(max_initial_velocity, organism.vel.x))
            vel_y = max(-max_initial_velocity, min(max_initial_velocity, organism.vel.y))
            body.velocity = (vel_x, vel_y)

            # Validate shape points and create polygon
            valid_vertices = []
            for x, y in organism.shape_points:
                if not (np.isnan(x) or np.isnan(y)):
                    valid_vertices.append((float(x), float(y)))

            # If insufficient valid vertices, create default circle shape
            if len(valid_vertices) < 3:
                logging.warning(f"Insufficient valid vertices for organism {id(organism)}, using circle shape")
                shape = pymunk.Circle(body, organism.size)
            else:
                shape = pymunk.Poly(body, valid_vertices)

            shape.elasticity = self.config.ELASTICITY
            shape.friction = self.config.FRICTION
            shape.collision_type = self.config.COLLISION_TYPE_ORGANISM

            # Store reference to organism
            shape.organism = organism
            body.organism = organism

            self.space.add(body, shape)
            return body
        except Exception as e:
            logging.error(f"Error adding organism to physics: {e}")
            return None

    def handle_collision(self, arbiter, space, data):
        """Handle collision between organisms"""
        try:
            # Get colliding organisms
            shape_a, shape_b = arbiter.shapes
            org_a, org_b = shape_a.organism, shape_b.organism

            # Add to interaction set
            interaction_pair = tuple(sorted([id(org_a), id(org_b)]))
            self.current_interactions.add(interaction_pair)

            # Calculate collision response with validation
            restitution = max(0, min(1, self.config.ELASTICITY))
            j = -(1 + restitution) * arbiter.total_ke / 2

            # Validate impulse
            if not np.isnan(j):
                # Clamp impulse to prevent extreme values
                max_impulse = 1000.0
                j = max(-max_impulse, min(max_impulse, j))

                body_a = shape_a.body
                body_b = shape_b.body
                
                normal = arbiter.normal
                point = arbiter.contact_point_set.points[0].point_a

                # Apply impulse along the collision normal
                body_a.apply_impulse_at_world_point(j * normal, point)
                body_b.apply_impulse_at_world_point(-j * normal, point)

            return True
        except Exception as e:
            logging.error(f"Error handling collision: {e}")
            return False

    def process_interactions(self, organisms: List[FractalOrganism]):
        """Process all current interactions"""
        try:
            # Process collision-based interactions
            for org_a_id, org_b_id in self.current_interactions:
                org_a = next((org for org in organisms if id(org) == org_a_id), None)
                org_b = next((org for org in organisms if id(org) == org_b_id), None)

                if org_a and org_b and org_a.alive and org_b.alive:
                    # Neural interaction
                    shared_thoughts = org_a.brain.interact_with(org_b.brain)

                    # Energy transfer based on neural activity
                    energy_diff = org_a.brain.total_energy - org_b.brain.total_energy
                    transfer = max(-10.0, min(10.0, energy_diff * 0.1))  # Limit transfer amount
                    org_a.brain.total_energy = max(0, org_a.brain.total_energy - transfer)
                    org_b.brain.total_energy = max(0, org_b.brain.total_energy + transfer)

            # Clear interactions for next frame
            self.current_interactions.clear()

            # Process proximity-based interactions
            for i, org_a in enumerate(organisms):
                if not org_a.alive:
                    continue

                for org_b in organisms[i+1:]:
                    if not org_b.alive:
                        continue

                    # Calculate distance with validation
                    dx = org_b.pos.x - org_a.pos.x
                    dy = org_b.pos.y - org_a.pos.y
                    
                    if np.isnan(dx) or np.isnan(dy):
                        continue
                        
                    distance = np.sqrt(dx*dx + dy*dy)

                    if distance < self.config.INTERACTION_RADIUS:
                        # Calculate interaction strength based on distance
                        strength = 1.0 - (distance / self.config.INTERACTION_RADIUS)

                        # Neural field effect with reduced strength
                        field_interaction = strength * 0.5

                        # Calculate force with validation
                        force_magnitude = field_interaction * self.config.FORCE_SCALE
                        force_angle = np.arctan2(dy, dx)

                        if not (np.isnan(force_magnitude) or np.isnan(force_angle)):
                            force_x = np.cos(force_angle) * force_magnitude
                            force_y = np.sin(force_angle) * force_magnitude

                            # Clamp forces
                            max_force = 100.0
                            force_x = max(-max_force, min(max_force, force_x))
                            force_y = max(-max_force, min(max_force, force_y))

                            # Apply forces through physics bodies
                            body_a = next((body for body in self.space.bodies 
                                         if hasattr(body, 'organism') and body.organism == org_a), None)
                            body_b = next((body for body in self.space.bodies 
                                         if hasattr(body, 'organism') and body.organism == org_b), None)

                            if body_a and body_b:
                                body_a.apply_force_at_local_point((-force_x, -force_y), (0, 0))
                                body_b.apply_force_at_local_point((force_x, force_y), (0, 0))

                            # Apply direct forces to organisms as well
                            org_a.apply_force((-force_x, -force_y))
                            org_b.apply_force((force_x, force_y))

        except Exception as e:
            logging.error(f"Error processing interactions: {e}")

# ==============================
# Visualization with PyGame
# ==============================

class NeuralVisualizer:
    def __init__(self, width: int, height: int, config: VisualizationConfig):
        self.width = width
        self.height = height
        self.config = config
        self.neuron_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.connection_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    def _apply_pattern_overlay(self, organism, surface):
        """Apply visual pattern overlay based on organism type"""
        try:
            if not organism.alive:
                return

            pattern_alpha = int(255 * organism.pattern_intensity)
            pattern_color = (
                255 - organism.color[0],
                255 - organism.color[1],
                255 - organism.color[2],
                pattern_alpha
            )

            # Create pattern surface
            pattern_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

            if organism.pattern_type == 'stripes':
                # Draw alternating stripes
                stride = max(5, int(organism.size * 0.5))
                x, y = int(organism.pos.x), int(organism.pos.y)
                for i in range(-2, 3):
                    offset = i * stride
                    pygame.draw.line(pattern_surface, pattern_color,
                                (x - organism.size, y + offset),
                                (x + organism.size, y + offset), 2)

            elif organism.pattern_type == 'spots':
                # Draw spots in a circular pattern
                x, y = int(organism.pos.x), int(organism.pos.y)
                spot_size = max(2, int(organism.size * 0.2))
                for angle in range(0, 360, 45):
                    spot_x = x + int(np.cos(np.radians(angle)) * organism.size * 0.7)
                    spot_y = y + int(np.sin(np.radians(angle)) * organism.size * 0.7)
                    pygame.draw.circle(pattern_surface, pattern_color,
                                    (spot_x, spot_y), spot_size)

            else:  # gradient
                # Draw radial gradient
                x, y = int(organism.pos.x), int(organism.pos.y)
                max_radius = int(organism.size * 1.2)
                for radius in range(max_radius, 0, -2):
                    alpha = int((radius / max_radius) * pattern_alpha)
                    current_color = (*pattern_color[:3], alpha)
                    pygame.draw.circle(pattern_surface, current_color,
                                    (x, y), radius, 1)

            # Blend pattern with surface
            surface.blit(pattern_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)

        except Exception as e:
            logging.error(f"Error applying pattern overlay: {e}")

    def draw_brain_state(self, organism: FractalOrganism, surface: pygame.Surface):
        """Draw neural activity visualization with patterns and NaN handling"""
        try:
            # Clear previous state
            self.neuron_surface.fill((0, 0, 0, 0))
            self.connection_surface.fill((0, 0, 0, 0))

            # Get brain vitals with NaN check
            vitals = organism.brain.get_vitals()
            if any(np.isnan(value) for value in vitals.values() if isinstance(value, (int, float))):
                logging.warning(f"NaN detected in vitals for organism {id(organism)}. Marking for death.")
                organism.alive = False
                return

            # Calculate neural positions based on fractal pattern
            def plot_neural_layer(center, radius, neurons, depth=0):
                if depth >= 3 or not neurons:
                    return
                
                # Convert neurons to list if it's a ModuleList
                if hasattr(neurons, 'sub_neurons'):
                    neurons = neurons.sub_neurons
                
                if not neurons or len(neurons) == 0:
                    return

                angle_step = 2 * np.pi / len(neurons)
                for i, neuron in enumerate(neurons):
                    try:
                        angle = i * angle_step
                        x = center[0] + radius * np.cos(angle)
                        y = center[1] + radius * np.sin(angle)

                        # NaN check for coordinates
                        if np.isnan(x) or np.isnan(y):
                            logging.warning(f"NaN coordinates detected for neuron {i}. Skipping.")
                            continue

                        # Draw neuron
                        activation = float(neuron.state.activation)
                        connections = int(neuron.state.connections)

                        # Check for NaNs in neuron state
                        if np.isnan(activation) or np.isnan(connections):
                            logging.warning(f"NaN detected in neuron state. Marking organism for death.")
                            organism.alive = False
                            return

                        # Ensure coordinates are valid integers
                        x_pos = int(np.clip(x, 0, self.width))
                        y_pos = int(np.clip(y, 0, self.height))

                        color = self._get_neuron_color(activation, connections)
                        pygame.draw.circle(self.neuron_surface, color, (x_pos, y_pos), 5)

                        # Draw connections with safety checks
                        if connections > 0:
                            alpha = int(255 * min(connections / self.config.MAX_NEURAL_CONNECTIONS, 1))
                            if not np.isnan(alpha):
                                connection_color = (*self.config.CONNECTION_COLOR[:3], alpha)
                                pygame.draw.line(
                                    self.connection_surface,
                                    connection_color,
                                    (x_pos, y_pos),
                                    (int(center[0]), int(center[1])),
                                    2
                                )

                        # Recursively draw sub-neurons
                        if hasattr(neuron, 'sub_neurons') and neuron.sub_neurons:
                            child_radius = radius * 0.5
                            child_center = (x, y)
                            plot_neural_layer(child_center, child_radius, neuron.sub_neurons, depth + 1)
                    except Exception as e:
                        logging.error(f"Error plotting neuron {i}: {e}")
                        continue

            # Draw neural network
            try:
                center = (organism.pos.x, organism.pos.y)
                if not (np.isnan(center[0]) or np.isnan(center[1])):
                    plot_neural_layer(center, organism.size * 2,
                                    [organism.brain.visual_cortex,
                                    organism.brain.thought_processor,
                                    organism.brain.action_generator])
            except Exception as e:
                logging.error(f"Error plotting neural network: {e}")

            # Apply pattern overlay with safety checks
            try:
                self._apply_pattern_overlay(organism, surface)
            except Exception as e:
                logging.error(f"Error applying pattern overlay: {e}")

            # Combine surfaces with alpha blending
            surface.blit(self.connection_surface, (0, 0))
            surface.blit(self.neuron_surface, (0, 0))

        except Exception as e:
            logging.error(f"Error in draw_brain_state: {e}")

    def _get_neuron_color(self, activation: float, connections: int) -> Tuple[int, int, int]:
        """Generate color based on neuron state"""
        try:
            # Use HSV color space for smooth transitions
            hue = (activation + 1) / 2  # Map -1,1 to 0,1
            saturation = min(connections / self.config.MAX_NEURAL_CONNECTIONS, 1)
            value = 0.8 + 0.2 * activation

            # Convert to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            return tuple(int(255 * x) for x in rgb)
        except Exception as e:
            logging.error(f"Error in _get_neuron_color: {e}. Defaulting to gray.")
            return (100, 100, 100)

class SimulationVisualizer:
    def __init__(self, width: int, height: int, config: VisualizationConfig):
        pygame.init()
        self.width = width
        self.height = height
        self.config = config
        
        # Enable double buffering and vsync
        self.screen = pygame.display.set_mode(
            (width, height), 
            pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.SCALED, 
            vsync=1
        )
        pygame.display.set_caption("Fractal Life Simulation")
        
        # Create off-screen surfaces for double buffering
        self.buffer = pygame.Surface((width, height), pygame.SRCALPHA)
        self.neural_viz = NeuralVisualizer(width, height, config)
        self.background = pygame.Surface((width, height))
        self.background.fill(config.BACKGROUND_COLOR)
        
        # Additional surfaces for layered rendering
        self.organism_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.interaction_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.stats_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    def _validate_color(self, color):
        """Validate and ensure color values are proper RGB integers"""
        try:
            if len(color) >= 3:
                return (
                    max(0, min(255, int(color[0]))),
                    max(0, min(255, int(color[1]))),
                    max(0, min(255, int(color[2])))
                )
            return (100, 100, 100)  # Default fallback color
        except Exception as e:
            logging.error(f"Error validating color: {e}")
            return (100, 100, 100)



    def draw_frame(self, organisms: List[FractalOrganism], stats: Dict):
        """Draw complete frame with all visualizations"""
        try:
            # Clear all off-screen surfaces
            self.organism_surface.fill((0, 0, 0, 0))
            self.interaction_surface.fill((0, 0, 0, 0))
            self.stats_surface.fill((0, 0, 0, 0))

            # Draw organisms and their neural states
            for organism in organisms:
                if organism.alive:
                    self._draw_organism(organism, self.organism_surface)
                    self.neural_viz.draw_brain_state(organism, self.interaction_surface)

            # Draw statistics
            self._draw_stats(stats, self.stats_surface)

            # Blit off-screen surfaces to the main display surface
            self.screen.blit(self.background, (0, 0))  # Background layer
            self.screen.blit(self.organism_surface, (0, 0))
            self.screen.blit(self.interaction_surface, (0, 0))
            self.screen.blit(self.stats_surface, (0, 0))

            # Flip display buffers to avoid flickering
            pygame.display.flip()
        except Exception as e:
            logging.error(f"Error in draw_frame: {e}")
            
    def _draw_organism(self, organism: FractalOrganism, surface: pygame.Surface):
        """Draw organism body with color and pattern"""
        try:
            # Validate color before drawing
            safe_color = self._validate_color(organism.color)
            
            # Draw shape with validated color
            points = []
            for x, y in organism.shape_points:
                try:
                    px = organism.pos.x + x
                    py = organism.pos.y + y
                    if not (np.isnan(px) or np.isnan(py)):
                        points.append((px, py))
                    else:
                        logging.warning(f"NaN detected in shape points for organism {id(organism)}. Skipping point.")
                except Exception as e:
                    logging.error(f"Error processing shape points for organism {id(organism)}: {e}")
                    continue

            if len(points) >= 3:
                pygame.draw.polygon(surface, safe_color, points)
            else:
                logging.warning(f"Insufficient valid points to draw organism {id(organism)}. Skipping drawing.")

            # Draw energy bar
            energy_percentage = min(max(organism.brain.total_energy / 1000.0, 0), 1)
            bar_width = organism.size * 2
            bar_height = 4
            bar_pos = (organism.pos.x - bar_width / 2, organism.pos.y - organism.size - 10)
            pygame.draw.rect(surface, (50, 50, 50), (*bar_pos, bar_width, bar_height))
            pygame.draw.rect(surface, (0, 255, 0),  # Using direct color value for energy bar
                             (*bar_pos, bar_width * energy_percentage, bar_height))
        except Exception as e:
            logging.error(f"Error in _draw_organism: {e}")

    def _draw_stats(self, stats: Dict, surface: pygame.Surface):
        """Draw simulation statistics"""
        try:
            font = pygame.font.Font(None, 24)
            y_pos = 10
            for key, value in stats.items():
                text = font.render(f"{key.capitalize()}: {value}", True, (255, 255, 255))
                surface.blit(text, (10, y_pos))
                y_pos += 25
        except Exception as e:
            logging.error(f"Error in _draw_stats: {e}")

    def cleanup(self):
        """Clean up pygame resources"""
        try:
            pygame.quit()
        except Exception as e:
            logging.error(f"Error during pygame cleanup: {e}")

# ==============================
# Interaction Field (Optional Enhancement)
# ==============================

class InteractionField:
    def __init__(self, width: int, height: int, resolution: int = 50):
        self.width = max(1, width)
        self.height = max(1, height)
        self.resolution = max(10, min(resolution, 100))  # Bound resolution

        # Create field grid with safe dimensions
        self.grid_w = max(1, self.width // self.resolution)
        self.grid_h = max(1, self.height // self.resolution)
        self.field = np.zeros((self.grid_w, self.grid_h, 3))

    def update(self, organisms: List[FractalOrganism]):
        """Update field based on organism neural activity with safety checks"""
        try:
            # Decay field
            self.field *= 0.9
            np.clip(self.field, 0, 1, out=self.field)

            for org in organisms:
                if not org.alive:
                    continue

                try:
                    # Safe position to grid conversion
                    pos_x = max(0, min(float(org.pos.x), self.width))
                    pos_y = max(0, min(float(org.pos.y), self.height))
                    
                    grid_x = int((pos_x / self.width) * (self.grid_w - 1))
                    grid_y = int((pos_y / self.height) * (self.grid_h - 1))

                    # Get neural activity with safety checks
                    vitals = org.brain.get_vitals()
                    activity_color = np.array([
                        max(0.0, min(1.0, float(vitals['activation']))),
                        max(0.0, min(1.0, float(vitals['energy']) / 1000.0)),
                        max(0.0, min(1.0, float(vitals['connections']) / 100.0))
                    ])

                    # Apply to field with falloff
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            x = (grid_x + dx) % self.grid_w
                            y = (grid_y + dy) % self.grid_h
                            
                            distance = np.sqrt(dx * dx + dy * dy)
                            if distance < 3:
                                intensity = max(0.0, min(1.0, (3 - distance) / 3))
                                self.field[x, y] += activity_color * intensity

                except (ValueError, TypeError, ZeroDivisionError) as e:
                    logging.error(f"Error updating field for organism {id(org)}: {e}")
                    continue  # Skip problematic organisms

            # Ensure field values stay in valid range
            np.clip(self.field, 0, 1, out=self.field)
        except Exception as e:
            logging.error(f"Error in InteractionField.update: {e}")

    def get_field_at(self, x: float, y: float) -> np.ndarray:
        """Safely get field value at position"""
        try:
            x = max(0, min(float(x), self.width))
            y = max(0, min(float(y), self.height))
            
            grid_x = int((x / self.width) * (self.grid_w - 1))
            grid_y = int((y / self.height) * (self.grid_h - 1))
            
            return self.field[grid_x, grid_y]
        except (ValueError, TypeError, IndexError) as e:
            logging.error(f"Error in get_field_at for position ({x}, {y}): {e}")
            return np.zeros(3)

# ==============================
# Simulation State Tracking
# ==============================

class SimulationState:
    """Tracks the current state of the simulation"""
    def __init__(self):
        self.running = False
        self.paused = False
        self.step_count = 0
        self.stats = {
            'population': 0,
            'avg_energy': 0.0,
            'avg_neurons': 0.0,
            'avg_connections': 0.0,
            'total_interactions': 0
        }
        self.selected_organism: Optional[FractalOrganism] = None

# ==============================
# Main Simulation Class
# ==============================

class FractalLifeSimulation:
    def __init__(self, config: SimulationConfig, shared_data: Dict):
        self.config = config
        self.state = SimulationState()

        # Shared data for Gradio interface
        self.shared_data = shared_data
        self.shared_lock = threading.Lock()

        # Initialize systems
        visualization_config = VisualizationConfig()
        self.visualizer = SimulationVisualizer(config.WIDTH, config.HEIGHT, visualization_config)
        self.physics = PhysicsEngine(config.WIDTH, config.HEIGHT, PhysicsConfig())
        self.field = InteractionField(config.WIDTH, config.HEIGHT, resolution=50)

        # Event queues for thread communication
        self.event_queue = Queue()
        self.stats_queue = Queue()

        # Initialize organisms
        self.organisms: List[FractalOrganism] = []
        self._init_organisms()

        # Pygame threading control
        self.running = False
        self.thread = None

    def _init_organisms(self):
        """Initialize starting organisms"""
        try:
            for _ in range(self.config.MIN_ORGANISMS):
                x = random.uniform(0, self.config.WIDTH)
                y = random.uniform(0, self.config.HEIGHT)
                organism = FractalOrganism(
                    x=x, y=y,
                    feature_dim=32,
                    max_neurons=self.config.MAX_NEURONS
                )
                self.organisms.append(organism)
                self.physics.add_organism(organism)
        except Exception as e:
            logging.error(f"Error initializing organisms: {e}")

    def _process_reproduction(self):
        """Handle organism reproduction"""
        try:
            new_organisms = []
            for org in self.organisms:
                if not org.alive or len(self.organisms) + len(new_organisms) >= self.config.MAX_ORGANISMS:
                    continue

                if org.brain.total_energy > self.config.REPRODUCTION_ENERGY:
                    # Find a mate (simple random selection for demonstration)
                    potential_mates = [o for o in self.organisms if o != org and o.alive and o.brain.total_energy > self.config.REPRODUCTION_ENERGY]
                    if potential_mates:
                        mate = random.choice(potential_mates)
                        child = org.reproduce(mate, mutation_rate=self.config.MUTATION_RATE)
                        if child:
                            new_organisms.append(child)
                            self.physics.add_organism(child)

            self.organisms.extend(new_organisms)
        except Exception as e:
            logging.error(f"Error processing reproduction: {e}")

    def _update_stats(self):
        """Update simulation statistics"""
        try:
            living_organisms = [org for org in self.organisms if org.alive]
            population = len(living_organisms)

            if population > 0:
                self.state.stats.update({
                    'population': population,
                    'avg_energy': sum(org.brain.total_energy for org in living_organisms) / population,
                    'avg_neurons': sum(org.brain.total_neurons for org in living_organisms) / population,
                    'avg_connections': sum(sum(n.state.connections for n in [org.brain.visual_cortex,
                                                                            org.brain.thought_processor,
                                                                            org.brain.action_generator])
                                       for org in living_organisms) / population,
                    'total_interactions': len(self.physics.current_interactions)
                })
            else:
                self.state.stats.update({
                    'population': 0,
                    'avg_energy': 0.0,
                    'avg_neurons': 0.0,
                    'avg_connections': 0.0,
                    'total_interactions': 0
                })

            with self.shared_lock:
                self.shared_data['stats'] = self.state.stats.copy()
        except Exception as e:
            logging.error(f"Error updating statistics: {e}")

    def _main_loop(self):
        """Main simulation loop"""
        try:
            clock = pygame.time.Clock()

            while self.running:
                dt = clock.tick(self.config.TARGET_FPS) / 1000.0  # Delta time in seconds

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop()

                # Process external events
                while not self.event_queue.empty():
                    event = self.event_queue.get()
                    self._handle_event(event)

                if not self.state.paused:
                    # Update physics
                    self.physics.update(dt)

                    # Update neural field
                    self.field.update(self.organisms)

                    # Update organisms
                    for org in self.organisms:
                        if org.alive:
                            # Update organism state
                            org.update(self.config.WIDTH, self.config.HEIGHT, self.organisms)

                            # Process field interactions (placeholder for actual implementation)
                            field_value = self.field.get_field_at(org.pos.x, org.pos.y)
                            # org.process_field_input(field_value)  # Implement if needed

                            # Energy decay
                            org.brain.total_energy = max(0.0, org.brain.total_energy - self.config.ENERGY_DECAY)

                    # Process physical interactions
                    self.physics.process_interactions(self.organisms)

                    # Handle reproduction
                    self._process_reproduction()

                    # Remove dead organisms
                    self.organisms = [org for org in self.organisms if org.alive]

                    # Maintain minimum population
                    while len(self.organisms) < self.config.MIN_ORGANISMS and len(self.organisms) < self.config.MAX_ORGANISMS:
                        x = random.uniform(0, self.config.WIDTH)
                        y = random.uniform(0, self.config.HEIGHT)
                        organism = FractalOrganism(
                            x=x, y=y,
                            feature_dim=32,
                            max_neurons=self.config.MAX_NEURONS
                        )
                        self.organisms.append(organism)
                        self.physics.add_organism(organism)

                    # Update statistics
                    self._update_stats()

                    # Draw frame
                    self.visualizer.draw_frame(self.organisms, self.state.stats)
        except Exception as e:
            logging.error(f"Exception in main loop: {e}")
        finally:
            # Cleanup when simulation stops
            self.visualizer.cleanup()

    def start(self):
        """Start simulation in separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._main_loop)
            self.thread.start()
            logging.info("Simulation started.")

    def stop(self):
        """Stop simulation"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
            logging.info("Simulation stopped.")

    def pause(self):
        """Pause/unpause simulation"""
        self.state.paused = not self.state.paused
        logging.info(f"Simulation {'paused' if self.state.paused else 'resumed'}.")

    def _handle_event(self, event: Dict):
        """Handle external events"""
        try:
            if event['type'] == 'select_organism':
                organism_id = event['organism_id']
                self.state.selected_organism = next(
                    (org for org in self.organisms if id(org) == organism_id),
                    None
                )
                logging.info(f"Organism {organism_id} selected.")
            elif event['type'] == 'add_energy':
                if self.state.selected_organism:
                    self.state.selected_organism.brain.total_energy += event['amount']
                    logging.info(f"Added {event['amount']} energy to organism {id(self.state.selected_organism)}.")
            elif event['type'] == 'modify_neurons':
                if self.state.selected_organism:
                    # Placeholder for neuron modification logic
                    logging.info(f"Modify neurons event received for organism {id(self.state.selected_organism)}.")
        except Exception as e:
            logging.error(f"Error handling event {event}: {e}")

# ==============================
# Gradio Interface
# ==============================

class FractalLifeInterface:
    def __init__(self):
        self.simulation: Optional[FractalLifeSimulation] = None
        self.history = {
            'population': [],
            'avg_energy': [],
            'avg_neurons': [],
            'time': []
        }
        self.selected_organism_id = None
        self.frame_image = None
        self.DEFAULT_WIDTH = 1024
        self.DEFAULT_HEIGHT = 768
        
        self.shared_data = {
            'stats': {
                'population': 0,
                'avg_energy': 0.0,
                'avg_neurons': 0.0,
                'avg_connections': 0.0,
                'total_interactions': 0
            }
        }

    def get_frame(self):
        """Retrieve the latest frame from Pygame."""
        try:
            if self.simulation and self.simulation.running:
                with self.simulation.shared_lock:
                    # Get pygame surface
                    screen = self.simulation.visualizer.screen
                    
                    # Get the size of the screen
                    width = screen.get_width()
                    height = screen.get_height()
                    
                    # Create a new surface with alpha channel
                    surf_alpha = pygame.Surface((width, height), pygame.SRCALPHA)
                    surf_alpha.blit(screen, (0, 0))
                    
                    # Convert Pygame surface to PIL Image
                    data_string = pygame.image.tostring(surf_alpha, 'RGBA')
                    image = Image.frombytes('RGBA', (width, height), data_string)
                    
                    # Convert to RGB
                    image = image.convert('RGB')
                    
                    return image
            else:
                # Return a blank image
                return Image.new('RGB', (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), (0, 0, 0))
                
        except Exception as e:
            logging.error(f"Error in get_frame: {e}")
            # Return a fallback image in case of error
            return Image.new('RGB', (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), (0, 0, 0))
        
    def update_display(self):
        """Update display by retrieving the latest frame and statistics."""
        try:
            # Get the frame as PIL Image
            image = self.get_frame()
            
            # Update statistics only if simulation is running
            if self.simulation and self.simulation.running:
                with self.simulation.shared_lock:
                    stats = self.shared_data['stats']
                    # Update history
                    self.history['time'].append(self.simulation.state.step_count)
                    self.history['population'].append(stats['population'])
                    self.history['avg_energy'].append(stats['avg_energy'])
                    self.history['avg_neurons'].append(stats['avg_neurons'])
                    
                    # Limit history length to prevent memory issues
                    max_history = 1000
                    if len(self.history['time']) > max_history:
                        for key in self.history:
                            self.history[key] = self.history[key][-max_history:]
            else:
                stats = self.shared_data['stats']

            # Update plots
            stats_fig = self._create_stats_plot()
            neural_fig = self._create_neural_plot()

            # Update organism list only if simulation is running
            organism_list = []
            if self.simulation and self.simulation.running:
                organism_list = [
                    (f"Organism {id(org)}", id(org))
                    for org in self.simulation.organisms
                    if org.alive
                ]

            # Update selected organism vitals
            vitals = None
            if self.selected_organism_id and self.simulation and self.simulation.running:
                org = next(
                    (org for org in self.simulation.organisms
                     if id(org) == self.selected_organism_id),
                    None
                )
                if org:
                    vitals = {
                        'energy': org.brain.total_energy,
                        'neurons': org.brain.total_neurons,
                        'age': org.age,
                        'connections': sum(n.state.connections for n in
                                        [org.brain.visual_cortex,
                                         org.brain.thought_processor,
                                         org.brain.action_generator])
                    }

            # Create new Dropdown choices instead of using update
            dropdown = gr.Dropdown(
                choices=organism_list,
                label="Select Organism",
                interactive=True
            )

            return [
                image,  # Return PIL Image directly
                stats_fig,
                neural_fig,
                dropdown,
                vitals
            ]
            
        except Exception as e:
            logging.error(f"Error in update_display: {e}")
            # Return default values in case of error
            blank_image = Image.new('RGB', (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), (0, 0, 0))
            empty_fig = go.Figure()
            empty_dropdown = gr.Dropdown(choices=[])
            return [blank_image, empty_fig, empty_fig, empty_dropdown, None]

    def create_interface(self):
        with gr.Blocks(title="Fractal Life Simulator") as interface:
            gr.Markdown("# 🧬 Fractal Life Simulator")

            with gr.Row():
                # Main simulation view and controls
                with gr.Column(scale=2):
                    canvas = gr.Image(label="Simulation View", interactive=False)

                    with gr.Row():
                        start_btn = gr.Button("Start Simulation", variant="primary")
                        pause_btn = gr.Button("Pause")
                        stop_btn = gr.Button("Stop")

                    with gr.Row():
                        population_slider = gr.Slider(
                            minimum=5, maximum=100,  # Increased maximum for flexibility
                            value=10, step=1,
                            label="Initial Population"
                        )
                        mutation_rate = gr.Slider(
                            minimum=0, maximum=1, value=0.1, step=0.01,
                            label="Mutation Rate"
                        )
                        max_population_slider = gr.Slider(
                            minimum=50, maximum=500, value=50, step=10,
                            label="Max Population"
                        )

                # Statistics and organism details
                with gr.Column(scale=1):
                    stats_plot = gr.Plot(label="Population Statistics")
                    neural_plot = gr.Plot(label="Neural Activity")

                    with gr.Group():
                        gr.Markdown("### Selected Organism")
                        organism_dropdown = gr.Dropdown(
                            label="Select Organism",
                            choices=[],
                            interactive=True
                        )
                        vitals_json = gr.JSON(label="Organism Vitals")

                        with gr.Row():
                            add_energy_btn = gr.Button("Add Energy")
                            add_neurons_btn = gr.Button("Add Neurons")

            # Advanced settings tab
            with gr.Tab("Advanced Settings"):
                with gr.Row():
                    with gr.Column():
                        brain_update_rate = gr.Slider(
                            minimum=1, maximum=30, value=10, step=1,
                            label="Brain Update Rate (Hz)"
                        )
                        max_neurons = gr.Slider(
                            minimum=100, maximum=5000, value=1000, step=100,
                            label="Max Neurons per Brain"
                        )
                        energy_decay = gr.Slider(
                            minimum=0, maximum=1, value=0.1, step=0.01,
                            label="Energy Decay Rate"
                        )

                    with gr.Column():
                        interaction_strength = gr.Slider(
                            minimum=0, maximum=1, value=0.5, step=0.01,
                            label="Interaction Strength"
                        )
                        field_resolution = gr.Slider(
                            minimum=10, maximum=100, value=50, step=5,
                            label="Field Resolution"
                        )

            # Event handlers
            def start_simulation(initial_population, mutation_rate_val, max_population_val, 
                            brain_update_rate_val, max_neurons_val, energy_decay_val, 
                            interaction_strength_val, field_resolution_val):
                try:
                    if self.simulation is None:
                        config = SimulationConfig()
                        config.MIN_ORGANISMS = int(initial_population)
                        config.MUTATION_RATE = mutation_rate_val
                        config.MAX_ORGANISMS = int(max_population_val)
                        config.BRAIN_UPDATE_RATE = int(brain_update_rate_val)
                        config.MAX_NEURONS = int(max_neurons_val)
                        config.ENERGY_DECAY = energy_decay_val

                        self.simulation = FractalLifeSimulation(config, self.shared_data)
                        self.simulation.start()
                        logging.info("Simulation started via interface.")
                        return "Simulation started"
                    else:
                        logging.warning("Simulation is already running.")
                        return "Simulation is already running."
                except Exception as e:
                    logging.error(f"Error starting simulation: {e}")
                    return "Failed to start simulation."

            def pause_simulation():
                try:
                    if self.simulation:
                        self.simulation.pause()
                        status = "paused" if self.simulation.state.paused else "resumed"
                        logging.info(f"Simulation {status} via interface.")
                        return f"Simulation {status}"
                    logging.warning("No simulation running to pause/resume.")
                    return "No simulation running"
                except Exception as e:
                    logging.error(f"Error pausing simulation: {e}")
                    return "Failed to pause simulation."

            def stop_simulation():
                try:
                    if self.simulation:
                        self.simulation.stop()
                        self.simulation = None
                        logging.info("Simulation stopped via interface.")
                        return "Simulation stopped"
                    logging.warning("No simulation running to stop.")
                    return "No simulation running"
                except Exception as e:
                    logging.error(f"Error stopping simulation: {e}")
                    return "Failed to stop simulation."

            def select_organism(organism_id):
                try:
                    self.selected_organism_id = organism_id
                    if self.simulation:
                        self.simulation.event_queue.put({
                            'type': 'select_organism',
                            'organism_id': organism_id
                        })
                        logging.info(f"Organism {organism_id} selected via interface.")
                except Exception as e:
                    logging.error(f"Error selecting organism: {e}")

            def add_energy_to_organism():
                try:
                    if self.simulation and self.selected_organism_id:
                        self.simulation.event_queue.put({
                            'type': 'add_energy',
                            'amount': 50.0
                        })
                        logging.info(f"Added energy to organism {self.selected_organism_id} via interface.")
                        return "Added energy to selected organism"
                    logging.warning("No organism selected or simulation not running to add energy.")
                    return "No organism selected or simulation not running"
                except Exception as e:
                    logging.error(f"Error adding energy to organism: {e}")
                    return "Failed to add energy"

            def add_neurons_to_organism():
                try:
                    if self.simulation and self.selected_organism_id:
                        self.simulation.event_queue.put({
                            'type': 'modify_neurons',
                            'amount': 10
                        })
                        logging.info(f"Added neurons to organism {self.selected_organism_id} via interface.")
                        return "Added neurons to selected organism"
                    logging.warning("No organism selected or simulation not running to add neurons.")
                    return "No organism selected or simulation not running"
                except Exception as e:
                    logging.error(f"Error adding neurons to organism: {e}")
                    return "Failed to add neurons"

            # Create a bound method for update_display
            def bound_update_display():
                return self.update_display()

            # Connect events
            start_btn.click(
                start_simulation, 
                inputs=[population_slider, mutation_rate, max_population_slider, 
                    brain_update_rate, max_neurons, energy_decay, 
                    interaction_strength, field_resolution], 
                outputs=gr.Textbox()
            )
            pause_btn.click(pause_simulation, outputs=gr.Textbox())
            stop_btn.click(stop_simulation, outputs=gr.Textbox())

            organism_dropdown.change(select_organism, inputs=[organism_dropdown], outputs=None)
            add_energy_btn.click(add_energy_to_organism, outputs=gr.Textbox())
            add_neurons_btn.click(add_neurons_to_organism, outputs=gr.Textbox())

            # Periodic update using Gradio's update mechanism
            interface.load(
                fn=bound_update_display,
                inputs=[],
                outputs=[canvas, stats_plot, neural_plot, organism_dropdown, vitals_json],
                every=1/30,  # 30 FPS updates
                queue=True
            )

            return interface

    def _create_stats_plot(self):
        """Create statistics plot using plotly"""
        try:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("Population", "Average Energy", "Average Neurons"))

            fig.add_trace(
                go.Scatter(x=self.history['time'], y=self.history['population'],
                           name="Population"),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=self.history['time'], y=self.history['avg_energy'],
                           name="Avg Energy"),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=self.history['time'], y=self.history['avg_neurons'],
                           name="Avg Neurons"),
                row=3, col=1
            )

            fig.update_layout(height=600, showlegend=True)
            return fig
        except Exception as e:
            logging.error(f"Error creating stats plot: {e}")
            return go.Figure()

    def _create_neural_plot(self):
        """Create neural activity plot"""
        try:
            if self.selected_organism_id and self.simulation and self.simulation.running:
                org = next(
                    (org for org in self.simulation.organisms
                     if id(org) == self.selected_organism_id),
                    None
                )
                if org:
                    # Create neural activity heatmap
                    neurons = [org.brain.visual_cortex,
                            org.brain.thought_processor,
                            org.brain.action_generator]

                    activities = []
                    for neuron in neurons:
                        layer_activities = [child.state.activation for child in neuron.sub_neurons] if hasattr(neuron, 'sub_neurons') and neuron.sub_neurons else [neuron.state.activation]
                        activities.append(layer_activities)

                    activities = np.array(activities)

                    fig = go.Figure(data=go.Heatmap(z=activities, colorscale='Viridis'))
                    fig.update_layout(
                        title="Neural Activity",
                        xaxis_title="Neuron Index",
                        yaxis_title="Layer"
                    )
                    return fig
            return go.Figure()
        except Exception as e:
            logging.error(f"Error creating neural plot: {e}")
            return go.Figure()

# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    interface = FractalLifeInterface()
    gr_interface = interface.create_interface()
    # Enable queue and allow for public access
    gr_interface.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=True  # This creates a public link
    )
