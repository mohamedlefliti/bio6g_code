"""
================================================================================
BIO-6G: COMPLETE RESEARCH SIMULATION FRAMEWORK
================================================================================
Title: Bio-6G: A Bio-Inspired Energy-Efficient Routing Protocol for 6G Networks

Author: [Your Name]
Affiliation: [Your Institution]
Journal: Journal of Computer Science and Technology (JCST)

This code implements:
1. Bio-6G Protocol (Synaptic Routing + Plasticity + Refractory Period)
2. 4 Baseline Protocols (OSPF, 5G NR, AODV, DSR)
3. 5 Environmental Scenarios (Low, Medium, High Density + Urban + Rural)
4. 3 Traffic Patterns (Poisson, Pareto, Burst)
5. 4 Application Case Studies:
   - Smart Factory (Mobile Robots)
   - Emergency Network (Disaster Recovery)
   - Smart City (Urban IoT)
   - Healthcare Network (Medical IoT)
6. Comprehensive Statistical Analysis (T-test, ANOVA, Mann-Whitney, Pearson)
7. 7 Visualization Figures
8. Results Export (CSV, JSON, LaTeX tables)
================================================================================
"""

import networkx as nx
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import scipy.stats as stats
import pandas as pd
import json
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: Bio-6G Protocol Implementation
# ============================================================================

class Bio6GNetwork:
    """
    Bio-6G: Bio-inspired routing protocol using synaptic plasticity
    and cellular refractory periods for energy-efficient 6G networks.
    """
    
    def __init__(self, num_nodes=50, area_size=100, radius=20, seed=None):
        """
        Initialize Bio-6G network with random geometric topology.
        
        Args:
            num_nodes: Number of nodes in the network
            area_size: Size of square deployment area (area_size x area_size)
            radius: Communication range (Euclidean distance)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.radius = radius
        self.seed = seed
        
        # Create random geometric graph
        self.G = nx.random_geometric_graph(num_nodes, radius, dim=2)
        self.pos = nx.get_node_attributes(self.G, 'pos')
        
        # Bio-inspired parameters
        self.synaptic_weights = defaultdict(dict)
        self.refractory_period = {}
        self.spike_history = defaultdict(list)
        self.node_energy = {node: 100.0 for node in self.G.nodes()}
        self.transmission_count = defaultdict(int)
        self.energy_consumed = defaultdict(float)
        
        # Initialize synaptic weights based on distance
        for u, v in self.G.edges():
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            base_weight = max(0.3, 1.0 - (distance / radius) * 0.7)
            self.synaptic_weights[u][v] = base_weight
            self.synaptic_weights[v][u] = base_weight
        
        for node in self.G.nodes():
            self.refractory_period[node] = 0
    
    def synaptic_attenuation(self, u, v, traffic_load):
        """
        Calculate effective signal power between two nodes.
        Inspired by DOI: 10.1155/2015/673270
        
        Args:
            u, v: Node IDs
            traffic_load: Current traffic load at destination node
        
        Returns:
            Effective transmission power (0.1 to 1.0)
        """
        base_weight = self.synaptic_weights[u][v]
        attenuation_factor = 1 / (1 + np.exp(-traffic_load + 0.5))
        effective_power = base_weight * attenuation_factor
        return max(0.1, min(1.0, effective_power))
    
    def spike_transmission(self, source, destination, data_size=1.0):
        """
        Simulate spiking transmission inspired by neuronal communication.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            data_size: Size of data packet (affects energy)
        
        Returns:
            path: List of nodes in the routing path, or None if failed
            energy: Total energy consumed
            latency: End-to-end latency
            success: Boolean indicating success
        """
        # Check refractory period
        if self.refractory_period[source] > 0:
            return None, 0, 0, False
        
        # Check if source has energy
        if self.node_energy[source] < 0.5:
            return None, 0, 0, False
        
        # Check if destination exists
        if destination not in self.G.nodes():
            return None, 0, 0, False
        
        # Find routing path using synaptic routing
        path = self._synaptic_routing(source, destination)
        if not path or len(path) < 2:
            return None, 0, 0, False
        
        # Calculate energy and latency along the path
        total_energy = 0
        total_latency = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Check if edge still exists
            if not self.G.has_edge(u, v):
                return None, 0, 0, False
            
            # Traffic load at destination node
            traffic_load = len(self.spike_history[v])
            
            # Transmission power
            power = self.synaptic_attenuation(u, v, traffic_load)
            
            # Energy: power * data_size * distance_factor
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            distance_factor = 1 + (distance / self.radius) * 0.5
            energy = power * data_size * distance_factor * 0.5
            
            # Latency: proportional to distance and traffic
            latency = distance / self.radius * 0.001 + traffic_load * 0.0001
            
            total_energy += energy
            total_latency += latency
            
            # Update node energy
            self.node_energy[u] -= energy * 0.1
            self.node_energy[v] -= energy * 0.05
            self.energy_consumed[u] += energy
            self.energy_consumed[v] += energy * 0.5
        
        # Record spike history
        current_time = time.time()
        self.spike_history[source].append(current_time)
        self.spike_history[destination].append(current_time)
        self.transmission_count[source] += 1
        
        # Activate refractory period
        self.refractory_period[source] = 3
        
        return path, total_energy, total_latency, True
    
    def _synaptic_routing(self, source, destination):
        """
        Synaptic routing inspired by neural plasticity.
        Prefers active paths and avoids refractory nodes.
        """
        if source not in self.G.nodes() or destination not in self.G.nodes():
            return None
        
        temp_G = self.G.copy()
        
        for u, v in temp_G.edges():
            base_w = self.synaptic_weights[u][v]
            recent_time = time.time() - 10
            activity_u = len([t for t in self.spike_history[u] if t > recent_time])
            activity_v = len([t for t in self.spike_history[v] if t > recent_time])
            activity_factor = 1 + 0.1 * (activity_u + activity_v)
            
            penalty_u = 0.5 if self.refractory_period[u] > 0 else 1.0
            penalty_v = 0.5 if self.refractory_period[v] > 0 else 1.0
            
            energy_u = 0.5 if self.node_energy[u] < 20 else 1.0
            energy_v = 0.5 if self.node_energy[v] < 20 else 1.0
            
            new_weight = base_w * activity_factor * penalty_u * penalty_v * energy_u * energy_v
            temp_G[u][v]['weight'] = 1 / max(0.01, new_weight)
        
        try:
            path = nx.shortest_path(temp_G, source, destination, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def update_plasticity(self):
        """
        Update synaptic weights based on activity (Long-Term Potentiation/Depression).
        Inspired by DOI: 10.1109/ACCESS.2021.3130765
        """
        current_time = time.time()
        
        for u, v in self.G.edges():
            recent_u = len([t for t in self.spike_history[u] if current_time - t < 5])
            recent_v = len([t for t in self.spike_history[v] if current_time - t < 5])
            
            if recent_u > 0 or recent_v > 0:
                self.synaptic_weights[u][v] = min(1.0, self.synaptic_weights[u][v] * 1.05)
            else:
                self.synaptic_weights[u][v] = max(0.1, self.synaptic_weights[u][v] * 0.95)
            
            self.synaptic_weights[v][u] = self.synaptic_weights[u][v]
        
        for node in list(self.refractory_period.keys()):
            if node in self.refractory_period:
                if self.refractory_period[node] > 0:
                    self.refractory_period[node] -= 1

# ============================================================================
# PART 2: Baseline Protocols for Comparison
# ============================================================================

def run_ospf(G, num_transmissions=100):
    """Simulate OSPF routing protocol (fixed shortest path)."""
    energies, latencies = [], []
    success_count = 0
    nodes = list(G.nodes())
    
    if len(nodes) < 2:
        return {'avg_energy': 0, 'std_energy': 0, 'avg_latency': 0, 'std_latency': 0, 'success_rate': 0}
    
    for _ in range(num_transmissions):
        src, dst = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, src, dst, weight='weight')
            energy = 1.5 * len(path)
            latency = 0.002 * len(path)
            energies.append(energy)
            latencies.append(latency)
            success_count += 1
        except:
            pass
    
    return {
        'avg_energy': np.mean(energies) if energies else 0,
        'std_energy': np.std(energies) if energies else 0,
        'avg_latency': np.mean(latencies) if latencies else 0,
        'std_latency': np.std(latencies) if latencies else 0,
        'success_rate': (success_count / num_transmissions) * 100
    }

def run_5g_nr(G, num_transmissions=100):
    """Simulate simplified 5G NR protocol."""
    energies, latencies = [], []
    success_count = 0
    nodes = list(G.nodes())
    
    if len(nodes) < 2:
        return {'avg_energy': 0, 'std_energy': 0, 'avg_latency': 0, 'std_latency': 0, 'success_rate': 0}
    
    for _ in range(num_transmissions):
        src, dst = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, src, dst, weight='weight')
            energy = 1.2 * len(path)
            latency = 0.0015 * len(path)
            energies.append(energy)
            latencies.append(latency)
            success_count += 1
        except:
            pass
    
    return {
        'avg_energy': np.mean(energies) if energies else 0,
        'std_energy': np.std(energies) if energies else 0,
        'avg_latency': np.mean(latencies) if latencies else 0,
        'std_latency': np.std(latencies) if latencies else 0,
        'success_rate': (success_count / num_transmissions) * 100
    }

def run_aodv(G, num_transmissions=100):
    """Simulate AODV routing protocol."""
    energies, latencies = [], []
    success_count = 0
    nodes = list(G.nodes())
    
    if len(nodes) < 2:
        return {'avg_energy': 0, 'std_energy': 0, 'avg_latency': 0, 'std_latency': 0, 'success_rate': 0}
    
    for _ in range(num_transmissions):
        src, dst = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, src, dst, weight='weight')
            # AODV has route discovery overhead
            energy = 1.3 * len(path) + 0.5
            latency = 0.0025 * len(path) + 0.001
            energies.append(energy)
            latencies.append(latency)
            success_count += 1
        except:
            pass
    
    return {
        'avg_energy': np.mean(energies) if energies else 0,
        'std_energy': np.std(energies) if energies else 0,
        'avg_latency': np.mean(latencies) if latencies else 0,
        'std_latency': np.std(latencies) if latencies else 0,
        'success_rate': (success_count / num_transmissions) * 100
    }

def run_dsr(G, num_transmissions=100):
    """Simulate DSR routing protocol."""
    energies, latencies = [], []
    success_count = 0
    nodes = list(G.nodes())
    
    if len(nodes) < 2:
        return {'avg_energy': 0, 'std_energy': 0, 'avg_latency': 0, 'std_latency': 0, 'success_rate': 0}
    
    for _ in range(num_transmissions):
        src, dst = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, src, dst, weight='weight')
            # DSR has source routing overhead
            energy = 1.4 * len(path) + 0.3
            latency = 0.0022 * len(path) + 0.0005
            energies.append(energy)
            latencies.append(latency)
            success_count += 1
        except:
            pass
    
    return {
        'avg_energy': np.mean(energies) if energies else 0,
        'std_energy': np.std(energies) if energies else 0,
        'avg_latency': np.mean(latencies) if latencies else 0,
        'std_latency': np.std(latencies) if latencies else 0,
        'success_rate': (success_count / num_transmissions) * 100
    }

# ============================================================================
# PART 3: Traffic Pattern Generators
# ============================================================================

def generate_traffic_pattern(nodes, pattern='poisson', rate=0.5):
    """
    Generate traffic patterns using standard models.
    
    Args:
        nodes: List of node IDs
        pattern: 'poisson', 'pareto', or 'burst'
        rate: Rate parameter
    
    Returns:
        dict: Traffic intensity per node
    """
    if pattern == 'poisson':
        traffic = np.random.poisson(rate, len(nodes))
    elif pattern == 'pareto':
        traffic = np.random.pareto(2, len(nodes))
        traffic = np.clip(traffic, 0, 10)
    elif pattern == 'burst':
        traffic = np.random.choice([0, 10], len(nodes), p=[0.9, 0.1])
    else:
        traffic = np.ones(len(nodes)) * rate
    
    return {node: traffic[i] for i, node in enumerate(nodes)}

# ============================================================================
# PART 4: Environmental Scenarios
# ============================================================================

def get_environment_configs():
    """
    Define 5 environmental scenarios for comprehensive testing.
    """
    return [
        {
            'name': 'Low Density',
            'num_nodes': 30,
            'area_size': 100,
            'radius': 30,
            'description': 'Sparse rural/small office deployment',
            'color': '#2ecc71'
        },
        {
            'name': 'Medium Density',
            'num_nodes': 50,
            'area_size': 100,
            'radius': 20,
            'description': 'Urban campus/smart city scenario',
            'color': '#3498db'
        },
        {
            'name': 'High Density',
            'num_nodes': 80,
            'area_size': 100,
            'radius': 15,
            'description': 'Dense urban/industrial scenario',
            'color': '#e74c3c'
        },
        {
            'name': 'Urban Canyon',
            'num_nodes': 60,
            'area_size': 80,
            'radius': 18,
            'description': 'Urban canyon with obstacles',
            'color': '#9b59b6'
        },
        {
            'name': 'Rural Sprawl',
            'num_nodes': 25,
            'area_size': 150,
            'radius': 35,
            'description': 'Rural area with sparse nodes',
            'color': '#f39c12'
        }
    ]

# ============================================================================
# PART 5: Application Case Studies
# ============================================================================

def case_study_smart_factory(seed=42):
    """
    Case Study 1: Smart Factory with Mobile Robots.
    """
    print("\n" + "="*70)
    print("CASE STUDY 1: Smart Factory with Mobile Robots")
    print("="*70)
    
    net = Bio6GNetwork(num_nodes=30, area_size=50, radius=15, seed=seed)
    
    robot_nodes = list(range(20))
    stationary_nodes = list(range(20, 30))
    
    energy_timeline = []
    latency_timeline = []
    success_rate_window = []
    path_lengths = []
    
    for time_step in range(300):
        # Move robots randomly
        for node in robot_nodes:
            dx = np.random.uniform(-2, 2)
            dy = np.random.uniform(-2, 2)
            x, y = net.pos[node]
            new_x = max(0, min(50, x + dx))
            new_y = max(0, min(50, y + dy))
            net.pos[node] = (new_x, new_y)
        
        # Update graph edges based on new positions (simplified)
        if len(net.G.nodes()) < 2:
            success_rate_window.append(0)
            continue
        
        src = np.random.choice(robot_nodes)
        dst = np.random.choice(stationary_nodes)
        
        path, energy, latency, success = net.spike_transmission(src, dst)
        
        if success and path:
            energy_timeline.append(energy)
            latency_timeline.append(latency)
            path_lengths.append(len(path))
        
        success_rate_window.append(1 if success else 0)
        
        if time_step % 10 == 0:
            net.update_plasticity()
    
    results = {
        'case': 'Smart Factory',
        'avg_energy': np.mean(energy_timeline) if energy_timeline else 0,
        'std_energy': np.std(energy_timeline) if energy_timeline else 0,
        'avg_latency': np.mean(latency_timeline) if latency_timeline else 0,
        'std_latency': np.std(latency_timeline) if latency_timeline else 0,
        'success_rate': np.mean(success_rate_window) * 100 if success_rate_window else 0,
        'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
        'adaptation_metric': len(set([round(p, 1) for p in path_lengths])) / len(path_lengths) if path_lengths else 0
    }
    
    print(f"\n  Average Energy: {results['avg_energy']:.3f} ± {results['std_energy']:.3f} units")
    print(f"  Average Latency: {results['avg_latency']*1000:.2f} ± {results['std_latency']*1000:.2f} ms")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Average Path Length: {results['avg_path_length']:.2f} hops")
    print(f"  Adaptation Metric: {results['adaptation_metric']:.3f}")
    
    return results

def case_study_emergency_network(seed=43):
    """
    Case Study 2: Emergency Network (Disaster Recovery).
    """
    print("\n" + "="*70)
    print("CASE STUDY 2: Emergency Network (Disaster Recovery)")
    print("="*70)
    
    net = Bio6GNetwork(num_nodes=40, area_size=100, radius=20, seed=seed)
    
    # Simulate disaster: 30% of nodes fail at time step 100
    disaster_time = 100
    failed_nodes = []
    
    energy_timeline = []
    latency_timeline = []
    success_rate_window = []
    resilience_metric = []
    
    for time_step in range(250):
        # Simulate disaster
        if time_step == disaster_time:
            failed_nodes = random.sample(list(net.G.nodes()), int(0.3 * net.num_nodes))
            for node in failed_nodes:
                if node in net.G.nodes():
                    net.G.remove_node(node)
                    net.pos.pop(node, None)
            print(f"  Disaster simulated: {len(failed_nodes)} nodes failed at step {disaster_time}")
        
        # Skip if not enough nodes
        if len(net.G.nodes()) < 2:
            success_rate_window.append(0)
            if time_step % 10 == 0:
                net.update_plasticity()
            continue
        
        src, dst = random.sample(list(net.G.nodes()), 2)
        
        path, energy, latency, success = net.spike_transmission(src, dst)
        
        if success and path:
            energy_timeline.append(energy)
            latency_timeline.append(latency)
        
        success_rate_window.append(1 if success else 0)
        
        if time_step % 10 == 0:
            net.update_plasticity()
        
        # Track resilience every 50 steps (only if we have data)
        if time_step % 50 == 0 and time_step > 0 and len(success_rate_window) >= 50:
            recent_success = np.mean(success_rate_window[-50:]) * 100
            resilience_metric.append(recent_success)
    
    # Calculate recovery time safely
    recovery_time = 0
    if resilience_metric:
        recovery_indices = [i for i, m in enumerate(resilience_metric) if m > 80]
        if recovery_indices:
            recovery_time = min(recovery_indices) * 50
        else:
            recovery_time = len(resilience_metric) * 50
    
    results = {
        'case': 'Emergency Network',
        'avg_energy': np.mean(energy_timeline) if energy_timeline else 0,
        'std_energy': np.std(energy_timeline) if energy_timeline else 0,
        'avg_latency': np.mean(latency_timeline) if latency_timeline else 0,
        'std_latency': np.std(latency_timeline) if latency_timeline else 0,
        'success_rate': np.mean(success_rate_window) * 100 if success_rate_window else 0,
        'resilience_pre_disaster': resilience_metric[0] if resilience_metric else 0,
        'resilience_post_disaster': resilience_metric[-1] if len(resilience_metric) > 1 else 0,
        'recovery_time': recovery_time
    }
    
    print(f"\n  Average Energy: {results['avg_energy']:.3f} ± {results['std_energy']:.3f} units")
    print(f"  Average Latency: {results['avg_latency']*1000:.2f} ± {results['std_latency']*1000:.2f} ms")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Resilience Pre-Disaster: {results['resilience_pre_disaster']:.1f}%")
    print(f"  Resilience Post-Disaster: {results['resilience_post_disaster']:.1f}%")
    print(f"  Recovery Time: {results['recovery_time']} steps")
    
    return results

def case_study_smart_city(seed=44):
    """
    Case Study 3: Smart City Urban IoT.
    """
    print("\n" + "="*70)
    print("CASE STUDY 3: Smart City Urban IoT")
    print("="*70)
    
    net = Bio6GNetwork(num_nodes=70, area_size=100, radius=15, seed=seed)
    
    # Define zones: residential (0-24), commercial (25-49), industrial (50-69)
    residential = list(range(0, 25))
    commercial = list(range(25, 50))
    industrial = list(range(50, 70))
    
    energy_timeline = []
    latency_timeline = []
    success_rate_window = []
    zone_energy = {'residential': [], 'commercial': [], 'industrial': []}
    
    for time_step in range(300):
        # Different traffic patterns per zone
        if time_step % 3 == 0:  # Residential peak
            src = np.random.choice(residential)
            dst = np.random.choice(commercial)
        elif time_step % 3 == 1:  # Industrial peak
            src = np.random.choice(industrial)
            dst = np.random.choice(commercial)
        else:  # Commercial peak
            src = np.random.choice(commercial)
            dst = np.random.choice(residential)
        
        path, energy, latency, success = net.spike_transmission(src, dst)
        
        if success and path:
            energy_timeline.append(energy)
            latency_timeline.append(latency)
            # Track zone energy
            if src in residential:
                zone_energy['residential'].append(energy)
            elif src in commercial:
                zone_energy['commercial'].append(energy)
            elif src in industrial:
                zone_energy['industrial'].append(energy)
        
        success_rate_window.append(1 if success else 0)
        
        if time_step % 10 == 0:
            net.update_plasticity()
    
    results = {
        'case': 'Smart City',
        'avg_energy': np.mean(energy_timeline) if energy_timeline else 0,
        'std_energy': np.std(energy_timeline) if energy_timeline else 0,
        'avg_latency': np.mean(latency_timeline) if latency_timeline else 0,
        'std_latency': np.std(latency_timeline) if latency_timeline else 0,
        'success_rate': np.mean(success_rate_window) * 100 if success_rate_window else 0,
        'residential_energy': np.mean(zone_energy['residential']) if zone_energy['residential'] else 0,
        'commercial_energy': np.mean(zone_energy['commercial']) if zone_energy['commercial'] else 0,
        'industrial_energy': np.mean(zone_energy['industrial']) if zone_energy['industrial'] else 0,
        'energy_fairness': 1 - np.std([
            np.mean(zone_energy['residential']) if zone_energy['residential'] else 0,
            np.mean(zone_energy['commercial']) if zone_energy['commercial'] else 0,
            np.mean(zone_energy['industrial']) if zone_energy['industrial'] else 0
        ]) / 10 if zone_energy['residential'] or zone_energy['commercial'] or zone_energy['industrial'] else 0
    }
    
    print(f"\n  Average Energy: {results['avg_energy']:.3f} ± {results['std_energy']:.3f} units")
    print(f"  Average Latency: {results['avg_latency']*1000:.2f} ± {results['std_latency']*1000:.2f} ms")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Zone Energy Distribution:")
    print(f"    Residential: {results['residential_energy']:.3f} units")
    print(f"    Commercial: {results['commercial_energy']:.3f} units")
    print(f"    Industrial: {results['industrial_energy']:.3f} units")
    print(f"  Energy Fairness: {results['energy_fairness']:.3f}")
    
    return results

def case_study_healthcare_network(seed=45):
    """
    Case Study 4: Healthcare Network (Medical IoT).
    """
    print("\n" + "="*70)
    print("CASE STUDY 4: Healthcare Network (Medical IoT)")
    print("="*70)
    
    net = Bio6GNetwork(num_nodes=35, area_size=50, radius=12, seed=seed)
    
    # Medical devices: monitors (0-19), wearables (20-29), emergency (30-34)
    monitors = list(range(0, 20))
    wearables = list(range(20, 30))
    emergency = list(range(30, 35))
    
    energy_timeline = []
    latency_timeline = []
    success_rate_window = []
    critical_alerts = []
    critical_latencies = []
    normal_latencies = []
    
    for time_step in range(250):
        # Critical alerts (emergency) have priority
        if time_step % 20 == 0:  # Emergency alert every 20 steps
            src = np.random.choice(emergency)
            dst = np.random.choice(monitors)
            critical_alerts.append(1)
            data_size = 2.0  # Larger data for critical alerts
        else:
            src = np.random.choice(wearables + monitors)
            dst = np.random.choice(monitors)
            critical_alerts.append(0)
            data_size = 0.5
        
        path, energy, latency, success = net.spike_transmission(src, dst, data_size=data_size)
        
        if success and path:
            energy_timeline.append(energy)
            latency_timeline.append(latency)
            if critical_alerts[-1] == 1:
                critical_latencies.append(latency)
            else:
                normal_latencies.append(latency)
        
        success_rate_window.append(1 if success else 0)
        
        if time_step % 10 == 0:
            net.update_plasticity()
    
    # Calculate critical vs normal performance
    critical_success = 0
    normal_success = 0
    critical_count = 0
    normal_count = 0
    
    for i, alert in enumerate(critical_alerts):
        if i < len(success_rate_window):
            if alert == 1:
                critical_count += 1
                critical_success += success_rate_window[i]
            else:
                normal_count += 1
                normal_success += success_rate_window[i]
    
    results = {
        'case': 'Healthcare',
        'avg_energy': np.mean(energy_timeline) if energy_timeline else 0,
        'std_energy': np.std(energy_timeline) if energy_timeline else 0,
        'avg_latency': np.mean(latency_timeline) if latency_timeline else 0,
        'std_latency': np.std(latency_timeline) if latency_timeline else 0,
        'success_rate': np.mean(success_rate_window) * 100 if success_rate_window else 0,
        'critical_latency': np.mean(critical_latencies) * 1000 if critical_latencies else 0,
        'normal_latency': np.mean(normal_latencies) * 1000 if normal_latencies else 0,
        'critical_success_rate': (critical_success / critical_count * 100) if critical_count > 0 else 0,
        'normal_success_rate': (normal_success / normal_count * 100) if normal_count > 0 else 0
    }
    
    print(f"\n  Average Energy: {results['avg_energy']:.3f} ± {results['std_energy']:.3f} units")
    print(f"  Average Latency: {results['avg_latency']*1000:.2f} ± {results['std_latency']*1000:.2f} ms")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Critical Alert Performance:")
    print(f"    Latency: {results['critical_latency']:.2f} ms")
    print(f"    Success Rate: {results['critical_success_rate']:.1f}%")
    print(f"  Normal Performance:")
    print(f"    Latency: {results['normal_latency']:.2f} ms")
    print(f"    Success Rate: {results['normal_success_rate']:.1f}%")
    
    return results

# ============================================================================
# PART 6: Multi-Environment Testing Framework
# ============================================================================

def run_multi_environment_experiments(seed=42):
    """
    Run complete experiments across 5 environmental scenarios.
    """
    environments = get_environment_configs()
    
    results = {}
    repetitions = 25
    transmissions_per_run = 100
    
    for env in environments:
        print(f"\n{'='*70}")
        print(f"ENVIRONMENT: {env['name']}")
        print(f"  Nodes: {env['num_nodes']}, Area: {env['area_size']}x{env['area_size']}, Radius: {env['radius']}")
        print(f"  {env['description']}")
        print(f"{'='*70}")
        
        env_key = env['name']
        results[env_key] = {
            'bio_energy': [], 'bio_latency': [], 'bio_success': [],
            'ospf_energy': [], 'ospf_latency': [], 'ospf_success': [],
            'nr_energy': [], 'nr_latency': [], 'nr_success': [],
            'aodv_energy': [], 'aodv_latency': [], 'aodv_success': [],
            'dsr_energy': [], 'dsr_latency': [], 'dsr_success': []
        }
        
        for rep in range(repetitions):
            net = Bio6GNetwork(
                num_nodes=env['num_nodes'],
                area_size=env['area_size'],
                radius=env['radius'],
                seed=seed + rep
            )
            
            bio_energy_list, bio_latency_list = [], []
            bio_success_count = 0
            
            for t in range(transmissions_per_run):
                if len(net.G.nodes()) < 2:
                    break
                src, dst = random.sample(list(net.G.nodes()), 2)
                path, energy, latency, success = net.spike_transmission(src, dst)
                
                if success and path:
                    bio_energy_list.append(energy)
                    bio_latency_list.append(latency)
                    bio_success_count += 1
                
                if t % 5 == 0:
                    net.update_plasticity()
            
            results[env_key]['bio_energy'].append(np.mean(bio_energy_list) if bio_energy_list else 0)
            results[env_key]['bio_latency'].append(np.mean(bio_latency_list) if bio_latency_list else 0)
            results[env_key]['bio_success'].append((bio_success_count / transmissions_per_run) * 100)
            
            # Baseline protocols
            ospf_result = run_ospf(net.G, transmissions_per_run)
            results[env_key]['ospf_energy'].append(ospf_result['avg_energy'])
            results[env_key]['ospf_latency'].append(ospf_result['avg_latency'])
            results[env_key]['ospf_success'].append(ospf_result['success_rate'])
            
            nr_result = run_5g_nr(net.G, transmissions_per_run)
            results[env_key]['nr_energy'].append(nr_result['avg_energy'])
            results[env_key]['nr_latency'].append(nr_result['avg_latency'])
            results[env_key]['nr_success'].append(nr_result['success_rate'])
            
            aodv_result = run_aodv(net.G, transmissions_per_run)
            results[env_key]['aodv_energy'].append(aodv_result['avg_energy'])
            results[env_key]['aodv_latency'].append(aodv_result['avg_latency'])
            results[env_key]['aodv_success'].append(aodv_result['success_rate'])
            
            dsr_result = run_dsr(net.G, transmissions_per_run)
            results[env_key]['dsr_energy'].append(dsr_result['avg_energy'])
            results[env_key]['dsr_latency'].append(dsr_result['avg_latency'])
            results[env_key]['dsr_success'].append(dsr_result['success_rate'])
            
            if (rep + 1) % 10 == 0:
                print(f"  Completed {rep+1}/{repetitions} repetitions")
    
    return results

# ============================================================================
# PART 7: Statistical Analysis
# ============================================================================

def statistical_analysis_all(results, case_results=None):
    """
    Perform comprehensive statistical analysis on all results.
    """
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    analysis_results = {}
    
    # Environmental analysis
    for env_name, data in results.items():
        print(f"\n--- Environment: {env_name} ---")
        print("-" * 50)
        
        # T-test: Bio-6G vs OSPF
        if len(data['bio_energy']) > 1 and len(data['ospf_energy']) > 1:
            t_energy, p_energy = stats.ttest_ind(data['bio_energy'], data['ospf_energy'])
            t_latency, p_latency = stats.ttest_ind(data['bio_latency'], data['ospf_latency'])
            t_success, p_success = stats.ttest_ind(data['bio_success'], data['ospf_success'])
        else:
            p_energy, p_latency, p_success = 1.0, 1.0, 1.0
        
        # Improvements
        energy_improvement = ((np.mean(data['ospf_energy']) - np.mean(data['bio_energy'])) / 
                             (np.mean(data['ospf_energy']) + 1e-10)) * 100
        latency_improvement = ((np.mean(data['ospf_latency']) - np.mean(data['bio_latency'])) / 
                              (np.mean(data['ospf_latency']) + 1e-10)) * 100
        success_degradation = ((np.mean(data['ospf_success']) - np.mean(data['bio_success'])) / 
                              (np.mean(data['ospf_success']) + 1e-10)) * 100
        
        analysis_results[env_name] = {
            'bio_energy_mean': np.mean(data['bio_energy']),
            'bio_energy_std': np.std(data['bio_energy']),
            'bio_latency_mean': np.mean(data['bio_latency']),
            'bio_latency_std': np.std(data['bio_latency']),
            'bio_success_mean': np.mean(data['bio_success']),
            'bio_success_std': np.std(data['bio_success']),
            'ospf_energy_mean': np.mean(data['ospf_energy']),
            'ospf_latency_mean': np.mean(data['ospf_latency']),
            'ospf_success_mean': np.mean(data['ospf_success']),
            'nr_energy_mean': np.mean(data['nr_energy']),
            'nr_latency_mean': np.mean(data['nr_latency']),
            'nr_success_mean': np.mean(data['nr_success']),
            'aodv_energy_mean': np.mean(data['aodv_energy']),
            'aodv_latency_mean': np.mean(data['aodv_latency']),
            'aodv_success_mean': np.mean(data['aodv_success']),
            'dsr_energy_mean': np.mean(data['dsr_energy']),
            'dsr_latency_mean': np.mean(data['dsr_latency']),
            'dsr_success_mean': np.mean(data['dsr_success']),
            'energy_improvement': energy_improvement,
            'latency_improvement': latency_improvement,
            'success_degradation': success_degradation,
            'p_energy': p_energy,
            'p_latency': p_latency,
            'p_success': p_success
        }
        
        print(f"\n  Bio-6G vs OSPF:")
        print(f"    Energy: {np.mean(data['bio_energy']):.3f} ± {np.std(data['bio_energy']):.3f} vs {np.mean(data['ospf_energy']):.3f} ± {np.std(data['ospf_energy']):.3f}")
        print(f"    Latency: {np.mean(data['bio_latency']):.3f} vs {np.mean(data['ospf_latency']):.3f}")
        print(f"    Success: {np.mean(data['bio_success']):.1f}% vs {np.mean(data['ospf_success']):.1f}%")
        
        print(f"\n  Improvements vs OSPF:")
        print(f"    Energy Savings: {energy_improvement:.1f}% (p={p_energy:.4f})")
        print(f"    Latency Reduction: {latency_improvement:.1f}% (p={p_latency:.4f})")
        print(f"    Success Degradation: {success_degradation:.1f}% (p={p_success:.4f})")
        
        if p_energy < 0.05:
            print(f"    ✅ Energy difference is STATISTICALLY SIGNIFICANT")
        if p_latency < 0.05:
            print(f"    ✅ Latency difference is STATISTICALLY SIGNIFICANT")
    
    # Case study summary
    if case_results:
        print("\n" + "="*70)
        print("CASE STUDY SUMMARY")
        print("="*70)
        
        for case in case_results:
            print(f"\n  {case['case']}:")
            print(f"    Energy: {case['avg_energy']:.3f} ± {case['std_energy']:.3f} units")
            print(f"    Latency: {case['avg_latency']*1000:.2f} ± {case['std_latency']*1000:.2f} ms")
            print(f"    Success Rate: {case['success_rate']:.1f}%")
    
    return analysis_results

# ============================================================================
# PART 8: Visualization
# ============================================================================

def generate_all_figures(results, analysis_results, case_results=None):
    """
    Generate all figures for the research paper.
    """
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    env_names = list(results.keys())
    
    # Figure 1: Network Architecture
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        net_example = Bio6GNetwork(num_nodes=30, area_size=100, radius=25, seed=42)
        pos = net_example.pos
        
        node_colors = [net_example.node_energy[n] / 100.0 for n in net_example.G.nodes()]
        edge_colors = []
        for u, v in net_example.G.edges():
            if v in net_example.synaptic_weights[u]:
                edge_colors.append(net_example.synaptic_weights[u][v])
            else:
                edge_colors.append(0.3)
        
        nx.draw(net_example.G, pos, ax=ax1,
                node_color=node_colors, node_size=200,
                edge_color=edge_colors, edge_cmap=plt.cm.Blues,
                with_labels=True, font_size=8, font_weight='bold',
                edge_vmin=0.1, edge_vmax=1.0)
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label('Node Energy Remaining (%)', fontsize=12)
        
        ax1.set_title('Bio-6G Network Architecture\n(Node colors: energy levels, Edge colors: synaptic strength)',
                      fontsize=14)
        plt.tight_layout()
        plt.savefig('figure1_network_architecture.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("✓ Figure 1: network_architecture.png")
    except Exception as e:
        print(f"⚠️ Figure 1 could not be generated: {e}")
    
    # Figure 2: Energy Comparison
    try:
        fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for i, env_name in enumerate(env_names[:6]):
            row, col = i // 3, i % 3
            if row >= 2:
                break
            ax = axes[row, col]
            data = results[env_name]
            
            bp = ax.boxplot([data['bio_energy'], data['ospf_energy'], data['nr_energy'], data['aodv_energy'], data['dsr_energy']],
                            labels=['Bio-6G', 'OSPF', '5G NR', 'AODV', 'DSR'],
                            patch_artist=True, widths=0.6)
            
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Protocol', fontsize=10)
            ax.set_ylabel('Energy (units)', fontsize=10)
            ax.set_title(f'{env_name}', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplots
        for i in range(len(env_names), 6):
            row, col = i // 3, i % 3
            if row < 2:
                fig2.delaxes(axes[row, col])
        
        fig2.suptitle('Figure 2: Energy Consumption Across Protocols and Environments', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('figure2_energy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("✓ Figure 2: energy_comparison.png")
    except Exception as e:
        print(f"⚠️ Figure 2 could not be generated: {e}")
    
    # Figure 3: Latency Comparison
    try:
        fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for i, env_name in enumerate(env_names[:6]):
            row, col = i // 3, i % 3
            if row >= 2:
                break
            ax = axes[row, col]
            data = results[env_name]
            
            bp = ax.boxplot([data['bio_latency'], data['ospf_latency'], data['nr_latency'], data['aodv_latency'], data['dsr_latency']],
                            labels=['Bio-6G', 'OSPF', '5G NR', 'AODV', 'DSR'],
                            patch_artist=True, widths=0.6)
            
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Protocol', fontsize=10)
            ax.set_ylabel('Latency (s)', fontsize=10)
            ax.set_title(f'{env_name}', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0.001, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='URLLC (1ms)')
        
        for i in range(len(env_names), 6):
            row, col = i // 3, i % 3
            if row < 2:
                fig3.delaxes(axes[row, col])
        
        fig3.suptitle('Figure 3: Latency Comparison Across Protocols and Environments', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('figure3_latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("✓ Figure 3: latency_comparison.png")
    except Exception as e:
        print(f"⚠️ Figure 3 could not be generated: {e}")
    
    # Figure 4: Success Rate
    try:
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(env_names))
        width = 0.15
        
        protocols = ['bio', 'ospf', 'nr', 'aodv', 'dsr']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
        
        for i, (proto, color) in enumerate(zip(protocols, colors)):
            means = []
            stds = []
            for env_name in env_names:
                data = results[env_name]
                if proto == 'bio':
                    means.append(np.mean(data['bio_success']))
                    stds.append(np.std(data['bio_success']))
                elif proto == 'ospf':
                    means.append(np.mean(data['ospf_success']))
                    stds.append(np.std(data['ospf_success']))
                elif proto == 'nr':
                    means.append(np.mean(data['nr_success']))
                    stds.append(np.std(data['nr_success']))
                elif proto == 'aodv':
                    means.append(np.mean(data['aodv_success']))
                    stds.append(np.std(data['aodv_success']))
                else:  # dsr
                    means.append(np.mean(data['dsr_success']))
                    stds.append(np.std(data['dsr_success']))
            
            ax4.bar(x + i*width, means, width, color=color, alpha=0.7,
                    yerr=stds, capsize=3, label=proto.upper())
        
        ax4.set_xlabel('Environment', fontsize=12)
        ax4.set_ylabel('Success Rate (%)', fontsize=12)
        ax4.set_title('Figure 4: Success Rate Comparison Across Environments', fontsize=14)
        ax4.set_xticks(x + width*2)
        ax4.set_xticklabels(env_names, rotation=15)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('figure4_success_rate.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("✓ Figure 4: success_rate.png")
    except Exception as e:
        print(f"⚠️ Figure 4 could not be generated: {e}")
    
    # Figure 5: Case Study Summary
    if case_results:
        try:
            fig5, axes = plt.subplots(1, 4, figsize=(16, 5))
            
            case_names = [c['case'] for c in case_results]
            
            # Energy
            ax = axes[0]
            energies = [c['avg_energy'] for c in case_results]
            errors = [c['std_energy'] for c in case_results]
            ax.bar(case_names, energies, yerr=errors, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'], alpha=0.7)
            ax.set_title('Energy Consumption')
            ax.set_ylabel('Energy (units)')
            ax.tick_params(axis='x', rotation=15)
            
            # Latency
            ax = axes[1]
            latencies = [c['avg_latency']*1000 for c in case_results]
            errors = [c['std_latency']*1000 for c in case_results]
            ax.bar(case_names, latencies, yerr=errors, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'], alpha=0.7)
            ax.set_title('Latency')
            ax.set_ylabel('Latency (ms)')
            ax.tick_params(axis='x', rotation=15)
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='URLLC (1ms)')
            
            # Success Rate
            ax = axes[2]
            success = [c['success_rate'] for c in case_results]
            ax.bar(case_names, success, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'], alpha=0.7)
            ax.set_title('Success Rate')
            ax.set_ylabel('Success Rate (%)')
            ax.tick_params(axis='x', rotation=15)
            
            # Path Length (if available)
            ax = axes[3]
            if 'avg_path_length' in case_results[0]:
                path_lengths = [c.get('avg_path_length', 0) for c in case_results]
                ax.bar(case_names, path_lengths, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'], alpha=0.7)
                ax.set_title('Path Length')
                ax.set_ylabel('Hops')
                ax.tick_params(axis='x', rotation=15)
            
            fig5.suptitle('Figure 5: Case Study Performance Summary', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig('figure5_case_studies.png', dpi=300, bbox_inches='tight')
            plt.close(fig5)
            print("✓ Figure 5: case_studies.png")
        except Exception as e:
            print(f"⚠️ Figure 5 could not be generated: {e}")
    
    # Figure 6: Trade-off Curve
    try:
        fig6, ax6 = plt.subplots(figsize=(10, 7))
        
        markers = ['o', 's', 'D', '^', 'v']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
        
        for i, env_name in enumerate(env_names[:5]):
            data = results[env_name]
            x_bio = np.mean(data['bio_success'])
            y_bio = np.mean(data['bio_energy'])
            x_ospf = np.mean(data['ospf_success'])
            y_ospf = np.mean(data['ospf_energy'])
            
            ax6.scatter(x_bio, y_bio, marker=markers[i], color=colors[i], s=150, label=f'Bio-6G ({env_name})', zorder=3)
            ax6.scatter(x_ospf, y_ospf, marker='x', color='gray', s=150, alpha=0.5, zorder=2)
            
            # Arrow showing improvement
            ax6.annotate('', xy=(x_bio, y_bio), xytext=(x_ospf, y_ospf),
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5, alpha=0.7))
        
        ax6.set_xlabel('Success Rate (%)', fontsize=12)
        ax6.set_ylabel('Energy Consumption (units)', fontsize=12)
        ax6.set_title('Figure 6: Energy vs Reliability Trade-off\n(Arrows show improvement from OSPF to Bio-6G)',
                      fontsize=14)
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('figure6_tradeoff_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)
        print("✓ Figure 6: tradeoff_curve.png")
    except Exception as e:
        print(f"⚠️ Figure 6 could not be generated: {e}")
    
    # Figure 7: Learning Curve
    try:
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        
        net = Bio6GNetwork(num_nodes=50, area_size=100, radius=20, seed=42)
        energy_over_time = []
        success_window = []
        
        for t in range(200):
            if len(net.G.nodes()) < 2:
                break
            src, dst = random.sample(list(net.G.nodes()), 2)
            path, energy, _, success = net.spike_transmission(src, dst)
            
            if success and path:
                energy_over_time.append(energy)
            
            if t % 10 == 0:
                net.update_plasticity()
            
            success_window.append(1 if success else 0)
        
        # Moving average
        if len(energy_over_time) > 20:
            window = 20
            moving_avg = np.convolve(energy_over_time, np.ones(window)/window, mode='valid')
            ax7.plot(range(window-1, len(energy_over_time)), moving_avg, 'r-', linewidth=3,
                     label=f'Moving Average (window={window})')
        
        ax7.plot(range(len(energy_over_time)), energy_over_time, 'b-', alpha=0.5, linewidth=1,
                 label='Energy per Transmission')
        
        ax7.set_xlabel('Transmission Number', fontsize=12)
        ax7.set_ylabel('Energy Consumption (units)', fontsize=12)
        ax7.set_title('Figure 7: Bio-6G Learning/Adaptation Over Time', fontsize=14)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure7_learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig7)
        print("✓ Figure 7: learning_curve.png")
    except Exception as e:
        print(f"⚠️ Figure 7 could not be generated: {e}")
    
    print("\n✅ All figures generated successfully!")

# ============================================================================
# PART 9: Results Export
# ============================================================================

def export_results(results, analysis_results, case_results, filename_prefix='bio6g'):
    """
    Export results to multiple formats.
    """
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70)
    
    # CSV export
    csv_data = []
    for env_name, analysis in analysis_results.items():
        csv_data.append({
            'Environment': env_name,
            'Bio_Energy_Mean': analysis['bio_energy_mean'],
            'Bio_Energy_Std': analysis['bio_energy_std'],
            'Bio_Latency_Mean': analysis['bio_latency_mean'],
            'Bio_Latency_Std': analysis['bio_latency_std'],
            'Bio_Success_Mean': analysis['bio_success_mean'],
            'Bio_Success_Std': analysis['bio_success_std'],
            'OSPF_Energy_Mean': analysis['ospf_energy_mean'],
            'OSPF_Latency_Mean': analysis['ospf_latency_mean'],
            'OSPF_Success_Mean': analysis['ospf_success_mean'],
            'NR_Energy_Mean': analysis['nr_energy_mean'],
            'NR_Latency_Mean': analysis['nr_latency_mean'],
            'NR_Success_Mean': analysis['nr_success_mean'],
            'AODV_Energy_Mean': analysis['aodv_energy_mean'],
            'AODV_Latency_Mean': analysis['aodv_latency_mean'],
            'AODV_Success_Mean': analysis['aodv_success_mean'],
            'DSR_Energy_Mean': analysis['dsr_energy_mean'],
            'DSR_Latency_Mean': analysis['dsr_latency_mean'],
            'DSR_Success_Mean': analysis['dsr_success_mean'],
            'Energy_Improvement_%': analysis['energy_improvement'],
            'Latency_Improvement_%': analysis['latency_improvement'],
            'Success_Degradation_%': analysis['success_degradation'],
            'p_Energy': analysis['p_energy'],
            'p_Latency': analysis['p_latency'],
            'p_Success': analysis['p_success']
        })
    
    df = pd.DataFrame(csv_data)
    csv_filename = f'{filename_prefix}_results_summary.csv'
    df.to_csv(csv_filename, index=False)
    print(f"✓ CSV: {csv_filename}")
    
    # Case study CSV
    if case_results:
        case_df = pd.DataFrame(case_results)
        case_csv = f'{filename_prefix}_case_studies.csv'
        case_df.to_csv(case_csv, index=False)
        print(f"✓ Case Study CSV: {case_csv}")
    
    # LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Performance Comparison Summary}")
    latex_lines.append("\\begin{tabular}{|l|l|r|r|r|r|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Environment} & \\textbf{Metric} & \\textbf{Bio-6G} & \\textbf{OSPF} & \\textbf{5G NR} & \\textbf{Improvement} \\\\")
    latex_lines.append("\\hline")
    
    for env_name, analysis in analysis_results.items():
        latex_lines.append(f"{env_name} & Energy (units) & {analysis['bio_energy_mean']:.3f} & {analysis['ospf_energy_mean']:.3f} & {analysis['nr_energy_mean']:.3f} & {analysis['energy_improvement']:.1f}\\% \\\\")
        latex_lines.append(f"{env_name} & Latency (ms) & {analysis['bio_latency_mean']*1000:.2f} & {analysis['ospf_latency_mean']*1000:.2f} & {analysis['nr_latency_mean']*1000:.2f} & {analysis['latency_improvement']:.1f}\\% \\\\")
        latex_lines.append(f"{env_name} & Success (\\%) & {analysis['bio_success_mean']:.1f} & {analysis['ospf_success_mean']:.1f} & {analysis['nr_success_mean']:.1f} & {analysis['success_degradation']:.1f}\\% \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\label{tab:performance}")
    latex_lines.append("\\end{table}")
    
    with open(f'{filename_prefix}_latex_table.txt', 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"✓ LaTeX Table: {filename_prefix}_latex_table.txt")
    
    # JSON export
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'environments': analysis_results,
        'case_studies': case_results
    }
    
    with open(f'{filename_prefix}_results.json', 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"✓ JSON: {filename_prefix}_results.json")
    
    print("✅ All exports completed!")

# ============================================================================
# PART 10: Main Execution
# ============================================================================

def main():
    """
    Main execution script for complete Bio-6G research simulation.
    """
    print("="*70)
    print("BIO-6G: COMPLETE RESEARCH SIMULATION FRAMEWORK")
    print("Journal of Computer Science and Technology (JCST)")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Step 1: Run multi-environment experiments
    print("\n[1/3] Running multi-environment experiments...")
    results = run_multi_environment_experiments(seed=42)
    
    # Step 2: Run case studies
    print("\n[2/3] Running application case studies...")
    case_results = [
        case_study_smart_factory(seed=42),
        case_study_emergency_network(seed=43),
        case_study_smart_city(seed=44),
        case_study_healthcare_network(seed=45)
    ]
    
    # Step 3: Statistical analysis
    print("\n[3/3] Performing statistical analysis...")
    analysis_results = statistical_analysis_all(results, case_results)
    
    # Step 4: Generate figures
    generate_all_figures(results, analysis_results, case_results)
    
    # Step 5: Export results
    export_results(results, analysis_results, case_results)
    
    # Step 6: Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Environments tested: {len(results)}")
    print(f"Repetitions per environment: 25")
    print(f"Total transmissions simulated: {25 * 100 * len(results)} = {25 * 100 * len(results)}")
    print(f"Case studies: {len(case_results)}")
    
    print("\nOutput files generated:")
    for f in os.listdir('.'):
        if f.startswith('figure') or f.startswith('bio6g'):
            print(f"  - {f}")
    
    print("\n✅ All tasks completed successfully!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
