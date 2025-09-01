# -*- coding: utf-8 -*-
"""
# 2D Synthetic Cell Membrane Masks for CellSynthesis
# Based on the original 3D version, adapted for 2D cellular structures
# Copyright (C) 2021 D. Eschweiler, M. Rethwisch, M. Jarchow, S. Koppers, J. Stegmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the Liceense at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""

import os
import numpy as np
import pandas as pd
from skimage import morphology, measure, segmentation, filters
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree, Voronoi
from sklearn.cluster import AgglomerativeClustering
# from utils.h5_converter import h5_writer  # Optional for standalone use
# from utils.utils import print_timestamp    # Optional for standalone use


def h5_writer(data_list, filename, group_names=None):
    """Simple h5 writer function - requires h5py"""
    import h5py
    with h5py.File(filename, 'w') as f:
        for i, data in enumerate(data_list):
            group_name = group_names[i] if group_names and i < len(group_names) else f'data_{i}'
            f.create_dataset(group_name, data=data)


def generate_data_2d(syn_class, save_path='Segmentations_2d_h5', experiment_name='synthetic_2d_data', img_count=100, param_dict={}):
    """Generate 2D synthetic cell membrane data"""
    
    synthesizer = syn_class(**param_dict)
    os.makedirs(save_path, exist_ok=True)
    
    for num_img in range(img_count):
        
        
        # Generate a new mask
        synthesizer.generate_instances()
        
        # Get and save the instance, boundary, centroid and distance masks
        instance_mask = synthesizer.get_instance_mask().astype(np.uint16)
        boundary_mask = synthesizer.get_boundary_mask().astype(np.uint8)
        distance_mask = synthesizer.get_distance_mask().astype(np.float32)
        centroid_mask = synthesizer.get_centroid_mask().astype(np.uint8)
        
        save_name = os.path.join(save_path, experiment_name+'_'+str(num_img)+'.h5')        
        h5_writer([instance_mask, boundary_mask, distance_mask, centroid_mask], 
                  save_name, group_names=['instances', 'boundary', 'distance', 'seeds'])


def agglomerative_clustering_2d(x_samples, y_samples, max_dist=10):
    """2D version of agglomerative clustering"""
    
    samples = np.array([x_samples, y_samples]).T
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=max_dist, linkage='complete').fit(samples)
    
    cluster_labels = clustering.labels_
    
    cluster_samples_x = []
    cluster_samples_y = []
    for label in np.unique(cluster_labels):
        cluster_samples_x.append(int(np.mean(x_samples[cluster_labels==label])))
        cluster_samples_y.append(int(np.mean(y_samples[cluster_labels==label])))
        
    cluster_samples_x = np.array(cluster_samples_x)
    cluster_samples_y = np.array(cluster_samples_y)
        
    return cluster_samples_x, cluster_samples_y


class SyntheticCellMembranes2D:
    """Base class for generating 2D synthetic cell membranes"""
    
    def __init__(self, width=182, height=256, cell_size_range=(20, 80), 
                 cell_density=0.3, irregularity=0.2, membrane_thickness=2):
        """
        Initialize 2D cell membrane synthesizer
        
        Args:
            width (int): Image width
            height (int): Image height
            cell_size_range (tuple): Min and max cell radius
            cell_density (float): Density of cells (0-1)
            irregularity (float): Shape irregularity factor (0-1)
            membrane_thickness (int): Thickness of cell membranes
        """
        
        self.width = width
        self.height = height
        self.cell_size_range = cell_size_range
        self.cell_density = cell_density
        self.irregularity = irregularity
        self.membrane_thickness = membrane_thickness
        
        self.instance_mask = None
        self.cell_centers = []
        self.cell_radii = []
        
    def _cart2polar(self, x, y):
        """Convert cartesian to polar coordinates"""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta
        
    def _polar2cart(self, r, theta):
        """Convert polar to cartesian coordinates"""
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
        
    def generate_instances(self):
        """Generate synthetic cell instances"""
        #print_timestamp('Generating 2D cell layout...')
        self._place_cell_centers()
        #print_timestamp('Creating cell shapes...')
        self._create_cell_shapes()
        #print_timestamp('Applying Voronoi tessellation...')
        self._voronoi_tessellation()
       # print_timestamp('Post-processing...')
        #self._post_processing()
        
    def _place_cell_centers(self):
        """Place cell centers using Poisson disk sampling for natural distribution"""
        
        # Calculate target number of cells more accurately
        avg_cell_radius = np.mean(self.cell_size_range)
        avg_cell_area = np.pi * (avg_cell_radius**2)
        total_area = self.width * self.height
        
        # Use density as a direct multiplier for number of cells
        target_cells = int(self.cell_density * total_area / avg_cell_area)
        
        # For high density, use smaller minimum distance
        min_distance = avg_cell_radius * 0.5  # Increased spacing to prevent merging
        
        centers = []
        radii = []
        attempts = 0
        max_attempts = target_cells * 100  # Many attempts for high density
        
        while len(centers) < target_cells and attempts < max_attempts:
            # Random position with very small margins
            margin = avg_cell_radius * 0.5
            x = np.random.uniform(margin, self.width - margin)
            y = np.random.uniform(margin, self.height - margin)
            
            # Check distance to existing centers
            valid = True
            for center, radius in zip(centers, radii):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                # Allow proper spacing between cells
                required_dist = (radius + min_distance) * 0.9
                if dist < required_dist:
                    valid = False
                    break
                    
            if valid:
                # Random cell size
                radius = np.random.uniform(*self.cell_size_range)
                centers.append([x, y])
                radii.append(radius)
                
            attempts += 1
            
        self.cell_centers = np.array(centers)
        self.cell_radii = np.array(radii)
        
        #print(f"Placed {len(centers)} cell centers (target was {target_cells})")
        
    def _create_cell_shapes(self):
        """Create irregular cell shapes"""
        
        self.cell_boundaries = []
        
        for center, radius in zip(self.cell_centers, self.cell_radii):
            # Create irregular boundary using polar coordinates
            angles = np.linspace(0, 2*np.pi, 24, endpoint=False)  # Fewer points for smoother shapes
            
            # Add irregularity to radius - reduced for rounder cells
            if self.irregularity > 0:
                radius_variation = 1 + self.irregularity * np.random.normal(0, 0.25, len(angles))
                radius_variation = np.maximum(radius_variation, 0.7)  # Prevent too much deformation
                
                # Smooth the variations more for rounder appearance
                radius_variation = filters.gaussian(radius_variation, sigma=2.0, mode='wrap')
            else:
                radius_variation = np.ones(len(angles))
            
            irregular_radii = radius * radius_variation
            
            # Convert to cartesian coordinates
            x_boundary = center[0] + irregular_radii * np.cos(angles)
            y_boundary = center[1] + irregular_radii * np.sin(angles)
            
            self.cell_boundaries.append(np.column_stack([x_boundary, y_boundary]))
            
    def _voronoi_tessellation(self):
        """Create Voronoi tessellation for realistic cell arrangement"""
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # Build KD-tree for efficient nearest neighbor search
        if len(self.cell_centers) > 0:
            tree = cKDTree(self.cell_centers)
            distances, indices = tree.query(coords)
            
            # Create instance mask
            self.instance_mask = indices.reshape(self.height, self.width) + 1
        else:
            self.instance_mask = np.zeros((self.height, self.width), dtype=np.uint16)
            
    def _post_processing(self):
        """Apply minimal post-processing to preserve cell count"""
        
        if self.instance_mask is None:
            return
            
        # MINIMAL post-processing to preserve as many cells as possible
        #print(f"Before post-processing: {np.max(self.instance_mask)} cells")
        
        # Only remove extremely small artifacts (< 10 pixels)
        min_cell_area = 10  # Very small threshold
        
        labels_to_remove = []
        for label in np.unique(self.instance_mask):
            if label == 0:
                continue
            cell_mask = self.instance_mask == label
            if np.sum(cell_mask) < min_cell_area:
                labels_to_remove.append(label)
        
        # Remove tiny artifacts only
        for label in labels_to_remove:
            self.instance_mask[self.instance_mask == label] = 0
        
        # Skip relabeling to avoid merging cells - this was the main problem!
        # Just ensure we have the right count
        final_count = len(np.unique(self.instance_mask)) - 1  # -1 for background
        
        #print(f"After post-processing: {final_count} cells remain")
        
    def get_instance_mask(self):
        """Get the instance segmentation mask"""
        return self.instance_mask
        
    def get_centroid_mask(self, data=None):
        """Generate centroid mask"""
        
        if data is None:
            if self.instance_mask is None:
                return None
            else:
                data = self.instance_mask
                
        centroid_mask = np.zeros(data.shape, dtype=bool)
        
        regions = measure.regionprops(data)
        for props in regions:
            c = props.centroid
            centroid_mask[int(c[0]), int(c[1])] = True
            
        return centroid_mask
        
    def get_boundary_mask(self, data=None):
        """Generate cell boundary/membrane mask"""
        
        if data is None:
            if self.instance_mask is None:
                return None
            else:
                data = self.instance_mask
                
        # Create membrane mask using gradient
        membrane_mask = segmentation.find_boundaries(data, mode='thick')
        
        # Dilate to desired thickness
        if self.membrane_thickness > 1:
            selem = morphology.disk(self.membrane_thickness // 2)
            membrane_mask = morphology.dilation(membrane_mask, selem)
            
        return membrane_mask
        
    def get_distance_mask(self, data=None):
        """Generate distance transform mask"""
        
        if data is None:
            if self.instance_mask is None:
                return None
            else:
                data = self.instance_mask
                
        # Create membrane mask using gradient
        membrane_mask = segmentation.find_boundaries(data, mode='thick')
        
        # Dilate to desired thickness
        if self.membrane_thickness > 1:
            selem = morphology.disk(self.membrane_thickness // 2)
            membrane_mask = morphology.dilation(membrane_mask, selem)
                
        distance_encoding = np.zeros(membrane_mask.shape, dtype=np.float32)
        
        # Invert the binary membrane_mask
        membrane_mask = np.logical_not(membrane_mask)
        
        # Get foreground distance (inside cells)
        distance_encoding = distance_transform_edt(membrane_mask > 0)
        
        # Get background distance (outside cells)
        #distance_encoding = distance_encoding - distance_transform_edt(data <= 0)
        
        return distance_encoding


class SyntheticTissue2D(SyntheticCellMembranes2D):
    """Specialized class for tissue-like 2D cell arrangements"""
    
    def __init__(self, width=512, height=512, cell_size_range=(15, 45), 
                 cell_density=0.4, irregularity=0.3, membrane_thickness=2,
                 tissue_pattern='random'):
        """
        Initialize tissue synthesizer
        
        Args:
            tissue_pattern (str): 'random', 'hexagonal', or 'clustered'
        """
        
        super().__init__(width, height, cell_size_range, cell_density, 
                        irregularity, membrane_thickness)
        self.tissue_pattern = tissue_pattern
        
    def _place_cell_centers(self):
        """Place cell centers according to tissue pattern"""
        
        if self.tissue_pattern == 'hexagonal':
            self._place_hexagonal_pattern()
        elif self.tissue_pattern == 'clustered':
            self._place_clustered_pattern()
        else:
            super()._place_cell_centers()  # Random pattern
            
    def _place_hexagonal_pattern(self):
        """Create hexagonal tissue pattern"""
        
        # Calculate spacing for hexagonal grid
        avg_radius = np.mean(self.cell_size_range)
        spacing = avg_radius * 2.2
        
        centers = []
        radii = []
        
        # Generate hexagonal grid
        for row in range(int(self.height // (spacing * 0.866))):
            y = row * spacing * 0.866 + avg_radius
            if y > self.height - avg_radius:
                break
                
            x_offset = (spacing / 2) if row % 2 else 0
            
            for col in range(int((self.width - x_offset) // spacing)):
                x = col * spacing + x_offset + avg_radius
                if x > self.width - avg_radius:
                    break
                    
                # Add some randomness
                x += np.random.normal(0, spacing * 0.1)
                y += np.random.normal(0, spacing * 0.1)
                
                if (avg_radius < x < self.width - avg_radius and 
                    avg_radius < y < self.height - avg_radius):
                    
                    radius = np.random.uniform(*self.cell_size_range)
                    centers.append([x, y])
                    radii.append(radius)
                    
        self.cell_centers = np.array(centers)
        self.cell_radii = np.array(radii)
        
    def _place_clustered_pattern(self):
        """Create clustered tissue pattern"""
        
        # Create several cluster centers
        num_clusters = np.random.randint(3, 8)
        cluster_centers = []
        
        for _ in range(num_clusters):
            cx = np.random.uniform(100, self.width - 100)
            cy = np.random.uniform(100, self.height - 100)
            cluster_centers.append([cx, cy])
            
        centers = []
        radii = []
        
        # Place cells around cluster centers
        for cluster_center in cluster_centers:
            cluster_size = np.random.randint(5, 15)
            cluster_radius = np.random.uniform(50, 100)
            
            for _ in range(cluster_size):
                # Random position within cluster
                angle = np.random.uniform(0, 2*np.pi)
                dist = np.random.uniform(0, cluster_radius)
                
                x = cluster_center[0] + dist * np.cos(angle)
                y = cluster_center[1] + dist * np.sin(angle)
                
                if (self.cell_size_range[1] < x < self.width - self.cell_size_range[1] and 
                    self.cell_size_range[1] < y < self.height - self.cell_size_range[1]):
                    
                    radius = np.random.uniform(*self.cell_size_range)
                    centers.append([x, y])
                    radii.append(radius)
                    
        self.cell_centers = np.array(centers)
        self.cell_radii = np.array(radii)


class SyntheticEpithelium2D(SyntheticCellMembranes2D):
    """Specialized class for epithelial-like cell arrangements"""
    
    def __init__(self, width=512, height=512, cell_size_range=(20, 60), 
                 cell_density=0.5, irregularity=0.15, membrane_thickness=3,
                 elongation_factor=1.2):
        """
        Initialize epithelium synthesizer
        
        Args:
            elongation_factor (float): Factor for cell elongation
        """
        
        super().__init__(width, height, cell_size_range, cell_density, 
                        irregularity, membrane_thickness)
        self.elongation_factor = elongation_factor
        
    def _create_cell_shapes(self):
        """Create elongated epithelial cell shapes"""
        
        self.cell_boundaries = []
        
        for center, radius in zip(self.cell_centers, self.cell_radii):
            # Create elongated cell shape
            angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
            
            # Create elongation in random direction
            elongation_angle = np.random.uniform(0, 2*np.pi)
            
            # Modulate radius based on angle relative to elongation direction
            angle_diff = np.abs(np.cos(angles - elongation_angle))
            elongated_radii = radius * (1 + (self.elongation_factor - 1) * angle_diff)
            
            # Add irregularity
            radius_variation = 1 + self.irregularity * np.random.normal(0, 0.2, len(angles))
            radius_variation = np.maximum(radius_variation, 0.6)
            radius_variation = filters.gaussian(radius_variation, sigma=1.5, mode='wrap')
            
            final_radii = elongated_radii * radius_variation
            
            # Convert to cartesian coordinates
            x_boundary = center[0] + final_radii * np.cos(angles)
            y_boundary = center[1] + final_radii * np.sin(angles)
            
            self.cell_boundaries.append(np.column_stack([x_boundary, y_boundary]))


# Example usage and parameter sets
def example_usage():
    """Example of how to use the 2D synthetic cell generators"""
    
    # Basic tissue-like cells
    tissue_params = {
        'width': 512,
        'height': 512,
        'cell_size_range': (20, 50),
        'cell_density': 0.35,
        'irregularity': 0.25,
        'membrane_thickness': 2,
        'tissue_pattern': 'random'
    }
    
    # Hexagonal epithelium
    epithelium_params = {
        'width': 512,
        'height': 512,
        'cell_size_range': (25, 45),
        'cell_density': 0.45,
        'irregularity': 0.15,
        'membrane_thickness': 3,
        'elongation_factor': 1.3
    }
    
    # Generate tissue data
    print("Generating tissue-like cells...")
    generate_data_2d(SyntheticTissue2D, 
                     save_path='synthetic_tissue_2d',
                     experiment_name='tissue_2d',
                     img_count=10,
                     param_dict=tissue_params)
    
    # Generate epithelium data
    print("Generating epithelial cells...")
    generate_data_2d(SyntheticEpithelium2D,
                     save_path='synthetic_epithelium_2d', 
                     experiment_name='epithelium_2d',
                     img_count=10,
                     param_dict=epithelium_params)


if __name__ == "__main__":
    example_usage()
