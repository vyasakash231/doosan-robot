import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation as R
from matplotlib.tri import Triangulation

# Define quaternion operations
def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Return the conjugate of a quaternion q = [w, x, y, z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def exp_m(a, m=np.array([1, 0, 0, 0])):
    """
    Exponential map from the tangent space at m to the UQ manifold
    a: point in the tangent space (pure quaternion: [0, x, y, z])
    m: base point on the manifold (unit quaternion: [w, x, y, z])
    """
    a_norm = np.linalg.norm(a[1:])
    
    if a_norm < 1e-10:  # Almost zero
        return m
    
    # Compute the exponential map
    cos_part = np.cos(a_norm)
    sin_part = np.sin(a_norm) / a_norm
    exp_result = np.array([cos_part, sin_part * a[1], sin_part * a[2], sin_part * a[3]])
    
    # Quaternion product with the base point
    return quaternion_multiply(exp_result, m)

def log_m(a, m=np.array([1, 0, 0, 0])):
    """
    Logarithmic map from the UQ manifold to the tangent space at m
    a: point on the manifold (unit quaternion: [w, x, y, z])
    m: base point on the manifold (unit quaternion: [w, x, y, z])
    """
    # Compute a * m^{-1} (a * conjugate(m) for unit quaternions)
    a_m_inv = quaternion_multiply(a, quaternion_conjugate(m))
    
    # Extract the vector part and compute its norm
    v = a_m_inv[0]  # Scalar part
    u = a_m_inv[1:]  # Vector part
    u_norm = np.linalg.norm(u)
    
    if u_norm < 1e-10:  # Almost zero
        return np.array([0, 0, 0, 0])
    
    # Compute the logarithm
    angle = np.arccos(np.clip(v, -1.0, 1.0))
    log_result = angle * u / u_norm
    
    # Return as a pure quaternion (zero scalar part)
    return np.array([0, log_result[0], log_result[1], log_result[2]])

def project_to_tangent_space(vector, base_point):
    """
    Project a vector onto the tangent space at base_point.
    For unit quaternions, the tangent space consists of vectors orthogonal to the base point.
    
    Parameters:
    - vector: The vector to project (as a quaternion [w, x, y, z])
    - base_point: The point on the manifold (as a unit quaternion [w, x, y, z])
    
    Returns:
    - The projected vector (as a quaternion [w, x, y, z])
    """
    # Compute the quaternion dot product
    dot_product = np.sum(vector * base_point)
    
    # Subtract the component parallel to the base point
    projected = vector - dot_product * base_point
    
    return projected

def generate_tangent_plane(base_point, grid_size=10, scale=0.6):
    """
    Generate a grid of points in the tangent plane at the base point.
    
    Parameters:
    - base_point: The point on the manifold (as a unit quaternion [w, x, y, z])
    - grid_size: Number of points along each axis of the grid
    - scale: Scale factor for the grid
    
    Returns:
    - X, Y, Z: 2D meshgrid arrays for the tangent plane
    - basis1, basis2: Basis vectors for the tangent plane
    """
    # For unit quaternions, we need to find vectors orthogonal to the base point
    
    # Convert base_point to 3D (ignore w)
    p_3d = base_point[1:]
    p_norm = np.linalg.norm(p_3d)
    
    # Find two orthogonal vectors in the tangent plane
    if p_norm < 1e-10:
        # At identity, tangent space is spanned by [0,1,0,0], [0,0,1,0], [0,0,0,1]
        basis1 = np.array([0, 1, 0, 0])
        basis2 = np.array([0, 0, 1, 0])
    else:
        # For non-identity points, construct a basis
        # First basis vector: project x-axis to tangent space
        v1 = np.array([0, 1, 0, 0])
        v1 = project_to_tangent_space(v1, base_point)
        v1_norm = np.linalg.norm(v1[1:])
        
        # If v1 is small (base_point is close to x-axis), use y-axis instead
        if v1_norm < 0.1:
            v1 = np.array([0, 0, 1, 0])
            v1 = project_to_tangent_space(v1, base_point)
            v1_norm = np.linalg.norm(v1[1:])
        
        v1 = v1 / v1_norm
        
        # Second basis vector: orthogonal to both base_point and v1
        # Use cross product of the 3D parts
        v2_3d = np.cross(base_point[1:], v1[1:])
        v2_3d = v2_3d / np.linalg.norm(v2_3d)
        v2 = np.array([0, v2_3d[0], v2_3d[1], v2_3d[2]])
        
        basis1, basis2 = v1, v2
    
    # Create meshgrid in the tangent plane
    u = np.linspace(-scale, scale, grid_size)
    v = np.linspace(-scale, scale, grid_size)
    
    X = np.zeros((grid_size, grid_size))
    Y = np.zeros((grid_size, grid_size))
    Z = np.zeros((grid_size, grid_size))
    
    # Base point 3D coordinates - this is where to center the tangent plane
    m_3d = base_point[1:]
    
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            # Compute point in tangent space
            tangent_vector = ui * basis1 + vj * basis2
            
            # Project to ensure it's in tangent space
            tangent_vector = project_to_tangent_space(tangent_vector, base_point)
            
            # Add the base point coordinates to position the plane correctly
            point_3d = m_3d + tangent_vector[1:]
            
            X[i, j] = point_3d[0]
            Y[i, j] = point_3d[1]
            Z[i, j] = point_3d[2]
    
    return X, Y, Z, basis1, basis2

def plot_unit_quaternion_with_tangent_plane():
    """
    Visualize the unit quaternion manifold with tangent plane in the same plot,
    with the tangent plane positioned correctly at the tangent point.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Unit Quaternion Manifold with Tangent Plane', fontsize=14)
    
    # Set up sphere surface plot
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot semitransparent sphere
    ax.plot_surface(x, y, z, color='green', alpha=0.2, edgecolor=None, zorder=1)
    
    # Base point m on the manifold (Let's use a non-identity point to better show the tangent plane positioning)
    # angle = np.pi/4
    # axis = np.array([1, 0, 1])
    # axis = axis / np.linalg.norm(axis)
    # m = np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])
    # m_3d = m[1:]  # 3D coordinates for visualization

    # Base point on the surface of the 3D sphere (For a point to appear exactly on the sphere's surface in your 3D visualization, 
    # the 3D components (x, y, z) need to have a norm of exactly 1, which means w would have to be 0. )
    axis = np.array([1, 2, 3])  # Any non-zero 3D vector
    axis = axis / np.linalg.norm(axis)  # Normalize to unit length
    m = np.array([0, axis[0], axis[1], axis[2]])  # w=0, so x,y,z form a unit vector
    m_3d = m[1:]  # 3D coordinates for visualization
    
    # Plot the base point
    ax.scatter(*m_3d, color='red', s=100, label='Base point m', zorder=3)
    
    # Generate tangent plane at m
    X, Y, Z, basis1, basis2 = generate_tangent_plane(m, grid_size=15, scale=0.6)
    
    # Plot the tangent plane
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.5, edgecolor='gray', linewidth=0.5, zorder=2)
    
    # Choose some random points in the tangent space
    np.random.seed(42)
    n_points = 2
    
    # Create distinct colors for the points
    colors = plt.cm.tab10(np.linspace(0, 1, n_points))
    
    # Label positions
    label_offset = 0.1
    
    for i in range(n_points):
        # Generate a random direction in the tangent plane
        u_coeff = np.random.uniform(-0.5, 0.5)
        v_coeff = np.random.uniform(-0.5, 0.5)
        
        # Create tangent vector
        tangent_vector = u_coeff * basis1 + v_coeff * basis2
        
        # Ensure it's in the tangent space
        tangent_vector = project_to_tangent_space(tangent_vector, m)
        
        # Point in tangent space (3D coordinates)
        tangent_point_3d = m_3d + tangent_vector[1:]
        
        # Apply exponential map to get point on manifold
        q = exp_m(tangent_vector, m)
        q_3d = q[1:]  # 3D coordinates
        
        # Plot the point in tangent space
        ax.scatter(*tangent_point_3d, color=colors[i], s=80, marker='o', zorder=4)
        ax.text(tangent_point_3d[0] + label_offset, tangent_point_3d[1] + label_offset, tangent_point_3d[2] + label_offset, f'a{i+1}', color=colors[i], fontsize=12)
        
        # Plot the point on the manifold
        ax.scatter(*q_3d, color=colors[i], s=80, marker='o', zorder=4)
        ax.text(q_3d[0] + label_offset, q_3d[1] + label_offset, q_3d[2] + label_offset, f'A{i+1}', color=colors[i], fontsize=12)
        
        # Draw curve connecting tangent space to manifold (geodesic)
        t_values = np.linspace(0, 1, 20)
        curve_points = []
        
        for t in t_values:
            # Scale the tangent vector
            scaled_tv = tangent_vector * t
            # Apply exp map
            pt = exp_m(scaled_tv, m)
            curve_points.append(pt[1:])  # 3D coordinates
        
        curve_points = np.array(curve_points)
        ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], color=colors[i], linestyle='-', linewidth=2, zorder=3)
        
    # Label the tangent plane and manifold
    # Find a point on the edge of the tangent plane
    edge_idx = X.shape[0] - 1
    edge_pt = np.array([X[edge_idx, edge_idx], Y[edge_idx, edge_idx], Z[edge_idx, edge_idx]])
    
    ax.text(edge_pt[0], edge_pt[1], edge_pt[2] + 0.1, '$T_m\\mathcal{M}$', color='black', fontsize=14)
    
    # Label the manifold
    ax.text(0.5, 0.5, -0.8, '$\\mathcal{M}$', color='green', fontsize=16)
    
    # Set aspect ratio and labels
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_box_aspect([1, 1, 1])
    
    # Set viewpoint to better see the tangent plane
    ax.view_init(elev=30, azim=45)
    
    # Set limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.show()

# Run the visualizations
if __name__ == "__main__":
    print("Visualizing unit quaternion manifold with tangent plane in one plot...")
    plot_unit_quaternion_with_tangent_plane()
    