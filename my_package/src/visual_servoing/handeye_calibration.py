import numpy as np
from scipy.optimize import minimize
import quaternion  # numpy-quaternion package

class DualQuaternion:
    """Daniilidis's 1999 paper - Dual quaternion implementation for rigid transformations."""
    
    def __init__(self, real_part, dual_part=None):
        """Initialize a dual quaternion with a real quaternion and a dual part.
        
        Args:
            real_part: A quaternion representing rotation
            dual_part: A quaternion representing translation (optional)
        """
        self.real = np.quaternion(real_part.w, real_part.x, real_part.y, real_part.z) if hasattr(real_part, 'w') else real_part
        
        if dual_part is None:
            # If no dual part is provided, create one from zeroes
            self.dual = np.quaternion(0, 0, 0, 0)
        elif isinstance(dual_part, np.ndarray) and dual_part.size == 3:
            # If a translation vector is provided, convert it to a quaternion
            t = np.quaternion(0, dual_part[0], dual_part[1], dual_part[2])
            self.dual = 0.5 * t * self.real
        else:
            self.dual = dual_part
    
    def inverse(self):
        """Return the inverse of this dual quaternion."""
        real_conj = self.real.conjugate()
        real_norm_squared = self.real * self.real.conjugate()
        
        inv_real = real_conj / real_norm_squared
        inv_dual = -real_conj * self.dual * real_conj / (real_norm_squared * real_norm_squared)
        
        return DualQuaternion(inv_real, inv_dual)
    
    def log(self):
        """Return the logarithm of this dual quaternion."""
        # Ensure real part is unit quaternion
        real_norm = np.sqrt(self.real.w**2 + self.real.x**2 + self.real.y**2 + self.real.z**2)
        
        if real_norm < 1e-10:
            return DualQuaternion(np.quaternion(0, 0, 0, 0), np.quaternion(0, 0, 0, 0))
        
        unit_real = self.real / real_norm
        
        # For unit quaternion q = [cos(theta/2), sin(theta/2)*v], 
        # log(q) = [0, (theta/2)*v]
        theta = 2 * np.arccos(min(1.0, max(-1.0, unit_real.w)))
        
        if abs(theta) < 1e-10:
            real_log = np.quaternion(0, 0, 0, 0)
        else:
            v = np.array([unit_real.x, unit_real.y, unit_real.z])
            v = v / np.sin(theta/2) * (theta/2)
            real_log = np.quaternion(0, v[0], v[1], v[2])
        
        # For dual part, it's more complex
        dual_log = self.dual * self.real.conjugate() / real_norm
        
        return DualQuaternion(real_log, dual_log)
    
    def __mul__(self, other):
        """Multiply this dual quaternion with another dual quaternion."""
        if isinstance(other, DualQuaternion):
            result_real = self.real * other.real
            result_dual = self.real * other.dual + self.dual * other.real
            return DualQuaternion(result_real, result_dual)
        else:
            # Scalar multiplication
            return DualQuaternion(self.real * other, self.dual * other)
    
    def to_matrix(self):
        """Convert dual quaternion to 4x4 transformation matrix."""
        # Extract quaternion components
        qw, qx, qy, qz = self.real.w, self.real.x, self.real.y, self.real.z
        
        # Extract translation components from dual part
        t = 2 * (self.dual * self.real.conjugate())
        tx, ty, tz = t.x, t.y, t.z
        
        # Create rotation matrix from quaternion
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [tx, ty, tz]
        
        return T


def skew(v):
    """Create a skew-symmetric matrix from a 3-element vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def quaternion_from_axis_angle(axis_angle):
    """Convert axis-angle representation to quaternion."""
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-10:
        return np.quaternion(1, 0, 0, 0)
    
    axis = axis_angle / angle
    
    qw = np.cos(angle / 2)
    qx = axis[0] * np.sin(angle / 2)
    qy = axis[1] * np.sin(angle / 2)
    qz = axis[2] * np.sin(angle / 2)
    
    return np.quaternion(qw, qx, qy, qz)


def angle_axis_and_translation_to_screw(rvec, tvec):
    """Convert axis-angle rotation and translation to screw parameters."""
    theta = np.linalg.norm(rvec)
    
    if theta < 1e-10:
        # For pure translation, return appropriate values
        return 0, np.linalg.norm(tvec), np.array([0, 0, 0]), tvec / np.linalg.norm(tvec)
    
    # Unit rotation axis
    l = rvec / theta
    
    # Point on the screw axis
    c = np.cross(tvec, l) / theta
    
    # Pitch of the screw
    d = np.dot(l, tvec)
    
    # Moment of the screw axis
    m = 0.5 * np.cross(l, c)
    
    return theta, d, l, m


def screw_to_stranpose_block_of_t(a, a_prime, b, b_prime):
    """Create S transpose block of T matrix from screw parameters."""
    S_transpose = np.zeros((6, 8))
    
    skew_a_plus_b = skew(a + b)
    a_minus_b = a - b
    
    S_transpose[0:3, 0:1] = a_minus_b.reshape(3, 1)
    S_transpose[0:3, 1:4] = skew_a_plus_b
    S_transpose[3:6, 0:1] = (a_prime - b_prime).reshape(3, 1)
    S_transpose[3:6, 1:4] = skew(a_prime + b_prime)
    S_transpose[3:6, 4:5] = a_minus_b.reshape(3, 1)
    S_transpose[3:6, 5:8] = skew_a_plus_b
    
    return S_transpose


def axis_angle_to_stranpose_block_of_t(rvec1, tvec1, rvec2, tvec2):
    """Convert axis-angle rotations and translations to S transpose block of T matrix."""
    theta1, d1, l1, m1 = angle_axis_and_translation_to_screw(rvec1, tvec1)
    theta2, d2, l2, m2 = angle_axis_and_translation_to_screw(rvec2, tvec2)
    
    a = l1
    a_prime = m1
    b = l2
    b_prime = m2
    
    return screw_to_stranpose_block_of_t(a, a_prime, b, b_prime)


def solve_quadratic_equation(a, b, c):
    """Solve quadratic equation ax^2 + bx + c = 0."""
    delta2 = b * b - 4.0 * a * c
    
    if delta2 < 0.0:
        return None, None
    
    delta = np.sqrt(delta2)
    
    x1 = (-b + delta) / (2.0 * a)
    x2 = (-b - delta) / (2.0 * a)
    
    return x1, x2


def pose_error(params, rvec1, tvec1, rvec2, tvec2):
    """Calculate pose error for optimization."""
    # Extract quaternion and translation from params
    q = np.quaternion(params[0], params[1], params[2], params[3])
    t = params[4:7]
    
    # Create dual quaternion
    dq = DualQuaternion(q, t)
    
    # Convert axis-angle rotations to quaternions
    q1 = quaternion_from_axis_angle(rvec1)
    q2 = quaternion_from_axis_angle(rvec2)
    
    # Create dual quaternions
    dq1 = DualQuaternion(q1, tvec1)
    dq2 = DualQuaternion(q2, tvec2)
    
    # Calculate error
    dq1_ = dq * dq2 * dq.inverse()
    diff = (dq1.inverse() * dq1_).log()
    
    # Compute residual (squared norm of real and dual parts)
    real_part_norm = diff.real.w**2 + diff.real.x**2 + diff.real.y**2 + diff.real.z**2
    dual_part_norm = diff.dual.w**2 + diff.dual.x**2 + diff.dual.y**2 + diff.dual.z**2
    
    return real_part_norm + dual_part_norm


class HandEyeCalibration:
    """Hand-eye calibration implementation using dual quaternions."""
    
    verbose = True
    
    @staticmethod
    def set_verbose(on=True):
        """Set verbosity of the hand-eye calibration."""
        HandEyeCalibration.verbose = on
    
    @staticmethod
    def estimate_hand_eye_screw_initial(T, planar_motion=False):
        """Initial estimation of hand-eye transformation using SVD."""
        # SVD decomposition
        U, S, Vh = np.linalg.svd(T, full_matrices=True)
        
        # Get the vectors spanning the null space of T
        v6 = Vh[5, :]
        v7 = Vh[6, :]
        v8 = Vh[7, :]
        
        # If rank is 5 (planar motion)
        if planar_motion:
            if HandEyeCalibration.verbose:
                print("# INFO: No unique solution, returned an arbitrary one.")
            v7 += v6
        
        u1 = v7[0:4]
        v1 = v7[4:8]
        u2 = v8[0:4]
        v2 = v8[4:8]
        
        lambda1 = 0.0
        lambda2 = 0.0
        
        # Find lambdas for scaling the null space vectors
        if np.dot(u1, v1) == 0.0:
            u1, u2 = u2, u1
            v1, v2 = v2, v1
        
        if np.dot(u1, v1) != 0.0:
            a = np.dot(u1, v1)
            b = np.dot(u1, v2) + np.dot(u2, v1)
            c = np.dot(u2, v2)
            
            s1, s2 = solve_quadratic_equation(a, b, c)
            
            if s1 is None:
                raise RuntimeError("Could not solve quadratic equation. Check your input data.")
            
            # Find better solution for s
            t1 = s1 * s1 * np.dot(u1, u1) + 2 * s1 * np.dot(u1, u2) + np.dot(u2, u2)
            t2 = s2 * s2 * np.dot(u1, u1) + 2 * s2 * np.dot(u1, u2) + np.dot(u2, u2)
            
            idx = 0 if t1 > t2 else 1
            s = s1 if idx == 0 else s2
            
            discriminant = 4.0 * np.dot(u1, u2)**2 - 4.0 * (np.dot(u1, u1) * np.dot(u2, u2))
            if discriminant == 0.0 and HandEyeCalibration.verbose:
                print("# INFO: Noise-free case")
            
            lambda2 = np.sqrt(1.0 / (s1 * s1 * np.dot(u1, u1) + 2 * s1 * np.dot(u1, u2) + np.dot(u2, u2)))
            lambda1 = s * lambda2
        else:
            u1_norm = np.linalg.norm(u1)
            u2_norm = np.linalg.norm(u2)
            
            if u1_norm == 0 and u2_norm > 0:
                lambda1 = 0
                lambda2 = 1.0 / u2_norm
            elif u2_norm == 0 and u1_norm > 0:
                lambda1 = 1.0 / u1_norm
                lambda2 = 0
            else:
                error_msg = "Normalization could not be handled. Your rotations and translations "
                error_msg += "are probably either not aligned or not passed in properly."
                raise RuntimeError(error_msg)
        
        # Calculate quaternion coefficients
        q_coeffs = lambda1 * u1 + lambda2 * u2
        q_prime_coeffs = lambda1 * v1 + lambda2 * v2
        
        q = np.quaternion(q_coeffs[0], q_coeffs[1], q_coeffs[2], q_coeffs[3])
        d = np.quaternion(q_prime_coeffs[0], q_prime_coeffs[1], q_prime_coeffs[2], q_prime_coeffs[3])
        
        return DualQuaternion(q, d)
    
    @staticmethod
    def estimate_hand_eye_screw_refine(dq, rvecs1, tvecs1, rvecs2, tvecs2):
        """Refine hand-eye calibration using optimization."""
        # Initialize with the dual quaternion parameters
        H = dq.to_matrix()
        initial_params = [
            dq.real.w, dq.real.x, dq.real.y, dq.real.z,
            H[0, 3], H[1, 3], H[2, 3]
        ]
        
        # Define the objective function for all poses
        def objective(params):
            total_error = 0
            
            # Normalize quaternion part
            q_norm = np.sqrt(params[0]**2 + params[1]**2 + params[2]**2 + params[3]**2)
            params_normalized = params.copy()
            params_normalized[0:4] = params_normalized[0:4] / q_norm
            
            for i in range(len(rvecs1)):
                total_error += pose_error(params_normalized, rvecs1[i], tvecs1[i], rvecs2[i], tvecs2[i])
            
            return total_error
        
        # Define constraint to keep the quaternion normalized
        def quaternion_constraint(params):
            return params[0]**2 + params[1]**2 + params[2]**2 + params[3]**2 - 1.0
        
        constraint = {'type': 'eq', 'fun': quaternion_constraint}
        
        # Run optimization
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            constraints=[constraint],
            options={'maxiter': 500, 'disp': HandEyeCalibration.verbose}
        )
        
        if HandEyeCalibration.verbose:
            print(f"Optimization result: {result.message}")
            print(f"Final error: {result.fun}")
        
        # Extract optimized parameters
        optimized_params = result.x
        q = np.quaternion(optimized_params[0], optimized_params[1], optimized_params[2], optimized_params[3])
        t = optimized_params[4:7]
        
        return DualQuaternion(q, t)
    
    @staticmethod
    def estimate_hand_eye_screw(rvecs1, tvecs1, rvecs2, tvecs2, planar_motion=False):
        """Estimate hand-eye calibration using dual quaternion approach.
        
        Args:
            rvecs1: List of axis-angle rotations for the first transform.
            tvecs1: List of translations for the first transform.
            rvecs2: List of axis-angle rotations for the second transform.
            tvecs2: List of translations for the second transform.
            planar_motion: Flag indicating if the motion is planar.
            
        Returns:
            H_12: Homogeneous transformation matrix (4x4) from system 1 to system 2.
        """
        motion_count = len(rvecs1)
        
        # Create matrix T
        T = np.zeros((motion_count * 6, 8))
        
        for i in range(motion_count):
            rvec1 = rvecs1[i]
            tvec1 = tvecs1[i]
            rvec2 = rvecs2[i]
            tvec2 = tvecs2[i]
            
            # Skip cases with zero rotation
            if np.linalg.norm(rvec1) == 0 or np.linalg.norm(rvec2) == 0:
                continue
            
            T[i*6:(i+1)*6, :] = axis_angle_to_stranpose_block_of_t(rvec1, tvec1, rvec2, tvec2)
        
        # Initial estimation
        dq = HandEyeCalibration.estimate_hand_eye_screw_initial(T, planar_motion)
        
        H_12 = dq.to_matrix()
        if HandEyeCalibration.verbose:
            print("# INFO: Before refinement: H_12 =")
            print(H_12)
        
        # Refinement
        dq = HandEyeCalibration.estimate_hand_eye_screw_refine(dq, rvecs1, tvecs1, rvecs2, tvecs2)
        
        H_12 = dq.to_matrix()
        if HandEyeCalibration.verbose:
            print("# INFO: After refinement: H_12 =")
            print(H_12)
        
        return H_12


# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    import random
    
    # True transformation to be estimated
    true_rotation = np.array([0.1, 0.2, 0.3])  # axis-angle
    true_translation = np.array([0.5, -0.2, 0.7])
    
    # Create true transformation matrix
    true_q = quaternion_from_axis_angle(true_rotation)
    true_dq = DualQuaternion(true_q, true_translation)
    true_H = true_dq.to_matrix()
    
    print("True transformation:")
    print(true_H)
    
    # Generate random poses
    n_poses = 10
    rvecs1 = []
    tvecs1 = []
    rvecs2 = []
    tvecs2 = []
    
    for _ in range(n_poses):
        # Generate random pose for system 1
        r1 = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        r1 = r1 / np.linalg.norm(r1) * random.uniform(0.1, 0.5)  # Scale to reasonable angle
        t1 = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        
        # Convert to dual quaternion
        q1 = quaternion_from_axis_angle(r1)
        dq1 = DualQuaternion(q1, t1)
        
        # Calculate pose for system 2 (A = X*B*X^-1)
        dq2 = true_dq.inverse() * dq1 * true_dq
        
        # Convert back to axis-angle and translation
        H2 = dq2.to_matrix()
        q2 = dq2.real
        t2 = H2[0:3, 3]
        
        # Convert quaternion to axis-angle
        angle = 2 * np.arccos(min(1.0, max(-1.0, q2.w)))
        if abs(angle) < 1e-10:
            r2 = np.zeros(3)
        else:
            axis = np.array([q2.x, q2.y, q2.z]) / np.sin(angle/2)
            r2 = axis * angle
        
        rvecs1.append(r1)
        tvecs1.append(t1)
        rvecs2.append(r2)
        tvecs2.append(t2)
    
    # Estimate transformation
    HandEyeCalibration.set_verbose(True)
    H_12 = HandEyeCalibration.estimate_hand_eye_screw(rvecs1, tvecs1, rvecs2, tvecs2)
    
    print("\nEstimated transformation:")
    print(H_12)
    
    print("\nError (difference):")
    print(H_12 - true_H)