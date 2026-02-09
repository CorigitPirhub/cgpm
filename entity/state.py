import numpy as np

class SE2Pose:
    """
    Represents the pose g in SE(2): (x, y, theta).
    Corresponds to \\hat{g} in Definition 2.3.
    """
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    @property
    def vector(self) -> np.ndarray:
        """Returns the pose state as a vector [x, y, theta]."""
        return np.array([self.x, self.y, self.theta])

    def to_matrix(self) -> np.ndarray:
        """Returns the 3x3 homogeneous transformation matrix."""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0, 1.0]
        ])

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'SE2Pose':
        """Creates an SE2Pose from a 3x3 homogeneous matrix."""
        x = matrix[0, 2]
        y = matrix[1, 2]
        theta = np.arctan2(matrix[1, 0], matrix[0, 0])
        return cls(x, y, theta)

    def __repr__(self):
        return f"SE2Pose(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f})"


class StateVector:
    """
    Represents the full state vector for tracking, including kinematics.
    State x = [x, y, theta, v, omega]^T
    
    x, y: Position
    theta: Heading
    v: Linear velocity (longitudinal)
    omega: Angular velocity (yaw rate)
    """
    def __init__(self, 
                 x: float = 0.0, 
                 y: float = 0.0, 
                 theta: float = 0.0, 
                 v: float = 0.0, 
                 omega: float = 0.0):
        self.pose = SE2Pose(x, y, theta) # Composition
        self.v = float(v)
        self.omega = float(omega)

    @property
    def x(self) -> float: return self.pose.x
    
    @property
    def y(self) -> float: return self.pose.y

    @property
    def theta(self) -> float: return self.pose.theta

    @property
    def vector(self) -> np.ndarray:
        """Returns the full state as a numpy array [x, y, theta, v, omega]."""
        return np.array([self.x, self.y, self.theta, self.v, self.omega])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'StateVector':
        """Creates a StateVector from an array-like [x, y, theta, v, omega]."""
        if len(vec) < 3: 
           return cls(*vec) # Hope for the best or error
        if len(vec) == 3: # just pose
           return cls(vec[0], vec[1], vec[2])
        return cls(vec[0], vec[1], vec[2], vec[3], vec[4] if len(vec) > 4 else 0.0)

    def __repr__(self):
        return f"StateVector(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f}, v={self.v:.2f}, w={self.omega:.2f})"


class StateCovariance:
    """
    Represents the covariance matrix Sigma_g (3x3) defined on the Lie algebra se(2).
    Corresponds to \\Sigma_g in Definition 2.3.
    """
    def __init__(self, cov_matrix: np.ndarray = None):
        if cov_matrix is None:
            self.matrix = np.eye(3) * 1e-6 # Default small uncertainty
        else:
            self.matrix = np.array(cov_matrix)
            # assert self.matrix.shape == (3, 3), "Covariance matrix must be 3x3" 
            # Note: For full state, this might need to be 5x5 eventually, 
            # but strictly Definition 2.3 defines Sigma_g for pose.
            # We will keep it as is for now, or users might use a 5x5 for the full StateVector.

    def __repr__(self):
        return f"StateCovariance(\n{self.matrix}\n)"
