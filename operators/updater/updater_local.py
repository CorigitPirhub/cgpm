import numpy as np
from typing import List

from entity.entity import Entity, PiecewiseCatmullRomModel
from operators.associator.associator import AssociationResult
from .state_estimator import StateEstimator
from .geometry_fuser import GeometryFuser, CatmullRomGeometryFuser

class LocalUpdater:
    """
    Coordinates the "Where?" (StateEstimator) and "What?" (GeometryFuser) updates.
    """
    def __init__(self, obs_sigma: float = 0.05, geometry_reg_lambda: float = 1.0):
        self.state_estimator = StateEstimator(obs_sigma=obs_sigma)
        self.geometry_fuser = GeometryFuser(regularization_lambda=geometry_reg_lambda)
        self.catmullrom_fuser = CatmullRomGeometryFuser()

    def process(self, entities: List[Entity], associations: List[AssociationResult], observations: np.ndarray) -> List[Entity]:
        """
        Performs the update step:
        1. State Estimation (Pose Update) - Using current associations & fixed shape.
        2. Geometry Fusion (Shape Update) - Using all evidence & fixed pose.
        """
        if not associations and not any(len(e.evidence.get_all()) > 0 for e in entities):
            return entities

        # 1. Group Associations by Entity
        # We process each entity independently
        assoc_map = {i: [] for i in range(len(entities))}
        for assoc in associations:
            assoc_map[assoc.entity_index].append(assoc)
            
        # 2. Iterate Entities
        for idx, entity in enumerate(entities):
            entity_assocs = assoc_map[idx]
            
            # Note regarding Evidence: 
            # The Associator ALREADY adds the new associations to the entity's evidence set.
            # So `entity.evidence` contains both history and current frame data.
            # `entity_assocs` contains just the current frame data (needed for EKF).
            
            # --- Step 1: Link 'Where is it?' ---
            # Update Pose using EKF on current frame associations
            if entity_assocs:
                entity = self.state_estimator.estimate(entity, entity_assocs, observations)
            
            # --- Step 2: Link 'What does it look like?' ---
            # Update Shape using model-specific fusion
            if isinstance(entity.model, PiecewiseCatmullRomModel):
                entity = self.catmullrom_fuser.fuse(entity, entity_assocs, observations)
            else:
                total_evidence_count = len(entity.evidence.get_all())
                if total_evidence_count > 3:
                    entity = self.geometry_fuser.fuse(entity, entity_assocs, observations)
            
            # Note: Evidence Accumulation (Step 3) is implicit because Associator adds to set, 
            # and `fuse` uses that set.
        
        return entities
