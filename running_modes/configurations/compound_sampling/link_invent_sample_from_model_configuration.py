from dataclasses import dataclass
from typing import List

@dataclass
class LinkInventSampleFromModelConfiguration:
    model_path: str
    output_smiles_path: str
    # smiles: str
    warheads: str
    num_smiles: int = 128
    batch_size: int = 10
    with_likelihood: bool = False