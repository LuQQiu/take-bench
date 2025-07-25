import lance
import pyarrow as pa
from torch.utils.data import Dataset
from typing import List, Optional, Union, Dict
import numpy as np


class LanceMapDataset(Dataset):
    """
    A PyTorch MapDataset implementation that uses Lance's take API for efficient data loading.
    
    This dataset supports batch loading through __getitems__ for improved performance.
    """
    
    def __init__(
        self, 
        dataset_path: str, 
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        transform=None
    ):
        """
        Initialize the LanceMapDataset.
        
        Args:
            dataset_path: Path to the Lance dataset
            columns: Optional list of column names to load, or dict of column names to SQL expressions
            transform: Optional transform to apply to the data
        """
        self.dataset_path = dataset_path
        self.columns = columns
        self.transform = transform
        self._initialize_dataset()
        
    def _initialize_dataset(self):
        """Initialize dataset with storage options."""
        import os
        
        storage_options = {
            "client_max_retries": "30", 
            "client_retry_timeout": "1800",
            "timeout": "60s",
            "connect_timeout": "30s",
            "aws_region": "us-west-2"
        }
        
        # Check for AWS credentials in environment
        if os.environ.get('AWS_ACCESS_KEY_ID'):
            storage_options["aws_access_key_id"] = os.environ['AWS_ACCESS_KEY_ID']
        if os.environ.get('AWS_SECRET_ACCESS_KEY'):
            storage_options["aws_secret_access_key"] = os.environ['AWS_SECRET_ACCESS_KEY']
        if os.environ.get('AWS_SESSION_TOKEN'):
            storage_options["aws_session_token"] = os.environ['AWS_SESSION_TOKEN']
            
        self.dataset = lance.dataset(self.dataset_path, storage_options=storage_options)
        self._len = self.dataset.count_rows()
        
    def __getstate__(self):
        """Prepare for pickling."""
        state = self.__dict__.copy()
        del state['dataset']
        del state['_len']
        return state
        
    def __setstate__(self, state):
        """Restore after pickling."""
        self.__dict__.update(state)
        self._initialize_dataset()
        
    def __len__(self):
        """Return the total number of rows in the dataset."""
        return self._len
    
    def __getitem__(self, idx: int):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            The data at the specified index
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} rows")
        
        # Use take API with single index
        table = self.dataset.take([idx], columns=self.columns)
        
        # Convert to dict for easier handling
        item = table.to_pydict()
        
        # If single item, extract from lists
        for key in item:
            if isinstance(item[key], list) and len(item[key]) == 1:
                item[key] = item[key][0]
        
        if self.transform:
            item = self.transform(item)
            
        return item
    
    def __getitems__(self, indices: List[int]):
        """
        Get multiple items from the dataset efficiently using Lance's take API.
        
        This is the key method for efficient batch loading in PyTorch DataLoader.
        
        Args:
            indices: List of indices to retrieve
            
        Returns:
            List of items at the specified indices
        """
        # Validate indices
        max_idx = max(indices) if indices else 0
        if max_idx >= len(self):
            raise IndexError(f"Index {max_idx} out of range for dataset with {len(self)} rows")
        
        # Use Lance's take API for efficient batch retrieval
        table = self.dataset.take(indices, columns=self.columns)
        
        return table.to_pylist()
    