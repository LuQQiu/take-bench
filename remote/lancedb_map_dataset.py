import lancedb
import pyarrow as pa
from torch.utils.data import Dataset
from typing import List, Optional, Union, Dict
import numpy as np
import asyncio


class LanceDBMapDataset(Dataset):
    """
    A PyTorch MapDataset implementation that uses LanceDB's remote take API for efficient data loading.
    
    This dataset supports batch loading through __getitems__ for improved performance.
    """
    
    def __init__(
        self, 
        table_name: str,
        db_uri: str = "db://my-db",
        api_key: str = "sk_localtest",
        host_override: str = "http://localhost:10024",
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        transform=None
    ):
        """
        Initialize the LanceDBMapDataset.
        
        Args:
            table_name: Name of the table in LanceDB
            db_uri: Database URI
            api_key: API key for authentication
            host_override: Host override for local testing
            columns: Optional list of column names to load, or dict of column names to SQL expressions
            transform: Optional transform to apply to the data
        """
        # Connect to remote LanceDB
        self.db = lancedb.connect(db_uri, api_key=api_key, host_override=host_override)
        self.table = self.db.open_table(table_name)
        self.columns = columns
        self.transform = transform
        
        # Get table size
        # Note: count_rows() might not be available for remote tables, 
        # so we'll need to get this info differently
        # For now, assume the table size is provided or use a query
        result = self.table.search().limit(0).to_arrow()
        self._len = self.table.count_rows() if hasattr(self.table, 'count_rows') else None
        
        if self._len is None:
            # Fallback: this is inefficient but works
            # In production, you'd want to store this metadata separately
            raise NotImplementedError("Remote table size detection not implemented. Please provide table size.")
    
    def set_length(self, length: int):
        """Manually set the dataset length if count_rows() is not available."""
        self._len = length
        
    def __len__(self):
        """Return the total number of rows in the dataset."""
        if self._len is None:
            raise RuntimeError("Dataset length not set. Call set_length() first.")
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
        result = self.table.take([idx], columns=self.columns)
        table = self._read_recordbatch_stream(result)
        
        if table is None or len(table) == 0:
            raise RuntimeError(f"Failed to retrieve data at index {idx}")
        
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
        """Get multiple items from the dataset efficiently using LanceDB's take API."""
        # Validate indices
        max_idx = max(indices) if indices else 0
        if max_idx >= len(self):
            raise IndexError(f"Index {max_idx} out of range for dataset with {len(self)} rows")
        
        result = self.table.take(indices, columns=self.columns)
        
        items = []
        
        if hasattr(result, 'read_all'):
            table = result.read_all()
            items = table.to_pylist()
        else:
            # Most efficient: convert each batch directly
            async def process_batches():
                all_items = []
                async for batch in result:
                    # RecordBatch also has to_pylist() method!
                    all_items.extend(batch.to_pylist())
                return all_items
            
            items = asyncio.run(process_batches())
        
        # Apply transforms if needed
        if self.transform:
            items = [self.transform(item) for item in items]
        
        return items