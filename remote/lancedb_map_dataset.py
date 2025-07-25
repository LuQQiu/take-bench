import lancedb
from lancedb import ClientConfig, RetryConfig, TimeoutConfig
import pyarrow as pa
from torch.utils.data import Dataset
from typing import List, Optional, Union, Dict
import numpy as np
import asyncio
from datetime import timedelta


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
        host_override: str = "http://internal-k8s-lancedb-lancedbq-09d85d35db-1743442719.us-west-2.elb.amazonaws.com:80",
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        transform=None,
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
        # Store connection parameters instead of connecting immediately
        # This allows the dataset to be pickled for multiprocessing
        self.table_name = table_name
        self.db_uri = db_uri
        self.api_key = api_key
        self.host_override = host_override
        self.columns = columns
        self.transform = transform

        # Connection will be created lazily in each worker process
        self._db = None
        self._table = None
        self._len = None

    def _ensure_connection(self):
        """Ensure connection is established. Called lazily in worker processes."""
        if self._db is None:
            # Configure enlarged timeouts to handle 504 Gateway Timeout errors
            timeout_config = TimeoutConfig(
                connect_timeout=timedelta(
                    seconds=300
                ),  # 5 minutes (increased from 2 minutes)
                read_timeout=timedelta(
                    seconds=1200
                ),  # 20 minutes (increased from 5 minutes)
                pool_idle_timeout=timedelta(seconds=600),  # 10 minutes
            )

            # Configure retry strategy including 504 errors
            retry_config = RetryConfig(
                retries=5,
                connect_retries=3,
                read_retries=5,
                backoff_factor=2.0,  # Exponential backoff
                backoff_jitter=1.0,
                statuses=[429, 500, 502, 503, 504],  # Include 504
            )

            client_config = ClientConfig(
                retry_config=retry_config, timeout_config=timeout_config
            )

            self._db = lancedb.connect(
                self.db_uri,
                api_key=self.api_key,
                host_override=self.host_override,
                client_config=client_config,
            )
            self._table = self._db.open_table(self.table_name)

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
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self)} rows"
            )

        # Ensure connection is established
        self._ensure_connection()

        # Use take API with single index
        result = self._table.take([idx], columns=self.columns)

        # Convert result to pylist (list of dicts)
        items = result.to_pylist()

        if not items:
            raise RuntimeError(f"Failed to retrieve data at index {idx}")

        # Get the first (and only) item
        item = items[0]

        # No need to extract from lists since to_pylist() already returns dict per row

        if self.transform:
            item = self.transform(item)

        return item

    def __getitems__(self, indices: List[int]):
        """Get multiple items from the dataset efficiently using LanceDB's take API."""
        # Validate indices
        max_idx = max(indices) if indices else 0
        if max_idx >= len(self):
            raise IndexError(
                f"Index {max_idx} out of range for dataset with {len(self)} rows"
            )

        # Ensure connection is established
        self._ensure_connection()

        result = self._table.take(indices, columns=self.columns)

        items = []

        if hasattr(result, "read_all"):
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