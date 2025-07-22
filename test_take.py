# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import numpy as np
import pyarrow as pa
import pytest
import lancedb


def create_test_data():
    """Create test data for take operations."""
    return pa.table({
        "id": pa.array([0, 1, 2, 3, 4], type=pa.int32()),
        "value": pa.array([10, 20, 30, 40, 50], type=pa.int32()),
        "name": pa.array(["alice", "bob", "charlie", "diana", "eve"], type=pa.string()),
        "vector": pa.array([[1.0, 2.0, 3.0, 4.0], 
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0],
                           [13.0, 14.0, 15.0, 16.0],
                           [17.0, 18.0, 19.0, 20.0]], type=pa.list_(pa.float32()))
    })


def read_recordbatch_stream(result):
    """Helper to read RecordBatchStream and convert to PyArrow Table."""
    if hasattr(result, 'read_all'):
        # Standard PyArrow RecordBatchReader
        return result.read_all()
    else:
        # RecordBatchStream only supports async iteration
        # For sync tests, we need to collect the data using asyncio
        import asyncio
        
        async def collect_batches():
            batches = []
            async for batch in result:
                batches.append(batch)
            return batches
        
        batches = asyncio.run(collect_batches())
        if batches:
            return pa.Table.from_batches(batches, schema=result.schema)
        return None


def test_take_basic():
    """Test basic take functionality with row indices."""
    # Connect to remote database
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    
    # Create table
    data = create_test_data()
    table = db.create_table("test_take", data=data, mode="overwrite")
    
    # Take specific rows
    result = table.take([0, 2, 4])
    result_table = read_recordbatch_stream(result)
    
    assert result_table is not None
    assert len(result_table) == 3
    
    # Verify correct rows were returned
    df = result_table.to_pandas()
    assert df['id'].tolist() == [0, 2, 4]
    assert df['value'].tolist() == [10, 30, 50]
    assert df['name'].tolist() == ["alice", "charlie", "eve"]


def test_take_with_columns():
    """Test take with column selection."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_cols", data=data, mode="overwrite")
    
    # Take specific rows and columns
    result = table.take([1, 3], columns=["id", "name"])
    result_table = read_recordbatch_stream(result)
    
    assert result_table is not None
    assert len(result_table) == 2
    assert result_table.column_names == ["id", "name"]
    
    # Verify data
    df = result_table.to_pandas()
    assert df['id'].tolist() == [1, 3]
    assert df['name'].tolist() == ["bob", "diana"]


def test_take_with_schema_columns():
    """Test take with columns specified as PyArrow schema."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_schema", data=data, mode="overwrite")
    
    # Create schema with subset of columns
    schema = pa.schema([
        pa.field("value", pa.int32()),
        pa.field("vector", pa.list_(pa.float32()))
    ])
    
    # Take with schema
    result = table.take([0, 4], columns=schema)
    result_table = read_recordbatch_stream(result)
    
    assert result_table is not None
    assert len(result_table) == 2
    assert set(result_table.column_names) == {"value", "vector"}
    
    # Verify data
    df = result_table.to_pandas()
    assert df['value'].tolist() == [10, 50]


def test_take_numpy_indices():
    """Test take with numpy array indices."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_numpy", data=data, mode="overwrite")
    
    # Use numpy array for indices
    indices = np.array([1, 2, 3])
    result = table.take(indices)
    result_table = read_recordbatch_stream(result)
    
    assert result_table is not None
    assert len(result_table) == 3
    
    # Verify correct rows
    df = result_table.to_pandas()
    assert df['id'].tolist() == [1, 2, 3]


def test_take_single_index():
    """Test take with a single index."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_single", data=data, mode="overwrite")
    
    # Take single row
    result = table.take([2])
    result_table = read_recordbatch_stream(result)
    
    assert result_table is not None
    assert len(result_table) == 1
    
    # Verify data
    df = result_table.to_pandas()
    assert df['id'].tolist() == [2]
    assert df['name'].tolist() == ["charlie"]


def test_take_empty_indices():
    """Test take with empty indices - should raise error."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_empty", data=data, mode="overwrite")
    
    # Empty indices should raise an error
    with pytest.raises(Exception, match="indices cannot be empty"):
        table.take([])


def test_take_out_of_bounds():
    """Test take with out of bounds indices."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_bounds", data=data, mode="overwrite")
    
    # Out of bounds indices should raise an error
    with pytest.raises(Exception):
        table.take([0, 10])  # Index 10 is out of bounds


def test_take_duplicate_indices():
    """Test take with duplicate indices."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_dup", data=data, mode="overwrite")
    
    # Take with duplicate indices
    result = table.take([1, 1, 3, 3])
    result_table = read_recordbatch_stream(result)
    
    assert result_table is not None
    assert len(result_table) == 4
    
    # Verify duplicates are returned
    df = result_table.to_pandas()
    assert df['id'].tolist() == [1, 1, 3, 3]
    assert df['name'].tolist() == ["bob", "bob", "diana", "diana"]


def test_take_all_rows():
    """Test take with all row indices."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_all", data=data, mode="overwrite")
    
    # Take all rows in order
    result = table.take([0, 1, 2, 3, 4])
    result_table = read_recordbatch_stream(result)
    
    assert result_table is not None
    assert len(result_table) == 5
    
    # Should be same as original
    assert result_table.to_pandas().equals(data.to_pandas())


def test_take_reverse_order():
    """Test take with indices in reverse order."""
    db = lancedb.connect("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = db.create_table("test_take_reverse", data=data, mode="overwrite")
    
    # Take rows in reverse order
    result = table.take([4, 3, 2, 1, 0])
    result_table = read_recordbatch_stream(result)
    
    assert result_table is not None
    assert len(result_table) == 5
    
    # Verify reverse order
    df = result_table.to_pandas()
    assert df['id'].tolist() == [4, 3, 2, 1, 0]
    assert df['name'].tolist() == ["eve", "diana", "charlie", "bob", "alice"]


@pytest.mark.asyncio
async def test_take_async():
    """Test async take functionality."""
    # Connect to remote database asynchronously
    db = await lancedb.connect_async("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    
    # Create table
    data = create_test_data()
    table = await db.create_table("test_take_async", data=data, mode="overwrite")
    
    # Take specific rows
    result = await table.take([0, 2, 4])
    
    # RecordBatchStream for async iteration
    batches = []
    async for batch in result:
        batches.append(batch)
    result_table = pa.Table.from_batches(batches, schema=result.schema)
    
    assert len(result_table) == 3
    
    # Verify correct rows were returned
    df = result_table.to_pandas()
    assert df['id'].tolist() == [0, 2, 4]


@pytest.mark.asyncio
async def test_take_async_with_columns():
    """Test async take with column selection."""
    db = await lancedb.connect_async("db://my-db", api_key="sk_localtest", host_override="http://localhost:10024")
    data = create_test_data()
    table = await db.create_table("test_take_async_cols", data=data, mode="overwrite")
    
    # Take specific rows and columns
    result = await table.take([1, 3], columns=["id", "name"])
    
    # Collect batches from RecordBatchStream
    batches = []
    async for batch in result:
        batches.append(batch)
    result_table = pa.Table.from_batches(batches, schema=result.schema)
    
    assert len(result_table) == 2
    assert result_table.column_names == ["id", "name"]
    
    # Verify data
    df = result_table.to_pandas()
    assert df['id'].tolist() == [1, 3]
    assert df['name'].tolist() == ["bob", "diana"]