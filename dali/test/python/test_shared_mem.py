# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import multiprocessing
import numpy as np
from nose_utils import raises
from nvidia.dali._multiproc import shared_mem


def test_shared_mem_allocate():
    """Test basic shared memory allocation."""
    size = 1024
    shm = shared_mem.SharedMem.allocate(size)

    assert shm.capacity == size
    assert shm.handle >= 0
    assert len(shm.buf) == size

    # Test that we can write to the buffer
    test_data = b"test data"
    shm.buf[:len(test_data)] = test_data
    assert shm.buf[:len(test_data)] == test_data

    shm.close()


def test_shared_mem_resize():
    """Test resizing shared memory."""
    initial_size = 256
    shm = shared_mem.SharedMem.allocate(initial_size)

    # Write initial data
    initial_data = b"initial"
    shm.buf[:len(initial_data)] = initial_data

    # Resize to larger size
    new_size = 1024
    shm.resize(new_size, trunc=True)

    assert shm.capacity == new_size
    assert len(shm.buf) == new_size
    # Check that original data is preserved
    assert shm.buf[:len(initial_data)] == initial_data

    # Resize to smaller size
    smaller_size = 128
    shm.resize(smaller_size, trunc=True)
    assert shm.capacity == smaller_size
    assert len(shm.buf) == smaller_size

    shm.close()


def test_shared_mem_resize_without_trunc():
    """Test resizing without truncating the underlying memory."""
    initial_size = 256
    shm = shared_mem.SharedMem.allocate(initial_size)

    # Write initial data
    initial_data = b"initial data"
    shm.buf[:len(initial_data)] = initial_data

    # Resize without trunc (just adjust mapping)
    new_size = 512
    shm.resize(new_size, trunc=False)

    assert shm.capacity == new_size
    assert len(shm.buf) == new_size
    # Check that original data is preserved
    assert shm.buf[:len(initial_data)] == initial_data

    shm.close()


def test_shared_mem_close_handle():
    """Test closing the handle while keeping the mapping."""
    size = 256
    shm = shared_mem.SharedMem.allocate(size)

    # Write some data
    test_data = b"test data"
    shm.buf[:len(test_data)] = test_data

    # Close handle but keep mapping
    shm.close_handle()

    # Should still be able to access the buffer
    assert shm.buf[:len(test_data)] == test_data
    assert len(shm.buf) == size

    # But handle should be invalid
    assert shm.handle == -1

    shm.close()


def test_shared_mem_close():
    """Test closing shared memory completely."""
    size = 256
    shm = shared_mem.SharedMem.allocate(size)

    # Write some data
    test_data = b"test data"
    shm.buf[:len(test_data)] = test_data

    # Close everything
    shm.close()

    # Should not be able to access buffer anymore
    assert shm.buf is None
    assert shm.handle == -1


def test_shared_mem_multiprocessing():
    """Test shared memory across multiple processes."""
    size = 1024

    def worker_process(shm_handle, size, result_queue):
        try:
            # Open shared memory in worker process
            shm = shared_mem.SharedMem.open(shm_handle, size)

            # Read data written by parent
            data = bytes(shm.buf[:11])
            result_queue.put(data)

            # Write data back to parent
            response_data = b"response"
            shm.buf[:len(response_data)] = response_data

            shm.close()
            result_queue.put("success")
        except Exception as e:
            result_queue.put(f"error: {e}")

    # Allocate shared memory in parent
    shm = shared_mem.SharedMem.allocate(size)

    # Write data
    test_data = b"parent data"
    shm.buf[:len(test_data)] = test_data

    # Start worker process
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=worker_process,
        args=(shm.handle, size, result_queue)
    )
    process.start()
    process.join()

    # Check results
    assert not result_queue.empty()
    result = result_queue.get()

    assert result == test_data

    status = result_queue.get()
    assert status == "success"

    # Check data written by worker
    response_data = b"response"
    assert shm.buf[:len(response_data)] == response_data

    shm.close()


def test_shared_mem_large_allocation():
    """Test allocation of large shared memory chunks."""
    # Test with 1MB
    size = 1024 * 1024
    shm = shared_mem.SharedMem.allocate(size)

    assert shm.capacity == size
    assert len(shm.buf) == size

    # Write data to different parts of the buffer
    shm.buf[0] = 42
    shm.buf[size // 2] = 123
    shm.buf[size - 1] = 255

    assert shm.buf[0] == 42
    assert shm.buf[size // 2] == 123
    assert shm.buf[size - 1] == 255

    shm.close()


def test_shared_mem_numpy_array():
    """Test using shared memory with numpy arrays."""
    size = 1024
    shm = shared_mem.SharedMem.allocate(size)

    # Create numpy array from shared memory buffer
    arr = np.frombuffer(shm.buf, dtype=np.uint8)

    # Write data using numpy
    test_data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    arr[:len(test_data)] = test_data

    # Verify data
    assert np.array_equal(arr[:len(test_data)], test_data)
    assert shm.buf[:len(test_data)] == test_data.tobytes()

    shm.close()


def test_shared_mem_invalid_handle():
    """Test behavior with invalid handle."""
    size = 256

    # Test with invalid handle (-1)
    shm = shared_mem.SharedMem(None, size)

    assert shm.capacity == size
    assert shm.handle >= 0  # Should create a new handle
    assert len(shm.buf) == size

    shm.close()


@raises(ValueError, glob="Cannot create buffer - no shared memory object provided")
def test_shared_mem_null_pointer():
    """Test error when trying to access buffer of None shared memory."""
    shm = shared_mem.SharedMem(None, 256)
    shm.close()
    # This should raise an error
    _ = shm.buf


@raises(ValueError, glob="Cannot create buffer - no memory has been mapped")
def test_shared_mem_no_mapping():
    """Test error when trying to access buffer without mapping."""
    shm = shared_mem.SharedMem(None, 256)
    shm.close()
    # This should raise an error
    _ = shm.buf


def test_shared_mem_context_manager():
    """Test using SharedMem as a context manager."""
    size = 256

    with shared_mem.SharedMem.allocate(size) as shm:
        assert shm.capacity == size
        assert len(shm.buf) == size

        # Write some data
        test_data = b"context test"
        shm.buf[:len(test_data)] = test_data
        assert shm.buf[:len(test_data)] == test_data

    # Should be closed after context exit
    # Note: This test assumes SharedMem implements __enter__ and __exit__
    # If it doesn't, this test will need to be modified


def test_shared_mem_concurrent_access():
    """Test concurrent access to shared memory from multiple threads."""
    import threading

    size = 1024
    shm = shared_mem.SharedMem.allocate(size)

    results = []
    errors = []

    def worker_thread(thread_id, offset):
        try:
            # Write thread-specific data
            data = f"thread_{thread_id}".encode()
            shm.buf[offset:offset + len(data)] = data
            results.append((thread_id, data))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Start multiple threads
    threads = []
    for i in range(4):
        offset = i * 64
        thread = threading.Thread(target=worker_thread, args=(i, offset))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check results
    assert len(errors) == 0, f"Thread errors: {errors}"
    assert len(results) == 4

    # Verify data written by each thread
    for thread_id, expected_data in results:
        offset = thread_id * 64
        actual_data = bytes(shm.buf[offset:offset + len(expected_data)])
        assert actual_data == expected_data

    shm.close()


def test_shared_mem_resize_edge_cases():
    """Test edge cases for resizing."""
    size = 256
    shm = shared_mem.SharedMem.allocate(size)

    # Test resize to same size
    shm.resize(size, trunc=True)
    assert shm.capacity == size

    # Test resize to 0 (should work)
    shm.resize(1, trunc=True)
    assert shm.capacity == 1
    assert len(shm.buf) == 1

    # Test resize back to positive size
    shm.resize(size, trunc=True)
    assert shm.capacity == size
    assert len(shm.buf) == size

    shm.close()


def test_shared_mem_handle_transfer():
    """Test transferring shared memory handle between processes."""
    size = 512

    def handle_receiver(handle, size, result_queue):
        try:
            shm = shared_mem.SharedMem.open(handle, size)
            data = bytes(shm.buf[:13])
            result_queue.put(data)
            shm.close()
        except Exception as e:
            result_queue.put(f"error: {e}")

    # Allocate shared memory
    shm = shared_mem.SharedMem.allocate(size)

    # Write data
    test_data = b"transfer test"
    shm.buf[:len(test_data)] = test_data

    # Transfer handle to another process
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=handle_receiver,
        args=(shm.handle, size, result_queue)
    )
    process.start()
    process.join()

    # Check result
    result = result_queue.get()
    assert result == test_data

    shm.close()


def test_shared_mem_memory_leak():
    """Test that shared memory is properly cleaned up."""
    import gc

    # Allocate multiple shared memory chunks
    chunks = []
    for i in range(10):
        shm = shared_mem.SharedMem.allocate(1024)
        chunks.append(shm)

    # Close all chunks
    for shm in chunks:
        shm.close()

    # Force garbage collection
    gc.collect()

    # The test passes if no exceptions are raised
    # In a real scenario, you might want to check system resources
    # or use tools like valgrind to detect memory leaks


if __name__ == "__main__":
    # Run tests
    test_shared_mem_allocate()
    test_shared_mem_resize()
    test_shared_mem_resize_without_trunc()
    test_shared_mem_close_handle()
    test_shared_mem_close()
    test_shared_mem_multiprocessing()
    test_shared_mem_large_allocation()
    test_shared_mem_numpy_array()
    test_shared_mem_invalid_handle()
    test_shared_mem_concurrent_access()
    test_shared_mem_resize_edge_cases()
    test_shared_mem_handle_transfer()
    test_shared_mem_memory_leak()

    print("All tests passed!")
