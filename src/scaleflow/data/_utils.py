import concurrent.futures
from collections.abc import Iterable, Mapping
from typing import Any

import anndata as ad
import numpy as np
import tqdm
import zarr
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec


def _get_size(shape: tuple[int, ...], chunk_size: int, shard_size: int) -> tuple[int, int]:
    shard_size_used = shard_size
    chunk_size_used = chunk_size
    if chunk_size > shape[0]:
        chunk_size_used = shard_size_used = shape[0]
    elif chunk_size < shape[0] and shard_size > shape[0]:
        chunk_size_used = shard_size_used = shape[0]
    return chunk_size_used, shard_size_used


def write_single_array(group, key: str, arr: np.ndarray, chunk_size: int, shard_size: int) -> str:
    """Write a single array - designed for threading"""
    chunk_size_used, shard_size_used = _get_size(arr.shape, chunk_size, shard_size)
    group.create_array(
        name=key,
        data=arr,
        chunks=(chunk_size_used, arr.shape[1]) if len(arr.shape) > 1 else (chunk_size_used,),
        shards=(shard_size_used, arr.shape[1]) if len(arr.shape) > 1 else (shard_size_used,),
        compressors=None,
    )
    return key


def write_dist_data_threaded(
    group,
    dist_data: dict[int, np.ndarray],
    chunk_size: int,
    shard_size: int,
    max_workers: int = 24,
) -> None:
    """Write distribution data using threading for I/O parallelism"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all write tasks
        future_to_key = {
            executor.submit(
                write_single_array,
                group,
                str(k),
                dist_data[k],
                chunk_size,
                shard_size,
            ): k
            for k in dist_data.keys()
        }

        # Process results with progress bar
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_key), total=len(future_to_key), desc=f"Writing {group.name}"
        ):
            key = future_to_key[future]
            try:
                future.result()  # This will raise any exceptions
            except Exception as exc:
                print(f"Array {key} generated an exception: {exc}")
                raise


def write_nested_dist_data_threaded(
    group,
    dist_data: dict[int, dict[str, np.ndarray]],
    chunk_size: int,
    shard_size: int,
    max_workers: int = 24,
) -> None:
    """Write nested distribution data (dict of dicts) using threading for I/O parallelism.
    
    For each distribution, concatenates all arrays into one long array and stores:
    - Sorted keys in .attrs
    - Split positions in .attrs to reconstruct individual arrays
    - Single concatenated array as zarr array
    
    Parameters
    ----------
    group
        Zarr group to write to
    dist_data
        Nested distribution data: {dist_id: {col_name: array}}
    chunk_size
        Chunk size for arrays
    shard_size
        Shard size for arrays
    max_workers
        Number of threads for parallel writing
    """
    # Prepare concatenated arrays for each distribution
    concatenated_data = {}
    
    for dist_id, col_dict in dist_data.items():
        # Sort keys for consistent ordering
        sorted_keys = sorted(col_dict.keys())
        
        # Concatenate arrays and track positions
        arrays_to_concat = [col_dict[key] for key in sorted_keys]
        concatenated = np.concatenate(arrays_to_concat, axis=0)
        
        # Calculate split positions
        positions = [0]
        for arr in arrays_to_concat:
            positions.append(positions[-1] + len(arr))
        
        # Store the concatenated array
        concatenated_data[dist_id] = concatenated
        
        # Create subgroup and store metadata
        dist_group = group.create_group(str(dist_id))
        dist_group.attrs['keys'] = sorted_keys
        dist_group.attrs['positions'] = positions
    
    # Write all concatenated arrays in parallel using existing function
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for dist_id, concatenated in concatenated_data.items():
            dist_group = group[str(dist_id)]
            future = executor.submit(
                write_single_array,
                dist_group,
                "data",
                concatenated,
                chunk_size,
                shard_size,
            )
            futures[future] = dist_id
        
        # Wait for all writes to complete
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Writing {group.name}"
        ):
            try:
                future.result()
            except Exception as exc:
                dist_id = futures[future]
                print(f"Array write for dist {dist_id} generated an exception: {exc}")
                raise

def write_sharded(
    group: zarr.Group,
    data: dict[str, Any],
    name: str,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    compressors: Iterable[BytesBytesCodec] = (
        BloscCodec(
            cname="lz4",
            clevel=3,
        ),
    ),
):
    """Function to write data to a zarr group in a sharded format.

    Parameters
    ----------
    group
        The zarr group to write to.
    data
        The data to write.
    chunk_size
        The chunk size.
    shard_size
        The shard size.
    """
    # TODO: this is a copy of the function in arrayloaders
    # when it is no longer public we should use the function from arrayloaders
    # https://github.com/laminlabs/arrayloaders/blob/main/arrayloaders/io/store_creation.py
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr

    def callback(
        func: ad.experimental.Write,
        g: zarr.Group,
        k: str,
        elem: ad.typing.RWAble,
        dataset_kwargs: Mapping[str, Any],
        iospec: ad.experimental.IOSpec,
    ):
        if iospec.encoding_type in {"array"}:
            # Calculate greatest common divisor for first dimension
            # or use smallest dimension as chunk size

            shape = elem.shape if hasattr(elem, "shape") else (len(elem),)
            chunk_size_used, shard_size_used = _get_size(shape, chunk_size, shard_size)

            dataset_kwargs = {
                "shards": (shard_size_used,) + (shape[1:]),  # only shard over 1st dim
                "chunks": (chunk_size_used,) + (shape[1:]),  # only chunk over 1st dim
                "compressors": compressors,
                **dataset_kwargs,
            }
        elif iospec.encoding_type in {"csr_matrix", "csc_matrix"}:
            dataset_kwargs = {
                "shards": (shard_size,),
                "chunks": (chunk_size,),
                "compressors": compressors,
                **dataset_kwargs,
            }

        func(g, k, elem, dataset_kwargs=dataset_kwargs)

    ad.experimental.write_dispatched(group, name, data, callback=callback)


def _to_list(x: list[Any] | tuple[Any] | Any) -> list[Any] | tuple[Any]:
    """Converts x to a list if it is not already a list or tuple."""
    if isinstance(x, (list | tuple)):
        return x
    return [x]


def _flatten_list(x: Iterable[Iterable[Any]]) -> list[Any]:
    """Flattens a list of lists."""
    return [item for sublist in x for item in sublist]
