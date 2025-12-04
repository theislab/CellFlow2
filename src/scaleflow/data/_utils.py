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

    Uses a CSR-like format: each column is stored as one contiguous array across
    all distributions, with metadata storing the index pointers (indptr) for each dist_id.

    Structure on disk:
        group/
            {col_name_1}: contiguous array (all dists concatenated)
            {col_name_2}: contiguous array (all dists concatenated)
            ...
        group.attrs['dist_ids']: sorted list of distribution IDs
        group.attrs['indptr_{col_name}']: index pointers for each column

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
    if not dist_data:
        return

    # Get sorted dist_ids for consistent ordering
    sorted_dist_ids = sorted(dist_data.keys())

    # Get all column names (assume all dists have the same columns)
    first_dist = dist_data[sorted_dist_ids[0]]
    col_names = sorted(first_dist.keys())

    # For each column, concatenate all distributions and compute indptr
    concatenated_cols = {}
    indptrs = {}

    for col_name in col_names:
        arrays_to_concat = []
        indptr = [0]

        for dist_id in sorted_dist_ids:
            arr = dist_data[dist_id][col_name]
            arrays_to_concat.append(arr)
            indptr.append(indptr[-1] + len(arr))

        concatenated_cols[col_name] = np.concatenate(arrays_to_concat, axis=0)
        indptrs[col_name] = indptr

    # Store metadata
    group.attrs["dist_ids"] = sorted_dist_ids
    for col_name, indptr in indptrs.items():
        group.attrs[f"indptr_{col_name}"] = indptr

    # Write all concatenated columns in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for col_name, concatenated in concatenated_cols.items():
            future = executor.submit(
                write_single_array,
                group,
                col_name,
                concatenated,
                chunk_size,
                shard_size,
            )
            futures[future] = col_name

        # Wait for all writes to complete
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc=f"Writing {group.name}"
        ):
            try:
                future.result()
            except Exception as exc:
                col_name = futures[future]
                print(f"Array write for {col_name} generated an exception: {exc}")
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
