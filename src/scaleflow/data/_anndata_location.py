from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["AnnDataLocation"]


class AnnDataLocation:
    """An object that stores a sequence of access operations on an AnnData object."""

    def __init__(self, path=None):
        # The path is a list of tuples, e.g., [('getattr', 'obsm'), ('getitem', 's')]
        self._path = path if path is not None else []

    @classmethod
    def from_path(cls, path: list[list[str]]) -> "AnnDataLocation":
        """Create an AnnDataLocation from a serialized path.

        Parameters
        ----------
        path
            A list of lists, where each inner list is [op_type, op_arg].
            For example: [["getattr", "obsm"], ["getitem", "X_pca"]]

        Returns
        -------
        AnnDataLocation
            A new AnnDataLocation instance with the given path.
        """
        return cls(path=[tuple(op) for op in path])

    @classmethod
    def from_json(cls, json_str: str) -> "AnnDataLocation":
        """Create an AnnDataLocation from a JSON string.

        Parameters
        ----------
        json_str
            A JSON string representation of the path.

        Returns
        -------
        AnnDataLocation
            A new AnnDataLocation instance with the given path.
        """
        import json

        path = json.loads(json_str)
        return cls.from_path(path)

    def to_path(self) -> list[list[str]]:
        """Serialize the path to a list of lists for storage.

        Returns
        -------
        list[list[str]]
            The path as a list of lists, e.g. [["getattr", "obsm"], ["getitem", "X_pca"]]
        """
        return [list(op) for op in self._path]

    def to_json(self) -> str:
        """Serialize the path to a JSON string for storage.

        Returns
        -------
        str
            A JSON string representation of the path.
        """
        import json

        return json.dumps(self.to_path())

    def __getattr__(self, name):
        """Handles attribute access, like .obs or .X."""
        if name.startswith("__") and name.endswith("__"):
            # Avoid interfering with special methods
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        new_path = self._path + [("getattr", name)]
        return AnnDataLocation(new_path)

    @staticmethod
    def _serialize_key(key):
        """Convert a key to a JSON-serializable format."""
        if isinstance(key, slice):
            return {"__slice__": [key.start, key.stop, key.step]}
        return key

    @staticmethod
    def _deserialize_key(key):
        """Convert a serialized key back to its original form."""
        if isinstance(key, dict) and "__slice__" in key:
            start, stop, step = key["__slice__"]
            return slice(start, stop, step)
        return key

    def __getitem__(self, key):
        """Handles item access, like ['my_key'] or slices like [0:10]."""
        serializable_key = self._serialize_key(key)
        new_path = self._path + [("getitem", serializable_key)]
        return AnnDataLocation(new_path)

    def __call__(self, adata: "AnnData"):
        """Executes the stored path of operations on the provided AnnData object."""
        target = adata
        try:
            for op_type, op_arg in self._path:
                if op_type == "getattr":
                    target = getattr(target, op_arg)
                elif op_type == "getitem":
                    key = self._deserialize_key(op_arg)
                    target = target[key]
            return target
        except (AttributeError, KeyError) as e:
            raise type(e)(f"Failed to resolve location {self!r} on the AnnData object. Reason: {e}") from e

    def __repr__(self):
        """Provides a user-friendly string representation of the stored path."""
        representation = "AnnDataAccessor()"
        for op_type, op_arg in self._path:
            if op_type == "getattr":
                representation += f".{op_arg}"
            elif op_type == "getitem":
                # Use repr() to correctly handle string keys with quotes
                representation += f"[{repr(op_arg)}]"
        return f"<AnnDataLocation: {representation}>"
