from models.tensor import Tensor


class Matrix(Tensor):
    """
    A class representing a 2D matrix, inheriting from Tensor.

    Attributes:
        dimension (tuple): Shape of the matrix as (rows, columns).
        data (list): Values stored row-wise in the matrix.

    Raises:
        TypeError: If `dimension` is not a tuple.
        ValueError: If `dimension` has an invalid format.
    """

    def __init__(self, dimension, data):
        if not isinstance(dimension, tuple):
            raise TypeError(f"invalid type: '{type(dimension).__name__}'")
        if len(dimension) != 2:
            raise ValueError("invalid dimension")

        super().__init__(dimension, data)

    def conv_rc2i(self, r, c):
        """
        Convert (row, column) indices to a flat list index.

        Args:
            r (int): Row index.
            c (int): Column index.

        Returns:
            int: Corresponding flat list index.

        Raises:
            IndexError: If the indices are out of bounds.
        """
        m, n = self.dimension
        if not (-m <= r < m and -n <= c < n):
            raise IndexError("matrix index out of range")
        return (r % m) * n + (c % n)

    def conv_i2rc(self, i):
        """
        Convert a flat list index to (row, column) indices.

        Args:
            i (int): Flat list index.

        Returns:
            tuple: (row, column) indices.

        Raises:
            IndexError: If the index is out of bounds.
        """
        rows, cols = self.dimension
        if not (0 <= i < rows * cols):
            raise IndexError("matrix index out of range")
        return divmod(i, cols)

    def __str__(self):
        """
        Return a formatted string representation of the matrix.
        Columns are right-aligned based on the width of the largest element.
        """
        max_width = len(str(max(self._data, key=abs)))  # Determine column width
        rows, cols = self.dimension

        result = "[\n"
        for r in range(rows):
            row_values = [f"{self._data[self.conv_rc2i(r, c)]:>{max_width}}" for c in range(cols)]
            result += "  " + "  ".join(row_values) + "\n"
        result += "]"

        return result

    def _process_indexes(self, indexes, dim=1):
        """
        Convert different types of indices (int, list, slice) into a list of valid indices.

        Args:
            indexes (int, list, slice, tuple): The index or indices to process.
            dim (int): Dimension type (1 for rows, 2 for columns).

        Returns:
            list: A list of valid indices.

        Raises:
            IndexError: If an index is out of bounds.
            TypeError: If an invalid index type is provided.
        """
        m, n = self.dimension
        if isinstance(indexes, int):
            if dim == 1 and -m <= indexes < m:
                return (indexes,),
            if dim == 2 and -n <= indexes < n:
                return (indexes,),
            raise IndexError("matrix index out of range")

        if isinstance(indexes, list):
            if any(not isinstance(idx, int) for idx in indexes):
                raise TypeError("invalid list index type")
            if dim == 1 and all(-m <= idx < m for idx in indexes):
                return indexes,
            if dim == 2 and all(-n <= idx < n for idx in indexes):
                return indexes,
            raise IndexError("matrix index out of range")

        if isinstance(indexes, slice):
            if dim == 1:
                return tuple(range(*indexes.indices(m))),
            if dim == 2:
                return tuple(range(*indexes.indices(n))),

        if isinstance(indexes, tuple):
            if any(isinstance(idx, tuple) for idx in indexes) or len(indexes) != 2:
                raise TypeError("invalid tuple index type")
            d1 = self._process_indexes(indexes[0], 1)
            d2 = self._process_indexes(indexes[1], 2)
            return *d1, *d2

        raise TypeError(f"invalid index type: {type(indexes).__name__}")

    def __getitem__(self, key):
        """
        Retrieve elements or submatrices using various indexing types: int, list, slice, or tuple.
        Supports advanced indexing such as multiple row/column selection.

        Args:
            key (int, list, slice, tuple): The indexing parameter specifying the selection.

        Returns:
            Matrix or scalar: A submatrix or a single element based on the indexing.

        Raises:
            IndexError: If indices are out of range or tuple length is incorrect.
            TypeError: If an invalid index type is provided.
        """
        m, n = self.dimension
        indexes = self._process_indexes(key)

        if len(indexes) == 1:
            indexes = (*indexes, tuple(i for i in range(n)))

        if len(indexes) == 2:
            d1, d2 = indexes
            if len(d1) == len(d2) == 1:
                idx = self.conv_rc2i(d1[0], d2[0])
                return self._data[idx]
            new_data = [self._data[self.conv_rc2i(i, j)] for i in d1 for j in d2]
            return Matrix((len(d1), len(d2)), new_data)

    def __len__(self):
        return len(self._data)