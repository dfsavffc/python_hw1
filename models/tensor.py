import math


class Tensor:
    """
    Base class representing a multidimensional tensor.

    Attributes:
        dimension (int | tuple): The shape of the tensor.
        data (list): The values stored in the tensor.

    Raises:
        TypeError: If `dimension` is not an int or tuple, or if `data` is not a list.
        ValueError: If `dimension` does not match the length of `data`.
    """

    def __init__(self, dimension, data):
        if not isinstance(dimension, (int, tuple)):
            raise TypeError(f"invalid type: '{type(dimension).__name__}'")

        if not isinstance(data, list):
            raise TypeError(f"invalid type: '{type(data).__name__}'")

        if not all(isinstance(d, int) and d > 0 for d in dimension):
            raise ValueError("invalid dimension")

        if isinstance(dimension, int) and dimension != len(data):
            raise ValueError("invalid dimension")

        if isinstance(dimension, tuple) and math.prod(dimension) != len(data):
            raise ValueError("invalid dimension")

        self.dimension = dimension
        self._data = data

    def __str__(self):
        """
        Return a string representation of the tensor data.
        """
        return str(self._data)



