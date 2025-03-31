import unittest
from models.matrix import Matrix


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.data = [i for i in range(100)]
        self.matrix = Matrix((10, 10), self.data)

    def test_single_element(self):
        self.assertEqual(self.matrix[1, 1], 11)

    def test_single_row(self):
        expected = Matrix((1, 10), self.data[10:20])
        self.assertEqual(str(self.matrix[1]), str(expected))

    def test_negative_row(self):
        expected = Matrix((1, 10), self.data[90:100])
        self.assertEqual(str(self.matrix[-1]), str(expected))

    def test_row_slice(self):
        expected = Matrix((3, 10), self.data[10:40])
        self.assertEqual(str(self.matrix[1:4]), str(expected))

    def test_row_slice_start(self):
        expected = Matrix((4, 10), self.data[:40])
        self.assertEqual(str(self.matrix[:4]), str(expected))

    def test_row_slice_end(self):
        expected = Matrix((6, 10), self.data[40:])
        self.assertEqual(str(self.matrix[4:]), str(expected))

    def test_full_matrix(self):
        expected = Matrix((10, 10), self.data)
        self.assertEqual(str(self.matrix[:]), str(expected))

    def test_step_slice(self):
        expected_data = self.data[10:20] + self.data[30:40] + self.data[50:60]
        expected = Matrix((3, 10), expected_data)
        self.assertEqual(str(self.matrix[1:7:2]), str(expected))

    def test_column(self):
        expected = Matrix((10, 1), [self.data[i] for i in range(1, 100, 10)])
        self.assertEqual(str(self.matrix[:, 1]), str(expected))

    def test_submatrix(self):
        expected = Matrix((3, 3), [11, 12, 13, 21, 22, 23, 31, 32, 33])
        self.assertEqual(str(self.matrix[1:4, 1:4]), str(expected))

    def test_mixed_slices(self):
        expected = Matrix((3, 4), [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
        self.assertEqual(str(self.matrix[1:4, :4]), str(expected))

    def test_list_indexing(self):
        expected = Matrix((2, 10), self.data[10:20] + self.data[40:50])
        self.assertEqual(str(self.matrix[[1, 4]]), str(expected))

    def test_column_list_indexing(self):
        expected = Matrix((10, 2), [i + j for i in range(1, 100, 10) for j in range(0, 4, 3)])
        self.assertEqual(str(self.matrix[:, [1, 4]]), str(expected))

    def test_double_list_indexing(self):
        expected = Matrix((2, 2), [11, 14, 41, 44])
        self.assertEqual(str(self.matrix[[1, 4], [1, 4]]), str(expected))


if __name__ == "__main__":
    unittest.main()