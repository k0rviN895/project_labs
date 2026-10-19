import numpy as np
from grader_contracts.numpy_tasks import (BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput, TimeSeriesInput,)
from numpy_tasks import (sum_prod, binarize, unique_rows, unique_columns, matrix_statistics, chess, draw_rectangle, draw_ellipse, analyze_time_series, one_hot,)


def test_sum_prod():
    A = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    x = np.array([[[1], [2]], [[3], [4]]])
    result = sum_prod(MatrixVectorBatchInput(matrices=A, vectors=x))
    expected = np.array([[1 + 6], [2 + 8]])  
    assert np.allclose(result, expected)


def test_binarize():
    M = np.array([[1.0, 5.0], [3.0, 2.0]])
    out = binarize(BinarizeInput(matrix=M, threshold=2.5))
    assert np.array_equal(out, np.array([[0, 1], [1, 0]]))
    assert M[0, 1] == 5.0  


def test_unique_rows_columns():
    M = np.array([[1, 2, 2], [3, 3, 4]])
    rows = unique_rows(MatrixInput(matrix=M))
    cols = unique_columns(MatrixInput(matrix=M))
    assert [list(r) for r in rows] == [[1, 2], [3, 4]]
    assert [list(c) for c in cols] == [[1, 3], [2, 3], [2, 4]]


def test_matrix_statistics():
    stats = matrix_statistics(RandomMatrixInput(
        rows=3, columns=4, mean=0.0, std=1.0, seed=42))
    assert stats.matrix.shape == (3, 4)
    assert stats.row_means.shape == (3,)
    assert stats.column_means.shape == (4,)
    assert stats.row_variances.shape == (3,)
    assert stats.column_variances.shape == (4,)


def test_chess():
    out = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = np.array([[0, 1, 0, 1],
                         [1, 0, 1, 0],
                         [0, 1, 0, 1]])
    assert np.array_equal(out, expected)


def test_draw_rectangle():
    out = draw_rectangle(RectangleInput(
        width=4, height=2, image_height=10, image_width=10,
        shape_color=(255, 0, 0), background_color=(0, 0, 0)))
    assert out.shape == (10, 10, 3)
    # центр должен быть красным
    assert tuple(out[5, 5]) == (255, 0, 0)
    # угол — фон
    assert tuple(out[0, 0]) == (0, 0, 0)


def test_draw_ellipse():
    out = draw_ellipse(EllipseInput(
        semi_axis_x=3, semi_axis_y=3, image_height=11, image_width=11,
        shape_color=(0, 255, 0), background_color=(0, 0, 0)))
    assert out.shape == (11, 11, 3)
    assert tuple(out[5, 5]) == (0, 255, 0)   
    assert tuple(out[0, 0]) == (0, 0, 0)     


def test_analyze_time_series():
    series = TimeSeriesInput(values=[1, 3, 2, 5, 4], window=2)
    stats = analyze_time_series(series)
    assert stats.mean == 3.0
    assert stats.local_maxima == [1, 3]
    assert stats.local_minima == [2]
    assert np.allclose(stats.moving_average, [2.0, 2.5, 3.5, 4.5])


def test_one_hot():
    out = one_hot(OneHotInput(labels=[0, 2, 3, 0], class_count=None))
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
    assert np.array_equal(out, expected)

    out2 = one_hot(OneHotInput(labels=[1, 0], class_count=5))
    assert out2.shape == (2, 5)