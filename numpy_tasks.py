import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)

def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    # matrices: (p, n, n), vectors: (p, n, 1)
    result = np.zeros_like(vectors[0])
    for A, x in zip(matrices, vectors):
        result = result + A @ x
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(np.int64)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(row)) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(col)) for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    row_vars = matrix.var(axis=1)
    col_vars = matrix.var(axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=col_means,
        row_variances=row_vars,
        column_variances=col_vars,
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    rr, cc = np.indices((rows, columns))
    return np.where((rr + cc) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    # Центр изображения
    cy = image_height // 2
    cx = image_width // 2

    top = cy - height // 2
    bottom = top + height
    left = cx - width // 2
    right = left + width

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, image_height)
    right = min(right, image_width)

    image[top:bottom, left:right] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    cy = image_height // 2
    cx = image_width // 2

    yy, xx = np.indices((image_height, image_width))

    a = max(semi_axis_x, 1e-9)
    b = max(semi_axis_y, 1e-9)
    mask = ((xx - cx) ** 2) / (a * a) + ((yy - cy) ** 2) / (b * b) <= 1.0

    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)

    mean = float(np.mean(values))
    variance = float(np.var(values))
    std = float(np.std(values))

    local_maxima = []
    local_minima = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_maxima.append(i)
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima.append(i)

    # Скользящее среднее
    if window <= 0 or window > len(values):
        moving_average = np.array([], dtype=float)
    else:
        kernel = np.ones(window) / window
        moving_average = np.convolve(values, kernel, mode="valid")

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima=local_maxima,
        local_minima=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)

    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size > 0 else 0

    result = np.zeros((labels.size, class_count), dtype=np.int64)
    for i, label in enumerate(labels):
        if 0 <= label < class_count:
            result[i, label] = 1
    return result