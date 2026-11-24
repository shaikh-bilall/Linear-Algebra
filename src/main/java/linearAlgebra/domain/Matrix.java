package linearAlgebra.domain;

import com.google.common.base.Preconditions;

import java.util.Arrays;

/**
 * A rectangular matrix of finite real numbers.
 * This class stores an immutable copy of the data and does not allow any modifications.
 *
 * @param data entries of the matrix
 */
public record Matrix(
    double[][] data
) {

    /**
     * Stores a matrix for the given data.
     *
     * @param data entries of the matrix
     */
    public Matrix(
        final double[][] data
    ) {
        checkMatrix(data);

        this.data = deepCopy(data);
    }

    /**
     * @return the entries of this matrix
     */
    @Override
    public double[][] data() {
        return deepCopy(this.data);
    }

    /**
     * Returns the row, at the specified index, as a {@link Vector}.
     *
     * @param rowIndex the zero-based index of the row
     * @return the row at the given index
     */
    public Vector getRow(
        final int rowIndex
    ) {
        Preconditions.checkElementIndex(
            rowIndex,
            rows(),
            "Matrix.getRow: row index"
        );

        return new Vector(data[rowIndex].clone());
    }

    /**
     * Returns the entry at the specified row and column indices.
     *
     * @param rowIndex the zero-based row index
     * @param colIndex the zero-based column index
     * @return the entry at the given indices
     */
    public double get(
        final int rowIndex,
        final int colIndex
    ) {
        Preconditions.checkElementIndex(
            rowIndex,
            rows(),
            "Matrix.get: row index"
        );
        Preconditions.checkElementIndex(
            colIndex,
            cols(),
            "Matrix.get: col index"
        );

        return data[rowIndex][colIndex];
    }

    /**
     * @return the number of rows.
     */
    public int rows() {
        return data.length;
    }

    /**
     * @return the number of columns.
     */
    public int cols() {
        return data[0].length;
    }

    /**
     * @return a new matrix with the same entries
     */
    public Matrix copy() {
        return new Matrix(deepCopy(this.data));
    }

    /**
     * Compares this matrix to another object for equality.
     *
     * @param otherObject the object to compare with
     * @return {@code true} if {@code otherObject} is a matrix with the same dimensions and entries
     */
    @Override
    public boolean equals(
        final Object otherObject
    ) {
        boolean equals = true;
        if (!(otherObject instanceof Matrix otherMatrix)) {
            equals = false;
        } else {
            if (
                rows() != otherMatrix.rows() ||
                cols() != otherMatrix.cols()
            ) {
                equals = false;
            }
            int i = 0;
            while (
                equals &&
                i < rows()
            ) {
                equals = Vector.entriesEqual(
                    data[i],
                    otherMatrix.data[i]
                );
                i++;
            }
        }
        return equals;
    }

    /**
     * @return the hash code that agrees with {@link #equals}
     */
    @Override
    public int hashCode() {
        int result = 1;
        for (double[] row : data) {
            result = 31 * result + Vector.entriesHashCode(row);
        }
        return result;
    }

    /**
     * Returns the identity matrix of the given dimension.
     *
     * @param dimension the number of rows and columns
     * @return the identity matrix
     */
    public static Matrix identity(
        final int dimension
    ) {
        Preconditions.checkArgument(
            dimension > 0,
            "Matrix.identity: dimension must be positive"
        );

        double[][] identityData = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            identityData[i][i] = 1;
        }
        return new Matrix(identityData);
    }

    // Checks that there are at least one row and column, no null rows, rectangular shape.
    private static void checkMatrix(
        final double[][] data
    ) {
        Preconditions.checkNotNull(
            data,
            "Matrix: data must not be null"
        );
        Preconditions.checkArgument(
            data.length > 0,
            "Matrix: matrix must have at least one row"
        );
        Preconditions.checkNotNull(
            data[0],
            "Matrix: row must not be null"
        );
        Preconditions.checkArgument(
            data[0].length > 0,
            "Matrix: matrix must have at least one column"
        );
        for (var row : data) {
            Preconditions.checkNotNull(
                row,
                "Matrix: row must not be null"
            );
            Preconditions.checkArgument(
                row.length == data[0].length,
                "Matrix: all rows must have the same number of columns"
            );
            Preconditions.checkArgument(
                Arrays.stream(row).allMatch(Double::isFinite),
                "Matrix: all entries must be finite numbers"
            );
        }
    }

    // Returns a deep copy of this matrix's data
    private static double[][] deepCopy(
        final double[][] data
    ) {
        double[][] copy = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            copy[i] = data[i].clone();
        }
        return copy;
    }
}
