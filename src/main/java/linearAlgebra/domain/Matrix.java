package linearAlgebra.domain;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.exceptions.MatrixMultiplyDimMismatchException;

import java.util.Arrays;
import java.util.List;

/**
 * A rectangular matrix of finite real numbers, with structural operations such as transpose, merge and row reordering.
 *
 * @param data the entries, indexed as {@code data[row][column]}; {@code data()} returns a copy of this array
 */
public record Matrix(double[][] data) {

    /**
     * Creates a matrix from the given entries.
     *
     * @param data the entries, indexed as {@code data[row][column]}
     */
    public Matrix(
        final double[][] data
    ) {
        this.data = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            this.data[i] = data[i].clone();
        }

        checkMatrix();
    }

    /**
     * Returns the entries.
     *
     * @return a deep copy of the entries; changing it does not affect this matrix
     */
    @Override
    public double[][] data() {
        checkMatrix();

        double[][] copy = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            copy[i] = data[i].clone();
        }
        return copy;
    }
    /**
     * Returns the identity matrix of the given dimension.
     *
     * @param dimension the number of rows and columns, which must be positive
     * @return the identity matrix
     */
    public static Matrix identity(
        final int dimension
    ) {
        Preconditions.checkArgument(
            dimension > 0,
            "Matrix.identity: dimension must be positive: %s",
            dimension
        );

        double[][] identityData = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            identityData[i][i] = 1;
        }
        return new Matrix(identityData);
    }
    /**
     * Returns the largest absolute value among the entries.
     *
     * @return the largest absolute value of any entry, never negative
     */
    public double maxAbs() {
        checkMatrix();

        double maxAbs = 0;
        for (double[] row : data) {
            for (double entry : row) {
                maxAbs = Math.max(maxAbs, Math.abs(entry));
            }
        }
        return maxAbs;
    }
    /**
     * Returns the transpose, with rows and columns swapped.
     *
     * @return a new matrix; this one is unchanged
     */
    public Matrix transpose() {
        checkMatrix();

        final int rows = data.length;
        final int cols = data[0].length;
        double[][] transposedData = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposedData[j][i] = data[i][j];
            }
        }
        return new Matrix(transposedData);
    }
    /**
     * Multiplies this matrix by a vector, giving one entry per row.
     *
     * @param vector the vector to multiply, whose length must equal the number of columns
     * @return the product of this matrix and {@code vector}
     * @throws MatrixMultiplyDimMismatchException if the length of {@code vector} differs from the number of columns
     */
    public Vector multiply(
        final Vector vector
    )
        throws MatrixMultiplyDimMismatchException
    {
        checkMatrix();
        Preconditions.checkNotNull(
            vector,
            "Matrix.multiply: vector must not be null"
        );
        if (data[0].length != vector.length()) {
            throw new MatrixMultiplyDimMismatchException();
        }

        double[] resultData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            double sum = 0;
            for (int j = 0; j < data[0].length; j++) {
                sum += data[i][j] * vector.get(j);
            }
            Preconditions.checkState(
                Double.isFinite(sum),
                "Matrix.multiply: product entry must be finite: row %s",
                i
            );
            resultData[i] = sum;
        }
        return new Vector(resultData);
    }
    /**
     * Places another matrix to the right of this one, as when forming an augmented matrix {@code [A | b]}.
     *
     * @param other the matrix whose columns are appended, with the same number of rows as this one
     * @return a new matrix with the columns of both; neither operand is changed
     */
    public Matrix merge(
        final Matrix other
    ) {
        checkMatrix();
        Preconditions.checkNotNull(
            other,
            "Matrix.merge: other matrix must not be null"
        );
        Preconditions.checkArgument(
            data.length == other.rows(),
            "Matrix.merge: matrices must have the same number of rows: %s != %s",
            data.length,
            other.rows()
        );

        final int cols = data[0].length;
        final int otherCols = other.cols();
        double[][] mergedData = new double[data.length][cols + otherCols];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, mergedData[i], 0, cols);
            System.arraycopy(other.data[i], 0, mergedData[i], cols, otherCols);
        }
        return new Matrix(mergedData);
    }
    /**
     * Reorders the rows: row {@code i} of the result is row {@code shuffledIndices.get(i)} of this matrix.
     *
     * @param shuffledIndices a permutation of the row indices, each of {@code 0} to {@code rows() - 1} exactly once
     * @return a new matrix with the rows reordered
     */
    public Matrix permuteRows(
        final List<Integer> shuffledIndices
    ) {
        checkMatrix();
        Permutations.checkPermutation(
            shuffledIndices,
            data.length,
            "Matrix.permuteRows"
        );

        double[][] shuffledData = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            shuffledData[i] = data[shuffledIndices.get(i)];
        }
        return new Matrix(shuffledData);
    }
    /**
     * Returns one row as a vector.
     *
     * @param rowIndex the zero-based index of the row
     * @return a copy of the row; changing it does not affect this matrix
     */
    public Vector getRow(
        final int rowIndex
    ) {
        checkMatrix();
        Preconditions.checkArgument(
            rowIndex >= 0 &&
                rowIndex < data.length,
            "Matrix.getRow: row index out of bounds: %s",
            rowIndex
        );

        return new Vector(data[rowIndex]);
    }
    /**
     * Returns a single entry.
     *
     * @param rowIndex the zero-based row index
     * @param colIndex the zero-based column index
     * @return the entry at that position
     */
    public double get(
        final int rowIndex,
        final int colIndex
    ) {
        checkMatrix();
        Preconditions.checkArgument(
            rowIndex >= 0 &&
                rowIndex < data.length,
            "Matrix.get: row index out of bounds: %s",
            rowIndex
        );
        Preconditions.checkArgument(
            colIndex >= 0 &&
                colIndex < data[0].length,
            "Matrix.get: column index out of bounds: %s",
            colIndex
        );

        return data[rowIndex][colIndex];
    }
    /**
     * Returns the number of rows.
     *
     * @return the row count, always at least 1
     */
    public int rows() {
        checkMatrix();

        return data.length;
    }
    /**
     * Returns the number of columns.
     *
     * @return the column count, always at least 1
     */
    public int cols() {
        checkMatrix();

        return data[0].length;
    }
    /**
     * Returns an independent copy of this matrix.
     *
     * @return a new matrix with the same entries
     */
    public Matrix copy() {
        checkMatrix();

        return new Matrix(data);
    }
    /**
     * Compares this matrix to another object for equality, entry by entry. Two matrices are equal when they have the
     * same dimensions and every pair of matching entries is numerically equal, so {@code 0.0} and {@code -0.0} count
     * as the same entry.
     *
     * @param otherObject the object to compare with
     * @return {@code true} if {@code otherObject} is a matrix with the same dimensions and entries
     */
    @Override
    public boolean equals(
        final Object otherObject
    ) {
        checkMatrix();

        boolean equals = true;
        if (!(otherObject instanceof Matrix otherMatrix)) {
            equals = false;
        } else {
            otherMatrix.checkMatrix();
            if (
                data.length != otherMatrix.data.length ||
                data[0].length != otherMatrix.data[0].length
            ) {
                equals = false;
            }
            int i = 0;
            while (equals && i < data.length) {
                if (data[i].length != otherMatrix.data[i].length) {
                    equals = false;
                } else {
                    equals = Vector.entriesEqual(data[i], otherMatrix.data[i]);
                }
                i++;
            }
        }
        return equals;
    }
    /**
     * Returns a hash code that agrees with {@link #equals}: matrices that are equal have the same hash code.
     *
     * @return the hash code of the entries
     */
    @Override
    public int hashCode() {
        checkMatrix();

        int result = 1;
        for (double[] row : data) {
            result = 31 * result + Vector.entriesHashCode(row);
        }
        return result;
    }

    // Checks the cheap structural invariants: at least one row and column, no null rows, rectangular shape.
    private void checkMatrix() {
        Preconditions.checkNotNull(
            data,
            "Matrix: data must not be null"
        );
        Preconditions.checkState(
            data.length > 0,
            "Matrix: matrix must have at least one row"
        );
        Preconditions.checkNotNull(
            data[0],
            "Matrix: row must not be null"
        );
        Preconditions.checkState(
            data[0].length > 0,
            "Matrix: matrix must have at least one column"
        );
        for (double[] row : data) {
            Preconditions.checkNotNull(
                row,
                "Matrix: row must not be null"
            );
            Preconditions.checkState(
                row.length == data[0].length,
                "Matrix: all rows must have the same number of columns"
            );
            Preconditions.checkState(
                Arrays.stream(row).allMatch(Double::isFinite),
                "Matrix: all entries must be finite numbers"
            );
        }
    }

}
