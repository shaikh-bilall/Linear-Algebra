package ca.umanitoba.cs.shaikhb2.domain;

import com.google.common.base.Preconditions;

import java.util.Arrays;
import java.util.List;

/**
 * A rectangular matrix of finite real numbers, with structural operations such as transpose, merge and row reordering.
 *
 * @param data the entries, indexed as {@code data[row][column]}; {@code data()} returns this array itself, not a copy
 */
public record Matrix(double[][] data) {

    public Matrix(
            final double[][] data
    ) {
        Preconditions.checkNotNull(data, "Matrix data must not be null");
        this.data = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            Preconditions.checkNotNull(data[i], "Matrix row must not be null");
            this.data[i] = data[i].clone();
        }
        checkMatrix();
    }

    /**
     * Returns the transpose, with rows and columns swapped.
     *
     * @return a new matrix; this one is unchanged
     * @throws IllegalStateException if this matrix was corrupted through {@code data()}
     */
    public Matrix transpose() {
        double[][] transposedData = new double[cols()][rows()];
        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                transposedData[j][i] = data()[i][j];
            }
        }
        Matrix transposed = new Matrix(transposedData);
        Preconditions.checkState(transposed.rows() == cols() && transposed.cols() == rows(), "Transpose must swap the dimensions");
        return transposed;
    }
    /**
     * Multiplies this matrix by a vector, giving one entry per row.
     *
     * @param vector the vector to multiply, whose length must equal the number of columns
     * @return the product of this matrix and {@code vector}
     * @throws NullPointerException if {@code vector} is {@code null}
     * @throws IllegalStateException if the dimensions do not match, the product overflows, or this matrix was
     *                               corrupted through {@code data()}
     */
    public Vector multiply(
            final Vector vector
    ) {
        Preconditions.checkNotNull(vector, "Vector must not be null");
        Preconditions.checkState(cols() == vector.length(), "Incompatible matrix and vector dimensions for multiplication");
        double[] resultData = new double[rows()];
        for (int i = 0; i < rows(); i++) {
            double sum = 0;
            for (int j = 0; j < cols(); j++) {
                sum += data[i][j] * vector.get(j);
            }
            resultData[i] = sum;
        }
        Vector product = new Vector(resultData);
        Preconditions.checkState(product.length() == rows(), "Product must have one entry per row");
        return product;
    }
    /**
     * Places another matrix to the right of this one, as when forming an augmented matrix {@code [A | b]}.
     *
     * @param other the matrix whose columns are appended, with the same number of rows as this one
     * @return a new matrix with the columns of both; neither operand is changed
     * @throws NullPointerException if {@code other} is {@code null}
     * @throws IllegalStateException if the row counts differ, or either matrix was corrupted through {@code data()}
     */
    public Matrix merge(
            final Matrix other
    ) {
        Preconditions.checkNotNull(other, "Other matrix must not be null");
        Preconditions.checkState(this.rows() == other.rows(), "Matrices must have the same number of rows to be merged");

        double[][] mergedData = new double[this.rows()][this.cols() + other.cols()];
        for (int i = 0; i < this.rows(); i++) {
            System.arraycopy(this.data[i], 0, mergedData[i], 0, this.cols());
            System.arraycopy(other.data[i], 0, mergedData[i], this.cols(), other.cols());
        }
        Matrix merged = new Matrix(mergedData);
        Preconditions.checkState(merged.cols() == this.cols() + other.cols(), "Merged matrix must have the combined number of columns");
        return merged;
    }
    /**
     * Reorders the rows: row {@code i} of the result is row {@code shuffledIndices.get(i)} of this matrix.
     *
     * @param shuffledIndices a permutation of the row indices, each of {@code 0} to {@code rows() - 1} exactly once
     * @return a new matrix with the rows reordered
     * @throws NullPointerException if {@code shuffledIndices} or any of its elements is {@code null}
     * @throws IllegalStateException if the list is not a permutation of the row indices, or this matrix was
     *                               corrupted through {@code data()}
     */
    public Matrix permuteRows(
            final List<Integer> shuffledIndices
    ) {
        Preconditions.checkNotNull(shuffledIndices, "Shuffled indices list must not be null");
        Preconditions.checkState(shuffledIndices.size() == rows(), "Shuffled indices list size must match number of rows in the matrix");
        checkPermutation(shuffledIndices);

        double[][] shuffledData = new double[rows()][cols()];
        for (int i = 0; i < rows(); i++) {
            shuffledData[i] = data[shuffledIndices.get(i)].clone();
        }

        Matrix shuffled = new Matrix(shuffledData);
        Preconditions.checkState(shuffled.rows() == rows() && shuffled.cols() == cols(), "Permuting the rows must keep the dimensions");
        return shuffled;
    }
    /**
     * Returns one row as a vector.
     *
     * @param rowIndex the zero-based index of the row
     * @return a copy of the row; changing it does not affect this matrix
     * @throws IllegalStateException if {@code rowIndex} is out of range, or this matrix was corrupted through
     *                               {@code data()}
     */
    public Vector getRow(
            final int rowIndex
    ) {
        Preconditions.checkState(rowIndex >= 0 && rowIndex < rows(), "Row index out of bounds");

        Vector row = new Vector(data[rowIndex].clone());
        Preconditions.checkState(row.length() == cols(), "Row must have one entry per column");
        return row;
    }
    /**
     * Returns a single entry.
     *
     * @param rowIndex the zero-based row index
     * @param colIndex the zero-based column index
     * @return the entry at that position
     * @throws IllegalStateException if either index is out of range, or this matrix was corrupted through
     *                               {@code data()}
     */
    public double get(
            final int rowIndex,
            final int colIndex
    ) {
        Preconditions.checkState(rowIndex >= 0 && rowIndex < rows(), "Row index out of bounds");
        Preconditions.checkState(colIndex >= 0 && colIndex < cols(), "Column index out of bounds");

        double value = data[rowIndex][colIndex];
        Preconditions.checkState(Double.isFinite(value), "Entry must be finite");
        return value;
    }
    /**
     * Returns the number of rows.
     *
     * @return the row count, always at least 1
     * @throws IllegalStateException if this matrix was corrupted through {@code data()}
     */
    public int rows() {
        Preconditions.checkState(data.length > 0, "Matrix must have at least one row");
        return data.length;
    }
    /**
     * Returns the number of columns.
     *
     * @return the column count, always at least 1
     * @throws IllegalStateException if this matrix was corrupted through {@code data()}
     */
    public int cols() {
        Preconditions.checkState(data[0] != null && data[0].length > 0, "Matrix must have at least one column");
        return data[0].length;
    }
    /**
     * Returns an independent copy, which can be changed in place without affecting this matrix.
     *
     * @return a new matrix with the same entries
     * @throws IllegalStateException if this matrix was corrupted through {@code data()}
     */
    public Matrix copy() {
        double[][] clonedData = new double[rows()][cols()];
        for (int i = 0; i < rows(); i++) {
            System.arraycopy(data[i], 0, clonedData[i], 0, cols());
        }
        Matrix copy = new Matrix(clonedData);
        Preconditions.checkState(copy.rows() == rows() && copy.cols() == cols(), "Copy must have the same dimensions");
        return copy;
    }

    /**
     * Compares this matrix to another object for equality, entry by entry. Two matrices are equal when they have the
     * same dimensions and every pair of matching entries is numerically equal, so {@code 0.0} and {@code -0.0} count
     * as the same entry.
     *
     * @param otherObject the object to compare with
     * @return {@code true} if {@code otherObject} is a matrix with the same dimensions and entries
     * @throws IllegalStateException if this matrix, or the other one, was corrupted through {@code data()}
     */
    @Override
    public boolean equals(
            final Object otherObject
    ) {
        checkMatrix();
        if (this == otherObject) {
            return true;
        }
        if (!(otherObject instanceof Matrix otherMatrix)) {
            return false;
        }
        otherMatrix.checkMatrix();
        if (data.length != otherMatrix.data.length) {
            return false;
        }
        for (int i = 0; i < data.length; i++) {
            if (!Vector.entriesEqual(data[i], otherMatrix.data[i])) {
                return false;
            }
        }
        return true;
    }
    /**
     * Returns a hash code that agrees with {@link #equals}: matrices that are equal have the same hash code.
     *
     * @return the hash code of the entries
     * @throws IllegalStateException if this matrix was corrupted through {@code data()}
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

    // Checks that the matrix is well-formed: non-null, rectangular, and with finite entries.
    private void checkMatrix() {
        Preconditions.checkNotNull(data, "Matrix data must not be null");
        Preconditions.checkState(data.length > 0, "Matrix must have at least one row");
        Preconditions.checkState(data[0].length > 0, "Matrix must have at least one column");
        for (var row : data) {
            Preconditions.checkNotNull(row, "Matrix row must not be null");
            Preconditions.checkState(row.length == data[0].length, "All rows must have the same number of columns");
            Preconditions.checkState(Arrays.stream(row).allMatch(Double::isFinite), "All entries must be finite numbers");
        }
    }
    // The list must hold every row index from 0 to rows - 1 exactly once (its size is checked before this runs).
    private void checkPermutation(
            final List<Integer> indices
    ) {
        boolean[] seen = new boolean[rows()];
        for (Integer index : indices) {
            Preconditions.checkNotNull(index, "Shuffled indices must not contain null");
            Preconditions.checkState(index >= 0 && index < seen.length, "Shuffled index out of bounds: %s", index);
            Preconditions.checkState(!seen[index], "Shuffled indices must not repeat: %s", index);
            seen[index] = true;
        }
    }

}