package ca.umanitoba.cs.shaikhb2.domain;

import com.google.common.base.Preconditions;

import java.util.Arrays;
import java.util.List;

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
    public Vector getRow(
            final int rowIndex
    ) {
        Preconditions.checkState(rowIndex >= 0 && rowIndex < rows(), "Row index out of bounds");

        Vector row = new Vector(data[rowIndex].clone());
        Preconditions.checkState(row.length() == cols(), "Row must have one entry per column");
        return row;
    }
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
    public int rows() {
        Preconditions.checkState(data.length > 0, "Matrix must have at least one row");
        return data.length;
    }
    public int cols() {
        Preconditions.checkState(data[0] != null && data[0].length > 0, "Matrix must have at least one column");
        return data[0].length;
    }
    public Matrix copy() {
        double[][] clonedData = new double[rows()][cols()];
        for (int i = 0; i < rows(); i++) {
            System.arraycopy(data[i], 0, clonedData[i], 0, cols());
        }
        Matrix copy = new Matrix(clonedData);
        Preconditions.checkState(copy.rows() == rows() && copy.cols() == cols(), "Copy must have the same dimensions");
        return copy;
    }
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