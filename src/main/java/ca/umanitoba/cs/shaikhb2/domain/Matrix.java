package ca.umanitoba.cs.shaikhb2.domain;

import com.google.common.base.Preconditions;

import java.util.List;

public record Matrix(double[][] data) {

    public Matrix(final double[][] data) {
        this.data = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
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
        return new Matrix(transposedData);
    }
    public Vector multiply(final Vector vector) {
        Preconditions.checkNotNull(vector, "Vector must not be null");
        Preconditions.checkArgument(cols() == vector.length(), "Incompatible matrix and vector dimensions for multiplication");
        double[] resultData = new double[rows()];
        for (int i = 0; i < rows(); i++) {
            double sum = 0;
            for (int j = 0; j < cols(); j++) {
                sum += data[i][j] * vector.get(j);
            }
            resultData[i] = sum;
        }
        return new Vector(resultData);
    }
    public Matrix merge(final Matrix other) {
        Preconditions.checkNotNull(other, "Other matrix must not be null");
        Preconditions.checkArgument(this.rows() == other.rows(), "Matrices must have the same number of rows to be merged");

        double[][] mergedData = new double[this.rows()][this.cols() + other.cols()];
        for (int i = 0; i < this.rows(); i++) {
            System.arraycopy(this.data[i], 0, mergedData[i], 0, this.cols());
            System.arraycopy(other.data[i], 0, mergedData[i], this.cols(), other.cols());
        }
        return new Matrix(mergedData);
    }
    public Matrix permuteRows(final List<Integer> shuffledIndices) {
        Preconditions.checkNotNull(shuffledIndices, "Shuffled indices list must not be null");
        Preconditions.checkArgument(shuffledIndices.size() == rows(), "Shuffled indices list size must match number of rows in the matrix");

        double[][] shuffledData = new double[rows()][cols()];
        for (int i = 0; i < rows(); i++) {
            shuffledData[i] = data[shuffledIndices.get(i)].clone();
        }

        return new Matrix(shuffledData);
    }

    public Vector getRow(final int rowIndex) {
        Preconditions.checkArgument(rowIndex >= 0 && rowIndex < rows(), "Row index out of bounds");

        return new Vector(data[rowIndex].clone());
    }
    public double get(final int rowIndex, final int colIndex) {
        Preconditions.checkArgument(rowIndex >= 0 && rowIndex < rows(), "Row index out of bounds");
        Preconditions.checkArgument(colIndex >= 0 && colIndex < cols(), "Column index out of bounds");

        return data[rowIndex][colIndex];
    }

    public int rows() {
        return data.length;
    }
    public int cols() {
        return data[0].length;
    }

    public Matrix copy() {
        double[][] clonedData = new double[rows()][cols()];
        for (int i = 0; i < rows(); i++) {
            System.arraycopy(data[i], 0, clonedData[i], 0, cols());
        }
        return new Matrix(clonedData);
    }

    private void checkMatrix() {
        Preconditions.checkNotNull(data, "Matrix data must not be null");
        for (var row : data) {
            Preconditions.checkNotNull(row, "Matrix row must not be null");
            Preconditions.checkState(row.length == cols(), "All rows must have the same number of columns");

        }
        Preconditions.checkState(rows() > 0, "Matrix must have at least one row");
        Preconditions.checkState(cols() > 0, "Matrix must have at least one column");

    }

}