package linearAlgebra.logic;

import com.google.common.base.Preconditions;
import linearAlgebra.domain.Permutations;
import linearAlgebra.domain.Vector;
import linearAlgebra.logic.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;

import java.util.List;

/**
 * Defines basic linear algebra operations for a {@link Matrix}.
 */
public final class MatrixLogic {

    // Prevents instantiation.
    private MatrixLogic() {
    }

    /**
     * Finds the largest absolute value among the entries of a {@link Matrix}.
     *
     * @param matrix the {@link Matrix} to find the largest absolute value among its entries
     * @return the largest absolute value among the entries
     */
    public static double maxAbs(
        final Matrix matrix
    ) {
        Preconditions.checkNotNull(
            matrix,
            "MatrixLogic.maxAbs: matrix must not be null"
        );

        final double[][] data = matrix.data();

        double maxAbs = 0;
        for (double[] row : data) {
            for (double entry : row) {
                maxAbs = Math.max(maxAbs, Math.abs(entry));
            }
        }
        return maxAbs;
    }

    /**
     * Finds the transpose of a {@link Matrix}.
     *
     * @param matrix the {@link Matrix} to transpose
     * @return the transpose {@link Matrix}, with rows and columns flipped
     */
    public static Matrix transpose(
        final Matrix matrix
    ) {
        Preconditions.checkNotNull(
            matrix,
            "MatrixLogic.transpose: matrix must not be null"
        );

        final int rows = matrix.rows();
        final int cols = matrix.cols();

        final double[][] data = matrix.data();
        final double[][] transposedData = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposedData[j][i] = data[i][j];
            }
        }
        return new Matrix(transposedData);
    }

    /**
     * Multiplies the left {@link Matrix} by another on its right.
     *
     * @param left the left-hand factor, with as many columns as the other {@link Matrix}'s rows
     * @param right the right-hand factor, with as many rows as the other {@link Matrix}'s columns
     * @return the product, as a new {@link Matrix}
     * @throws MatrixMultiplyDimMismatchException if the dimensions do not allow the product
     */
    public static Matrix multiply(
        final Matrix left,
        final Matrix right
    )
        throws MatrixMultiplyDimMismatchException
    {
        Preconditions.checkNotNull(
            left,
            "MatrixLogic.multiply: left matrix must not be null"
        );
        Preconditions.checkNotNull(
            right,
            "MatrixLogic.multiply: right matrix must not be null"
        );
        if (left.cols() != right.rows()) {
            throw new MatrixMultiplyDimMismatchException();
        }

        final double[][] leftData = left.data();
        final double[][] rightData = right.data();

        final double[][] resultData = new double[leftData.length][rightData[0].length];

        for (int i = 0; i < leftData.length; i++) {
            for (int j = 0; j < rightData[0].length; j++) {
                double sum = 0;
                for (int k = 0; k < rightData.length; k++) {
                    sum += leftData[i][k] * rightData[k][j];
                }

                Preconditions.checkState(
                    Double.isFinite(sum),
                    "MatrixLogic.multiply: product entry must be finite"
                );

                resultData[i][j] = sum;
            }
        }

        return new Matrix(resultData);
    }

    /**
     * Multiplies a {@link Matrix} on the left, by a {@link Vector} on the right.
     *
     * @param matrix the {@link Matrix} to multiply
     * @param vector      the {@link Vector} to multiply, whose length must equal the number of columns
     * @return the product of this {@link Matrix} and {@link Vector}
     * @throws MatrixMultiplyDimMismatchException if the length of {@link Vector} differs from the number
     * of columns
     */
    public static Vector multiply(
        final Matrix matrix,
        final Vector vector
    )
        throws MatrixMultiplyDimMismatchException
    {
        Preconditions.checkNotNull(
            matrix,
            "MatrixLogic.multiply: matrix must not be null"
        );
        Preconditions.checkNotNull(
            vector,
            "MatrixLogic.multiply: vector must not be null"
        );
        if (matrix.cols() != vector.dimension()) {
            throw new MatrixMultiplyDimMismatchException();
        }

        final double[][] matrixData = matrix.data();
        final double[] vectorData = vector.data();

        final double[] resultData = new double[matrix.rows()];

        for (int i = 0; i < matrix.rows(); i++) {
            double sum = 0;
            for (int j = 0; j < matrix.cols(); j++) {
                sum += matrixData[i][j] * vectorData[j];
            }

            Preconditions.checkState(
                Double.isFinite(sum),
                "MatrixLogic.multiply: product entry must be finite"
            );

            resultData[i] = sum;
        }

        return new Vector(resultData);
    }

    /**
     * Merges two {@link Matrix} objects column-wise,
     * as when forming an augmented {@link Matrix} {@code [left | right]}.
     *
     * @param left   the {@link Matrix} whose columns are appended to the left
     * @param right  the {@link Matrix} whose columns are appended to the right
     * @return a {@link Matrix} with the columns of both
     */
    public static Matrix merge(
        final Matrix left,
        final Matrix right
    ) {
        Preconditions.checkNotNull(
            left,
            "MatrixLogic.merge: left matrix must not be null"
        );
        Preconditions.checkNotNull(
            right,
            "MatrixLogic.merge: right matrix must not be null"
        );
        Preconditions.checkArgument( //fixme: add a new custom exception
            left.rows() == right.rows(),
            "MatrixLogic.merge: both matrices must have the same number of rows"
        );

        final int cols = left.cols();
        final int otherCols = right.cols();

        final double[][] leftData = left.data();
        final double[][] rightData = right.data();

        final double[][] mergedData = new double[left.rows()][cols + otherCols];

        for (int i = 0; i < left.rows(); i++) {
            System.arraycopy(leftData[i], 0, mergedData[i], 0, cols);
            System.arraycopy(rightData[i], 0, mergedData[i], cols, otherCols);
        }

        return new Matrix(mergedData);
    }

    /**
     * Reorders the rows of a {@link Matrix} according to the provided specification.
     *
     * @param matrix          the {@link Matrix} to reorder
     * @param shuffledIndices a permutation of the row indices
     * @return a {@link Matrix} with the reordered rows
     */
    public static Matrix permuteRows(
        final Matrix matrix,
        final List<Integer> shuffledIndices
    ) {
        Preconditions.checkNotNull(
            matrix,
            "MatrixLogic.permuteRows: matrix must not be null"
        );
        Preconditions.checkNotNull(
            shuffledIndices,
            "MatrixLogic.permuteRows: shuffledIndices must not be null"
        );
        Permutations.checkPermutation( //fixme: add a new custom exception
            shuffledIndices,
            matrix.rows()
        );

        final double[][] data = matrix.data();
        final double[][] shuffledData = new double[matrix.rows()][matrix.cols()];

        for (int i = 0; i < matrix.rows(); i++) {
            shuffledData[i] = data[shuffledIndices.get(i)].clone();
        }

        return new Matrix(shuffledData);
    }
}
