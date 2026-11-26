package linearAlgebra.logic;

import com.google.common.base.Preconditions;
import linearAlgebra.domain.Vector;
import linearAlgebra.logic.exceptions.MatrixMergeDimMismatchException;
import linearAlgebra.logic.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;

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
     * @throws MatrixMergeDimMismatchException if the matrices have different numbers of rows
     */
    public static Matrix merge(
        final Matrix left,
        final Matrix right
    )
        throws MatrixMergeDimMismatchException
    {
        Preconditions.checkNotNull(
            left,
            "MatrixLogic.merge: left matrix must not be null"
        );
        Preconditions.checkNotNull(
            right,
            "MatrixLogic.merge: right matrix must not be null"
        );
        if (left.rows() != right.rows()) {
            throw new MatrixMergeDimMismatchException();
        }

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
}
