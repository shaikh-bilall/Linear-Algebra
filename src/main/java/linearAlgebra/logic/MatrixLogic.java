package linearAlgebra.logic;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;

/**
 * Defines basic linear algebra operations for a {@link Matrix}.
 */
public final class MatrixLogic {

    // Prevents instantiation.
    private MatrixLogic() {
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
}
