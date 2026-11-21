package linearAlgebra.logic;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;

/**
 * Basic linear algebra operations on a {@link Matrix}.
 */
public final class MatrixLogic {

    private final Matrix A;

    /**
     * Creates logical operations to be performed on a {@link Matrix}.
     *
     * @param A the {@link Matrix} to operate on
     */
    public MatrixLogic(
        final Matrix A
    ) {
        this.A = A;

        checkMatrixLogic();
    }

    /**
     * Multiplies the {@link Matrix} by another on its right.
     *
     * @param other the right-hand factor, with as many rows as the {@link Matrix} has columns
     * @return the product, as a new {@link Matrix}
     * @throws MatrixMultiplyDimMismatchException if the dimensions do not allow the product
     */
    public Matrix multiply(
        final Matrix other
    )
        throws MatrixMultiplyDimMismatchException
    {
        checkMatrixLogic();
        Preconditions.checkNotNull(
            other,
            "MatrixLogic.multiply: other matrix must not be null"
        );

        if (A.cols() != other.rows()) {
            throw new MatrixMultiplyDimMismatchException();
        }

        final double[][] left = A.data();
        final double[][] right = other.data();
        final double[][] resultData = new double[left.length][right[0].length];

        for (int i = 0; i < left.length; i++) {
            for (int j = 0; j < right[0].length; j++) {
                double sum = 0;
                for (int k = 0; k < right.length; k++) {
                    sum += left[i][k] * right[k][j];
                }
                Preconditions.checkState(
                    Double.isFinite(sum),
                    "MatrixLogic.multiply: product entry must be finite: row %s, column %s",
                    i,
                    j
                );
                resultData[i][j] = sum;
            }
        }

        return new Matrix(resultData);
    }

    // Checks that the matrix being worked on exists.
    private void checkMatrixLogic() {
        Preconditions.checkNotNull(
            A,
            "MatrixLogic: matrix must not be null"
        );
    }
}
