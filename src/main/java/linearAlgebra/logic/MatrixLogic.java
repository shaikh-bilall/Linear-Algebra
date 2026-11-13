package linearAlgebra.logic;

import com.google.common.base.Preconditions;

import linearAlgebra.logic.exceptions.DeterminantDoesNotExistException;
import linearAlgebra.logic.exceptions.InverseNotAvailableException;
import linearAlgebra.domain.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.rref.RowReducer;
import linearAlgebra.domain.rref.Type1;
import linearAlgebra.domain.rref.Type2;

/**
 * Linear-algebra operations on a fixed matrix: product, reduced row echelon form, determinant and
 * inverse. The matrix is never modified.
 */
public final class MatrixLogic {

    private final Matrix A;

    /**
     * Creates operations for a fixed matrix.
     *
     * @param A the matrix to operate on
     */
    public MatrixLogic(
        final Matrix A
    ) {
        this.A = A;

        checkMatrixLogic();
    }

    /**
     * Multiplies the matrix by another on its right.
     *
     * @param other the right-hand factor, with as many rows as the matrix has columns
     * @return the product, as a new matrix
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
    /**
     * Returns the reduced row echelon form of the matrix.
     *
     * @return a matrix in reduced row echelon form
     */
    public Matrix rref() {
        checkMatrixLogic();

        return RowReducer.reduce(A).reduced();
    }
    /**
     * Returns the inverse of the matrix.
     *
     * @return the inverse, as a new matrix
     * @throws InverseNotAvailableException if the matrix is not square or is singular
     */
    public Matrix inverse()
        throws InverseNotAvailableException
    {
        checkMatrixLogic();

        if (A.rows() != A.cols()) {
            throw new InverseNotAvailableException();
        }

        final RowReducer.Reduction reduction = RowReducer.reduce(A);
        if (getDeterminant(reduction) == 0) {
            throw new InverseNotAvailableException();
        }
        return RowReducer.applyAll(reduction.operations(), Matrix.identity(A.rows()));
    }
    /**
     * Returns the determinant of the matrix.
     *
     * @return the determinant, which is 0 for a singular matrix
     * @throws DeterminantDoesNotExistException if the matrix is not square
     */
    public double determinant()
        throws DeterminantDoesNotExistException
    {
        checkMatrixLogic();

        if (A.rows() != A.cols()) {
            throw new DeterminantDoesNotExistException();
        }
        return getDeterminant(RowReducer.reduce(A));
    }

    // Checks that the matrix being worked on exists.
    private void checkMatrixLogic() {
        Preconditions.checkNotNull(
            A,
            "MatrixLogic: matrix must not be null"
        );
    }

    // Returns the determinant from the operations recorded while reducing a square matrix.
    private static double getDeterminant(
        final RowReducer.Reduction reduction
    ) {
        final Matrix reduced = reduction.reduced();
        for (int i = 0; i < reduced.rows(); i++) {
            if (Math.abs(reduced.get(i, i)) < 0.5) {
                return 0;
            }
        }

        // Type1 flips the sign, Type2 divides by its scalar, and Type3 leaves the determinant unchanged.
        double determinant = 1;
        for (var operation : reduction.operations()) {
            if (operation instanceof Type1) {
                determinant = -determinant;
            } else if (operation instanceof Type2 scale) {
                determinant /= scale.scalar();
            }
        }
        return determinant;
    }

}