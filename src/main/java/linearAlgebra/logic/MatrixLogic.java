package linearAlgebra.logic;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.exceptions.DeterminantDoesNotExistException;
import linearAlgebra.domain.exceptions.InverseNotAvailableException;
import linearAlgebra.domain.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.rref.RowReducer;
import linearAlgebra.domain.rref.Type1;
import linearAlgebra.domain.rref.Type2;

/**
 * Row-reduction based operations on a fixed matrix: RREF, determinant, inverse and product. The matrix is never
 * modified.
 */
public final class MatrixLogic {

    private final Matrix A;

    public MatrixLogic(
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            A,
            "Matrix must not be null"
        );
        this.A = A;
    }

    /**
     * Multiplies the matrix by another on its right.
     *
     * @param other the right-hand factor, with as many rows as the matrix has columns
     * @return the product, as a new matrix
     * @throws MatrixMultiplyDimMismatchException if the dimensions do not allow the product
     * @throws NullPointerException if {@code other} is {@code null}
     */
    public Matrix multiply(
        final Matrix other
    )
        throws MatrixMultiplyDimMismatchException
    {
        Preconditions.checkNotNull(other, "Other matrix must not be null");

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
        if (A.rows() != A.cols()) {
            throw new DeterminantDoesNotExistException();
        }
        return getDeterminant(RowReducer.reduce(A));
    }

    // Returns the determinant of a reduced square matrix, computed from the recorded operations.
    private static double getDeterminant(
        final RowReducer.Reduction reduction
    ) {
        // A square matrix is singular iff its RREF is not the identity (some column got no pivot).
        // In that case the first pivotless column leaves a 0 on the diagonal.
        final Matrix reduced = reduction.reduced();
        for (int i = 0; i < reduced.rows(); i++) {
            if (Math.abs(reduced.get(i, i)) < 0.5) {
                return 0;
            }
        }

        // RREF(A) = I and det(I) = 1. Type1 flips the sign, Type2 scales the determinant
        // by its scalar, Type3 leaves it unchanged. So det(A) = (-1)^swaps / product(scalars).
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