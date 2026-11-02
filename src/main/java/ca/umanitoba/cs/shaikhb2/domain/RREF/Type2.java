package ca.umanitoba.cs.shaikhb2.domain.RREF;

import ca.umanitoba.cs.shaikhb2.domain.Matrix;
import ca.umanitoba.cs.shaikhb2.domain.Vector;
import ca.umanitoba.cs.shaikhb2.logic.MatrixLogic;
import ca.umanitoba.cs.shaikhb2.logic.VectorLogic;

/**
 * The elementary operation that multiplies a row by a nonzero scalar.
 *
 * @param targetRow the zero-based index of the row to scale
 * @param scalar    the factor, which must be nonzero and finite
 */
public record Type2(int targetRow, double scalar) implements OperationRREF {

    /**
     * Multiplies the row of a matrix by the scalar, in place.
     *
     * @param A the matrix to modify
     * @throws NullPointerException     if {@code A} is {@code null}
     * @throws IllegalArgumentException if the row index is out of range, or the scalar is zero or not finite
     */
    @Override
    public void apply(Matrix A) {
        MatrixLogic.multiplyRow(A, targetRow, scalar);
    }
    /**
     * Multiplies the entry of a vector at the same index by the scalar, in place.
     *
     * @param v the vector to modify
     * @throws NullPointerException     if {@code v} is {@code null}
     * @throws IllegalArgumentException if the index is out of range, or the scalar is zero or not finite
     */
    @Override
    public void apply(Vector v) {
        VectorLogic.multiplyElement(v, targetRow, scalar);
    }
}
