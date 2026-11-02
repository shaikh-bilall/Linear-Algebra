package ca.umanitoba.cs.shaikhb2.domain.RREF;

import ca.umanitoba.cs.shaikhb2.domain.Matrix;
import ca.umanitoba.cs.shaikhb2.domain.Vector;
import ca.umanitoba.cs.shaikhb2.logic.MatrixLogic;
import ca.umanitoba.cs.shaikhb2.logic.VectorLogic;

/**
 * The elementary operation that adds a multiple of one row to another row.
 *
 * @param sourceRow the zero-based index of the row whose multiple is added; it is left unchanged
 * @param targetRow the zero-based index of the row that receives the multiple
 * @param scalar    the multiple, which must be nonzero and finite
 */
public record Type3(int sourceRow, int targetRow, double scalar) implements OperationRREF {

    /**
     * Adds the multiple of the source row to the target row of a matrix, in place.
     *
     * @param A the matrix to modify
     * @throws NullPointerException     if {@code A} is {@code null}
     * @throws IllegalArgumentException if either row index is out of range, the two rows are the same, or the scalar
     *                                  is zero or not finite
     */
    @Override
    public void apply(Matrix A) {
        MatrixLogic.reduceRow(A, sourceRow, targetRow, scalar);
    }
    /**
     * Adds the multiple of the source entry to the target entry of a vector, in place.
     *
     * @param v the vector to modify
     * @throws NullPointerException     if {@code v} is {@code null}
     * @throws IllegalArgumentException if either index is out of range, the two indices are the same, or the scalar
     *                                  is zero or not finite
     */
    @Override
    public void apply(Vector v) {
        VectorLogic.reduceElement(v, sourceRow, targetRow, scalar);
    }
}
