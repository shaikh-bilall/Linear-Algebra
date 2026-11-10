package linearAlgebra.domain.RREF;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;
import linearAlgebra.logic.MatrixLogic;
import linearAlgebra.logic.VectorLogic;

/**
 * The elementary operation that swaps two rows.
 *
 * @param sourceRow the zero-based index of one of the rows
 * @param targetRow the zero-based index of the other row
 */
public record Type1(int sourceRow, int targetRow) implements OperationRREF {

    /**
     * Swaps the two rows of a matrix, in place.
     *
     * @param A the matrix to modify
     * @throws NullPointerException     if {@code A} is {@code null}
     * @throws IllegalArgumentException if either row index is out of range
     */
    @Override
    public void apply(Matrix A) {
        MatrixLogic.swapRows(A, sourceRow, targetRow);
    }
    /**
     * Swaps the two entries of a vector at the same indices, in place.
     *
     * @param v the vector to modify
     * @throws NullPointerException     if {@code v} is {@code null}
     * @throws IllegalArgumentException if either index is out of range
     */
    @Override
    public void apply(Vector v) {
        VectorLogic.swapElements(v, sourceRow, targetRow);
    }
}
