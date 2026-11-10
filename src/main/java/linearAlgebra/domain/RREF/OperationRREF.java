package linearAlgebra.domain.RREF;

import linearAlgebra.domain.Vector;
import linearAlgebra.domain.Matrix;

/**
 * An elementary row operation. It acts on the rows of a matrix and, in the same way, on the entries of a vector, so a
 * sequence of operations can transform a matrix and a vector consistently. Implementations check their indices and
 * scalar when applied, not when created.
 */
public interface OperationRREF {
    /**
     * Applies the operation to the rows of a matrix, in place.
     *
     * @param A the matrix to modify
     * @throws NullPointerException     if {@code A} is {@code null}
     * @throws IllegalArgumentException if the operation does not fit {@code A}, for instance a row index out of range
     */
    void apply(Matrix A);
    /**
     * Applies the operation to the entries of a vector, in place.
     *
     * @param v the vector to modify
     * @throws NullPointerException     if {@code v} is {@code null}
     * @throws IllegalArgumentException if the operation does not fit {@code v}, for instance an index out of range
     */
    void apply(Vector v);
}
