package linearAlgebra.domain.rref;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * An elementary row operation. It acts on the rows of a matrix and, in the same way, on the entries of a vector, so a
 * sequence of operations can transform a matrix and a vector consistently. An operation checks that it fits its
 * operand when applied.
 */
public interface OperationRREF {
    /**
     * Applies the operation to the rows of a matrix. The matrix itself is unchanged.
     *
     * @param A the matrix to operate on
     * @return a new matrix with the operation applied
     */
    Matrix apply(Matrix A);
    /**
     * Applies the operation to the entries of a vector. The vector itself is unchanged.
     *
     * @param v the vector to operate on
     * @return a new vector with the operation applied
     */
    Vector apply(Vector v);
}