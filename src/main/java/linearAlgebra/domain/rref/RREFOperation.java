package linearAlgebra.domain.rref;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * An elementary row operation that can be applied consistently to a matrix or vector.
 */
public interface RREFOperation {
    /**
     * Applies the operation to the rows of a {@link Matrix}. The {@link Matrix} itself is unchanged.
     *
     * @param A the {@link Matrix} to operate on
     * @return the resulting matrix
     */
    Matrix apply(Matrix A);
    /**
     * Applies the operation to the entries of a {@link Vector}. The {@link Vector} itself is unchanged.
     *
     * @param v the {@link Vector} to operate on
     * @return the resulting vector
     */
    Vector apply(Vector v);
}
