package linearAlgebra.domain.rref;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * An elementary row operation. It acts on the rows of a {@link Matrix} and, in the same way, on the entries of a
 * {@link Vector}, so a sequence of operations can transform a {@link Matrix} and a {@link Vector} consistently.
 */
public interface RREFOperation {
    /**
     * Applies the operation to the rows of a {@link Matrix}. The {@link Matrix} itself is unchanged.
     *
     * @param A the {@link Matrix} to operate on
     * @return a new {@link Matrix} with the operation applied
     */
    Matrix apply(Matrix A);
    /**
     * Applies the operation to the entries of a {@link Vector}. The {@link Vector} itself is unchanged.
     *
     * @param v the {@link Vector} to operate on
     * @return a new {@link Vector} with the operation applied
     */
    Vector apply(Vector v);
}
