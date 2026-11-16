package linearAlgebra.domain.rref;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * An elementary row operation that can be applied consistently to a {@link Matrix} or a {@link Vector}.
 */
public interface RREFOperation {

    /**
     * Applies the operation to the rows of a {@link Matrix} and leaves the {@link Matrix} unchanged.
     *
     * @param A the {@link Matrix} to operate on
     * @return the resulting {@link Matrix}
     */
    Matrix apply(
        final Matrix A
    );

    /**
     * Applies the operation to the entries of a {@link Vector} and leaves the {@link Vector} unchanged.
     *
     * @param v the {@link Vector} to operate on
     * @return the resulting {@link Vector}
     */
    Vector apply(
        final Vector v
    );
}
