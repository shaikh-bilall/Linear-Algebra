package linearAlgebra.domain.rref;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * An elementary row operation that can be applied consistently to a {@link Matrix} or a {@link Vector}.
 */
public interface RowOperation {

    /**
     * Applies the operation to the rows of a {@link Matrix}.
     *
     * @param A the {@link Matrix} to operate on
     * @return a {@link Matrix} with the operation applied; {@code A} is unchanged
     */
    Matrix apply(
        final Matrix A
    );

    /**
     * Applies the operation to the entries of a {@link Vector}.
     *
     * @param v the {@link Vector} to operate on
     * @return a {@link Vector} with the operation applied; {@code v} is unchanged
     */
    Vector apply(
        final Vector v
    );
}
