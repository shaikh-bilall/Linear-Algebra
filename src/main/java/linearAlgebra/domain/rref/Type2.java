package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * The elementary operation that multiplies a row by a nonzero {@code scalar}.
 *
 * @param targetRow the zero-based index of the row to scale
 * @param scalar    the nonzero, finite {@code scalar}
 */
public record Type2(
    int targetRow,
    double scalar
) implements RREFOperation {

    /**
     * Creates the operation.
     *
     * @param targetRow the zero-based index of the row to scale
     * @param scalar    the nonzero, finite {@code scalar}
     */
    public Type2(
        final int targetRow,
        final double scalar
    ) {
        this.targetRow = targetRow;
        this.scalar = scalar;

        checkType2();
    }

    /**
     * Returns the index of the row to scale.
     *
     * @return the zero-based row index
     */
    @Override
    public int targetRow() {
        checkType2();

        return targetRow;
    }

    /**
     * Returns the {@code scalar}.
     *
     * @return the nonzero, finite {@code scalar}
     */
    @Override
    public double scalar() {
        checkType2();

        return scalar;
    }

    /**
     * Multiplies the row of a {@link Matrix} by the {@code scalar}.
     *
     * @param A the {@link Matrix} to operate on
     * @return a new {@link Matrix} with the row scaled; {@code A} is unchanged
     */
    @Override
    public Matrix apply(
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            A,
            "Type2.apply: matrix must not be null"
        );
        checkType2();

        return RowReducer.apply(this, A);
    }

    /**
     * Multiplies the entry of a {@link Vector} at the same index by the {@code scalar}.
     *
     * @param v the {@link Vector} to operate on
     * @return a new {@link Vector} with the entry scaled; {@code v} is unchanged
     */
    @Override
    public Vector apply(
        final Vector v
    ) {
        Preconditions.checkNotNull(
            v,
            "Type2.apply: vector must not be null"
        );
        checkType2();

        return RowReducer.apply(this, v);
    }

    // Checks that the index is non-negative and the scalar is nonzero and finite.
    private void checkType2() {
        Preconditions.checkState(
            targetRow >= 0,
            "Type2: target row index must be non-negative: %s",
            targetRow
        );
        Preconditions.checkState(
            Double.isFinite(scalar),
            "Type2: scalar must be finite"
        );
        Preconditions.checkState(
            scalar != 0,
            "Type2: scalar must be non-zero"
        );
    }
}
