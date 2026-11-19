package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * The elementary operation that multiplies a row by a nonzero {@code scalar}.
 * This class stores an immutable copy of the data and does not allow any modifications.
 *
 * @param targetRow the zero-based index of the row to scale
 * @param scalar    the nonzero, finite {@code scalar}
 */
public record RowScale(
    int targetRow,
    double scalar
) implements RowOperation {

    /**
     * Stores the scaling {@link RowOperation}.
     *
     * @param targetRow the zero-based index of the row to scale
     * @param scalar    the nonzero, finite {@code scalar}
     */
    public RowScale {
        checkRowScale(targetRow, scalar);
    }

    /**
     * @return the zero-based index of the row to scale
     */
    @Override
    public int targetRow() {
        return targetRow;
    }

    /**
     * @return the nonzero, finite {@code scalar}
     */
    @Override
    public double scalar() {
        return scalar;
    }

    /**
     * @return a new scaling {@link RowOperation} with the same index and scalar
     */
    public RowScale copy() {
        return new RowScale(targetRow, scalar);
    }

    /**
     * Multiplies the entire {@code targetRow} of a {@link Matrix} by the {@code scalar}.
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
            "RowScale.apply: matrix must not be null"
        );

        return RowReducer.apply(this, A);
    }

    /**
     * Multiplies the {@code targetRow} entry of a {@link Vector} by the {@code scalar}.
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
            "RowScale.apply: vector must not be null"
        );

        return RowReducer.apply(this, v);
    }

    // Checks that the index is non-negative and the scalar is nonzero and finite.
    private static void checkRowScale(
        final int targetRow,
        final double scalar
    ) {
        Preconditions.checkArgument(
            targetRow >= 0,
            "RowScale: target row index must be non-negative"
        );
        Preconditions.checkArgument(
            Double.isFinite(scalar),
            "RowScale: scalar must be finite"
        );
        Preconditions.checkArgument(
            scalar != 0,
            "RowScale: scalar must be non-zero"
        );
    }
}
