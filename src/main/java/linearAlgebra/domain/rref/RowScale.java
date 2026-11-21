package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

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
     * Applying the scaling {@link RowOperation} divides the determinant by the scalar.
     *
     * @param determinant The determinant before applying this operation.
     * @return the determinant after applying this operation.
     */
    public double updatedDeterminant(
        final double determinant
    ) {
        return determinant/scalar;
    }

    /**
     * @return the zero-based index of the row to scale
     */
    public int targetRow() {
        return targetRow;
    }

    /**
     * @return the nonzero, finite {@code scalar}
     */
    public double scalar() {
        return scalar;
    }

    /**
     * @return a new scaling {@link RowOperation} with the same index and scalar
     */
    public RowScale copy() {
        return new RowScale(targetRow, scalar);
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
