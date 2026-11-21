package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

/**
 * The elementary operation that adds a {@code scalar} multiple of one row to another row.
 * This class stores an immutable copy of the data and does not allow any modifications.
 *
 * @param sourceRow the zero-based index of the row whose {@code scalar} multiple is added;
 *                  it is left unchanged
 * @param targetRow the zero-based index of the row that receives the {@code scalar} multiple
 * @param scalar    the nonzero, finite {@code scalar} multiple
 */
public record RowCombine(
    int sourceRow,
    int targetRow,
    double scalar
) implements RowOperation {

    /**
     * Stores the linear combination {@link RowOperation}.
     *
     * @param sourceRow the zero-based index of the row whose {@code scalar} multiple is added
     * @param targetRow the zero-based index of the row that receives the {@code scalar} multiple;
     *                  it must differ from {@code sourceRow}
     * @param scalar    the nonzero, finite {@code scalar} multiple
     */
    public RowCombine {
        checkRowCombine(sourceRow, targetRow, scalar);
    }

    /**
     * Applying the linear combination {@link RowOperation} does not modify the determinant.
     *
     * @param determinant The determinant before applying this operation.
     * @return the determinant after applying this operation.
     */
    public double updatedDeterminant(
        final double determinant
    ) {
        return determinant;
    }

    /**
     * @return the zero-based index of the row whose {@code scalar} multiple is added
     */
    public int sourceRow() {
        return sourceRow;
    }

    /**
     * @return the zero-based index of the row that receives the {@code scalar} multiple
     */
    public int targetRow() {
        return targetRow;
    }

    /**
     * @return the nonzero, finite {@code scalar} multiple
     */
    public double scalar() {
        return scalar;
    }

    /**
     * @return a new linear combination {@link RowOperation} with the same indices and scalar
     */
    public RowCombine copy() {
        return new RowCombine(sourceRow, targetRow, scalar);
    }

    // Checks that the indices are non-negative and different, and the scalar is nonzero and finite.
    private static void checkRowCombine(
        final int sourceRow,
        final int targetRow,
        final double scalar
    ) {
        Preconditions.checkArgument(
            sourceRow >= 0,
            "RowCombine: source row index must be non-negative"
        );
        Preconditions.checkArgument(
            targetRow >= 0,
            "RowCombine: target row index must be non-negative"
        );
        Preconditions.checkArgument(
            sourceRow != targetRow,
            "RowCombine: source and target rows must differ"
        );
        Preconditions.checkArgument(
            Double.isFinite(scalar),
            "RowCombine: scalar must be finite"
        );
        Preconditions.checkArgument(
            scalar != 0,
            "RowCombine: scalar must be non-zero"
        );
    }
}
