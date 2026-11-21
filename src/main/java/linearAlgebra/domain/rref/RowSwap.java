package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

/**
 * The elementary operation that swaps two rows.
 * This class stores an immutable copy of the data and does not allow any modifications.
 *
 * @param sourceRow the zero-based index of one of the rows
 * @param targetRow the zero-based index of the other row; it may equal {@code sourceRow}
 */
public record RowSwap(
    int sourceRow,
    int targetRow
) implements RowOperation {

    /**
     * Stores the swap {@link RowOperation}.
     *
     * @param sourceRow the zero-based index of one of the rows
     * @param targetRow the zero-based index of the other row; it may equal {@code sourceRow}
     */
    public RowSwap {
        checkRowSwap(sourceRow, targetRow);
    }

    /**
     * Applying the swap {@link RowOperation} negates the determinant.
     *
     * @param determinant The determinant before applying this operation.
     * @return the determinant after applying this operation.
     */
    public double updatedDeterminant(
        final double determinant
    ) {
        return -determinant;
    }

    /**
     * @return the zero-based index of the first row to swap
     */
    public int sourceRow() {
        return sourceRow;
    }

    /**
     * @return the zero-based index of the second row to swap
     */
    public int targetRow() {
        return targetRow;
    }

    /**
     * @return a new swap {@link RowOperation} with the same indices
     */
    public RowSwap copy() {
        return new RowSwap(sourceRow, targetRow);
    }

    // Checks that both indices are non-negative.
    private static void checkRowSwap(
        final int sourceRow,
        final int targetRow
    ) {
        Preconditions.checkArgument(
            sourceRow >= 0,
            "RowSwap: source row index must be non-negative"
        );
        Preconditions.checkArgument(
            targetRow >= 0,
            "RowSwap: target row index must be non-negative"
        );
    }
}
