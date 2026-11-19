package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

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
     * @return the zero-based index of the first row to swap
     */
    @Override
    public int sourceRow() {
        return sourceRow;
    }

    /**
     * @return the zero-based index of the second row to swap
     */
    @Override
    public int targetRow() {
        return targetRow;
    }

    /**
     * @return a new swap {@link RowOperation} with the same indices
     */
    public RowSwap copy() {
        return new RowSwap(sourceRow, targetRow);
    }

    /**
     * Swaps the two rows of a {@link Matrix}.
     *
     * @param A the {@link Matrix} to operate on
     * @return a new {@link Matrix} with the rows swapped; {@code A} is unchanged
     */
    @Override
    public Matrix apply(
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            A,
            "RowSwap.apply: matrix must not be null"
        );

        return RowReducer.apply(this, A);
    }

    /**
     * Swaps the two entries of a {@link Vector} at the same indices.
     *
     * @param v the {@link Vector} to operate on
     * @return a new {@link Vector} with the entries swapped; {@code v} is unchanged
     */
    @Override
    public Vector apply(
        final Vector v
    ) {
        Preconditions.checkNotNull(
            v,
            "RowSwap.apply: vector must not be null"
        );

        return RowReducer.apply(this, v);
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
