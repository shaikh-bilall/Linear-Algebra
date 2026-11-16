package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * The elementary operation that swaps two rows.
 *
 * @param sourceRow the zero-based index of one of the rows
 * @param targetRow the zero-based index of the other row; it may equal {@code sourceRow}
 */
public record Type1(
    int sourceRow,
    int targetRow
) implements RREFOperation {

    /**
     * Creates the operation.
     *
     * @param sourceRow the zero-based index of one of the rows
     * @param targetRow the zero-based index of the other row; it may equal {@code sourceRow}
     */
    public Type1(
        final int sourceRow,
        final int targetRow
    ) {
        Preconditions.checkArgument(
            sourceRow >= 0,
            "Type1: source row index must be non-negative: %s",
            sourceRow
        );
        Preconditions.checkArgument(
            targetRow >= 0,
            "Type1: target row index must be non-negative: %s",
            targetRow
        );

        this.sourceRow = sourceRow;
        this.targetRow = targetRow;

        checkType1();
    }

    /**
     * Returns the index of one of the swapped rows.
     *
     * @return the zero-based row index
     */
    @Override
    public int sourceRow() {
        checkType1();

        return sourceRow;
    }

    /**
     * Returns the index of the other swapped row.
     *
     * @return the zero-based row index
     */
    @Override
    public int targetRow() {
        checkType1();

        return targetRow;
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
            "Type1.apply: matrix must not be null"
        );
        checkType1();

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
            "Type1.apply: vector must not be null"
        );
        checkType1();

        return RowReducer.apply(this, v);
    }

    // Checks that both indices are non-negative.
    private void checkType1() {
        Preconditions.checkState(
            sourceRow >= 0 &&
            targetRow >= 0,
            "Type1: row indices must be non-negative"
        );
    }
}
