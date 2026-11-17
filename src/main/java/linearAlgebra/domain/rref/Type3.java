package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * The elementary operation that adds a {@code scalar} multiple of one row to another row.
 *
 * @param sourceRow the zero-based index of the row whose {@code scalar} multiple is added;
 *                  it is left unchanged
 * @param targetRow the zero-based index of the row that receives the {@code scalar} multiple
 * @param scalar    the nonzero, finite {@code scalar} multiple
 */
public record Type3(
    int sourceRow,
    int targetRow,
    double scalar
) implements RREFOperation {

    /**
     * Creates the operation.
     *
     * @param sourceRow the zero-based index of the row whose {@code scalar} multiple is added
     * @param targetRow the zero-based index of the row that receives the {@code scalar} multiple;
     *                  it must differ from {@code sourceRow}
     * @param scalar    the nonzero, finite {@code scalar} multiple
     */
    public Type3(
        final int sourceRow,
        final int targetRow,
        final double scalar
    ) {
        this.sourceRow = sourceRow;
        this.targetRow = targetRow;
        this.scalar = scalar;

        checkType3();
    }

    /**
     * Returns the index of the row whose {@code scalar} multiple is added.
     *
     * @return the zero-based row index
     */
    @Override
    public int sourceRow() {
        checkType3();

        return sourceRow;
    }

    /**
     * Returns the index of the row that receives the {@code scalar} multiple.
     *
     * @return the zero-based row index
     */
    @Override
    public int targetRow() {
        checkType3();

        return targetRow;
    }

    /**
     * Returns the {@code scalar} multiple.
     *
     * @return the nonzero, finite {@code scalar} multiple
     */
    @Override
    public double scalar() {
        checkType3();

        return scalar;
    }

    /**
     * Adds the {@code scalar} multiple of the source row to the target row of a {@link Matrix}.
     *
     * @param A the {@link Matrix} to operate on
     * @return a new {@link Matrix} with the operation applied; {@code A} is unchanged
     */
    @Override
    public Matrix apply(
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            A,
            "Type3.apply: matrix must not be null"
        );
        checkType3();

        return RowReducer.apply(this, A);
    }

    /**
     * Adds the {@code scalar} multiple of the source entry to the target entry of a {@link Vector}.
     *
     * @param v the {@link Vector} to operate on
     * @return a new {@link Vector} with the operation applied; {@code v} is unchanged
     */
    @Override
    public Vector apply(
        final Vector v
    ) {
        Preconditions.checkNotNull(
            v,
            "Type3.apply: vector must not be null"
        );
        checkType3();

        return RowReducer.apply(this, v);
    }

    // Checks that the indices are non-negative and different, and the scalar is nonzero and finite.
    private void checkType3() {
        Preconditions.checkState(
            sourceRow >= 0,
            "Type3: source row index must be non-negative: %s",
            sourceRow
        );
        Preconditions.checkState(
            targetRow >= 0,
            "Type3: target row index must be non-negative: %s",
            targetRow
        );
        Preconditions.checkState(
            sourceRow != targetRow,
            "Type3: source and target rows must differ: %s",
            sourceRow
        );
        Preconditions.checkState(
            Double.isFinite(scalar),
            "Type3: scalar must be finite"
        );
        Preconditions.checkState(
            scalar != 0,
            "Type3: scalar must be non-zero"
        );
    }
}
