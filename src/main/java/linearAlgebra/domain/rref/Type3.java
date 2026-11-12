package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * The elementary operation that adds a multiple of one row to another row.
 *
 * @param sourceRow the zero-based index of the row whose multiple is added; it is left unchanged
 * @param targetRow the zero-based index of the row that receives the multiple
 * @param scalar    the multiple, which must be nonzero and finite
 */
public record Type3(int sourceRow, int targetRow, double scalar) implements OperationRREF {

    /**
     * Creates the operation.
     *
     * @param sourceRow the zero-based index of the row whose multiple is added
     * @param targetRow the zero-based index of the row that receives the multiple; it must differ from
     *                  {@code sourceRow}
     * @param scalar    the multiple, which must be nonzero and finite
     */
    public Type3(
        final int sourceRow,
        final int targetRow,
        final double scalar
    ) {
        Preconditions.checkArgument(
            sourceRow >= 0,
            "Type3: source row index must be non-negative: %s",
            sourceRow
        );
        Preconditions.checkArgument(
            targetRow >= 0,
            "Type3: target row index must be non-negative: %s",
            targetRow
        );
        Preconditions.checkArgument(
            sourceRow != targetRow,
            "Type3: source and target rows must differ: %s",
            sourceRow
        );
        Preconditions.checkArgument(
            Double.isFinite(scalar),
            "Type3: scalar must be finite"
        );
        Preconditions.checkArgument(
            scalar != 0,
            "Type3: scalar must be non-zero"
        );
        this.sourceRow = sourceRow;
        this.targetRow = targetRow;
        this.scalar = scalar;
        checkType3();
    }

    /**
     * Returns the index of the row whose multiple is added.
     *
     * @return the zero-based row index
     */
    @Override
    public int sourceRow() {
        checkType3();
        return sourceRow;
    }
    /**
     * Returns the index of the row that receives the multiple.
     *
     * @return the zero-based row index
     */
    @Override
    public int targetRow() {
        checkType3();
        return targetRow;
    }
    /**
     * Returns the multiple.
     *
     * @return the nonzero, finite scalar
     */
    @Override
    public double scalar() {
        checkType3();
        return scalar;
    }
    /**
     * Adds the multiple of the source row to the target row of a matrix.
     *
     * @param A the matrix to operate on
     * @return a new matrix with the operation applied; {@code A} is unchanged
     */
    @Override
    public Matrix apply(
        final Matrix A
    ) {
        checkType3();
        return RowReducer.apply(this, A);
    }
    /**
     * Adds the multiple of the source entry to the target entry of a vector.
     *
     * @param v the vector to operate on
     * @return a new vector with the operation applied; {@code v} is unchanged
     */
    @Override
    public Vector apply(
        final Vector v
    ) {
        checkType3();
        return RowReducer.apply(this, v);
    }

    // Checks that the indices are non-negative and different, and the scalar is nonzero and finite.
    private void checkType3() {
        Preconditions.checkState(
            sourceRow >= 0 &&
                targetRow >= 0,
            "Type3: row indices must be non-negative"
        );
        Preconditions.checkState(
            sourceRow != targetRow,
            "Type3: source and target rows must differ"
        );
        Preconditions.checkState(
            scalar != 0 &&
                Double.isFinite(scalar),
            "Type3: scalar must be non-zero and finite"
        );
    }
}