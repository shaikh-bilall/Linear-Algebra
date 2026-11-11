package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

/**
 * The elementary operation that multiplies a row by a nonzero scalar.
 *
 * @param targetRow the zero-based index of the row to scale
 * @param scalar    the factor, which must be nonzero and finite
 */
public record Type2(int targetRow, double scalar) implements OperationRREF {

    /**
     * Creates the operation.
     *
     * @param targetRow the zero-based index of the row to scale
     * @param scalar    the factor, which must be nonzero and finite
     */
    public Type2(
        final int targetRow,
        final double scalar
    ) {
        Preconditions.checkArgument(
            targetRow >= 0,
            "Type2: target row index must be non-negative: %s",
            targetRow
        );
        Preconditions.checkArgument(
            scalar != 0,
            "Type2: scalar must be non-zero"
        );
        Preconditions.checkArgument(
            Double.isFinite(scalar),
            "Type2: scalar must be finite"
        );
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
     * Returns the factor.
     *
     * @return the nonzero, finite scalar
     */
    @Override
    public double scalar() {
        checkType2();
        return scalar;
    }
    /**
     * Multiplies the row of a matrix by the scalar.
     *
     * @param A the matrix to operate on
     * @return a new matrix with the row scaled; {@code A} is unchanged
     */
    @Override
    public Matrix apply(
        final Matrix A
    ) {
        checkType2();
        return RowReducer.apply(this, A);
    }
    /**
     * Multiplies the entry of a vector at the same index by the scalar.
     *
     * @param v the vector to operate on
     * @return a new vector with the entry scaled; {@code v} is unchanged
     */
    @Override
    public Vector apply(
        final Vector v
    ) {
        checkType2();
        return RowReducer.apply(this, v);
    }

    // Checks that the index is non-negative and the scalar is nonzero and finite.
    private void checkType2() {
        Preconditions.checkState(
            targetRow >= 0,
            "Type2: target row index must be non-negative"
        );
        Preconditions.checkState(
            scalar != 0 &&
                Double.isFinite(scalar),
            "Type2: scalar must be non-zero and finite"
        );
    }
}