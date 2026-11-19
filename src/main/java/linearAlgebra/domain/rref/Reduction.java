package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;

/**
 * The result of reducing a {@link Matrix} to reduced row echelon form ({@code RREF}).
 * This class stores an immutable copy of the data and does not allow any modifications.
 *
 * @param reduced    the reduced {@link Matrix}
 * @param operations the {@link RowOperation}s that produced {@code reduced}, in order
 */
public record Reduction(
    Matrix reduced,
    RowOperations operations
) {

    /**
     * Stores the resulting {@link Matrix} and {@link RowOperations} of reducing some {@link Matrix}.
     *
     * @param reduced    the reduced {@link Matrix}
     * @param operations the {@link RowOperations} that produced {@code reduced}, in order
     */
    public Reduction {
        checkReduction(reduced, operations);
    }

    /**
     * @return the reduced {@link Matrix}
     */
    @Override
    public Matrix reduced() {
        return reduced;
    }

    /**
     * @return the {@link RowOperations} in order
     */
    @Override
    public RowOperations operations() {
        return operations;
    }

    /**
     * @return a new reduction with the same reduced matrix and operations
     */
    public Reduction copy() {
        return new Reduction(
            reduced.copy(),
            operations.copy()
        );
    }

    // Checks that the reduced matrix and the operations that produced it are not null.
    private static void checkReduction(
        final Matrix reduced,
        final RowOperations operations
    ) {
        Preconditions.checkNotNull(
            reduced,
            "Reduction: reduced matrix must not be null"
        );
        Preconditions.checkNotNull(
            operations,
            "Reduction: operations must not be null"
        );
    }
}
