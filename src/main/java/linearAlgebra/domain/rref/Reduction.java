package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;

import java.util.List;

/**
 * The result of reducing a {@link Matrix} to reduced row echelon form ({@code RREF}).
 *
 * @param reduced    the reduced {@link Matrix}
 * @param operations the {@link RowOperation}s that produced {@code reduced}, in order
 */
public record Reduction(
    Matrix reduced,
    RowOperations operations
) {

    /**
     * Creates the result of reducing a {@link Matrix}.
     *
     * @param reduced    the reduced {@link Matrix}
     * @param operations the {@link RowOperations} that produced {@code reduced}, in order
     */
    public Reduction {
        checkReduction(reduced, operations);
    }

    // Checks that the reduced matrix and the operations that produced it are not null.
    private static void checkReduction(
        Matrix reduced,
        RowOperations operations
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
