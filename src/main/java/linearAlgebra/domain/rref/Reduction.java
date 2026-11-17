package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;

import java.util.List;

/**
 * The result of reducing a {@link Matrix} to reduced row echelon form.
 *
 * @param reduced    the reduced {@link Matrix}
 * @param operations the {@link RREFOperation}s that produced {@code reduced}, in order
 */
public record Reduction(
    Matrix reduced,
    List<RREFOperation> operations
) {

    /**
     * Creates the result of reducing a {@link Matrix}.
     *
     * @param reduced    the reduced {@link Matrix}
     * @param operations the {@link RREFOperation}s that produced {@code reduced}, in order
     */
    public Reduction(
        final Matrix reduced,
        final List<RREFOperation> operations
    ) {
        checkReduction(reduced, operations);

        this.reduced = reduced;
        this.operations = List.copyOf(operations);
    }

    // Checks that the reduced matrix and the operations that produced it are not null.
    private static void checkReduction(Matrix reduced, List<RREFOperation> operations) {
        Preconditions.checkNotNull(
            reduced,
            "Reduction: reduced matrix must not be null"
        );
        Preconditions.checkNotNull(
            operations,
            "Reduction: operations must not be null"
        );
        for (var operation : operations) {
            Preconditions.checkNotNull(
                operation,
                "Reduction: operations list must not contain null elements"
            );
        }
    }
}
