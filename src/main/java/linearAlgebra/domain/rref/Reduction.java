package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;
import linearAlgebra.domain.Matrix;

import java.util.List;

/**
 * The result of performing {@link OperationRREF}s on  a matrix.
 *
 * @param reduced    the reduced row echelon form {@link Matrix}
 * @param operations the {@link OperationRREF}s that turn the original {@link Matrix}
 *                  into {@code reduced}
 */
public record Reduction(Matrix reduced, List<OperationRREF> operations) {

    /**
     * Creates a reduction from its reduced {@link Matrix} and
     * the {@link OperationRREF}s that produced it.
     *
     * @param reduced    the reduced row echelon form {@link Matrix}
     * @param operations the {@link OperationRREF}s that turn the original {@link Matrix}
     *                   into {@code reduced}, in order
     */
    public Reduction {
        Preconditions.checkNotNull(
            reduced,
            "Reduction: reduced matrix must not be null"
        );
        operations = List.copyOf(operations);
    }
}