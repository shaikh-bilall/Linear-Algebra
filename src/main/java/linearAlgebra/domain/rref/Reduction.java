package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;
import linearAlgebra.domain.Matrix;

import java.util.List;

/**
 * The result after performing {@link RREFOperation}s on  a {@link Matrix}.
 *
 * @param reduced    the reduced row echelon form {@link Matrix}
 * @param operations the {@link RREFOperation}s that turn the original {@link Matrix}
 *                  into {@code reduced}
 */
public record Reduction(Matrix reduced, List<RREFOperation> operations) {

    /**
     * Creates a {@code Reduction} object from its reduced {@link Matrix} and
     * the {@link RREFOperation}s that produced it.
     *
     * @param reduced    the reduced row echelon form {@link Matrix}
     * @param operations the {@link RREFOperation}s that turn the original {@link Matrix}
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