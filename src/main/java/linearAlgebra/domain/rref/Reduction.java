package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;
import linearAlgebra.domain.Matrix;

import java.util.List;

/**
 * The result of reducing a matrix to reduced row echelon form.
 *
 * @param reduced    the reduced matrix
 * @param operations the operations that transform the original matrix into {@code reduced}, in order
 */
public record Reduction(Matrix reduced, List<RREFOperation> operations) {

    /**
     * Creates the result of reducing a matrix.
     *
     * @param reduced    the reduced matrix
     * @param operations the operations that transform the original matrix into {@code reduced}, in order
     */
    public Reduction {
        Preconditions.checkNotNull(
            reduced,
            "Reduction: reduced matrix must not be null"
        );
        operations = List.copyOf(operations);
    }
}
