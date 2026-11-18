package linearAlgebra.domain.exceptions;

import linearAlgebra.domain.rref.RowOperation;

/**
 * Thrown when a {@link RowOperation} that the reduction algorithm does not support is used.
 */
public final class NonExistentRowOperationException extends IllegalArgumentException {
}
