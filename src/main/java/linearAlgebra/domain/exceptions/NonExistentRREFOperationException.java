package linearAlgebra.domain.exceptions;

import linearAlgebra.domain.rref.RREFOperation;

/**
 * Thrown when a {@link RREFOperation} that the reduction algorithm does not support is used.
 */
public final class NonExistentRREFOperationException extends IllegalArgumentException {
}
