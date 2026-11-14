package linearAlgebra.domain.exceptions;

import linearAlgebra.domain.rref.OperationRREF;

/**
 * Thrown when an {@link OperationRREF} that is not part of the
 * reduced row echelon form ({@code RREF}) algorithm is requested/used.
 */
public final class NonExistentRREFOperationException extends IllegalStateException{
}
