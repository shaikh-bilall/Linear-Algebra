package linearAlgebra.domain.exceptions;

/**
 * Thrown when a matrix is defective: an eigenvalue repeats more often than it has independent
 * eigenvectors, so the matrix cannot be diagonalized.
 */
public final class DefectiveMatrixException extends Exception {
}
