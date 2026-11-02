package ca.umanitoba.cs.shaikhb2.domain;

import com.google.common.base.Preconditions;

/**
 * The factorization {@code A = P D P^-1} of a diagonalizable matrix {@code A}. All three matrices are square and the
 * same size, and {@code D} is diagonal.
 *
 * @param P        the eigenvectors of {@code A}, one per column
 * @param D        the eigenvalues matching the columns of {@code P}, on the diagonal
 * @param inverseP the inverse of {@code P}
 */
public record Diagonalization(Matrix P, Matrix D, Matrix inverseP) {

    public Diagonalization(
            final Matrix P,
            final Matrix D,
            final Matrix inverseP
    ) {
        Preconditions.checkNotNull(P, "P must not be null");
        Preconditions.checkNotNull(D, "D must not be null");
        Preconditions.checkNotNull(inverseP, "Inverse of P must not be null");
        Preconditions.checkState(P.rows() == P.cols(), "P must be square");
        Preconditions.checkState(D.rows() == P.rows() && D.cols() == P.cols(), "D must have the same dimensions as P");
        Preconditions.checkState(inverseP.rows() == P.rows() && inverseP.cols() == P.cols(), "Inverse of P must have the same dimensions as P");
        checkDiagonal(D);

        this.P = P.copy();
        this.D = D.copy();
        this.inverseP = inverseP.copy();
    }

    // Checks that the matrix is diagonal, i.e. all off-diagonal elements are zero.
    private static void checkDiagonal(
            final Matrix D
    ) {
        for (int i = 0; i < D.rows(); i++) {
            for (int j = 0; j < D.cols(); j++) {
                if (i != j) {
                    Preconditions.checkState(D.get(i, j) == 0, "D must be diagonal");
                }
            }
        }
    }
}