package domain;

import com.google.common.base.Preconditions;

import java.util.Objects;

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

    /**
     * Compares this diagonalization to another object for equality: it is equal when {@code P}, {@code D} and the
     * inverse of {@code P} are each equal to those of the other, as defined by {@link Matrix#equals}.
     *
     * @param otherObject the object to compare with
     * @return {@code true} if {@code otherObject} is a diagonalization with equal matrices
     * @throws IllegalStateException if one of the matrices that gets compared was corrupted through {@code data()}
     */
    @Override
    public boolean equals(
            final Object otherObject
    ) {
        if (this == otherObject) {
            return true;
        }
        if (!(otherObject instanceof Diagonalization otherDiagonalization)) {
            return false;
        }
        return P.equals(otherDiagonalization.P)
                && D.equals(otherDiagonalization.D)
                && inverseP.equals(otherDiagonalization.inverseP);
    }
    /**
     * Returns a hash code that agrees with {@link #equals}: diagonalizations that are equal have the same hash code.
     *
     * @return the hash code of the three matrices
     * @throws IllegalStateException if one of the matrices was corrupted through {@code data()}
     */
    @Override
    public int hashCode() {
        return Objects.hash(P, D, inverseP);
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