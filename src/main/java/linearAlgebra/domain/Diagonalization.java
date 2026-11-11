package linearAlgebra.domain;

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
        Preconditions.checkNotNull(
            P,
            "P must not be null"
        );
        Preconditions.checkNotNull(
            D,
            "D must not be null"
        );
        Preconditions.checkNotNull(
            inverseP,
            "Inverse of P must not be null"
        );
        Preconditions.checkState(
            P.rows() == P.cols(),
            "P must be square"
        );
        Preconditions.checkState(
            D.rows() == P.rows() &&
                D.cols() == P.cols(),
            "D must have the same dimensions as P"
        );
        Preconditions.checkState(
            inverseP.rows() == P.rows() &&
                inverseP.cols() == P.cols(),
            "Inverse of P must have the same dimensions as P"
        );

        this.P = P;
        this.D = D;
        this.inverseP = inverseP;
        checkDiagonalization();
    }

    /**
     * Returns the eigenvectors, one per column.
     *
     * @return the matrix {@code P}
     */
    @Override
    public Matrix P() {
        checkDiagonalization();
        return P;
    }
    /**
     * Returns the eigenvalues on the diagonal.
     *
     * @return the matrix {@code D}
     */
    @Override
    public Matrix D() {
        checkDiagonalization();
        return D;
    }
    /**
     * Returns the inverse of the eigenvector matrix.
     *
     * @return the inverse of {@code P}
     */
    @Override
    public Matrix inverseP() {
        checkDiagonalization();
        return inverseP;
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
        checkDiagonalization();
        if (this == otherObject) {
            return true;
        }
        if (!(otherObject instanceof Diagonalization(Matrix p, Matrix d, Matrix inverseP1))) {
            return false;
        }
        return (
            P.equals(p) &&
                D.equals(d) &&
                inverseP.equals(inverseP1)
        );
    }
    /**
     * Returns a hash code that agrees with {@link #equals}.
     *
     * @return the hash code of the three matrices
     * @throws IllegalStateException if one of the matrices was corrupted through {@code data()}
     */
    @Override
    public int hashCode() {
        checkDiagonalization();
        return Objects.hash(P, D, inverseP);
    }

    // Checks the structural invariants: all three matrices exist, are square and the same size, and D is diagonal.
    private void checkDiagonalization() {
        Preconditions.checkState(
            P != null &&
                D != null &&
                inverseP != null,
            "P, D and the inverse of P must not be null"
        );
        Preconditions.checkState(
            P.rows() == P.cols(),
            "P must be square"
        );
        Preconditions.checkState(
            D.rows() == P.rows() &&
                D.cols() == P.cols(),
            "D must have the same dimensions as P"
        );
        Preconditions.checkState(
            inverseP.rows() == P.rows() &&
                inverseP.cols() == P.cols(),
            "Inverse of P must have the same dimensions as P"
        );
        for (int i = 0; i < D.rows(); i++) {
            for (int j = 0; j < D.cols(); j++) {
                if (i != j) {
                    Preconditions.checkState(
                        D.get(i, j) == 0,
                        "D must be diagonal"
                    );
                }
            }
        }
    }
}