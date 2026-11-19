package linearAlgebra.domain;

import com.google.common.base.Preconditions;

/**
 * The factorization {@code A = P D P^-1} of a diagonalizable {@link Matrix} {@code A}.
 * All three matrices are square and the same size, and {@code D} is diagonal.
 * This class stores an immutable copy of the data and does not allow any modifications.
 *
 * @param P        the eigenvector {@link Matrix} {@code P} of {@code A}
 * @param D        the {@link Matrix} with eigenvalues on the diagonal {@code D}
 * @param inverseP the inverse of the eigenvector {@link Matrix} {@code P}
 */
public record Diagonalization(
    Matrix P,
    Matrix D,
    Matrix inverseP
) {

    /**
     * Stores a diagonalization for a {@link Matrix}
     * from its eigenvector {@link Matrix},
     * {@link Matrix} with eigenvalues on the diagonal and
     * inverse of the eigenvector {@link Matrix}.
     *
     * @param P        the eigenvector {@link Matrix} {@code P} of {@code A}
     * @param D        the {@link Matrix} with eigenvalues on the diagonal {@code D}
     * @param inverseP the inverse of the eigenvector {@link Matrix} {@code P}
     */
    public Diagonalization {
        checkDiagonalization(P, D, inverseP);
    }

    /**
     * @return the eigenvector {@link Matrix} {@code P} of {@code A}
     */
    @Override
    public Matrix P() {
        return P;
    }

    /**
     * @return the {@link Matrix} with eigenvalues on the diagonal {@code D}
     */
    @Override
    public Matrix D() {
        return D;
    }

    /**
     * @return the inverse of the eigenvector {@link Matrix} {@code P}
     */
    @Override
    public Matrix inverseP() {
        return inverseP;
    }

    /**
     * @return a new diagonalization with equal matrices
     */
    public Diagonalization copy() {
        return new Diagonalization(
            P.copy(),
            D.copy(),
            inverseP.copy()
        );
    }

    // Checks that all three matrices exist, are square and the same size, and D is diagonal.
    private static void checkDiagonalization(
        final Matrix P,
        final Matrix D,
        final Matrix inverseP
    ) {
        Preconditions.checkNotNull(
            P,
            "Diagonalization: P must not be null"
        );
        Preconditions.checkNotNull(
            D,
            "Diagonalization: D must not be null"
        );
        Preconditions.checkNotNull(
            inverseP,
            "Diagonalization: inverse of P must not be null"
        );
        Preconditions.checkArgument(
            P.rows() == P.cols(),
            "Diagonalization: P must be square"
        );
        Preconditions.checkArgument(
            D.rows() == P.rows() &&
            D.cols() == P.cols(),
            "Diagonalization: D must have the same dimensions as P"
        );
        Preconditions.checkArgument(
            inverseP.rows() == P.rows() &&
            inverseP.cols() == P.cols(),
            "Diagonalization: inverse of P must have the same dimensions as P"
        );
        for (int i = 0; i < D.rows(); i++) {
            for (int j = 0; j < D.cols(); j++) {
                if (i != j) {
                    Preconditions.checkArgument(
                        D.get(i, j) == 0,
                        "Diagonalization: D must be diagonal"
                    );
                }
            }
        }
    }
}
