import linearAlgebra.domain.Diagonalization;
import linearAlgebra.domain.exceptions.ComplexEigenvaluesException;
import linearAlgebra.domain.exceptions.DefectiveMatrixException;
import linearAlgebra.domain.exceptions.DependentEigenvectorsException;
import linearAlgebra.domain.exceptions.DeterminantDoesNotExistException;
import linearAlgebra.domain.exceptions.InverseNotAvailableException;
import linearAlgebra.domain.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;
import linearAlgebra.logic.DiagonalizationLogic;
import linearAlgebra.logic.MatrixLogic;

import java.util.Locale;

/**
 * Runs the library on a few sample problems and prints the results to the console. It is a demonstration only: the
 * problems cover RREF, determinants and inverses, matrix products and diagonalization, including the inputs the
 * library rejects.
 */
public class Main {

    private static final String ENTRY_FORMAT = "%10.4f";
    // Entries smaller than this would print as 0.0000 or -0.0000, so they are shown as a plain zero.
    private static final double PRINT_ZERO = 0.5e-4;

    /**
     * Runs every sample problem in order.
     *
     * @param args unused
     */
    public static void main(
        final String[] args
    ) {
        solveLinearSystem();
        showInverse("2. Determinant and inverse of an invertible matrix",
            new Matrix(new double[][]{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}}));
        showInverse("3. A singular matrix has no inverse",
            new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));
        showProduct("4. Matrix product",
            new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}}),
            new Matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}}));
        showProduct("5. Matrix product with mismatched dimensions",
            new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}}),
            new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}}));
        showDiagonalization("6. Diagonalizing a matrix with distinct eigenvalues",
            new Matrix(new double[][]{{4, 1, 0}, {2, 3, 1}, {0, 1, 5}}));
        showDiagonalization("7. Diagonalizing a matrix with a repeated eigenvalue",
            new Matrix(new double[][]{{3, 1, 1}, {1, 3, 1}, {1, 1, 3}}));
        showDiagonalization("8. A rotation has complex eigenvalues, so it cannot be diagonalized over the reals",
            new Matrix(new double[][]{{0, -1}, {1, 0}}));
        showDiagonalization("9. A Jordan block is defective, so it cannot be diagonalized",
            new Matrix(new double[][]{{1, 1}, {0, 1}}));
    }

    // Solves A x = b by row reducing the augmented matrix [A | b]: the last column of the RREF is the solution.
    private static void solveLinearSystem() {
        Matrix A = new Matrix(new double[][]{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}});
        Matrix b = new Matrix(new double[][]{{8}, {-11}, {-3}});
        Matrix augmented = A.merge(b);

        printHeading("1. Solving A x = b with RREF (the solution is x = [2, 3, -1])");
        print("Augmented matrix [A | b]", augmented);
        print("RREF of [A | b]", new MatrixLogic(augmented).rref());
    }
    // Prints the determinant and, when it exists, the inverse together with a check that A * A^-1 is the identity.
    private static void showInverse(
        final String title,
        final Matrix A
    ) {
        printHeading(title);
        print("A", A);

        MatrixLogic logic = new MatrixLogic(A);
        try {
            System.out.println("Determinant: " + format(logic.determinant()));
        } catch (DeterminantDoesNotExistException e) {
            System.out.println("A has no determinant because it is not square.");
        }
        System.out.println();
        try {
            Matrix inverse = logic.inverse();
            print("A^-1", inverse);
            print("A * A^-1 (the identity)", product(A, inverse));
        } catch (InverseNotAvailableException e) {
            System.out.println("A has no inverse.");
            System.out.println();
        }
    }
    // Prints the product left * right, or a message when the dimensions do not allow it.
    private static void showProduct(
        final String title,
        final Matrix left,
        final Matrix right
    ) {
        printHeading(title);
        print("A", left);
        print("B", right);
        try {
            print("A * B", new MatrixLogic(left).multiply(right));
        } catch (MatrixMultiplyDimMismatchException e) {
            System.out.printf("Cannot multiply a %dx%d matrix by a %dx%d matrix.%n%n",
                left.rows(), left.cols(), right.rows(), right.cols());
        }
    }
    // Prints A = P D P^-1 with a check that multiplying the factors back together gives A, or why A has no such form.
    private static void showDiagonalization(
        final String title,
        final Matrix A
    ) {
        printHeading(title);
        print("A", A);
        try {
            Diagonalization diagonalization = new DiagonalizationLogic(A).diagonalize();
            print("P (eigenvectors as columns)", diagonalization.P());
            print("D (eigenvalues on the diagonal)", diagonalization.D());
            print("P^-1", diagonalization.inverseP());
            print("P * D * P^-1 (equals A)",
                product(product(diagonalization.P(), diagonalization.D()), diagonalization.inverseP()));
        } catch (ComplexEigenvaluesException e) {
            System.out.println("Not diagonalizable: the matrix has complex eigenvalues.");
            System.out.println();
        } catch (DefectiveMatrixException e) {
            System.out.println("Not diagonalizable: the matrix is defective.");
            System.out.println();
        } catch (DependentEigenvectorsException e) {
            System.out.println("Not diagonalizable: the eigenvectors are linearly dependent.");
            System.out.println();
        }
    }
    // Multiplies two matrices whose dimensions are known to match.
    private static Matrix product(
        final Matrix left,
        final Matrix right
    ) {
        try {
            return new MatrixLogic(left).multiply(right);
        } catch (MatrixMultiplyDimMismatchException e) {
            throw new IllegalStateException("Main.product: the dimensions were expected to match", e);
        }
    }
    private static void printHeading(
        final String title
    ) {
        System.out.println("=== " + title + " ===");
        System.out.println();
    }
    private static void print(
        final String label,
        final Matrix matrix
    ) {
        System.out.println(label + ":");
        for (int i = 0; i < matrix.rows(); i++) {
            StringBuilder row = new StringBuilder("  [");
            for (int j = 0; j < matrix.cols(); j++) {
                row.append(String.format(Locale.ROOT, ENTRY_FORMAT, shown(matrix.get(i, j))));
            }
            System.out.println(row + " ]");
        }
        System.out.println();
    }
    private static String format(
        final double value
    ) {
        return String.format(Locale.ROOT, "%.4f", shown(value));
    }
    private static double shown(
        final double value
    ) {
        return (Math.abs(value) < PRINT_ZERO) ? 0 : value;
    }

}