package ca.umanitoba.cs.shaikhb2.logic;

import ca.umanitoba.cs.shaikhb2.domain.Exceptions.InverseNotAvailableException;
import ca.umanitoba.cs.shaikhb2.domain.Matrix;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.platform.launcher.Launcher;
import org.junit.platform.launcher.LauncherDiscoveryRequest;
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder;
import org.junit.platform.launcher.core.LauncherFactory;
import org.junit.platform.launcher.listeners.SummaryGeneratingListener;
import org.junit.platform.launcher.listeners.TestExecutionSummary;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.platform.engine.discovery.DiscoverySelectors.selectPackage;

class MatrixLogicTest {

    private static final double TOL = 1e-9;

    public static void main(String[] args) {
        // Discovers every test class under this package, so tests added later are picked up automatically.
        // To run only this file, use selectClass(MatrixLogicTest.class) instead.
        LauncherDiscoveryRequest request = LauncherDiscoveryRequestBuilder.request()
                .selectors(selectPackage("ca.umanitoba.cs.shaikhb2"))
                .build();

        Launcher launcher = LauncherFactory.create();
        SummaryGeneratingListener listener = new SummaryGeneratingListener();
        launcher.execute(request, listener);

        TestExecutionSummary summary = listener.getSummary();
        long found = summary.getTestsFoundCount();
        long passed = summary.getTestsSucceededCount();
        long failed = summary.getTestsFailedCount();
        long skipped = summary.getTestsSkippedCount();
        long aborted = summary.getTestsAbortedCount();
        long millis = summary.getTimeFinished() - summary.getTimeStarted();

        if (!summary.getFailures().isEmpty()) {
            System.out.println("========== FAILURES ==========");
            for (TestExecutionSummary.Failure failure : summary.getFailures()) {
                System.out.println("FAIL: " + failure.getTestIdentifier().getDisplayName());
                System.out.println("      " + failure.getException().getMessage());
            }
            System.out.println();
        }

        System.out.println("========== SUMMARY ==========");
        System.out.printf("Total:   %d%n", found);
        System.out.printf("Passed:  %d%n", passed);
        System.out.printf("Failed:  %d%n", failed);
        System.out.printf("Skipped: %d%n", skipped);
        System.out.printf("Aborted: %d%n", aborted);
        System.out.printf("Time:    %d ms%n", millis);
        System.out.println(failed == 0 && aborted == 0 ? "RESULT: ALL TESTS PASSED" : "RESULT: SOME TESTS FAILED");

        System.exit(failed == 0 && aborted == 0 ? 0 : 1);
    }

    // name, input, expected RREF, expected determinant
    static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of("2x2 invertible",
                        new double[][]{{2, 4}, {1, 3}},
                        new double[][]{{1, 0}, {0, 1}}, 2.0),
                Arguments.of("2x2 needs a row swap",
                        new double[][]{{0, 1}, {1, 0}},
                        new double[][]{{1, 0}, {0, 1}}, -1.0),
                Arguments.of("2x2 singular",
                        new double[][]{{1, 2}, {2, 4}},
                        new double[][]{{1, 2}, {0, 0}}, 0.0),
                Arguments.of("3x3 invertible",
                        new double[][]{{2, 1, 1}, {4, -6, 0}, {-2, 7, 2}},
                        new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, -16.0),
                Arguments.of("3x3 identity (empty operation list)",
                        new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                        new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, 1.0),
                Arguments.of("2x2 zero matrix",
                        new double[][]{{0, 0}, {0, 0}},
                        new double[][]{{0, 0}, {0, 0}}, 0.0),
                Arguments.of("1x1",
                        new double[][]{{5}},
                        new double[][]{{1}}, 5.0),
                Arguments.of("2x3 wide (determinant undefined)",
                        new double[][]{{1, 2, 3}, {4, 5, 6}},
                        new double[][]{{1, 0, -1}, {0, 1, 2}}, Double.NaN),
                Arguments.of("3x2 tall (determinant undefined)",
                        new double[][]{{1, 2}, {3, 4}, {5, 6}},
                        new double[][]{{1, 0}, {0, 1}, {0, 0}}, Double.NaN)
        );
    }

    @ParameterizedTest(name = "RREF: {0}")
    @MethodSource("cases")
    void rrefMatchesExpected(String name, double[][] data, double[][] expectedRref, double expectedDet)
            throws Exception {
        Matrix actual = new MatrixLogic(new Matrix(data)).rref();
        assertMatrixEquals(new Matrix(expectedRref), actual);
    }

    @ParameterizedTest(name = "determinant: {0}")
    @MethodSource("cases")
    void determinantMatchesExpected(String name, double[][] data, double[][] expectedRref, double expectedDet) {
        // JUnit's assertEquals(double, double, delta) treats NaN == NaN as equal.
        assertEquals(expectedDet, new MatrixLogic(new Matrix(data)).determinant(), TOL);
    }

    @ParameterizedTest(name = "inverse: {0}")
    @MethodSource("cases")
    void inverseBehavesCorrectly(String name, double[][] data, double[][] expectedRref, double expectedDet)
            throws Exception {
        var logic = new MatrixLogic(new Matrix(data));
        boolean invertible = !Double.isNaN(expectedDet) && expectedDet != 0;

        if (invertible) {
            Matrix inverse = logic.inverse();
            assertMatrixEquals(identity(data.length), logic.multiply(inverse));
        } else {
            assertThrows(InverseNotAvailableException.class, logic::inverse);
        }
    }

    @Test
    void inputIsNeverMutated() throws Exception {
        double[][] data = {{2, 1, 1}, {4, -6, 0}, {-2, 7, 2}};
        var original = new Matrix(data);
        var logic = new MatrixLogic(original);
        logic.rref();
        logic.inverse();
        assertMatrixEquals(new Matrix(data), original);
    }

    @Test
    void randomMatricesGiveValidRrefAndCorrectDeterminant() throws Exception {
        Random random = new Random(42); // fixed seed keeps failures reproducible
        for (int trial = 0; trial < 500; trial++) {
            int rows = 1 + random.nextInt(5);
            int cols = 1 + random.nextInt(5);
            double[][] data = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    data[i][j] = random.nextInt(11) - 5; // small ints give plenty of singular/rank-deficient cases
                }
            }
            String label = Arrays.deepToString(data);
            var logic = new MatrixLogic(new Matrix(data));

            assertTrue(isRref(logic.rref()), "Not a valid RREF for " + label);

            if (rows == cols) {
                double expectedDet = laplaceDeterminant(data);
                assertEquals(expectedDet, logic.determinant(), 1e-6, "Wrong determinant for " + label);

                if (Math.abs(expectedDet) > 1e-9) {
                    assertMatrixEquals(identity(rows), logic.multiply(logic.inverse()), 1e-6);
                } else {
                    assertThrows(InverseNotAvailableException.class, logic::inverse);
                }
            }
        }
    }

    // ---------- helpers ----------

    // Checks the RREF definition: zero rows at the bottom, each pivot is 1, pivots move strictly right,
    // and every pivot column is zero everywhere else.
    private static boolean isRref(Matrix M) {
        int previousPivotCol = -1;
        boolean seenZeroRow = false;
        for (int i = 0; i < M.rows(); i++) {
            int pivotCol = -1;
            for (int j = 0; j < M.cols(); j++) {
                if (Math.abs(M.get(i, j)) > TOL) {
                    pivotCol = j;
                    break;
                }
            }
            if (pivotCol == -1) {
                seenZeroRow = true;
                continue;
            }
            if (seenZeroRow || pivotCol <= previousPivotCol) {
                return false;
            }
            if (Math.abs(M.get(i, pivotCol) - 1) > TOL) {
                return false;
            }
            for (int r = 0; r < M.rows(); r++) {
                if (r != i && Math.abs(M.get(r, pivotCol)) > TOL) {
                    return false;
                }
            }
            previousPivotCol = pivotCol;
        }
        return true;
    }

    // Independent oracle: cofactor expansion along the first row (fine for n <= 5).
    private static double laplaceDeterminant(double[][] m) {
        int n = m.length;
        if (n == 1) {
            return m[0][0];
        }
        double det = 0;
        for (int col = 0; col < n; col++) {
            double[][] minor = new double[n - 1][n - 1];
            for (int i = 1; i < n; i++) {
                int c = 0;
                for (int j = 0; j < n; j++) {
                    if (j != col) {
                        minor[i - 1][c++] = m[i][j];
                    }
                }
            }
            det += (col % 2 == 0 ? 1 : -1) * m[0][col] * laplaceDeterminant(minor);
        }
        return det;
    }

    private static Matrix identity(int n) {
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            data[i][i] = 1;
        }
        return new Matrix(data);
    }

    private static void assertMatrixEquals(Matrix expected, Matrix actual) {
        assertMatrixEquals(expected, actual, TOL);
    }

    private static void assertMatrixEquals(Matrix expected, Matrix actual, double tolerance) {
        assertEquals(expected.rows(), actual.rows(), "row count");
        assertEquals(expected.cols(), actual.cols(), "column count");
        for (int i = 0; i < expected.rows(); i++) {
            for (int j = 0; j < expected.cols(); j++) {
                assertEquals(expected.get(i, j), actual.get(i, j), tolerance,
                        "mismatch at (" + i + ", " + j + ")");
            }
        }
    }
}