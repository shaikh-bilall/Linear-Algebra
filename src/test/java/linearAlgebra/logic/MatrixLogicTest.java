package linearAlgebra.logic;

import linearAlgebra.logic.exceptions.DeterminantDoesNotExistException;
import linearAlgebra.logic.exceptions.InverseNotAvailableException;
import linearAlgebra.domain.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.rref.Type1;
import linearAlgebra.domain.rref.Type2;
import linearAlgebra.domain.rref.Type3;
import linearAlgebra.domain.Vector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static linearAlgebra.TestUtils.assertMatrixEquals;
import static linearAlgebra.TestUtils.identity;
import static linearAlgebra.TestUtils.product;
import static linearAlgebra.TestUtils.randomData;
import static linearAlgebra.TestUtils.randomScalar;
import static linearAlgebra.TestUtils.scaled;
import static linearAlgebra.TestUtils.shuffledIndices;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MatrixLogicTest {
    private static final double TOL = 1e-9;
    // For properties that compare results reached through different elimination orders or chained inverses,
    // where rounding differences are larger than in a single pass.
    private static final double LOOSE_TOL = 1e-6;

    // ---------- fixed cases: RREF, determinant, inverse ----------

    // name, input, expected RREF, expected determinant (NaN marks a non-square matrix, which has no determinant)
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
                new double[][]{{1, 0}, {0, 1}, {0, 0}}, Double.NaN),
            Arguments.of("1x1 zero",
                new double[][]{{0}},
                new double[][]{{0}}, 0.0),
            Arguments.of("1x1 negative",
                new double[][]{{-4}},
                new double[][]{{1}}, -4.0),
            Arguments.of("1x1 identity",
                new double[][]{{1}},
                new double[][]{{1}}, 1.0),
            Arguments.of("2x2 negative entries",
                new double[][]{{-1, -2}, {-3, -4}},
                new double[][]{{1, 0}, {0, 1}}, -2.0),
            Arguments.of("2x2 fractional entries",
                new double[][]{{0.5, 0.25}, {0.25, 0.5}},
                new double[][]{{1, 0}, {0, 1}}, 0.1875),
            Arguments.of("2x2 large magnitude",
                new double[][]{{1e6, 2e6}, {3e6, 4e6}},
                new double[][]{{1, 0}, {0, 1}}, -2e12),
            Arguments.of("2x2 small magnitude",
                new double[][]{{1e-6, 2e-6}, {3e-6, 4e-6}},
                new double[][]{{1, 0}, {0, 1}}, -2e-12),
            Arguments.of("3x3 cyclic permutation (two swaps)",
                new double[][]{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
                new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, 1.0),
            Arguments.of("3x3 singular (dependent rows)",
                new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                new double[][]{{1, 0, -1}, {0, 1, 2}, {0, 0, 0}}, 0.0),
            Arguments.of("3x3 singular (zero column)",
                new double[][]{{1, 0, 2}, {3, 0, 4}, {5, 0, 6}},
                new double[][]{{1, 0, 0}, {0, 0, 1}, {0, 0, 0}}, 0.0),
            Arguments.of("4x4 invertible",
                new double[][]{{1, 0, 2, -1}, {3, 0, 0, 5}, {2, 1, 4, -3}, {1, 0, 5, 0}},
                new double[][]{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}, 30.0),
            Arguments.of("1x3 row vector",
                new double[][]{{3, 0, 6}},
                new double[][]{{1, 0, 2}}, Double.NaN),
            Arguments.of("3x1 column vector",
                new double[][]{{0}, {2}, {4}},
                new double[][]{{1}, {0}, {0}}, Double.NaN),
            Arguments.of("2x3 leading zero column",
                new double[][]{{0, 1, 2}, {0, 3, 4}},
                new double[][]{{0, 1, 0}, {0, 0, 1}}, Double.NaN),
            Arguments.of("2x3 rank one",
                new double[][]{{1, 2, 3}, {2, 4, 6}},
                new double[][]{{1, 2, 3}, {0, 0, 0}}, Double.NaN),
            Arguments.of("3x2 rank one",
                new double[][]{{1, 2}, {2, 4}, {3, 6}},
                new double[][]{{1, 2}, {0, 0}, {0, 0}}, Double.NaN),
            Arguments.of("3x2 zero row in the middle",
                new double[][]{{1, 1}, {0, 0}, {0, 1}},
                new double[][]{{1, 0}, {0, 1}, {0, 0}}, Double.NaN)
        );
    }
    @ParameterizedTest(name = "RREF: {0}")
    @MethodSource("cases")
    void rrefMatchesExpected(
        String name,
        double[][] data,
        double[][] expectedRref,
        double expectedDet
    )
        throws Exception
    {
        Matrix actual = new MatrixLogic(new Matrix(data)).rref();
        assertMatrixEquals(new Matrix(expectedRref), actual, TOL);
    }
    @ParameterizedTest(name = "determinant: {0}")
    @MethodSource("cases")
    void determinantMatchesExpected(
        String name,
        double[][] data,
        double[][] expectedRref,
        double expectedDet
    )
        throws Exception
    {
        var logic = new MatrixLogic(new Matrix(data));
        if (Double.isNaN(expectedDet)) {
            assertThrows(DeterminantDoesNotExistException.class, logic::determinant, name);
        } else {
            assertDeterminantEquals(expectedDet, logic.determinant(), name);
        }
    }
    @ParameterizedTest(name = "inverse: {0}")
    @MethodSource("cases")
    void inverseBehavesCorrectly(
        String name,
        double[][] data,
        double[][] expectedRref,
        double expectedDet
    )
        throws Exception
    {
        var logic = new MatrixLogic(new Matrix(data));
        boolean invertible = !Double.isNaN(expectedDet) && expectedDet != 0;

        if (invertible) {
            Matrix inverse = logic.inverse();
            assertMatrixEquals(identity(data.length), logic.multiply(inverse), TOL);
        } else {
            assertThrows(InverseNotAvailableException.class, logic::inverse);
        }
    }
    @ParameterizedTest(name = "RREF of an RREF is unchanged: {0}")
    @MethodSource("cases")
    void rrefOfAnRrefIsUnchanged(
        String name,
        double[][] data,
        double[][] expectedRref,
        double expectedDet
    )
        throws Exception
    {
        Matrix once = new MatrixLogic(new Matrix(data)).rref();
        Matrix twice = new MatrixLogic(once).rref();
        assertMatrixEquals(once, twice, TOL);
    }

    // name, input, expected inverse
    static Stream<Arguments> inverseCases() {
        return Stream.of(
            Arguments.of("1x1",
                new double[][]{{4}},
                new double[][]{{0.25}}),
            Arguments.of("2x2 with unit determinant",
                new double[][]{{2, 1}, {1, 1}},
                new double[][]{{1, -1}, {-1, 2}}),
            Arguments.of("2x2 with fractional inverse",
                new double[][]{{4, 7}, {2, 6}},
                new double[][]{{0.6, -0.7}, {-0.2, 0.4}}),
            Arguments.of("2x2 permutation is its own inverse",
                new double[][]{{0, 1}, {1, 0}},
                new double[][]{{0, 1}, {1, 0}}),
            Arguments.of("3x3 diagonal",
                new double[][]{{2, 0, 0}, {0, 4, 0}, {0, 0, -5}},
                new double[][]{{0.5, 0, 0}, {0, 0.25, 0}, {0, 0, -0.2}}),
            Arguments.of("3x3 textbook example",
                new double[][]{{1, 2, 3}, {0, 1, 4}, {5, 6, 0}},
                new double[][]{{-24, 18, 5}, {20, -15, -4}, {-5, 4, 1}}),
            Arguments.of("3x3 cyclic permutation inverts to its transpose",
                new double[][]{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
                new double[][]{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}})
        );
    }
    @ParameterizedTest(name = "known inverse: {0}")
    @MethodSource("inverseCases")
    void inverseMatchesTheKnownInverse(
        String name,
        double[][] data,
        double[][] expectedInverse
    )
        throws Exception
    {
        assertMatrixEquals(expectedInverse, new MatrixLogic(new Matrix(data)).inverse(), TOL);
    }
    @ParameterizedTest(name = "inverse works on both sides: {0}")
    @MethodSource("inverseCases")
    void inverseIsATwoSidedInverse(
        String name,
        double[][] data,
        double[][] expectedInverse
    )
        throws Exception
    {
        var original = new Matrix(data);
        Matrix inverse = new MatrixLogic(original).inverse();
        assertMatrixEquals(identity(data.length), new MatrixLogic(original).multiply(inverse), TOL);
        assertMatrixEquals(identity(data.length), new MatrixLogic(inverse).multiply(original), TOL);
    }

    // ---------- zero and identity matrices of many shapes ----------

    @ParameterizedTest(name = "zero matrix {0}x{1}")
    @CsvSource({"1, 1", "1, 4", "4, 1", "3, 3", "2, 5", "5, 2"})
    void zeroMatricesReduceToThemselves(
        int rows,
        int cols
    )
        throws Exception
    {
        var zero = new Matrix(new double[rows][cols]);
        var logic = new MatrixLogic(zero);
        assertMatrixEquals(zero, logic.rref(), TOL);
        if (rows == cols) {
            assertEquals(0.0, logic.determinant(), 0.0);
            assertThrows(InverseNotAvailableException.class, logic::inverse);
        } else {
            assertThrows(DeterminantDoesNotExistException.class, logic::determinant);
        }
    }
    @ParameterizedTest(name = "identity {0}x{0}")
    @ValueSource(ints = {1, 2, 3, 4, 6})
    void identityMatricesAreTheirOwnRrefAndInverse(
        int n
    )
        throws Exception
    {
        var logic = new MatrixLogic(identity(n));
        assertMatrixEquals(identity(n), logic.rref(), TOL);
        assertMatrixEquals(identity(n), logic.inverse(), TOL);
        assertEquals(1.0, logic.determinant(), TOL);
    }

    // ---------- state, aliasing and constructor preconditions ----------

    @Test
    void inputIsNeverMutated()
        throws Exception
    {
        double[][] data = {{2, 1, 1}, {4, -6, 0}, {-2, 7, 2}};
        var original = new Matrix(data);
        var logic = new MatrixLogic(original);
        logic.rref();
        logic.inverse();
        assertMatrixEquals(new Matrix(data), original, TOL);
    }
    @Test
    void repeatedCallsGiveIdenticalResults()
        throws Exception
    {
        var logic = new MatrixLogic(new Matrix(new double[][]{{2, 1, 1}, {4, -6, 0}, {-2, 7, 2}}));
        assertMatrixEquals(logic.rref(), logic.rref(), 0.0);
        assertMatrixEquals(logic.inverse(), logic.inverse(), 0.0);
        assertEquals(logic.determinant(), logic.determinant(), 0.0);
    }
    @Test
    void constructorRejectsANullMatrix() {
        assertThrows(NullPointerException.class, () -> new MatrixLogic(null));
    }

    // ---------- RREF properties on random matrices ----------

    @Test
    void randomMatricesGiveValidRrefAndCorrectDeterminant()
        throws Exception
    {
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
    // The RREF is checked against the input itself, not against a second implementation: every column of A must
    // equal the combination of A's pivot columns whose coefficients are that column of the RREF.
    @Test
    void rrefReproducesTheColumnDependenciesOfTheInput() {
        Random random = new Random(21);
        for (int trial = 0; trial < 500; trial++) {
            double[][] data = randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5);
            var original = new Matrix(data);
            assertColumnsFollowRref(original, new MatrixLogic(original).rref(), Arrays.deepToString(data));
        }
    }
    // Row operations never change the RREF, so reordering, scaling or combining rows must give the same result.
    @Test
    void rrefIgnoresTheOrderOfTheRows() {
        Random random = new Random(22);
        for (int trial = 0; trial < 300; trial++) {
            int rows = 1 + random.nextInt(5);
            double[][] data = randomData(random, rows, 1 + random.nextInt(5), 5);
            var original = new Matrix(data);
            Matrix shuffled = original.permuteRows(shuffledIndices(random, rows));
            assertMatrixEquals(new MatrixLogic(original).rref(), new MatrixLogic(shuffled).rref(), LOOSE_TOL);
        }
    }
    @Test
    void rrefIgnoresScalingARow() {
        Random random = new Random(23);
        for (int trial = 0; trial < 300; trial++) {
            int rows = 1 + random.nextInt(5);
            var original = new Matrix(randomData(random, rows, 1 + random.nextInt(5), 5));
            Matrix scaled = new Type2(random.nextInt(rows), randomScalar(random)).apply(original);
            assertMatrixEquals(new MatrixLogic(original).rref(), new MatrixLogic(scaled).rref(), LOOSE_TOL);
        }
    }
    @Test
    void rrefIgnoresAddingAMultipleOfOneRowToAnother() {
        Random random = new Random(24);
        for (int trial = 0; trial < 300; trial++) {
            int rows = 2 + random.nextInt(4);
            var original = new Matrix(randomData(random, rows, 1 + random.nextInt(5), 5));
            List<Integer> picks = shuffledIndices(random, rows);
            var type3 = new Type3(picks.get(0), picks.get(1), randomScalar(random));
            Matrix combined = type3.apply(original);
            assertMatrixEquals(new MatrixLogic(original).rref(), new MatrixLogic(combined).rref(), LOOSE_TOL);
        }
    }
    // Exercises the relative tolerance: a matrix and a hugely scaled copy must reduce identically.
    @Test
    void rrefIgnoresScalingTheWholeMatrix() {
        Random random = new Random(25);
        double[] factors = {1e-3, 1e3, -3};
        for (int trial = 0; trial < 300; trial++) {
            double[][] data = randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5);
            double factor = factors[random.nextInt(factors.length)];
            assertMatrixEquals(
                new MatrixLogic(new Matrix(data)).rref(),
                new MatrixLogic(new Matrix(scaled(data, factor))).rref(),
                LOOSE_TOL
            );
        }
    }
    @Test
    void rrefOfAnRrefIsUnchangedOnRandomMatrices() {
        Random random = new Random(26);
        for (int trial = 0; trial < 300; trial++) {
            var original = new Matrix(randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5));
            Matrix once = new MatrixLogic(original).rref();
            assertMatrixEquals(once, new MatrixLogic(once).rref(), LOOSE_TOL);
        }
    }
    @Test
    void aMatrixAndItsTransposeHaveTheSameRank() {
        Random random = new Random(27);
        for (int trial = 0; trial < 300; trial++) {
            double[][] data = randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5);
            var original = new Matrix(data);
            int rank = rankOf(new MatrixLogic(original).rref());
            int transposedRank = rankOf(new MatrixLogic(original.transpose()).rref());
            assertEquals(rank, transposedRank, "rank of " + Arrays.deepToString(data));
        }
    }
    // A product of a rows x r matrix and an r x cols matrix has rank at most r, whatever the entries are.
    @Test
    void lowRankProductsReduceToAtMostThatManyPivots()
        throws Exception
    {
        Random random = new Random(28);
        for (int trial = 0; trial < 300; trial++) {
            int rows = 1 + random.nextInt(5);
            int cols = 1 + random.nextInt(5);
            int rank = 1 + random.nextInt(Math.min(rows, cols));
            double[][] data = product(randomData(random, rows, rank, 3), randomData(random, rank, cols, 3));
            String label = "rank " + rank + ": " + Arrays.deepToString(data);
            var original = new Matrix(data);
            var logic = new MatrixLogic(original);
            Matrix reduced = logic.rref();

            assertTrue(isRref(reduced), "Not a valid RREF for " + label);
            assertTrue(rankOf(reduced) <= rank, "Too many pivots for " + label);
            assertColumnsFollowRref(original, reduced, label);

            if (rows == cols && rank < rows) {
                assertEquals(0.0, logic.determinant(), 0.0, "Determinant must be zero for " + label);
                assertThrows(InverseNotAvailableException.class, logic::inverse, label);
            }
        }
    }

    // ---------- determinant properties ----------

    @Test
    void permutationMatricesHaveDeterminantEqualToTheirSign()
        throws Exception
    {
        Random random = new Random(11);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(6);
            List<Integer> order = shuffledIndices(random, n);
            double[][] data = new double[n][n];
            for (int i = 0; i < n; i++) {
                data[i][order.get(i)] = 1;
            }
            String label = Arrays.deepToString(data);
            var logic = new MatrixLogic(new Matrix(data));

            assertEquals(permutationSign(order), logic.determinant(), TOL, "Wrong determinant for " + label);
            assertMatrixEquals(identity(n), logic.rref(), TOL);
            assertMatrixEquals(new Matrix(data).transpose(), logic.inverse(), TOL);
        }
    }
    @Test
    void triangularMatricesHaveDeterminantEqualToTheDiagonalProduct()
        throws Exception
    {
        Random random = new Random(13);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(6);
            boolean upper = random.nextBoolean();
            double[][] data = randomData(random, n, n, 5);
            double product = 1;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (upper ? j < i : j > i) {
                        data[i][j] = 0;
                    }
                }
                product *= data[i][i];
            }
            assertDeterminantEquals(product, new MatrixLogic(new Matrix(data)).determinant(),
                "Wrong determinant for " + Arrays.deepToString(data));
        }
    }
    @Test
    void transposingDoesNotChangeTheDeterminant()
        throws Exception
    {
        Random random = new Random(14);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(5);
            double[][] data = randomData(random, n, n, 5);
            assertDeterminantEquals(laplaceDeterminant(data), new MatrixLogic(new Matrix(data).transpose()).determinant(),
                "Wrong determinant for the transpose of " + Arrays.deepToString(data));
        }
    }
    @Test
    void swappingTwoRowsNegatesTheDeterminant()
        throws Exception
    {
        Random random = new Random(15);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + random.nextInt(4);
            double[][] data = randomData(random, n, n, 5);
            List<Integer> picks = shuffledIndices(random, n);
            Matrix swapped = new Type1(picks.get(0), picks.get(1)).apply(new Matrix(data));
            assertDeterminantEquals(-laplaceDeterminant(data), new MatrixLogic(swapped).determinant(),
                "Wrong determinant after a swap in " + Arrays.deepToString(data));
        }
    }
    @Test
    void scalingARowScalesTheDeterminantByTheSameFactor()
        throws Exception
    {
        Random random = new Random(16);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(5);
            double[][] data = randomData(random, n, n, 5);
            double factor = randomScalar(random);
            Matrix scaled = new Type2(random.nextInt(n), factor).apply(new Matrix(data));
            assertDeterminantEquals(factor * laplaceDeterminant(data), new MatrixLogic(scaled).determinant(),
                "Wrong determinant after scaling a row of " + Arrays.deepToString(data));
        }
    }
    @Test
    void addingAMultipleOfOneRowToAnotherKeepsTheDeterminant()
        throws Exception
    {
        Random random = new Random(17);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + random.nextInt(4);
            double[][] data = randomData(random, n, n, 5);
            List<Integer> picks = shuffledIndices(random, n);
            var type3 = new Type3(picks.get(0), picks.get(1), randomScalar(random));
            Matrix combined = type3.apply(new Matrix(data));
            assertDeterminantEquals(laplaceDeterminant(data), new MatrixLogic(combined).determinant(),
                "Wrong determinant after a row combination in " + Arrays.deepToString(data));
        }
    }
    @Test
    void scalingTheWholeMatrixScalesTheDeterminantByTheNthPower()
        throws Exception
    {
        Random random = new Random(18);
        double[] factors = {-2, -1, 2, 3};
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(4);
            double[][] data = randomData(random, n, n, 5);
            double factor = factors[random.nextInt(factors.length)];
            assertDeterminantEquals(Math.pow(factor, n) * laplaceDeterminant(data),
                new MatrixLogic(new Matrix(scaled(data, factor))).determinant(),
                "Wrong determinant after scaling " + Arrays.deepToString(data) + " by " + factor);
        }
    }
    @Test
    void determinantOfAProductIsTheProductOfTheDeterminants()
        throws Exception
    {
        Random random = new Random(19);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(3);
            double[][] left = randomData(random, n, n, 3);
            double[][] right = randomData(random, n, n, 3);
            assertDeterminantEquals(laplaceDeterminant(left) * laplaceDeterminant(right),
                new MatrixLogic(new Matrix(product(left, right))).determinant(),
                "Wrong determinant for " + Arrays.deepToString(left) + " times " + Arrays.deepToString(right));
        }
    }
    @Test
    void matricesWithARepeatedOrZeroRowOrColumnAreSingular()
        throws Exception
    {
        Random random = new Random(20);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + random.nextInt(4);
            List<Integer> picks = shuffledIndices(random, n);

            double[][] repeatedRow = randomData(random, n, n, 5);
            repeatedRow[picks.get(1)] = repeatedRow[picks.get(0)].clone();

            double[][] zeroRow = randomData(random, n, n, 5);
            zeroRow[picks.get(0)] = new double[n];

            double[][] zeroColumn = randomData(random, n, n, 5);
            for (int i = 0; i < n; i++) {
                zeroColumn[i][picks.get(0)] = 0;
            }

            for (double[][] data : List.of(repeatedRow, zeroRow, zeroColumn)) {
                String label = Arrays.deepToString(data);
                var logic = new MatrixLogic(new Matrix(data));
                assertEquals(0.0, logic.determinant(), 0.0, "Determinant must be exactly zero for " + label);
                assertThrows(InverseNotAvailableException.class, logic::inverse, label);
            }
        }
    }

    // ---------- inverse properties ----------

    @Test
    void inverseOfTheInverseIsTheOriginal()
        throws Exception
    {
        Random random = new Random(29);
        for (int trial = 0; trial < 300; trial++) {
            int n = 1 + random.nextInt(4);
            double[][] data = randomData(random, n, n, 5);
            if (laplaceDeterminant(data) == 0) {
                continue;
            }
            var original = new Matrix(data);
            Matrix inverse = new MatrixLogic(original).inverse();
            assertMatrixEquals(original, new MatrixLogic(inverse).inverse(), LOOSE_TOL);
        }
    }
    @Test
    void inverseOfAProductIsTheReversedProductOfTheInverses()
        throws Exception
    {
        Random random = new Random(30);
        for (int trial = 0; trial < 300; trial++) {
            int n = 1 + random.nextInt(3);
            double[][] left = randomData(random, n, n, 3);
            double[][] right = randomData(random, n, n, 3);
            if (laplaceDeterminant(left) == 0 || laplaceDeterminant(right) == 0) {
                continue;
            }
            Matrix inverseOfProduct = new MatrixLogic(new Matrix(product(left, right))).inverse();
            Matrix leftInverse = new MatrixLogic(new Matrix(left)).inverse();
            Matrix rightInverse = new MatrixLogic(new Matrix(right)).inverse();
            assertMatrixEquals(inverseOfProduct, new MatrixLogic(rightInverse).multiply(leftInverse), LOOSE_TOL);
        }
    }

    // ---------- multiply ----------

    @Test
    void multiplyComputesAKnownProduct()
        throws Exception
    {
        var left = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        var right = new Matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        assertMatrixEquals(new double[][]{{58, 64}, {139, 154}}, new MatrixLogic(left).multiply(right), TOL);
    }
    @Test
    void multiplyBuildsOuterAndInnerProducts()
        throws Exception
    {
        var column = new Matrix(new double[][]{{1}, {2}, {3}});
        var row = new Matrix(new double[][]{{4, 5, 6}});
        assertMatrixEquals(
            new double[][]{{4, 5, 6}, {8, 10, 12}, {12, 15, 18}},
            new MatrixLogic(column).multiply(row),
            TOL
        );
        assertMatrixEquals(new double[][]{{32}}, new MatrixLogic(row).multiply(column), TOL);
    }
    @Test
    void multiplyByTheIdentityOnEitherSideLeavesTheMatrixUnchanged()
        throws Exception
    {
        Random random = new Random(32);
        for (int trial = 0; trial < 200; trial++) {
            int rows = 1 + random.nextInt(5);
            int cols = 1 + random.nextInt(5);
            var matrix = new Matrix(randomData(random, rows, cols, 5));
            assertMatrixEquals(matrix, new MatrixLogic(matrix).multiply(identity(cols)), TOL);
            assertMatrixEquals(matrix, new MatrixLogic(identity(rows)).multiply(matrix), TOL);
        }
    }
    @Test
    void multiplyByAZeroMatrixGivesAZeroMatrixOfTheRightShape()
        throws Exception
    {
        Random random = new Random(33);
        for (int trial = 0; trial < 100; trial++) {
            int rows = 1 + random.nextInt(5);
            int inner = 1 + random.nextInt(5);
            int cols = 1 + random.nextInt(5);
            var matrix = new Matrix(randomData(random, rows, inner, 5));
            var zero = new Matrix(new double[inner][cols]);
            assertMatrixEquals(new Matrix(new double[rows][cols]), new MatrixLogic(matrix).multiply(zero), TOL);
        }
    }
    @Test
    void multiplyIsAssociative()
        throws Exception
    {
        Random random = new Random(34);
        for (int trial = 0; trial < 200; trial++) {
            var a = new Matrix(randomData(random, 1 + random.nextInt(4), 1 + random.nextInt(4), 3));
            var b = new Matrix(randomData(random, a.cols(), 1 + random.nextInt(4), 3));
            var c = new Matrix(randomData(random, b.cols(), 1 + random.nextInt(4), 3));
            Matrix leftFirst = new MatrixLogic(new MatrixLogic(a).multiply(b)).multiply(c);
            Matrix rightFirst = new MatrixLogic(a).multiply(new MatrixLogic(b).multiply(c));
            assertMatrixEquals(leftFirst, rightFirst, TOL);
        }
    }
    @Test
    void multiplyFollowsTheTransposeRule()
        throws Exception
    {
        Random random = new Random(35);
        for (int trial = 0; trial < 200; trial++) {
            var a = new Matrix(randomData(random, 1 + random.nextInt(4), 1 + random.nextInt(4), 5));
            var b = new Matrix(randomData(random, a.cols(), 1 + random.nextInt(4), 5));
            Matrix transposeOfProduct = new MatrixLogic(a).multiply(b).transpose();
            Matrix productOfTransposes = new MatrixLogic(b.transpose()).multiply(a.transpose());
            assertMatrixEquals(transposeOfProduct, productOfTransposes, TOL);
        }
    }
    @Test
    void multiplyAgreesWithMatrixVectorMultiplication()
        throws Exception
    {
        Random random = new Random(36);
        for (int trial = 0; trial < 200; trial++) {
            int rows = 1 + random.nextInt(5);
            int cols = 1 + random.nextInt(5);
            var matrix = new Matrix(randomData(random, rows, cols, 5));
            double[] entries = new double[cols];
            double[][] column = new double[cols][1];
            for (int i = 0; i < cols; i++) {
                entries[i] = random.nextInt(11) - 5;
                column[i][0] = entries[i];
            }

            Vector expected = matrix.multiply(new Vector(entries));
            Matrix actual = new MatrixLogic(matrix).multiply(new Matrix(column));

            assertEquals(rows, actual.rows(), "row count");
            assertEquals(1, actual.cols(), "column count");
            for (int i = 0; i < rows; i++) {
                assertEquals(expected.get(i), actual.get(i, 0), TOL, "entry " + i);
            }
        }
    }
    @Test
    void multiplyRejectsIncompatibleDimensions() {
        var wide = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        var logic = new MatrixLogic(wide);
        assertThrows(MatrixMultiplyDimMismatchException.class, () -> logic.multiply(wide)); // 2x3 times 2x3
        assertThrows(MatrixMultiplyDimMismatchException.class, () -> logic.multiply(new Matrix(new double[][]{{1}}))); // 2x3 times 1x1
    }
    @Test
    void multiplyRejectsANullMatrix() {
        var logic = new MatrixLogic(new Matrix(new double[][]{{1, 2}, {3, 4}}));
        assertThrows(NullPointerException.class, () -> logic.multiply(null));
    }
    @Test
    void multiplyDoesNotChangeEitherOperand()
        throws Exception
    {
        double[][] leftData = {{1, 2, 3}, {4, 5, 6}};
        double[][] rightData = {{7, 8}, {9, 10}, {11, 12}};
        var left = new Matrix(leftData);
        var right = new Matrix(rightData);
        new MatrixLogic(left).multiply(right);
        assertMatrixEquals(new Matrix(leftData), left, TOL);
        assertMatrixEquals(new Matrix(rightData), right, TOL);
    }

    // ---------- helpers ----------

    // Checks the RREF definition: zero rows at the bottom, each pivot is 1, pivots move strictly right,
    // and every pivot column is zero everywhere else.
    private static boolean isRref(
        Matrix M
    ) {
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
    // For a valid RREF, the first non-zero entry of each non-zero row marks a pivot column.
    private static List<Integer> pivotColumns(
        Matrix reduced
    ) {
        List<Integer> pivots = new ArrayList<>();
        for (int i = 0; i < reduced.rows(); i++) {
            for (int j = 0; j < reduced.cols(); j++) {
                if (Math.abs(reduced.get(i, j)) > TOL) {
                    pivots.add(j);
                    break;
                }
            }
        }
        return pivots;
    }
    private static int rankOf(
        Matrix reduced
    ) {
        return pivotColumns(reduced).size();
    }
    // Ties the RREF back to the input: column j of A must equal sum over k of R[k][j] * (k-th pivot column of A).
    private static void assertColumnsFollowRref(
        Matrix original,
        Matrix reduced,
        String label
    ) {
        assertTrue(isRref(reduced), "Not a valid RREF for " + label);
        List<Integer> pivots = pivotColumns(reduced);
        for (int j = 0; j < original.cols(); j++) {
            for (int i = 0; i < original.rows(); i++) {
                double combination = 0;
                for (int k = 0; k < pivots.size(); k++) {
                    combination += reduced.get(k, j) * original.get(i, pivots.get(k));
                }
                assertEquals(original.get(i, j), combination, LOOSE_TOL,
                    "column " + j + ", row " + i + " of " + label);
            }
        }
    }
    // Independent oracle: cofactor expansion along the first row (fine for n <= 6).
    private static double laplaceDeterminant(
        double[][] m
    ) {
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
    // Relative tolerance, so that determinants of 1e12 and 1e-12 are held to the same standard.
    // An expected value of exactly zero must be matched exactly: singular matrices report 0, not "almost 0".
    private static void assertDeterminantEquals(
        double expected,
        double actual,
        String message
    ) {
        double tolerance = (expected == 0) ? 0 : TOL * Math.abs(expected);
        assertEquals(expected, actual, tolerance, message);
    }
    private static double permutationSign(
        List<Integer> order
    ) {
        int inversions = 0;
        for (int i = 0; i < order.size(); i++) {
            for (int j = i + 1; j < order.size(); j++) {
                if (order.get(i) > order.get(j)) {
                    inversions++;
                }
            }
        }
        return (inversions % 2 == 0) ? 1 : -1;
    }
}