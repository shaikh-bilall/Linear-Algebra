package linearAlgebra.logic;

import linearAlgebra.domain.Diagonalization;
import linearAlgebra.domain.exceptions.ComplexEigenvaluesException;
import linearAlgebra.domain.exceptions.DefectiveMatrixException;
import linearAlgebra.domain.exceptions.DependentEigenvectorsException;
import linearAlgebra.domain.Matrix;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DiagonalizationLogicTest {
    private static final double TOL = 1e-9;
    // For properties on matrices built from a random change of basis or from long products, where the eigenvectors
    // can be poorly conditioned and rounding differences are larger than in a single pass.
    private static final double LOOSE_TOL = 1e-6;
    private static final double[] FACTORS = {1e-6, 1e6, -1};

    // ---------- fixed cases that diagonalize ----------

    // name, input, expected eigenvalues (largest magnitude first, the larger value first when magnitudes tie)
    static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of("1x1",
                        new double[][]{{5}},
                        new double[]{5}),
                Arguments.of("1x1 zero",
                        new double[][]{{0}},
                        new double[]{0}),
                Arguments.of("1x1 negative",
                        new double[][]{{-4}},
                        new double[]{-4}),
                Arguments.of("2x2 diagonal, already in order",
                        new double[][]{{3, 0}, {0, 2}},
                        new double[]{3, 2}),
                Arguments.of("2x2 diagonal, needs reordering",
                        new double[][]{{1, 0}, {0, 5}},
                        new double[]{5, 1}),
                Arguments.of("2x2 scalar matrix",
                        new double[][]{{3, 0}, {0, 3}},
                        new double[]{3, 3}),
                Arguments.of("2x2 non-symmetric",
                        new double[][]{{4, 1}, {2, 3}},
                        new double[]{5, 2}),
                Arguments.of("2x2 with a negative eigenvalue",
                        new double[][]{{1, 2}, {2, 1}},
                        new double[]{3, -1}),
                Arguments.of("2x2 swap (eigenvalues of equal magnitude)",
                        new double[][]{{0, 1}, {1, 0}},
                        new double[]{1, -1}),
                Arguments.of("2x2 row-stochastic",
                        new double[][]{{0.9, 0.1}, {0.5, 0.5}},
                        new double[]{1, 0.4}),
                Arguments.of("2x2 column-stochastic",
                        new double[][]{{0.9, 0.5}, {0.1, 0.5}},
                        new double[]{1, 0.4}),
                Arguments.of("3x3 upper triangular",
                        new double[][]{{1, 2, 3}, {0, 4, 5}, {0, 0, 6}},
                        new double[]{6, 4, 1}),
                Arguments.of("3x3 lower triangular",
                        new double[][]{{2, 0, 0}, {1, 3, 0}, {4, 5, 1}},
                        new double[]{3, 2, 1}),
                Arguments.of("3x3 symmetric tridiagonal",
                        new double[][]{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}},
                        new double[]{2 + Math.sqrt(2), 2, 2 - Math.sqrt(2)}),
                Arguments.of("3x3 identity",
                        new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                        new double[]{1, 1, 1}),
                Arguments.of("3x3 zero",
                        new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
                        new double[]{0, 0, 0}),
                Arguments.of("3x3 repeated eigenvalue (symmetric)",
                        new double[][]{{3, 1, 1}, {1, 3, 1}, {1, 1, 3}},
                        new double[]{5, 2, 2}),
                Arguments.of("3x3 repeated eigenvalue (non-symmetric)",
                        bidiagonalSimilar(2, 2, 5),
                        new double[]{5, 2, 2}),
                Arguments.of("3x3 two absorbing states",
                        new double[][]{{1, 0, 0}, {0, 1, 0}, {0.2, 0.3, 0.5}},
                        new double[]{1, 1, 0.5}),
                Arguments.of("3x3 lazy random walk",
                        new double[][]{{0.5, 0.5, 0}, {0.25, 0.5, 0.25}, {0, 0.5, 0.5}},
                        new double[]{1, 0.5, 0}),
                Arguments.of("3x3 periodic random walk (eigenvalues 1, 0 and -1)",
                        new double[][]{{0, 1, 0}, {0.5, 0, 0.5}, {0, 1, 0}},
                        new double[]{1, -1, 0}),
                Arguments.of("4x4 periodic random walk (eigenvalues of both signs)",
                        new double[][]{{0, 1, 0, 0}, {0.5, 0, 0.5, 0}, {0, 0.5, 0, 0.5}, {0, 0, 1, 0}},
                        new double[]{1, -1, 0.5, -0.5}),
                Arguments.of("4x4 distinct (non-symmetric)",
                        bidiagonalSimilar(4, -3, 2, 1),
                        new double[]{4, -3, 2, 1}),
                Arguments.of("4x4 singular with a repeated zero eigenvalue",
                        bidiagonalSimilar(2, 0, -1, 0),
                        new double[]{2, -1, 0, 0}),
                Arguments.of("5x5 mixed signs and repeats",
                        bidiagonalSimilar(3, -3, 1, 1, -2),
                        new double[]{3, -3, -2, 1, 1})
        );
    }
    @ParameterizedTest(name = "eigenvalues: {0}")
    @MethodSource("cases")
    void eigenvaluesMatchExpected(
            String name,
            double[][] data,
            double[] expected
    )
            throws Exception
    {
        assertEigenvaluesEqual(expected, diagonalize(data), TOL * scaleOf(data), name);
    }
    @ParameterizedTest(name = "D is diagonal: {0}")
    @MethodSource("cases")
    void diagonalMatrixHasExactlyZeroOffDiagonalEntries(
            String name,
            double[][] data,
            double[] expected
    )
            throws Exception
    {
        assertDiagonal(diagonalize(data).D(), name);
    }
    @ParameterizedTest(name = "P D P^-1 = A: {0}")
    @MethodSource("cases")
    void factorizationReproducesTheInput(
            String name,
            double[][] data,
            double[] expected
    )
            throws Exception
    {
        Diagonalization d = diagonalize(data);
        double[][] rebuilt = product(product(d.P().data(), d.D().data()), d.inverseP().data());
        assertMatrixEquals(data, rebuilt, TOL * scaleOf(data), name);
    }
    @ParameterizedTest(name = "A P = P D: {0}")
    @MethodSource("cases")
    void everyColumnOfPIsAnEigenvectorForItsEigenvalue(
            String name,
            double[][] data,
            double[] expected
    )
            throws Exception
    {
        Diagonalization d = diagonalize(data);
        assertMatrixEquals(product(data, d.P().data()), product(d.P().data(), d.D().data()), TOL * scaleOf(data), name);
    }
    @ParameterizedTest(name = "inverse works on both sides: {0}")
    @MethodSource("cases")
    void inversePIsATwoSidedInverseOfP(
            String name,
            double[][] data,
            double[] expected
    )
            throws Exception
    {
        Diagonalization d = diagonalize(data);
        double[][] identity = identityData(data.length);
        assertMatrixEquals(identity, product(d.P().data(), d.inverseP().data()), TOL, name);
        assertMatrixEquals(identity, product(d.inverseP().data(), d.P().data()), TOL, name);
    }
    @ParameterizedTest(name = "unit eigenvectors with a positive largest entry: {0}")
    @MethodSource("cases")
    void eigenvectorsAreUnitLengthWithAPositiveLargestEntry(
            String name,
            double[][] data,
            double[] expected
    )
            throws Exception
    {
        assertUnitColumnsWithPositiveLargestEntry(diagonalize(data).P(), name);
    }
    @ParameterizedTest(name = "invariant under scaling the matrix: {0}")
    @MethodSource("cases")
    void scalingTheMatrixScalesTheEigenvaluesAndKeepsTheFactorization(
            String name,
            double[][] data,
            double[] expected
    )
            throws Exception
    {
        for (double factor : FACTORS) {
            double[][] scaledData = scaled(data, factor);
            double[] scaledExpected = new double[expected.length];
            for (int i = 0; i < expected.length; i++) {
                scaledExpected[i] = factor * expected[i];
            }

            Diagonalization d = diagonalize(scaledData);
            String label = name + " scaled by " + factor;
            assertEigenvaluesEqual(ordered(scaledExpected), d, TOL * scaleOf(scaledData), label);
            assertFactorization(label, scaledData, d, TOL);
        }
    }

    // ---------- fixed cases that do not diagonalize ----------

    // name, input
    static Stream<Arguments> defectiveCases() {
        return Stream.of(
                Arguments.of("2x2 Jordan block", new double[][]{{1, 1}, {0, 1}}),
                Arguments.of("2x2 lower Jordan block", new double[][]{{1, 0}, {1, 1}}),
                Arguments.of("2x2 nilpotent", new double[][]{{0, 1}, {0, 0}}),
                Arguments.of("3x3 Jordan block", new double[][]{{2, 1, 0}, {0, 2, 1}, {0, 0, 2}}),
                Arguments.of("3x3 nilpotent shift", new double[][]{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}}),
                Arguments.of("3x3 Jordan block next to a distinct eigenvalue",
                        new double[][]{{1, 1, 0}, {0, 1, 0}, {0, 0, 2}}),
                Arguments.of("3x3 eigenvalue with multiplicity 3 but only 2 eigenvectors",
                        new double[][]{{1, 1, 0}, {0, 1, 0}, {0, 0, 1}}),
                Arguments.of("4x4 Jordan block next to two distinct eigenvalues",
                        new double[][]{{2, 1, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, -1}}),
                Arguments.of("2x2 Jordan block after a change of basis",
                        similar(new double[][]{{1, 1}, {1, 2}}, new double[][]{{2, -1}, {-1, 1}}, new double[][]{{3, 1}, {0, 3}}))
        );
    }
    @ParameterizedTest(name = "defective: {0}")
    @MethodSource("defectiveCases")
    void defectiveMatricesAreRejected(
            String name,
            double[][] data
    ) {
        var logic = new DiagonalizationLogic(new Matrix(data));
        assertThrows(DefectiveMatrixException.class, logic::diagonalize, name);
    }

    // name, input
    static Stream<Arguments> complexCases() {
        return Stream.of(
                Arguments.of("2x2 rotation by 90 degrees", new double[][]{{0, -1}, {1, 0}}),
                Arguments.of("2x2 scaled rotation", new double[][]{{1, -2}, {2, 1}}),
                Arguments.of("3x3 cyclic permutation", new double[][]{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}),
                Arguments.of("4x4 cyclic permutation",
                        new double[][]{{0, 0, 0, 1}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}),
                Arguments.of("3x3 real eigenvalue and a complex pair",
                        new double[][]{{2, 0, 0}, {0, 0, -1}, {0, 1, 0}}),
                Arguments.of("5x5 two complex pairs and a real eigenvalue",
                        new double[][]{{0, -1, 0, 0, 0}, {1, 0, 0, 0, 0}, {0, 0, 1, -3, 0}, {0, 0, 3, 1, 0}, {0, 0, 0, 0, 3}}),
                Arguments.of("3x3 complex pair after a change of basis",
                        similar(new double[][]{{1, 1, 0}, {0, 1, 1}, {0, 0, 1}}, new double[][]{{1, -1, 1}, {0, 1, -1}, {0, 0, 1}},
                                new double[][]{{2, 0, 0}, {0, 1, -2}, {0, 2, 1}}))
        );
    }
    @ParameterizedTest(name = "complex: {0}")
    @MethodSource("complexCases")
    void matricesWithComplexEigenvaluesAreRejected(
            String name,
            double[][] data
    ) {
        var logic = new DiagonalizationLogic(new Matrix(data));
        assertThrows(ComplexEigenvaluesException.class, logic::diagonalize, name);
    }
    @ParameterizedTest(name = "rejection survives scaling: {0}")
    @MethodSource("defectiveCases")
    void defectiveMatricesAreRejectedAtEveryScale(
            String name,
            double[][] data
    ) {
        for (double factor : FACTORS) {
            double[][] scaledData = scaled(data, factor);
            assertThrows(DefectiveMatrixException.class,
                    () -> diagonalize(scaledData), name + " scaled by " + factor);
        }
    }
    @ParameterizedTest(name = "rejection survives scaling: {0}")
    @MethodSource("complexCases")
    void complexMatricesAreRejectedAtEveryScale(
            String name,
            double[][] data
    ) {
        for (double factor : FACTORS) {
            double[][] scaledData = scaled(data, factor);
            assertThrows(ComplexEigenvaluesException.class,
                    () -> diagonalize(scaledData), name + " scaled by " + factor);
        }
    }

    // ---------- zero, identity and diagonal matrices ----------

    @ParameterizedTest(name = "zero matrix {0}x{0}")
    @ValueSource(ints = {1, 2, 3, 5, 6})
    void zeroMatricesHaveOnlyZeroEigenvalues(
            int n
    )
            throws Exception
    {
        double[][] data = new double[n][n];
        Diagonalization d = diagonalize(data);
        assertEigenvaluesEqual(new double[n], d, TOL, "zero " + n);
        assertFactorization("zero " + n, data, d, TOL);
    }
    @ParameterizedTest(name = "identity {0}x{0}")
    @ValueSource(ints = {1, 2, 3, 5, 6})
    void identityMatricesHaveOnlyEigenvalueOne(
            int n
    )
            throws Exception
    {
        double[][] data = identityData(n);
        Diagonalization d = diagonalize(data);
        double[] ones = new double[n];
        Arrays.fill(ones, 1);
        assertEigenvaluesEqual(ones, d, TOL, "identity " + n);
        assertFactorization("identity " + n, data, d, TOL);
    }
    // A diagonal matrix is already diagonalized: P only has to put the eigenvalues in order.
    @Test
    void diagonalMatricesGiveAPermutationOfTheStandardBasis()
            throws Exception
    {
        Random random = new Random(51); // fixed seed keeps failures reproducible
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(8);
            List<Integer> values = shuffledIndices(random, 2 * n + 1);
            double[] diagonal = new double[n];
            for (int i = 0; i < n; i++) {
                diagonal[i] = values.get(i) - n; // distinct integers in [-n, n], zero included
            }
            double[][] data = diagonalData(diagonal);
            String label = Arrays.deepToString(data);
            Diagonalization d = diagonalize(data);

            assertEigenvaluesEqual(ordered(diagonal), d, TOL, label);
            double[][] P = d.P().data();
            for (int j = 0; j < n; j++) {
                int ones = 0;
                for (int i = 0; i < n; i++) {
                    double entry = P[i][j];
                    assertTrue(Math.abs(entry) < TOL || Math.abs(entry - 1) < TOL, "Entry " + entry + " in " + label);
                    if (Math.abs(entry - 1) < TOL) {
                        ones++;
                    }
                }
                assertEquals(1, ones, "Column " + j + " of " + label);
            }
            assertFactorization(label, data, d, TOL);
        }
    }

    // ---------- state, aliasing and constructor preconditions ----------

    @Test
    void constructorRejectsANullMatrix() {
        assertThrows(NullPointerException.class, () -> new DiagonalizationLogic(null));
    }
    @ParameterizedTest(name = "{0}x{1} is rejected")
    @CsvSource({"1, 2", "2, 1", "2, 3", "3, 2", "1, 5", "5, 1"})
    void constructorRejectsNonSquareMatrices(
            int rows,
            int cols
    ) {
        var matrix = new Matrix(new double[rows][cols]);
        assertThrows(IllegalArgumentException.class, () -> new DiagonalizationLogic(matrix));
    }
    @Test
    void inputIsNeverMutated()
            throws Exception
    {
        double[][] data = {{4, 1, 0}, {2, 3, 1}, {0, 1, 5}};
        var original = new Matrix(data);
        new DiagonalizationLogic(original).diagonalize();
        assertMatrixEquals(data, original.data(), 0.0, "input");
    }
    @Test
    void inputIsNeverMutatedWhenTheMatrixIsRejected() {
        double[][] data = {{0, -1}, {1, 0}};
        var original = new Matrix(data);
        assertThrows(ComplexEigenvaluesException.class, () -> new DiagonalizationLogic(original).diagonalize());
        assertMatrixEquals(data, original.data(), 0.0, "input");
    }
    @Test
    void repeatedCallsGiveIdenticalResults()
            throws Exception
    {
        var logic = new DiagonalizationLogic(new Matrix(new double[][]{{4, 1, 0}, {2, 3, 1}, {0, 1, 5}}));
        Diagonalization first = logic.diagonalize();
        Diagonalization second = logic.diagonalize();
        assertMatrixEquals(first.P().data(), second.P().data(), 0.0, "P");
        assertMatrixEquals(first.D().data(), second.D().data(), 0.0, "D");
        assertMatrixEquals(first.inverseP().data(), second.inverseP().data(), 0.0, "inverse of P");
    }
    @Test
    void returnedMatricesAreIndependentOfTheLogicObject()
            throws Exception
    {
        var logic = new DiagonalizationLogic(new Matrix(new double[][]{{2, 0}, {0, 3}}));
        Diagonalization first = logic.diagonalize();
        first.P().data()[0][0] = 99;
        first.D().data()[0][0] = 99;
        first.inverseP().data()[0][0] = 99;

        Diagonalization second = logic.diagonalize();
        assertMatrixEquals(new double[][]{{0, 1}, {1, 0}}, second.P().data(), TOL, "P");
        assertMatrixEquals(new double[][]{{3, 0}, {0, 2}}, second.D().data(), TOL, "D");
        assertMatrixEquals(new double[][]{{0, 1}, {1, 0}}, second.inverseP().data(), TOL, "inverse of P");
    }
    @Test
    void returnedMatricesAreIndependentOfTheInput()
            throws Exception
    {
        double[][] data = {{2, 0}, {0, 3}};
        var original = new Matrix(data);
        Diagonalization d = new DiagonalizationLogic(original).diagonalize();
        d.P().data()[0][0] = 99;
        d.D().data()[1][1] = 99;
        assertMatrixEquals(new double[][]{{2, 0}, {0, 3}}, original.data(), 0.0, "input");
    }

    // ---------- the test helpers themselves ----------

    @Test
    void randomBasisChangesAreExactIntegerInverses() {
        Random random = new Random(61);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(8);
            double[][][] basis = randomBasisChange(random, n, 2 * n);
            assertMatrixEquals(identityData(n), product(basis[0], basis[1]), 0.0, "S S^-1 for n = " + n);
            assertMatrixEquals(identityData(n), product(basis[1], basis[0]), 0.0, "S^-1 S for n = " + n);
        }
    }
    @ParameterizedTest(name = "{0} eigenvalues")
    @ValueSource(ints = {1, 2, 3, 6})
    void bidiagonalSimilarityHasTheRequestedSpectrum(
            int n
    )
            throws Exception
    {
        double[] spectrum = new double[n];
        for (int i = 0; i < n; i++) {
            spectrum[i] = i + 1;
        }
        double[][] data = bidiagonalSimilar(spectrum);
        // trace and determinant pin the spectrum without trusting the code under test
        double trace = 0;
        double product = 1;
        for (int i = 0; i < n; i++) {
            trace += data[i][i];
            product *= spectrum[i];
        }
        assertEquals(n * (n + 1) / 2.0, trace, 0.0);
        assertEquals(product, new MatrixLogic(new Matrix(data)).determinant(), 1e-9 * product);
    }

    // ---------- random matrices with a known spectrum ----------

    // A = S L S^-1 with integer S and S^-1, so the eigenvalues of A are exactly the entries of L.
    @Test
    void randomDiagonalizableMatricesGiveTheKnownSpectrumAndAValidFactorization()
            throws Exception
    {
        Random random = new Random(71);
        for (int trial = 0; trial < 400; trial++) {
            int n = 1 + random.nextInt(7);
            double[] spectrum = randomSpectrum(random, n, 5);
            double[][][] basis = randomBasisChange(random, n, 2 * n);
            double[][] data = similar(basis[0], basis[1], diagonalData(spectrum));
            String label = "spectrum " + Arrays.toString(spectrum) + " in " + Arrays.deepToString(data);

            Diagonalization d = diagonalize(data);
            assertEigenvaluesEqual(ordered(spectrum), d, LOOSE_TOL * scaleOf(data), label);
            assertFactorization(label, data, d, LOOSE_TOL);
        }
    }
    @Test
    void randomMatricesWithADefectiveEigenvalueAreRejected() {
        Random random = new Random(72);
        for (int trial = 0; trial < 300; trial++) {
            int n = 2 + random.nextInt(6);
            int blockSize = 2 + random.nextInt(Math.min(n, 4) - 1);
            double[] spectrum = randomSpectrum(random, n, 5);
            for (int i = 1; i < blockSize; i++) {
                spectrum[i] = spectrum[0]; // the Jordan block shares one eigenvalue
            }
            double[][] jordan = diagonalData(spectrum);
            for (int i = 0; i < blockSize - 1; i++) {
                jordan[i][i + 1] = 1;
            }
            double[][][] basis = randomBasisChange(random, n, 2 * n);
            double[][] data = similar(basis[0], basis[1], jordan);

            Exception thrown = assertThrows(Exception.class, () -> diagonalize(data),
                    "Jordan block of size " + blockSize + " in " + Arrays.deepToString(data));
            assertTrue(thrown instanceof DefectiveMatrixException || thrown instanceof ComplexEigenvaluesException,
                    "Expected a non-diagonalizable rejection but was: " + thrown);
        }
    }
    @Test
    void randomMatricesWithAComplexPairAreRejected() {
        Random random = new Random(73);
        for (int trial = 0; trial < 300; trial++) {
            int n = 2 + random.nextInt(6);
            double[] spectrum = randomSpectrum(random, n, 5);
            double[][] blocks = diagonalData(spectrum);
            int pairs = 1 + random.nextInt(n / 2);
            for (int p = 0; p < pairs; p++) {
                double real = random.nextInt(7) - 3;
                double imaginary = 1 + random.nextInt(3);
                blocks[2 * p][2 * p] = real;
                blocks[2 * p][2 * p + 1] = -imaginary;
                blocks[2 * p + 1][2 * p] = imaginary;
                blocks[2 * p + 1][2 * p + 1] = real;
            }
            double[][][] basis = randomBasisChange(random, n, 2 * n);
            double[][] data = similar(basis[0], basis[1], blocks);

            assertThrows(ComplexEigenvaluesException.class, () -> diagonalize(data),
                    pairs + " complex pair(s) in " + Arrays.deepToString(data));
        }
    }
    @Test
    void eigenvaluesAreOrderedByMagnitudeAndEigenvectorsCarryAPositiveLargestEntry()
            throws Exception
    {
        Random random = new Random(74);
        for (int trial = 0; trial < 300; trial++) {
            int n = 1 + random.nextInt(7);
            double[] spectrum = randomSpectrum(random, n, 5);
            double[][][] basis = randomBasisChange(random, n, 2 * n);
            double[][] data = similar(basis[0], basis[1], diagonalData(spectrum));
            String label = "spectrum " + Arrays.toString(spectrum);

            Diagonalization d = diagonalize(data);
            double[] values = eigenvaluesOf(d);
            for (int i = 0; i + 1 < n; i++) {
                assertTrue(Math.abs(values[i]) >= Math.abs(values[i + 1]) - LOOSE_TOL * scaleOf(data), "Magnitude order in " + label);
                if (Math.abs(Math.abs(values[i]) - Math.abs(values[i + 1])) < LOOSE_TOL * scaleOf(data)) {
                    assertTrue(values[i] >= values[i + 1], "Tie order in " + label);
                }
            }
            assertUnitColumnsWithPositiveLargestEntry(d.P(), label);
        }
    }
    // A basis change with rounded entries leaves rounding noise in A, so a repeated eigenvalue is only repeated to
    // within about 1e-15 (and a scalar matrix is only scalar to that precision). The exact integer matrices above
    // cannot show this.
    @Test
    void repeatedEigenvaluesSurviveRoundingNoiseInTheInput()
            throws Exception
    {
        Random random = new Random(76);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + random.nextInt(8);
            double[] spectrum = randomSpectrum(random, n, 1); // only -1, 0 and 1, so almost everything repeats
            double[][] data = noisySimilar(random, diagonalData(spectrum));
            String label = "spectrum " + Arrays.toString(spectrum) + " in " + Arrays.deepToString(data);

            Diagonalization d = diagonalize(data);
            assertEigenvaluesEqual(ordered(spectrum), d, LOOSE_TOL * scaleOf(data), label);
            assertFactorization(label, data, d, LOOSE_TOL);
        }
    }
    @Test
    void repeatedEigenvaluesWithRoundingNoiseSurviveAHugeOrTinyScale()
            throws Exception
    {
        Random random = new Random(78);
        for (int trial = 0; trial < 100; trial++) {
            int n = 2 + random.nextInt(8);
            double[] spectrum = randomSpectrum(random, n, 1);
            double[][] base = noisySimilar(random, diagonalData(spectrum));
            for (double factor : new double[]{1e-9, 1e9}) {
                double[][] data = scaled(base, factor);
                double[] expected = new double[n];
                for (int i = 0; i < n; i++) {
                    expected[i] = factor * spectrum[i];
                }
                String label = "spectrum " + Arrays.toString(spectrum) + " scaled by " + factor;

                Diagonalization d = diagonalize(data);
                assertEigenvaluesEqual(ordered(expected), d, LOOSE_TOL * scaleOf(data), label);
                assertFactorization(label, data, d, LOOSE_TOL);
            }
        }
    }
    @ParameterizedTest(name = "{0}x{0} scalar matrix hidden by a basis change")
    @ValueSource(ints = {2, 3, 5, 8, 10})
    void scalarMatricesHiddenByARoundedBasisChangeStillDiagonalize(
            int n
    )
            throws Exception
    {
        Random random = new Random(77 + n);
        for (int trial = 0; trial < 20; trial++) {
            double[] spectrum = new double[n];
            Arrays.fill(spectrum, 2);
            double[][] data = noisySimilar(random, diagonalData(spectrum));

            Diagonalization d = diagonalize(data);
            assertEigenvaluesEqual(spectrum, d, LOOSE_TOL * scaleOf(data), "scalar matrix " + Arrays.deepToString(data));
            assertFactorization("scalar matrix " + Arrays.deepToString(data), data, d, LOOSE_TOL);
        }
    }
    @Test
    void largerMatricesStillDiagonalize()
            throws Exception
    {
        Random random = new Random(75);
        for (int n : new int[]{10, 12, 15}) {
            double[] spectrum = new double[n];
            List<Integer> picks = shuffledIndices(random, 2 * n + 1);
            for (int i = 0; i < n; i++) {
                spectrum[i] = picks.get(i) - n; // distinct integers
            }
            double[][][] basis = randomBasisChange(random, n, n);
            double[][] data = similar(basis[0], basis[1], diagonalData(spectrum));

            Diagonalization d = diagonalize(data);
            assertEigenvaluesEqual(ordered(spectrum), d, LOOSE_TOL * scaleOf(data), "n = " + n);
            assertFactorization("n = " + n, data, d, LOOSE_TOL);
        }
    }

    // ---------- symmetric matrices ----------

    @Test
    void randomSymmetricMatricesDiagonalizeWithOrthogonalEigenvectorsForDistinctEigenvalues()
            throws Exception
    {
        Random random = new Random(81);
        for (int trial = 0; trial < 300; trial++) {
            int n = 1 + random.nextInt(8);
            double[][] half = randomData(random, n, n, 5);
            double[][] data = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    data[i][j] = half[i][j] + half[j][i];
                }
            }
            String label = Arrays.deepToString(data);
            Diagonalization d = diagonalize(data);
            assertFactorization(label, data, d, TOL);

            // trace is the sum of the eigenvalues and the determinant (from MatrixLogic) is their product
            double[] values = eigenvaluesOf(d);
            double trace = 0;
            double product = 1;
            for (int i = 0; i < n; i++) {
                trace += data[i][i];
                product *= values[i];
            }
            double sum = Arrays.stream(values).sum();
            assertEquals(trace, sum, TOL * scaleOf(data) * n, "Trace of " + label);
            double determinant = new MatrixLogic(new Matrix(data)).determinant();
            assertEquals(determinant, product, 1e-8 * (1 + Math.abs(determinant)), "Determinant of " + label);

            double[][] P = d.P().data();
            for (int a = 0; a < n; a++) {
                for (int b = a + 1; b < n; b++) {
                    if (Math.abs(values[a] - values[b]) > LOOSE_TOL * scaleOf(data)) {
                        double dot = 0;
                        for (int i = 0; i < n; i++) {
                            dot += P[i][a] * P[i][b];
                        }
                        assertEquals(0.0, dot, 1e-8, "Columns " + a + " and " + b + " of " + label);
                    }
                }
            }
        }
    }

    // ---------- Markov chains ----------

    // A reversible chain (weights W symmetric, P[i][j] = W[i][j] / sum of row i) has real eigenvalues and is
    // diagonalizable. Its stationary distribution is pi[i] = sum of row i of W / sum of W, which is known without
    // running any eigen-code.
    @Test
    void rowStochasticChainsHaveEigenvalueOneFirstAndAnAllEqualRightEigenvector()
            throws Exception
    {
        Random random = new Random(91);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + random.nextInt(7);
            double[][] weights = randomSymmetricWeights(random, n);
            double[][] chain = rowStochastic(weights);
            String label = Arrays.deepToString(chain);

            Diagonalization d = diagonalize(chain);
            assertFactorization(label, chain, d, TOL);
            double[] values = eigenvaluesOf(d);
            assertEquals(1.0, values[0], TOL, "Largest eigenvalue of " + label);
            for (double value : values) {
                assertTrue(Math.abs(value) <= 1 + TOL, "Eigenvalue " + value + " of " + label);
            }
            for (int i = 0; i < n; i++) {
                assertEquals(1 / Math.sqrt(n), d.P().get(i, 0), TOL, "Entry " + i + " of the first eigenvector of " + label);
            }

            // the first row of P^-1 is the left eigenvector for 1, which is the stationary distribution
            double[] stationary = stationaryDistribution(weights);
            double total = 0;
            for (int j = 0; j < n; j++) {
                total += d.inverseP().get(0, j);
            }
            for (int j = 0; j < n; j++) {
                assertEquals(stationary[j], d.inverseP().get(0, j) / total, TOL, "Stationary entry " + j + " of " + label);
            }
        }
    }
    @Test
    void columnStochasticChainsHaveTheStationaryDistributionAsTheirFirstEigenvector()
            throws Exception
    {
        Random random = new Random(92);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + random.nextInt(7);
            double[][] weights = randomSymmetricWeights(random, n);
            double[][] chain = transposed(rowStochastic(weights));
            String label = Arrays.deepToString(chain);

            Diagonalization d = diagonalize(chain);
            assertFactorization(label, chain, d, TOL);
            assertEquals(1.0, d.D().get(0, 0), TOL, "Largest eigenvalue of " + label);

            double[] stationary = stationaryDistribution(weights);
            double total = 0;
            for (int i = 0; i < n; i++) {
                total += d.P().get(i, 0);
            }
            for (int i = 0; i < n; i++) {
                assertEquals(stationary[i], d.P().get(i, 0) / total, TOL, "Stationary entry " + i + " of " + label);
            }
        }
    }
    // The point of diagonalizing: A^k = P D^k P^-1, checked against k repeated multiplications.
    @Test
    void diagonalizationReproducesMatrixPowersOfAChain()
            throws Exception
    {
        Random random = new Random(93);
        for (int trial = 0; trial < 100; trial++) {
            int n = 2 + random.nextInt(6);
            double[][] chain = transposed(rowStochastic(randomSymmetricWeights(random, n)));
            Diagonalization d = diagonalize(chain);

            Matrix repeated = new Matrix(identityData(n));
            for (int k = 1; k <= 12; k++) {
                repeated = new MatrixLogic(repeated).multiply(new Matrix(chain));

                double[][] poweredD = new double[n][n];
                for (int i = 0; i < n; i++) {
                    poweredD[i][i] = Math.pow(d.D().get(i, i), k);
                }
                double[][] viaDiagonalization = product(product(d.P().data(), poweredD), d.inverseP().data());
                assertMatrixEquals(repeated.data(), viaDiagonalization, TOL, "A^" + k + " of " + Arrays.deepToString(chain));
            }
        }
    }
    @ParameterizedTest(name = "chain with {0} states")
    @ValueSource(ints = {20, 40})
    void largeChainsStillDiagonalize(
            int n
    )
            throws Exception
    {
        Random random = new Random(94 + n);
        double[][] chain = transposed(rowStochastic(randomSymmetricWeights(random, n)));
        Diagonalization d = diagonalize(chain);
        assertFactorization("chain " + n, chain, d, 1e-8);
        assertEquals(1.0, d.D().get(0, 0), 1e-8);
    }

    // ---------- eigenvalue invariance ----------

    @Test
    void aMatrixAndItsTransposeHaveTheSameEigenvalues()
            throws Exception
    {
        Random random = new Random(101);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(7);
            double[] spectrum = randomSpectrum(random, n, 5);
            double[][][] basis = randomBasisChange(random, n, 2 * n);
            double[][] data = similar(basis[0], basis[1], diagonalData(spectrum));

            double tolerance = LOOSE_TOL * scaleOf(data);
            assertEigenvaluesEqual(ordered(spectrum), diagonalize(transposed(data)), tolerance, "transpose of " + Arrays.deepToString(data));
        }
    }
    @Test
    void shiftingTheMatrixByAMultipleOfTheIdentityShiftsTheEigenvalues()
            throws Exception
    {
        Random random = new Random(102);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(7);
            double[] spectrum = randomSpectrum(random, n, 5);
            double[][][] basis = randomBasisChange(random, n, 2 * n);
            double[][] data = similar(basis[0], basis[1], diagonalData(spectrum));
            double shift = random.nextInt(9) - 4;

            double[][] shifted = new double[n][];
            double[] shiftedSpectrum = new double[n];
            for (int i = 0; i < n; i++) {
                shifted[i] = data[i].clone();
                shifted[i][i] += shift;
                shiftedSpectrum[i] = spectrum[i] + shift;
            }
            assertEigenvaluesEqual(ordered(shiftedSpectrum), diagonalize(shifted), LOOSE_TOL * scaleOf(shifted),
                    "shift " + shift + " of " + Arrays.deepToString(data));
        }
    }
    @Test
    void changingTheBasisDoesNotChangeTheEigenvalues()
            throws Exception
    {
        Random random = new Random(103);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(7);
            double[] spectrum = randomSpectrum(random, n, 5);
            double[][][] first = randomBasisChange(random, n, 2 * n);
            double[][][] second = randomBasisChange(random, n, 2 * n);
            double[][] data = similar(first[0], first[1], diagonalData(spectrum));
            double[][] changed = similar(second[0], second[1], data);

            assertEigenvaluesEqual(ordered(spectrum), diagonalize(changed), LOOSE_TOL * scaleOf(changed),
                    "basis change of " + Arrays.deepToString(data));
        }
    }
    @Test
    void squaringTheMatrixSquaresTheEigenvalues()
            throws Exception
    {
        Random random = new Random(104);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(6);
            double[] spectrum = randomSpectrum(random, n, 4);
            double[][][] basis = randomBasisChange(random, n, 2 * n);
            double[][] data = similar(basis[0], basis[1], diagonalData(spectrum));
            double[][] squared = product(data, data);
            double[] squaredSpectrum = new double[n];
            for (int i = 0; i < n; i++) {
                squaredSpectrum[i] = spectrum[i] * spectrum[i];
            }
            assertEigenvaluesEqual(ordered(squaredSpectrum), diagonalize(squared), LOOSE_TOL * scaleOf(squared),
                    "square of " + Arrays.deepToString(data));
        }
    }

    // ---------- helpers ----------

    private static Diagonalization diagonalize(
            final double[][] data
    )
            throws ComplexEigenvaluesException,
            DefectiveMatrixException,
            DependentEigenvectorsException
    {
        return new DiagonalizationLogic(new Matrix(data)).diagonalize();
    }
    // Everything a diagonalization of A must satisfy, each checked directly against the input.
    private static void assertFactorization(
            final String label,
            final double[][] data,
            final Diagonalization d,
            final double tolerance
    ) {
        int n = data.length;
        assertEquals(n, d.P().rows(), "rows of P in " + label);
        assertEquals(n, d.P().cols(), "columns of P in " + label);
        assertEquals(n, d.D().rows(), "rows of D in " + label);
        assertEquals(n, d.inverseP().rows(), "rows of the inverse of P in " + label);

        double[][] P = d.P().data();
        double[][] inverse = d.inverseP().data();
        double scale = scaleOf(data);
        assertDiagonal(d.D(), label);
        assertUnitColumnsWithPositiveLargestEntry(d.P(), label);
        assertMatrixEquals(identityData(n), product(P, inverse), tolerance, "P P^-1 for " + label);
        assertMatrixEquals(identityData(n), product(inverse, P), tolerance, "P^-1 P for " + label);
        assertMatrixEquals(product(data, P), product(P, d.D().data()), tolerance * scale, "A P = P D for " + label);
        assertMatrixEquals(data, product(product(P, d.D().data()), inverse), tolerance * scale, "P D P^-1 = A for " + label);
    }
    private static void assertDiagonal(
            final Matrix D,
            final String label
    ) {
        for (int i = 0; i < D.rows(); i++) {
            for (int j = 0; j < D.cols(); j++) {
                if (i != j) {
                    assertEquals(0.0, D.get(i, j), 0.0, "D(" + i + ", " + j + ") in " + label);
                }
            }
        }
    }
    // Each column has length 1, and its entry of largest magnitude (the first of any near tie) is positive.
    private static void assertUnitColumnsWithPositiveLargestEntry(
            final Matrix P,
            final String label
    ) {
        for (int j = 0; j < P.cols(); j++) {
            double lengthSquared = 0;
            double largest = 0;
            for (int i = 0; i < P.rows(); i++) {
                lengthSquared += P.get(i, j) * P.get(i, j);
                largest = Math.max(largest, Math.abs(P.get(i, j)));
            }
            assertEquals(1.0, Math.sqrt(lengthSquared), TOL, "Length of column " + j + " in " + label);
            for (int i = 0; i < P.rows(); i++) {
                if (Math.abs(P.get(i, j)) >= largest * (1 - 1e-8)) {
                    assertTrue(P.get(i, j) > 0, "Largest entry of column " + j + " is negative in " + label);
                    break;
                }
            }
        }
    }
    private static double[] eigenvaluesOf(
            final Diagonalization d
    ) {
        double[] values = new double[d.D().rows()];
        for (int i = 0; i < values.length; i++) {
            values[i] = d.D().get(i, i);
        }
        return values;
    }
    private static void assertEigenvaluesEqual(
            final double[] expected,
            final Diagonalization actual,
            final double tolerance,
            final String label
    ) {
        double[] values = eigenvaluesOf(actual);
        assertEquals(expected.length, values.length, "number of eigenvalues in " + label);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values[i], tolerance,
                    "eigenvalue " + i + " of " + label + ": expected " + Arrays.toString(expected) + " but was " + Arrays.toString(values));
        }
    }
    // The order the linearAlgebra.logic promises: largest magnitude first, the larger value first when magnitudes tie.
    private static double[] ordered(
            final double[] eigenvalues
    ) {
        return Arrays.stream(eigenvalues).boxed()
                .sorted(Comparator.comparingDouble((Double x) -> Math.abs(x)).thenComparingDouble(x -> x).reversed())
                .mapToDouble(Double::doubleValue)
                .toArray();
    }
    // Largest absolute entry (1 for the zero matrix), so that tolerances scale with the matrix being tested.
    private static double scaleOf(
            final double[][] data
    ) {
        double maxAbs = 0;
        for (double[] row : data) {
            for (double entry : row) {
                maxAbs = Math.max(maxAbs, Math.abs(entry));
            }
        }
        return (maxAbs == 0) ? 1 : maxAbs;
    }
    // A change of basis with integer entries whose inverse is also integer, built by elementary row operations: S gets
    // each operation and S^-1 gets the inverse operation as a column operation. Returns {S, S^-1}.
    private static double[][][] randomBasisChange(
            final Random random,
            final int n,
            final int steps
    ) {
        double[][] basis = identityData(n);
        double[][] inverse = identityData(n);
        for (int step = 0; step < steps && n > 1; step++) {
            int i = random.nextInt(n);
            int j = (i + 1 + random.nextInt(n - 1)) % n;
            if (random.nextInt(4) == 0) {
                double[] row = basis[i];
                basis[i] = basis[j];
                basis[j] = row;
                for (double[] inverseRow : inverse) {
                    double entry = inverseRow[i];
                    inverseRow[i] = inverseRow[j];
                    inverseRow[j] = entry;
                }
            } else {
                double factor = random.nextBoolean() ? 1 : -1;
                for (int k = 0; k < n; k++) {
                    basis[i][k] += factor * basis[j][k]; // row i += factor * row j
                    inverse[k][j] -= factor * inverse[k][i]; // column j -= factor * column i
                }
            }
        }
        return new double[][][]{basis, inverse};
    }
    // S L S^-1 for a random Gaussian S (kept reasonably conditioned) whose inverse comes from MatrixLogic. Unlike the
    // integer basis changes this is not exact, so the result carries rounding noise.
    private static double[][] noisySimilar(
            final Random random,
            final double[][] middle
    )
            throws Exception
    {
        int n = middle.length;
        while (true) {
            double[][] basis = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    basis[i][j] = random.nextGaussian();
                }
            }
            double[][] inverse = new MatrixLogic(new Matrix(basis)).inverse().data();
            if (scaleOf(basis) * scaleOf(inverse) < 100) {
                return similar(basis, inverse, middle);
            }
        }
    }
    private static double[][] similar(
            final double[][] basis,
            final double[][] inverse,
            final double[][] middle
    ) {
        return product(product(basis, middle), inverse);
    }
    // S L S^-1 with S upper bidiagonal (ones on the diagonal and superdiagonal), whose inverse has entries (-1)^(j - i).
    // The result is an integer matrix with exactly the given eigenvalues and non-orthogonal eigenvectors.
    private static double[][] bidiagonalSimilar(
            final double... eigenvalues
    ) {
        int n = eigenvalues.length;
        double[][] basis = identityData(n);
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            if (i + 1 < n) {
                basis[i][i + 1] = 1;
            }
            for (int j = i; j < n; j++) {
                inverse[i][j] = ((j - i) % 2 == 0) ? 1 : -1;
            }
        }
        return similar(basis, inverse, diagonalData(eigenvalues));
    }
    // Integer eigenvalues in [-bound, bound], repeats allowed.
    private static double[] randomSpectrum(
            final Random random,
            final int n,
            final int bound
    ) {
        double[] spectrum = new double[n];
        for (int i = 0; i < n; i++) {
            spectrum[i] = random.nextInt(2 * bound + 1) - bound;
        }
        return spectrum;
    }
    // Integer entries in [-bound, bound].
    private static double[][] randomData(
            final Random random,
            final int rows,
            final int cols,
            final int bound
    ) {
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = random.nextInt(2 * bound + 1) - bound;
            }
        }
        return data;
    }
    // Symmetric positive weights W with integer entries in [1, 9].
    private static double[][] randomSymmetricWeights(
            final Random random,
            final int n
    ) {
        double[][] weights = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                weights[i][j] = 1 + random.nextInt(9);
                weights[j][i] = weights[i][j];
            }
        }
        return weights;
    }
    private static double[][] rowStochastic(
            final double[][] weights
    ) {
        int n = weights.length;
        double[][] chain = new double[n][n];
        for (int i = 0; i < n; i++) {
            double rowSum = Arrays.stream(weights[i]).sum();
            for (int j = 0; j < n; j++) {
                chain[i][j] = weights[i][j] / rowSum;
            }
        }
        return chain;
    }
    // For the chain built from symmetric weights the stationary probability of state-i is its row sum over the total.
    private static double[] stationaryDistribution(
            final double[][] weights
    ) {
        int n = weights.length;
        double[] stationary = new double[n];
        double total = 0;
        for (int i = 0; i < n; i++) {
            stationary[i] = Arrays.stream(weights[i]).sum();
            total += stationary[i];
        }
        for (int i = 0; i < n; i++) {
            stationary[i] /= total;
        }
        return stationary;
    }
    private static List<Integer> shuffledIndices(
            final Random random,
            final int size
    ) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        return indices;
    }
    private static double[][] diagonalData(
            final double... diagonal
    ) {
        double[][] data = new double[diagonal.length][diagonal.length];
        for (int i = 0; i < diagonal.length; i++) {
            data[i][i] = diagonal[i];
        }
        return data;
    }
    private static double[][] identityData(
            final int n
    ) {
        double[] ones = new double[n];
        Arrays.fill(ones, 1);
        return diagonalData(ones);
    }
    private static double[][] transposed(
            final double[][] data
    ) {
        double[][] result = new double[data[0].length][data.length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                result[j][i] = data[i][j];
            }
        }
        return result;
    }
    private static double[][] scaled(
            final double[][] data,
            final double factor
    ) {
        double[][] result = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                result[i][j] = factor * data[i][j];
            }
        }
        return result;
    }
    // Plain triple loop, deliberately independent of MatrixLogic.multiply.
    private static double[][] product(
            final double[][] left,
            final double[][] right
    ) {
        double[][] result = new double[left.length][right[0].length];
        for (int i = 0; i < left.length; i++) {
            for (int j = 0; j < right[0].length; j++) {
                double sum = 0;
                for (int k = 0; k < right.length; k++) {
                    sum += left[i][k] * right[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
    private static void assertMatrixEquals(
            final double[][] expected,
            final double[][] actual,
            final double tolerance,
            final String label
    ) {
        assertEquals(expected.length, actual.length, "row count of " + label);
        assertEquals(expected[0].length, actual[0].length, "column count of " + label);
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                assertEquals(expected[i][j], actual[i][j], tolerance,
                        "mismatch at (" + i + ", " + j + ") of " + label);
            }
        }
    }
}
