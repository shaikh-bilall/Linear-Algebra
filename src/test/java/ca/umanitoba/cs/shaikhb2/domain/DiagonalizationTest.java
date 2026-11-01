package ca.umanitoba.cs.shaikhb2.domain;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DiagonalizationTest {

    // ---------- constructor ----------

    @Test
    void constructorStoresThePDAndInverseMatrices() {
        var d = new Diagonalization(sampleP(), sampleD(), sampleInverseP());
        assertMatrixEquals(new double[][]{{1, 1}, {0, 1}}, d.P());
        assertMatrixEquals(new double[][]{{2, 0}, {0, 3}}, d.D());
        assertMatrixEquals(new double[][]{{1, -1}, {0, 1}}, d.inverseP());
    }
    @Test
    void constructorAcceptsASingleEntry() {
        var d = new Diagonalization(
                new Matrix(new double[][]{{1}}), new Matrix(new double[][]{{-4}}), new Matrix(new double[][]{{1}})
        );
        assertEquals(-4.0, d.D().get(0, 0), 0.0);
    }
    @Test
    void constructorAcceptsZerosOnTheDiagonalOfD() {
        var d = new Diagonalization(sampleP(), new Matrix(new double[][]{{0, 0}, {0, 0}}), sampleInverseP());
        assertMatrixEquals(new double[][]{{0, 0}, {0, 0}}, d.D());
    }
    @Test
    void constructorCopiesEveryMatrix() {
        var p = sampleP();
        var diagonal = sampleD();
        var inverse = sampleInverseP();
        var d = new Diagonalization(p, diagonal, inverse);
        p.data()[0][0] = 99;
        diagonal.data()[0][0] = 99;
        inverse.data()[0][0] = 99;
        assertNotSame(p, d.P());
        assertMatrixEquals(new double[][]{{1, 1}, {0, 1}}, d.P());
        assertMatrixEquals(new double[][]{{2, 0}, {0, 3}}, d.D());
        assertMatrixEquals(new double[][]{{1, -1}, {0, 1}}, d.inverseP());
    }
    @Test
    void constructorRejectsANullMatrix() {
        assertThrows(NullPointerException.class, () -> new Diagonalization(null, sampleD(), sampleInverseP()));
        assertThrows(NullPointerException.class, () -> new Diagonalization(sampleP(), null, sampleInverseP()));
        assertThrows(NullPointerException.class, () -> new Diagonalization(sampleP(), sampleD(), null));
    }
    @ParameterizedTest(name = "{0}x{1} is rejected")
    @CsvSource({"2, 3", "3, 2", "1, 4"})
    void constructorRejectsANonSquareP(
            int rows,
            int cols
    ) {
        var nonSquare = new Matrix(new double[rows][cols]);
        assertThrows(IllegalStateException.class, () -> new Diagonalization(nonSquare, nonSquare, nonSquare));
    }
    @Test
    void constructorRejectsMatricesOfDifferentSizes() {
        var bigger = new Matrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        assertThrows(IllegalStateException.class, () -> new Diagonalization(sampleP(), bigger, sampleInverseP()));
        assertThrows(IllegalStateException.class, () -> new Diagonalization(sampleP(), sampleD(), bigger));
        assertThrows(IllegalStateException.class, () -> new Diagonalization(bigger, sampleD(), sampleInverseP()));
    }
    @ParameterizedTest(name = "off-diagonal entry {0} is rejected")
    @ValueSource(doubles = {1, -1, 1e-12, 1e-300})
    void constructorRejectsAnyNonzeroOffDiagonalEntryInD(
            double entry
    ) {
        var above = new Matrix(new double[][]{{2, entry}, {0, 3}});
        var below = new Matrix(new double[][]{{2, 0}, {entry, 3}});
        assertThrows(IllegalStateException.class, () -> new Diagonalization(sampleP(), above, sampleInverseP()));
        assertThrows(IllegalStateException.class, () -> new Diagonalization(sampleP(), below, sampleInverseP()));
    }

    // ---------- helpers ----------

    // A fresh matrix each call, so tests can mutate one copy and compare against another.
    private static Matrix sampleP() {
        return new Matrix(new double[][]{{1, 1}, {0, 1}});
    }
    private static Matrix sampleD() {
        return new Matrix(new double[][]{{2, 0}, {0, 3}});
    }
    private static Matrix sampleInverseP() {
        return new Matrix(new double[][]{{1, -1}, {0, 1}});
    }
    private static void assertMatrixEquals(
            double[][] expected,
            Matrix actual
    ) {
        assertEquals(expected.length, actual.rows(), "row count");
        assertEquals(expected[0].length, actual.cols(), "column count");
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                assertEquals(expected[i][j], actual.get(i, j), 0.0, "mismatch at (" + i + ", " + j + ")");
            }
        }
    }
}