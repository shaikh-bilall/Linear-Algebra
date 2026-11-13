package linearAlgebra.domain;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.HashSet;
import java.util.Set;

import static linearAlgebra.TestUtils.assertMatrixEquals;
import static org.junit.jupiter.api.Assertions.*;

class DiagonalizationTest {

    // ---------- constructor ----------

    @Test
    void constructorStoresThePDAndInverseMatrices() {
        var d = new Diagonalization(sampleP(), sampleD(), sampleInverseP());
        assertMatrixEquals(new double[][]{{1, 1}, {0, 1}}, d.P(), 0.0);
        assertMatrixEquals(new double[][]{{2, 0}, {0, 3}}, d.D(), 0.0);
        assertMatrixEquals(new double[][]{{1, -1}, {0, 1}}, d.inverseP(), 0.0);
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
        assertMatrixEquals(new double[][]{{0, 0}, {0, 0}}, d.D(), 0.0);
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

    // ---------- equals and hashCode ----------

    @Test
    void equalsHoldsForDiagonalizationsWithTheSameMatrices() {
        var diagonalization = sampleDiagonalization();
        var same = sampleDiagonalization();
        assertEquals(diagonalization, diagonalization);
        assertEquals(diagonalization, same);
        assertEquals(same, diagonalization);
        assertEquals(diagonalization.hashCode(), same.hashCode());
    }
    @Test
    void equalsFailsWhenAnyOfTheThreeMatricesDiffers() {
        var diagonalization = sampleDiagonalization();
        var other = new Matrix(new double[][]{{5, 0}, {0, 7}});
        assertNotEquals(diagonalization, new Diagonalization(other, diagonalization.D(), diagonalization.inverseP()));
        assertNotEquals(diagonalization, new Diagonalization(diagonalization.P(), other, diagonalization.inverseP()));
        assertNotEquals(diagonalization, new Diagonalization(diagonalization.P(), diagonalization.D(), other));
    }
    @Test
    void equalsFailsWhenPAndTheInverseOfPAreSwapped() {
        var diagonalization = sampleDiagonalization();
        var swapped = new Diagonalization(diagonalization.inverseP(), diagonalization.D(), diagonalization.P());
        assertNotEquals(diagonalization, swapped);
        assertNotEquals(swapped, diagonalization);
    }
    @Test
    void equalsFailsForNullAndForOtherTypes() {
        var diagonalization = sampleDiagonalization();
        assertFalse(diagonalization.equals(null));
        assertFalse(diagonalization.equals("diagonalization"));
        assertFalse(diagonalization.equals(diagonalization.P()));
    }
    @Test
    void equalsTreatsNegativeZeroLikeZero() {
        var diagonalization = sampleDiagonalization();
        var negativeZeros = new Diagonalization(
            diagonalization.P(),
            new Matrix(new double[][]{{2, -0.0}, {-0.0, 3}}),
            diagonalization.inverseP());
        assertEquals(diagonalization, negativeZeros);
        assertEquals(diagonalization.hashCode(), negativeZeros.hashCode());
    }
    @Test
    void equalDiagonalizationsAreInterchangeableAsHashKeys() {
        Set<Diagonalization> diagonalizations = new HashSet<>();
        diagonalizations.add(sampleDiagonalization());
        assertTrue(diagonalizations.contains(sampleDiagonalization()));
        diagonalizations.add(sampleDiagonalization());
        assertEquals(1, diagonalizations.size());
    }

    // ---------- helpers ----------

    private static Matrix sampleP() {
        return new Matrix(new double[][]{{1, 1}, {0, 1}});
    }
    private static Matrix sampleD() {
        return new Matrix(new double[][]{{2, 0}, {0, 3}});
    }
    private static Matrix sampleInverseP() {
        return new Matrix(new double[][]{{1, -1}, {0, 1}});
    }
    // A diagonalization of an upper triangular matrix: P D P^-1 with D = diag(2, 3).
    private static Diagonalization sampleDiagonalization() {
        return new Diagonalization(
            new Matrix(new double[][]{{1, 1}, {0, 1}}),
            new Matrix(new double[][]{{2, 0}, {0, 3}}),
            new Matrix(new double[][]{{1, -1}, {0, 1}}));
    }

}
