package linearAlgebra.domain;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.*;

import static linearAlgebra.TestUtils.assertVectorEquals;
import static linearAlgebra.TestUtils.inverseOf;
import static linearAlgebra.TestUtils.randomEntries;
import static linearAlgebra.TestUtils.randomScalar;
import static linearAlgebra.TestUtils.shuffledIndices;
import static linearAlgebra.TestUtils.vector;
import static org.junit.jupiter.api.Assertions.*;

class VectorTest {

    private static final double TOL = 1e-12;

    // ---------- constructor from an array ----------

    @Test
    void constructorStoresTheEntriesInOrder() {
        var v = vector(1, -2, 3.5);
        assertVectorEquals(new double[]{1, -2, 3.5}, v);
        assertEquals(3, v.length());
    }
    @Test
    void constructorAcceptsASingleEntry() {
        assertVectorEquals(new double[]{7}, vector(7));
    }
    @Test
    void constructorCopiesTheArray() {
        double[] entries = {1, 2, 3};
        var v = new Vector(entries);
        entries[0] = 99;
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @Test
    void constructorRejectsANullArray() {
        assertThrows(NullPointerException.class, () -> new Vector((double[]) null));
    }
    @Test
    void constructorRejectsAnEmptyArray() {
        assertThrows(IllegalStateException.class, () -> new Vector(new double[0]));
    }
    @ParameterizedTest(name = "entry {0} is rejected")
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void constructorRejectsNonFiniteEntries(
        double entry
    ) {
        assertThrows(IllegalStateException.class, () -> new Vector(new double[]{1, entry, 3}));
        assertThrows(IllegalStateException.class, () -> new Vector(new double[]{entry}));
    }

    // ---------- constructor from a size ----------

    @ParameterizedTest(name = "size {0}")
    @ValueSource(ints = {1, 2, 5, 100})
    void sizeConstructorCreatesAZeroVector(
        int size
    ) {
        var v = new Vector(size);
        assertEquals(size, v.length());
        for (int i = 0; i < size; i++) {
            assertEquals(0.0, v.get(i), 0.0);
        }
    }
    @ParameterizedTest(name = "size {0} is rejected")
    @ValueSource(ints = {0, -1, Integer.MIN_VALUE})
    void sizeConstructorRejectsNonPositiveSizes(
        int size
    ) {
        assertThrows(IllegalStateException.class, () -> new Vector(size));
    }

    // ---------- data ----------

    @Test
    void dataReturnsACopy() {
        var v = vector(1, 2, 3);
        double[] first = v.data();
        double[] second = v.data();
        assertNotSame(first, second);
        assertTrue(Arrays.equals(first, second));

        first[1] = 20;
        assertVectorEquals(new double[]{1, 2, 3}, v);
        assertEquals(2.0, v.get(1), 0.0);
    }

    // ---------- add ----------

    @Test
    void addComputesTheEntrywiseSum() {
        assertVectorEquals(new double[]{5, 7, 9}, vector(1, 2, 3).add(vector(4, 5, 6)));
    }
    @Test
    void addIsCommutativeAndAssociative() {
        Random random = new Random(1); // fixed seed keeps failures reproducible
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            var a = new Vector(randomEntries(random, size));
            var b = new Vector(randomEntries(random, size));
            var c = new Vector(randomEntries(random, size));
            assertVectorEquals(a.add(b).data(), b.add(a));
            assertVectorEquals(a.add(b).add(c).data(), a.add(b.add(c)));
        }
    }
    @Test
    void addingTheZeroVectorChangesNothing() {
        Random random = new Random(2);
        for (int trial = 0; trial < 100; trial++) {
            int size = 1 + random.nextInt(8);
            var a = new Vector(randomEntries(random, size));
            assertVectorEquals(a.data(), a.add(new Vector(size)));
        }
    }
    @Test
    void addingTheNegationGivesTheZeroVector() {
        Random random = new Random(3);
        for (int trial = 0; trial < 100; trial++) {
            int size = 1 + random.nextInt(8);
            var a = new Vector(randomEntries(random, size));
            assertVectorEquals(new double[size], a.add(a.scale(-1)));
        }
    }
    @Test
    void addLeavesBothOperandsUnchangedAndReturnsANewVector() {
        var a = vector(1, 2);
        var b = vector(3, 4);
        var sum = a.add(b);
        assertNotSame(a, sum);
        assertNotSame(b, sum);
        assertVectorEquals(new double[]{1, 2}, a);
        assertVectorEquals(new double[]{3, 4}, b);
    }
    @Test
    void addRejectsVectorsOfDifferentLengths() {
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2).add(vector(1, 2, 3)));
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2, 3).add(vector(1, 2)));
    }
    @Test
    void addRejectsANullVector() {
        assertThrows(NullPointerException.class, () -> vector(1, 2).add(null));
    }
    @Test
    void addRejectsASumThatOverflows() {
        assertThrows(IllegalStateException.class, () -> vector(Double.MAX_VALUE).add(vector(Double.MAX_VALUE)));
    }

    // ---------- scale ----------

    @Test
    void scaleMultipliesEveryEntry() {
        assertVectorEquals(new double[]{-2, 4, -6}, vector(1, -2, 3).scale(-2));
    }
    @Test
    void scaleByZeroGivesTheZeroVector() {
        assertVectorEquals(new double[]{0, 0, 0}, vector(1, -2, 3).scale(0));
    }
    @Test
    void scaleByOneChangesNothing() {
        assertVectorEquals(new double[]{1, -2, 3}, vector(1, -2, 3).scale(1));
    }
    @Test
    void scaleDistributesOverAdditionAndComposes() {
        Random random = new Random(4);
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            var a = new Vector(randomEntries(random, size));
            var b = new Vector(randomEntries(random, size));
            double s = randomScalar(random);
            double t = randomScalar(random);
            assertVectorEquals(a.add(b).scale(s).data(), a.scale(s).add(b.scale(s)));
            assertVectorEquals(a.scale(s * t).data(), a.scale(s).scale(t));
        }
    }
    @Test
    void scaleLeavesTheOperandUnchangedAndReturnsANewVector() {
        var a = vector(1, 2);
        var scaled = a.scale(3);
        assertNotSame(a, scaled);
        assertVectorEquals(new double[]{1, 2}, a);
    }
    @ParameterizedTest(name = "scalar {0} is rejected")
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void scaleRejectsNonFiniteScalars(
        double scalar
    ) {
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2).scale(scalar));
    }
    @Test
    void scaleRejectsAProductThatOverflows() {
        assertThrows(IllegalStateException.class, () -> vector(1e308).scale(10));
    }

    // ---------- dot ----------

    @Test
    void dotComputesAKnownProduct() {
        assertEquals(32.0, vector(1, 2, 3).dot(vector(4, 5, 6)), TOL);
    }
    @Test
    void dotOfOrthogonalVectorsIsZero() {
        assertEquals(0.0, vector(1, 2).dot(vector(-2, 1)), TOL);
        assertEquals(0.0, vector(3, 0, 0).dot(vector(0, 5, 7)), TOL);
    }
    @Test
    void dotIsCommutativeAndLinear() {
        Random random = new Random(5);
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            var a = new Vector(randomEntries(random, size));
            var b = new Vector(randomEntries(random, size));
            var c = new Vector(randomEntries(random, size));
            double s = randomScalar(random);
            assertEquals(a.dot(b), b.dot(a), 0.0);
            assertEquals(s * a.dot(b), a.scale(s).dot(b), TOL);
            assertEquals(a.dot(b) + a.dot(c), a.dot(b.add(c)), TOL);
        }
    }
    @Test
    void dotWithTheZeroVectorIsZero() {
        assertEquals(0.0, vector(4, -5, 6).dot(new Vector(3)), 0.0);
    }
    @Test
    void dotWithItselfIsTheSquaredNorm() {
        Random random = new Random(6);
        for (int trial = 0; trial < 200; trial++) {
            var a = new Vector(randomEntries(random, 1 + random.nextInt(8)));
            assertEquals(a.dot(a), a.norm() * a.norm(), 1e-9);
        }
    }
    @Test
    void dotRejectsVectorsOfDifferentLengths() {
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2).dot(vector(1, 2, 3)));
    }
    @Test
    void dotRejectsANullVector() {
        assertThrows(NullPointerException.class, () -> vector(1, 2).dot(null));
    }
    @Test
    void dotRejectsAProductThatOverflows() {
        assertThrows(IllegalStateException.class, () -> vector(1e200).dot(vector(1e200)));
    }

    // ---------- norm ----------

    @Test
    void normComputesAKnownLength() {
        assertEquals(5.0, vector(3, 4).norm(), TOL);
        assertEquals(7.0, vector(-7).norm(), TOL);
    }
    @Test
    void normOfTheZeroVectorIsZero() {
        assertEquals(0.0, new Vector(4).norm(), 0.0);
    }
    @Test
    void normIsNonNegativeAndScalesWithTheAbsoluteValueOfTheScalar() {
        Random random = new Random(7);
        for (int trial = 0; trial < 200; trial++) {
            var a = new Vector(randomEntries(random, 1 + random.nextInt(8)));
            double s = randomScalar(random);
            assertTrue(a.norm() >= 0);
            assertEquals(Math.abs(s) * a.norm(), a.scale(s).norm(), 1e-9);
        }
    }
    @Test
    void normSatisfiesTheTriangleInequality() {
        Random random = new Random(8);
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            var a = new Vector(randomEntries(random, size));
            var b = new Vector(randomEntries(random, size));
            assertTrue(a.add(b).norm() <= a.norm() + b.norm() + 1e-9);
        }
    }
    @Test
    void normSurvivesEntriesWhoseSquaresWouldOverflow() {
        assertEquals(Math.sqrt(2) * 1e200, vector(1e200, 1e200).norm(), 1e-12 * 1e200);
    }
    @Test
    void normRejectsALengthThatOverflows() {
        assertThrows(IllegalStateException.class, () -> vector(Double.MAX_VALUE, Double.MAX_VALUE).norm());
    }

    // ---------- permute ----------

    @Test
    void permuteReordersTheEntries() {
        assertVectorEquals(new double[]{30, 10, 40, 20}, vector(10, 20, 30, 40).permute(List.of(2, 0, 3, 1)));
    }
    @Test
    void permuteWithTheIdentityOrderChangesNothing() {
        assertVectorEquals(new double[]{10, 20, 30}, vector(10, 20, 30).permute(List.of(0, 1, 2)));
    }
    @Test
    void permuteIsUndoneByTheInversePermutation() {
        Random random = new Random(9);
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            var a = new Vector(randomEntries(random, size));
            List<Integer> order = shuffledIndices(random, size);
            assertVectorEquals(a.data(), a.permute(order).permute(inverseOf(order)));
        }
    }
    @Test
    void permuteKeepsTheSameEntriesAndLeavesTheOriginalUnchanged() {
        Random random = new Random(10);
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            double[] entries = randomEntries(random, size);
            var a = new Vector(entries);
            double[] permuted = a.permute(shuffledIndices(random, size)).data().clone();
            double[] expected = entries.clone();
            Arrays.sort(permuted);
            Arrays.sort(expected);
            assertEquals(Arrays.toString(expected), Arrays.toString(permuted));
            assertVectorEquals(entries, a);
        }
    }
    @Test
    void permuteRejectsAListOfTheWrongSize() {
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2, 3).permute(List.of(0, 1)));
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2, 3).permute(List.of(0, 1, 2, 0)));
    }
    @Test
    void permuteRejectsAListThatIsNotAPermutation() {
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2, 3).permute(List.of(0, 0, 1))); // repeated index
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2, 3).permute(List.of(-1, 0, 1))); // negative index
        assertThrows(IllegalArgumentException.class, () -> vector(1, 2, 3).permute(List.of(0, 1, 3))); // index too large
    }
    @Test
    void permuteRejectsANullListAndANullIndex() {
        assertThrows(NullPointerException.class, () -> vector(1, 2, 3).permute(null));
        assertThrows(NullPointerException.class, () -> vector(1, 2, 3).permute(Arrays.asList(0, null, 2)));
    }

    // ---------- get ----------

    @Test
    void getReturnsEachEntry() {
        var v = vector(7, 8, 9);
        assertEquals(7.0, v.get(0), 0.0);
        assertEquals(8.0, v.get(1), 0.0);
        assertEquals(9.0, v.get(2), 0.0);
    }
    @ParameterizedTest(name = "index {0} is rejected")
    @ValueSource(ints = {-1, 3, 100, Integer.MIN_VALUE, Integer.MAX_VALUE})
    void getRejectsOutOfBoundsIndex(
        int index
    ) {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> v.get(index));
    }

    // ---------- length ----------

    @ParameterizedTest(name = "length {0}")
    @ValueSource(ints = {1, 2, 3, 10})
    void lengthMatchesTheNumberOfEntries(
        int size
    ) {
        assertEquals(size, new Vector(new double[size]).length());
    }

    // ---------- iterator ----------

    @Test
    void iteratorVisitsEveryEntryInOrder() {
        List<Double> seen = new ArrayList<>();
        for (double entry : vector(4, 5, 6)) {
            seen.add(entry);
        }
        assertEquals(List.of(4.0, 5.0, 6.0), seen);
    }
    @Test
    void iteratorReportsTheEndAndRefusesToGoPastIt() {
        Iterator<Double> iterator = vector(1, 2).iterator();
        assertTrue(iterator.hasNext());
        assertTrue(iterator.hasNext()); // asking twice must not advance
        assertEquals(1.0, iterator.next(), 0.0);
        assertEquals(2.0, iterator.next(), 0.0);
        assertFalse(iterator.hasNext());
        assertThrows(NoSuchElementException.class, iterator::next);
    }
    @Test
    void iteratorsAreIndependent() {
        var v = vector(1, 2, 3);
        Iterator<Double> first = v.iterator();
        Iterator<Double> second = v.iterator();
        first.next();
        first.next();
        assertEquals(1.0, second.next(), 0.0);
    }
    @Test
    void iterationAgreesWithIndexedAccessOnRandomVectors() {
        Random random = new Random(11);
        for (int trial = 0; trial < 100; trial++) {
            var a = new Vector(randomEntries(random, 1 + random.nextInt(8)));
            int index = 0;
            for (double entry : a) {
                assertEquals(a.get(index), entry, 0.0);
                index++;
            }
            assertEquals(a.length(), index);
        }
    }

    // ---------- copy ----------

    @Test
    void copyHasTheSameEntriesAndIsEqualToTheOriginal() {
        var vector = new Vector(new double[]{1, -2, 3.5});
        Vector copy = vector.copy();
        assertVectorEquals(new double[]{1, -2, 3.5}, copy);
        assertEquals(vector, copy);
    }
    @Test
    void copyIsANewInstanceAndChangingReturnedDataDoesNotChangeEitherVector() {
        var vector = new Vector(new double[]{1, 2, 3});
        Vector copy = vector.copy();
        double[] copiedData = copy.data();
        copiedData[0] = 99;
        assertNotSame(vector, copy);
        assertVectorEquals(new double[]{1, 2, 3}, vector);
        assertVectorEquals(new double[]{1, 2, 3}, copy);
    }
    @Test
    void repeatedCopiesPreserveEntries() {
        var vector = new Vector(new double[]{-4, 0, 7});
        assertEquals(vector, vector.copy().copy());
    }

    // ---------- equals and hashCode ----------

    @Test
    void equalsHoldsForVectorsWithTheSameEntries() {
        var vector = new Vector(new double[]{1, 2, 3});
        var same = new Vector(new double[]{1, 2, 3});
        assertEquals(vector, vector);
        assertEquals(vector, same);
        assertEquals(same, vector);
        assertEquals(vector.hashCode(), same.hashCode());
        assertEquals(new Vector(new double[3]), new Vector(3));
    }
    @Test
    void equalsFailsWhenAnySingleEntryDiffers() {
        var vector = new Vector(new double[]{1, 2, 3});
        for (int i = 0; i < 3; i++) {
            double[] changed = {1, 2, 3};
            changed[i] += 1;
            assertNotEquals(vector, new Vector(changed));
            assertNotEquals(new Vector(changed), vector);
        }
    }
    @Test
    void equalsHasNoToleranceForTinyDifferences() {
        assertNotEquals(new Vector(new double[]{1, 2}), new Vector(new double[]{1, Math.nextUp(2.0)}));
    }
    @Test
    void equalsFailsForDifferentLengths() {
        var vector = new Vector(new double[]{1, 2, 3});
        assertNotEquals(new Vector(new double[]{1, 2}), vector);
        assertNotEquals(new Vector(new double[]{1, 2, 3, 4}), vector);
        assertNotEquals(new Vector(new double[]{1}), vector);
    }
    @Test
    void equalsFailsForNullAndForOtherTypes() {
        var vector = new Vector(new double[]{1, 2});
        assertFalse(vector.equals(null));
        assertFalse(vector.equals("[1.0, 2.0]"));
        assertFalse(vector.equals(new double[]{1, 2}));
        assertFalse(vector.equals(new Matrix(new double[][]{{1, 2}})));
    }
    @Test
    void equalsTreatsPositiveAndNegativeZeroAsTheSameEntry() {
        var positive = new Vector(new double[]{0.0, 1, 0.0});
        var negative = new Vector(new double[]{-0.0, 1, -0.0});
        assertEquals(positive, negative);
        assertEquals(negative, positive);
        assertEquals(positive.hashCode(), negative.hashCode());
    }
    @Test
    void equalsAndHashCodeAgreeWithEntrywiseComparisonOnRandomVectors() {
        Random random = new Random(12);
        for (int trial = 0; trial < 500; trial++) {
            int size = 1 + random.nextInt(5);
            double[] data = new double[size];
            for (int i = 0; i < size; i++) {
                data[i] = random.nextInt(5) - 2;
            }
            double[] other = data.clone();
            if (random.nextBoolean()) {
                other[random.nextInt(size)] += 1;
            }
            boolean expected = Arrays.equals(data, other);
            var vector = new Vector(data);
            var otherVector = new Vector(other);
            assertEquals(expected, vector.equals(otherVector));
            assertEquals(expected, otherVector.equals(vector));
            if (expected) {
                assertEquals(vector.hashCode(), otherVector.hashCode());
            }
        }
    }
    @Test
    void equalVectorsAreInterchangeableAsHashKeys() {
        Set<Vector> vectors = new HashSet<>();
        vectors.add(new Vector(new double[]{1, 2, 3}));
        assertTrue(vectors.contains(new Vector(new double[]{1, 2, 3})));
        assertFalse(vectors.contains(new Vector(new double[]{3, 2, 1})));
        vectors.add(new Vector(new double[]{1, 2, 3}));
        assertEquals(1, vectors.size());
    }
}
