package ca.umanitoba.cs.shaikhb2.domain;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {

    private static final double TOL = 1e-12;
    private static final double[] SCALARS = {-3, -2, -1, -0.5, 0.5, 1, 2, 3};

    // ---------- constructor ----------

    @Test
    void constructorStoresTheEntriesAndReportsTheShape() {
        var matrix = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertEquals(2, matrix.rows());
        assertEquals(3, matrix.cols());
        assertMatrixEquals(new double[][]{{1, 2, 3}, {4, 5, 6}}, matrix);
    }
    @Test
    void constructorAcceptsASingleEntry() {
        assertMatrixEquals(new double[][]{{7}}, new Matrix(new double[][]{{7}}));
    }
    @Test
    void constructorCopiesEveryRowOfTheArray() {
        double[][] source = {{1, 2}, {3, 4}};
        var matrix = new Matrix(source);
        source[0][0] = 99;
        source[1] = new double[]{7, 7};
        assertMatrixEquals(new double[][]{{1, 2}, {3, 4}}, matrix);
    }
    @Test
    void constructorRejectsANullArrayAndANullRow() {
        assertThrows(NullPointerException.class, () -> new Matrix(null));
        assertThrows(NullPointerException.class, () -> new Matrix(new double[][]{{1, 2}, null}));
        assertThrows(NullPointerException.class, () -> new Matrix(new double[][]{null, {1, 2}}));
    }
    @Test
    void constructorRejectsAMatrixWithNoRowsOrNoColumns() {
        assertThrows(IllegalStateException.class, () -> new Matrix(new double[0][0]));
        assertThrows(IllegalStateException.class, () -> new Matrix(new double[2][0]));
    }
    @Test
    void constructorRejectsRowsOfDifferentLengths() {
        assertThrows(IllegalStateException.class, () -> new Matrix(new double[][]{{1, 2}, {3}}));
        assertThrows(IllegalStateException.class, () -> new Matrix(new double[][]{{1}, {2, 3}}));
    }
    @ParameterizedTest(name = "entry {0} is rejected")
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void constructorRejectsNonFiniteEntries(
            double entry
    ) {
        assertThrows(IllegalStateException.class, () -> new Matrix(new double[][]{{1, entry}, {3, 4}}));
        assertThrows(IllegalStateException.class, () -> new Matrix(new double[][]{{1, 2}, {3, entry}}));
        assertThrows(IllegalStateException.class, () -> new Matrix(new double[][]{{entry}}));
    }

    // ---------- rows, cols and data ----------

    @ParameterizedTest(name = "{0}x{1}")
    @CsvSource({"1, 1", "1, 4", "4, 1", "3, 3", "2, 5"})
    void rowsAndColsReportTheShape(
            int rows,
            int cols
    ) {
        var matrix = new Matrix(new double[rows][cols]);
        assertEquals(rows, matrix.rows());
        assertEquals(cols, matrix.cols());
    }
    // The logic classes mutate matrices through data(), so the accessor must hand out the live array.
    @Test
    void dataExposesTheLiveStorage() {
        var matrix = new Matrix(sampleData());
        assertSame(matrix.data(), matrix.data());
        matrix.data()[1][0] = 30;
        assertEquals(30.0, matrix.get(1, 0), 0.0);
    }

    // ---------- get ----------

    @Test
    void getReturnsEachEntry() {
        var matrix = new Matrix(sampleData());
        double[][] expected = sampleData();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expected[i][j], matrix.get(i, j), 0.0);
            }
        }
    }
    @ParameterizedTest(name = "get({0}, {1}) is rejected")
    @CsvSource({"-1, 0", "0, -1", "3, 0", "0, 2", "-1, -1", "3, 2", "-2147483648, 0", "0, 2147483647"})
    void getRejectsOutOfBoundsIndices(
            int row,
            int col
    ) {
        var matrix = new Matrix(sampleData());
        assertThrows(IllegalStateException.class, () -> matrix.get(row, col));
    }

    // ---------- getRow ----------

    @Test
    void getRowReturnsTheRequestedRow() {
        var matrix = new Matrix(sampleData());
        assertVectorEquals(new double[]{1, 2}, matrix.getRow(0));
        assertVectorEquals(new double[]{3, 4}, matrix.getRow(1));
        assertVectorEquals(new double[]{5, 6}, matrix.getRow(2));
    }
    @Test
    void getRowReturnsACopyOfTheRow() {
        var matrix = new Matrix(sampleData());
        Vector row = matrix.getRow(1);
        row.data()[0] = 99;
        assertMatrixEquals(sampleData(), matrix);
    }
    @Test
    void getRowAgreesWithGetOnRandomMatrices() {
        Random random = new Random(1); // fixed seed keeps failures reproducible
        for (int trial = 0; trial < 100; trial++) {
            var matrix = new Matrix(randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5));
            for (int i = 0; i < matrix.rows(); i++) {
                Vector row = matrix.getRow(i);
                assertEquals(matrix.cols(), row.length());
                for (int j = 0; j < matrix.cols(); j++) {
                    assertEquals(matrix.get(i, j), row.get(j), 0.0);
                }
            }
        }
    }
    @ParameterizedTest(name = "row {0} is rejected")
    @ValueSource(ints = {-1, 3, 100, Integer.MIN_VALUE})
    void getRowRejectsOutOfBoundsIndex(
            int row
    ) {
        var matrix = new Matrix(sampleData());
        assertThrows(IllegalStateException.class, () -> matrix.getRow(row));
    }

    // ---------- copy ----------

    @Test
    void copyHasTheSameShapeAndEntries() {
        var matrix = new Matrix(sampleData());
        Matrix copy = matrix.copy();
        assertNotSame(matrix, copy);
        assertMatrixEquals(sampleData(), copy);
    }
    @Test
    void copyIsIndependentOfTheOriginal() {
        var matrix = new Matrix(sampleData());
        Matrix copy = matrix.copy();
        copy.data()[0][0] = 99;
        assertMatrixEquals(sampleData(), matrix);
        matrix.data()[2][1] = -99;
        assertEquals(6.0, copy.get(2, 1), 0.0);
    }
    @Test
    void copyMatchesTheOriginalOnRandomMatrices() {
        Random random = new Random(2);
        for (int trial = 0; trial < 100; trial++) {
            var matrix = new Matrix(randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5));
            assertMatrixEquals(matrix, matrix.copy().copy());
        }
    }

    // ---------- transpose ----------

    @Test
    void transposeSwapsRowsAndColumns() {
        var transposed = new Matrix(sampleData()).transpose();
        assertEquals(2, transposed.rows());
        assertEquals(3, transposed.cols());
        assertMatrixEquals(new double[][]{{1, 3, 5}, {2, 4, 6}}, transposed);
    }
    @Test
    void transposeTurnsARowIntoAColumnAndBack() {
        var row = new Matrix(new double[][]{{1, 2, 3}});
        assertMatrixEquals(new double[][]{{1}, {2}, {3}}, row.transpose());
        assertMatrixEquals(new double[][]{{1, 2, 3}}, row.transpose().transpose());
    }
    @Test
    void transposeOfASymmetricMatrixIsItself() {
        var symmetric = new Matrix(new double[][]{{1, 2, 3}, {2, 4, 5}, {3, 5, 6}});
        assertMatrixEquals(symmetric, symmetric.transpose());
    }
    @Test
    void transposeIsItsOwnInverseAndMovesEveryEntry() {
        Random random = new Random(3);
        for (int trial = 0; trial < 200; trial++) {
            var matrix = new Matrix(randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5));
            Matrix transposed = matrix.transpose();
            assertEquals(matrix.cols(), transposed.rows());
            assertEquals(matrix.rows(), transposed.cols());
            for (int i = 0; i < matrix.rows(); i++) {
                for (int j = 0; j < matrix.cols(); j++) {
                    assertEquals(matrix.get(i, j), transposed.get(j, i), 0.0);
                }
            }
            assertMatrixEquals(matrix, transposed.transpose());
        }
    }
    // (A^T x) . y = x . (A y) ties transpose to multiply and dot without re-implementing either.
    @Test
    void transposeMovesAcrossTheDotProduct() {
        Random random = new Random(4);
        for (int trial = 0; trial < 200; trial++) {
            int rows = 1 + random.nextInt(5);
            int cols = 1 + random.nextInt(5);
            var matrix = new Matrix(randomData(random, rows, cols, 5));
            var x = new Vector(randomEntries(random, rows));
            var y = new Vector(randomEntries(random, cols));
            assertEquals(matrix.transpose().multiply(x).dot(y), x.dot(matrix.multiply(y)), TOL);
        }
    }
    @Test
    void transposeLeavesTheOriginalUnchangedAndReturnsAnIndependentMatrix() {
        var matrix = new Matrix(sampleData());
        Matrix transposed = matrix.transpose();
        transposed.data()[0][0] = 99;
        assertMatrixEquals(sampleData(), matrix);
    }

    // ---------- multiply (matrix times vector) ----------

    @Test
    void multiplyComputesAKnownProduct() {
        var matrix = new Matrix(new double[][]{{1, 2}, {3, 4}});
        assertVectorEquals(new double[]{17, 39}, matrix.multiply(vector(5, 6)));
    }
    @Test
    void multiplyOfARectangularMatrixHasOneEntryPerRow() {
        var wide = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertVectorEquals(new double[]{-2, -2}, wide.multiply(vector(1, 0, -1)));
        assertVectorEquals(new double[]{5, 11, 17}, new Matrix(sampleData()).multiply(vector(1, 2)));
    }
    @Test
    void multiplyByTheIdentityAndByZeroBehaveAsExpected() {
        Random random = new Random(5);
        for (int trial = 0; trial < 100; trial++) {
            int n = 1 + random.nextInt(6);
            var v = new Vector(randomEntries(random, n));
            assertVectorEquals(v.data(), identity(n).multiply(v));
            assertVectorEquals(new double[n], new Matrix(new double[n][n]).multiply(v));
        }
    }
    @Test
    void multiplyIsLinear() {
        Random random = new Random(6);
        for (int trial = 0; trial < 200; trial++) {
            int rows = 1 + random.nextInt(5);
            int cols = 1 + random.nextInt(5);
            var matrix = new Matrix(randomData(random, rows, cols, 5));
            var x = new Vector(randomEntries(random, cols));
            var y = new Vector(randomEntries(random, cols));
            double s = randomScalar(random);
            assertVectorEquals(matrix.multiply(x).add(matrix.multiply(y)).data(), matrix.multiply(x.add(y)));
            assertVectorEquals(matrix.multiply(x).scale(s).data(), matrix.multiply(x.scale(s)));
        }
    }
    @Test
    void multiplyChangesNeitherOperand() {
        var matrix = new Matrix(sampleData());
        var v = vector(1, 2);
        matrix.multiply(v);
        assertMatrixEquals(sampleData(), matrix);
        assertVectorEquals(new double[]{1, 2}, v);
    }
    @Test
    void multiplyRejectsAVectorOfTheWrongLength() {
        var matrix = new Matrix(sampleData());
        assertThrows(IllegalStateException.class, () -> matrix.multiply(vector(1, 2, 3)));
        assertThrows(IllegalStateException.class, () -> matrix.multiply(vector(1)));
    }
    @Test
    void multiplyRejectsANullVector() {
        assertThrows(NullPointerException.class, () -> new Matrix(sampleData()).multiply(null));
    }
    @Test
    void multiplyRejectsAProductThatOverflows() {
        var huge = new Matrix(new double[][]{{1e308, 1e308}});
        assertThrows(IllegalStateException.class, () -> huge.multiply(vector(10, 10)));
    }

    // ---------- merge ----------

    @Test
    void mergeAppendsTheColumnsOfTheOtherMatrix() {
        var left = new Matrix(new double[][]{{1, 2}, {3, 4}});
        var right = new Matrix(new double[][]{{5}, {6}});
        Matrix merged = left.merge(right);
        assertEquals(2, merged.rows());
        assertEquals(3, merged.cols());
        assertMatrixEquals(new double[][]{{1, 2, 5}, {3, 4, 6}}, merged);
    }
    @Test
    void mergeWithItselfRepeatsTheColumns() {
        var matrix = new Matrix(new double[][]{{1, 2}, {3, 4}});
        assertMatrixEquals(new double[][]{{1, 2, 1, 2}, {3, 4, 3, 4}}, matrix.merge(matrix));
    }
    @Test
    void mergeIsAssociativeAndAddsTheColumns() {
        Random random = new Random(7);
        for (int trial = 0; trial < 200; trial++) {
            int rows = 1 + random.nextInt(5);
            var a = new Matrix(randomData(random, rows, 1 + random.nextInt(4), 5));
            var b = new Matrix(randomData(random, rows, 1 + random.nextInt(4), 5));
            var c = new Matrix(randomData(random, rows, 1 + random.nextInt(4), 5));
            Matrix merged = a.merge(b).merge(c);
            assertEquals(rows, merged.rows());
            assertEquals(a.cols() + b.cols() + c.cols(), merged.cols());
            assertMatrixEquals(merged, a.merge(b.merge(c)));
        }
    }
    @Test
    void mergeLeavesBothOperandsUnchangedAndReturnsAnIndependentMatrix() {
        var left = new Matrix(new double[][]{{1, 2}, {3, 4}});
        var right = new Matrix(new double[][]{{5}, {6}});
        Matrix merged = left.merge(right);
        merged.data()[0][0] = 99;
        assertMatrixEquals(new double[][]{{1, 2}, {3, 4}}, left);
        assertMatrixEquals(new double[][]{{5}, {6}}, right);
    }
    @Test
    void mergeRejectsMatricesWithDifferentRowCounts() {
        var matrix = new Matrix(sampleData());
        assertThrows(IllegalStateException.class, () -> matrix.merge(new Matrix(new double[][]{{1}, {2}})));
        assertThrows(IllegalStateException.class, () -> new Matrix(new double[][]{{1}}).merge(matrix));
    }
    @Test
    void mergeRejectsANullMatrix() {
        assertThrows(NullPointerException.class, () -> new Matrix(sampleData()).merge(null));
    }

    // ---------- permuteRows ----------

    @Test
    void permuteRowsReordersTheRows() {
        var permuted = new Matrix(sampleData()).permuteRows(List.of(2, 0, 1));
        assertMatrixEquals(new double[][]{{5, 6}, {1, 2}, {3, 4}}, permuted);
    }
    @Test
    void permuteRowsWithTheIdentityOrderChangesNothing() {
        assertMatrixEquals(sampleData(), new Matrix(sampleData()).permuteRows(List.of(0, 1, 2)));
    }
    @Test
    void permuteRowsIsUndoneByTheInversePermutation() {
        Random random = new Random(8);
        for (int trial = 0; trial < 200; trial++) {
            int rows = 1 + random.nextInt(6);
            var matrix = new Matrix(randomData(random, rows, 1 + random.nextInt(5), 5));
            List<Integer> order = shuffledIndices(random, rows);
            assertMatrixEquals(matrix, matrix.permuteRows(order).permuteRows(inverseOf(order)));
        }
    }
    @Test
    void permuteRowsMovesWholeRowsAndLeavesTheOriginalUnchanged() {
        Random random = new Random(9);
        for (int trial = 0; trial < 200; trial++) {
            int rows = 1 + random.nextInt(6);
            double[][] data = randomData(random, rows, 1 + random.nextInt(5), 5);
            var matrix = new Matrix(data);
            List<Integer> order = shuffledIndices(random, rows);
            Matrix permuted = matrix.permuteRows(order);
            for (int i = 0; i < rows; i++) {
                assertVectorEquals(data[order.get(i)], permuted.getRow(i));
            }
            assertMatrixEquals(data, matrix);
        }
    }
    @Test
    void permuteRowsRejectsAListOfTheWrongSize() {
        var matrix = new Matrix(sampleData());
        assertThrows(IllegalStateException.class, () -> matrix.permuteRows(List.of(0, 1)));
        assertThrows(IllegalStateException.class, () -> matrix.permuteRows(List.of(0, 1, 2, 0)));
    }
    @Test
    void permuteRowsRejectsAListThatIsNotAPermutation() {
        var matrix = new Matrix(sampleData());
        assertThrows(IllegalStateException.class, () -> matrix.permuteRows(List.of(0, 0, 1))); // repeated index
        assertThrows(IllegalStateException.class, () -> matrix.permuteRows(List.of(-1, 0, 1))); // negative index
        assertThrows(IllegalStateException.class, () -> matrix.permuteRows(List.of(0, 1, 3))); // index too large
    }
    @Test
    void permuteRowsRejectsANullListAndANullIndex() {
        var matrix = new Matrix(sampleData());
        assertThrows(NullPointerException.class, () -> matrix.permuteRows(null));
        assertThrows(NullPointerException.class, () -> matrix.permuteRows(Arrays.asList(0, null, 2)));
    }

    // ---------- equals and hashCode ----------

    @Test
    void equalsHoldsForMatricesWithTheSameEntries() {
        var matrix = new Matrix(sampleData());
        var same = new Matrix(sampleData());
        assertEquals(matrix, matrix);
        assertEquals(matrix, same);
        assertEquals(same, matrix);
        assertEquals(matrix.hashCode(), same.hashCode());
    }
    @Test
    void equalsIsTransitive() {
        var first = new Matrix(sampleData());
        var second = new Matrix(sampleData());
        var third = new Matrix(sampleData());
        assertEquals(first, second);
        assertEquals(second, third);
        assertEquals(first, third);
    }
    @Test
    void equalsFailsWhenAnySingleEntryDiffers() {
        var matrix = new Matrix(sampleData());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                double[][] changed = sampleData();
                changed[i][j] += 1;
                assertNotEquals(matrix, new Matrix(changed));
                assertNotEquals(new Matrix(changed), matrix);
            }
        }
    }
    @Test
    void equalsHasNoToleranceForTinyDifferences() {
        double[][] changed = sampleData();
        changed[0][0] = Math.nextUp(changed[0][0]);
        assertNotEquals(new Matrix(sampleData()), new Matrix(changed));
    }
    @Test
    void equalsFailsForDifferentDimensions() {
        var matrix = new Matrix(sampleData()); // 3x2
        assertNotEquals(matrix, new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}})); // 2x3
        assertNotEquals(matrix, new Matrix(new double[][]{{1, 2}, {3, 4}})); // fewer rows, same leading entries
        assertNotEquals(matrix, new Matrix(new double[][]{{1}, {3}, {5}})); // fewer columns, same leading entries
        assertNotEquals(new Matrix(new double[][]{{1, 2, 3, 4}}), new Matrix(new double[][]{{1, 2}, {3, 4}})); // same entry count
    }
    @Test
    void equalsFailsForNullAndForOtherTypes() {
        var matrix = new Matrix(new double[][]{{1, 2}});
        assertFalse(matrix.equals(null));
        assertFalse(matrix.equals("[[1.0, 2.0]]"));
        assertFalse(matrix.equals(new Vector(new double[]{1, 2})));
        assertFalse(matrix.equals(new double[][]{{1, 2}}));
    }
    @Test
    void equalsTreatsPositiveAndNegativeZeroAsTheSameEntry() {
        var positive = new Matrix(new double[][]{{0.0, 1}, {2, 0.0}});
        var negative = new Matrix(new double[][]{{-0.0, 1}, {2, -0.0}});
        assertEquals(positive, negative);
        assertEquals(negative, positive);
        assertEquals(positive.hashCode(), negative.hashCode());
    }
    // Scaling a row of zeros by a negative factor leaves -0.0 entries, which must not make a matrix differ.
    @Test
    void aMatrixWithNegativeZeroEntriesEqualsTheIdentity() {
        assertEquals(identity(2), new Matrix(new double[][]{{1, -0.0}, {-0.0, 1}}));
    }
    @Test
    void equalsHoldsBetweenAMatrixAndItsCopyAndDoubleTranspose() {
        Random random = new Random(10);
        for (int trial = 0; trial < 100; trial++) {
            var matrix = new Matrix(randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5));
            assertEquals(matrix, matrix.copy());
            assertEquals(matrix, matrix.transpose().transpose());
            assertEquals(matrix.hashCode(), matrix.copy().hashCode());
        }
    }
    @Test
    void equalsAndHashCodeAgreeWithEntrywiseComparisonOnRandomMatrices() {
        Random random = new Random(11);
        for (int trial = 0; trial < 500; trial++) {
            int rows = 1 + random.nextInt(4);
            int cols = 1 + random.nextInt(4);
            double[][] data = randomData(random, rows, cols, 2);
            double[][] other = new double[rows][];
            for (int i = 0; i < rows; i++) {
                other[i] = data[i].clone();
            }
            if (random.nextBoolean()) {
                other[random.nextInt(rows)][random.nextInt(cols)] += 1;
            }
            boolean expected = Arrays.deepEquals(data, other);
            var matrix = new Matrix(data);
            var otherMatrix = new Matrix(other);
            assertEquals(expected, matrix.equals(otherMatrix));
            assertEquals(expected, otherMatrix.equals(matrix));
            if (expected) {
                assertEquals(matrix.hashCode(), otherMatrix.hashCode());
            }
        }
    }
    @Test
    void equalMatricesAreInterchangeableAsHashKeys() {
        Set<Matrix> matrices = new HashSet<>();
        matrices.add(new Matrix(sampleData()));
        assertTrue(matrices.contains(new Matrix(sampleData())));
        assertFalse(matrices.contains(identity(2)));
        matrices.add(new Matrix(sampleData()));
        assertEquals(1, matrices.size());
    }
    @Test
    void equalityOnACorruptedMatrixIsRejected() {
        var healthy = new Matrix(sampleData());
        var withNaN = new Matrix(sampleData());
        withNaN.data()[0][0] = Double.NaN;
        assertThrows(IllegalStateException.class, () -> withNaN.equals(healthy));
        assertThrows(IllegalStateException.class, () -> healthy.equals(withNaN));
        assertThrows(IllegalStateException.class, () -> withNaN.equals(withNaN));
        assertThrows(IllegalStateException.class, withNaN::hashCode);

        var withEmptyRow = new Matrix(sampleData());
        withEmptyRow.data()[0] = new double[0];
        assertThrows(IllegalStateException.class, () -> withEmptyRow.equals(healthy));
        assertThrows(IllegalStateException.class, () -> healthy.equals(withEmptyRow));
        assertThrows(IllegalStateException.class, withEmptyRow::hashCode);
    }

    // ---------- corrupted state ----------

    // data() hands out the live arrays, so a test can break what the constructor guarantees.
    // Every instance method has to notice instead of passing a NaN along.
    @Test
    void operationsOnACorruptedMatrixAreRejected() {
        var corrupted = new Matrix(sampleData());
        corrupted.data()[0][0] = Double.NaN;
        assertThrows(IllegalStateException.class, () -> corrupted.get(0, 0));
        assertThrows(IllegalStateException.class, () -> corrupted.getRow(0));
        assertThrows(IllegalStateException.class, corrupted::copy);
        assertThrows(IllegalStateException.class, corrupted::transpose);
        assertThrows(IllegalStateException.class, () -> corrupted.multiply(vector(1, 1)));
        assertThrows(IllegalStateException.class, () -> corrupted.merge(new Matrix(new double[][]{{1}, {2}, {3}})));
        assertThrows(IllegalStateException.class, () -> corrupted.permuteRows(List.of(0, 1, 2)));
    }
    @Test
    void aMatrixWhoseFirstRowBecameEmptyReportsNoColumns() {
        var corrupted = new Matrix(sampleData());
        corrupted.data()[0] = new double[0];
        assertThrows(IllegalStateException.class, corrupted::cols);
        assertThrows(IllegalStateException.class, corrupted::transpose);
    }

    // ---------- helpers ----------

    private static double randomScalar(
            Random random
    ) {
        return SCALARS[random.nextInt(SCALARS.length)];
    }
    // Integer entries in [-bound, bound].
    private static double[][] randomData(
            Random random,
            int rows,
            int cols,
            int bound
    ) {
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = random.nextInt(2 * bound + 1) - bound;
            }
        }
        return data;
    }
    // Integer entries in [-10, 10].
    private static double[] randomEntries(
            Random random,
            int size
    ) {
        double[] entries = new double[size];
        for (int i = 0; i < size; i++) {
            entries[i] = random.nextInt(21) - 10;
        }
        return entries;
    }
    private static List<Integer> shuffledIndices(
            Random random,
            int size
    ) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        return indices;
    }
    // Permuting the rows by p and then by the inverse of p must restore the original order.
    private static List<Integer> inverseOf(
            List<Integer> permutation
    ) {
        Integer[] inverse = new Integer[permutation.size()];
        for (int i = 0; i < permutation.size(); i++) {
            inverse[permutation.get(i)] = i;
        }
        return Arrays.asList(inverse);
    }
    // A fresh 3x2 array each call, so tests can mutate one copy and compare against another.
    private static double[][] sampleData() {
        return new double[][]{{1, 2}, {3, 4}, {5, 6}};
    }
    private static Matrix identity(
            int n
    ) {
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            data[i][i] = 1;
        }
        return new Matrix(data);
    }
    private static Vector vector(
            double... values
    ) {
        return new Vector(values);
    }
    private static void assertVectorEquals(
            double[] expected,
            Vector actual
    ) {
        assertEquals(expected.length, actual.length(), "vector length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual.get(i), TOL, "mismatch at index " + i);
        }
    }
    private static void assertMatrixEquals(
            double[][] expected,
            Matrix actual
    ) {
        assertMatrixEquals(new Matrix(expected), actual);
    }
    private static void assertMatrixEquals(
            Matrix expected,
            Matrix actual
    ) {
        assertEquals(expected.rows(), actual.rows(), "row count");
        assertEquals(expected.cols(), actual.cols(), "column count");
        for (int i = 0; i < expected.rows(); i++) {
            for (int j = 0; j < expected.cols(); j++) {
                assertEquals(expected.get(i, j), actual.get(i, j), TOL,
                        "mismatch at (" + i + ", " + j + ")");
            }
        }
    }
}