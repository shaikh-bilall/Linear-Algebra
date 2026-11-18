package linearAlgebra.domain.rref;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Random;

import static linearAlgebra.TestUtils.assertMatrixEquals;
import static linearAlgebra.TestUtils.assertVectorEquals;
import static linearAlgebra.TestUtils.randomEntries;
import static linearAlgebra.TestUtils.randomScalar;
import static linearAlgebra.TestUtils.sampleData;
import static linearAlgebra.TestUtils.shuffledIndices;
import static linearAlgebra.TestUtils.vector;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RowOperationTest {

    private static final double TOL = 1e-12;

    // ---------- RowSwap: construction ----------

    @Test
    void type1ReportsItsIndices() {
        var swap = new RowSwap(2, 5);
        assertEquals(2, swap.sourceRow());
        assertEquals(5, swap.targetRow());
    }

    @Test
    void type1AcceptsEqualIndices() {
        var swap = new RowSwap(1, 1);
        assertEquals(1, swap.sourceRow());
        assertEquals(1, swap.targetRow());
    }

    @ParameterizedTest(name = "RowSwap({0}, {1}) is rejected")
    @CsvSource({"-1, 0", "0, -1", "-1, -1", "-2147483648, 0"})
    void type1RejectsNegativeIndices(
        final int source,
        final int target
    ) {
        assertThrows(IllegalArgumentException.class, () -> new RowSwap(source, target));
    }

    // ---------- RowScale: construction ----------

    @Test
    void type2ReportsItsRowAndScalar() {
        var scale = new RowScale(3, -0.5);
        assertEquals(3, scale.targetRow());
        assertEquals(-0.5, scale.scalar(), 0.0);
    }

    @ParameterizedTest(name = "row {0} is rejected")
    @ValueSource(ints = {-1, Integer.MIN_VALUE})
    void type2RejectsANegativeRow(
        final int row
    ) {
        assertThrows(IllegalArgumentException.class, () -> new RowScale(row, 2));
    }

    @Test
    void type2RejectsAZeroScalar() {
        assertThrows(IllegalArgumentException.class, () -> new RowScale(0, 0));
    }

    @ParameterizedTest(name = "scalar {0} is rejected")
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void type2RejectsNonFiniteScalars(
        final double scalar
    ) {
        assertThrows(IllegalArgumentException.class, () -> new RowScale(0, scalar));
    }

    // ---------- RowCombine: construction ----------

    @Test
    void type3ReportsItsRowsAndScalar() {
        var combine = new RowCombine(1, 4, 2.5);
        assertEquals(1, combine.sourceRow());
        assertEquals(4, combine.targetRow());
        assertEquals(2.5, combine.scalar(), 0.0);
    }

    @ParameterizedTest(name = "RowCombine({0}, {1}) is rejected")
    @CsvSource({"-1, 0", "0, -1", "-2147483648, 0", "0, -2147483648"})
    void type3RejectsNegativeIndices(
        final int source,
        final int target
    ) {
        assertThrows(IllegalArgumentException.class, () -> new RowCombine(source, target, 2));
    }

    @ParameterizedTest(name = "row {0} as source and target is rejected")
    @ValueSource(ints = {0, 1, 7})
    void type3RejectsTheSameRowAsSourceAndTarget(
        final int row
    ) {
        assertThrows(IllegalArgumentException.class, () -> new RowCombine(row, row, 2));
    }

    @Test
    void type3RejectsAZeroScalar() {
        assertThrows(IllegalArgumentException.class, () -> new RowCombine(0, 1, 0));
    }

    @ParameterizedTest(name = "scalar {0} is rejected")
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void type3RejectsNonFiniteScalars(
        final double scalar
    ) {
        assertThrows(IllegalArgumentException.class, () -> new RowCombine(0, 1, scalar));
    }

    // ---------- all operations: operands ----------

    @Test
    void everyOperationLeavesItsOperandUnchangedAndReturnsANewObject() {
        for (RowOperation operation : operations()) {
            var matrix = new Matrix(sampleData());
            Matrix changedMatrix = operation.apply(matrix);
            assertNotSame(matrix, changedMatrix);
            assertMatrixEquals(sampleData(), matrix);

            var v = vector(1, 2, 3);
            Vector changedVector = operation.apply(v);
            assertNotSame(v, changedVector);
            assertVectorEquals(new double[]{1, 2, 3}, v);
        }
    }

    @Test
    void everyOperationRejectsANullOperand() {
        for (RowOperation operation : operations()) {
            assertThrows(NullPointerException.class, () -> operation.apply((Matrix) null));
            assertThrows(NullPointerException.class, () -> operation.apply((Vector) null));
        }
    }

    // ---------- RowSwap on a matrix ----------

    @Test
    void type1ExchangesTwoRowsAndLeavesTheRest() {
        Matrix result = new RowSwap(0, 2).apply(new Matrix(sampleData()));
        assertMatrixEquals(new double[][]{{5, 6}, {3, 4}, {1, 2}}, result);
    }

    @ParameterizedTest(name = "swap({0}, {1}) equals swap({1}, {0})")
    @CsvSource({"0, 1", "1, 2", "0, 2", "2, 0"})
    void type1IsSymmetricInItsIndicesOnAMatrix(
        final int i,
        final int j
    ) {
        var matrix = new Matrix(sampleData());
        assertMatrixEquals(new RowSwap(j, i).apply(matrix), new RowSwap(i, j).apply(matrix));
    }

    @Test
    void type1AppliedTwiceRestoresTheMatrix() {
        var swap = new RowSwap(0, 2);
        Matrix result = swap.apply(swap.apply(new Matrix(sampleData())));
        assertMatrixEquals(sampleData(), result);
    }

    @Test
    void type1WithEqualIndicesLeavesTheMatrixUnchanged() {
        Matrix result = new RowSwap(1, 1).apply(new Matrix(sampleData()));
        assertMatrixEquals(sampleData(), result);
    }

    @Test
    void type1WorksOnASingleRowMatrix() {
        Matrix result = new RowSwap(0, 0).apply(new Matrix(new double[][]{{1, 2, 3}}));
        assertMatrixEquals(new double[][]{{1, 2, 3}}, result);
    }

    @Test
    void type1MovesWholeRowsOfAWideMatrix() {
        Matrix result = new RowSwap(0, 1).apply(new Matrix(new double[][]{{1, 2, 3, 4}, {5, 6, 7, 8}}));
        assertMatrixEquals(new double[][]{{5, 6, 7, 8}, {1, 2, 3, 4}}, result);
    }

    // ---------- RowSwap on a vector ----------

    @Test
    void type1ExchangesTwoEntriesAndLeavesTheRest() {
        Vector result = new RowSwap(0, 3).apply(vector(1, 2, 3, 4));
        assertVectorEquals(new double[]{4, 2, 3, 1}, result);
    }

    @ParameterizedTest(name = "swap({0}, {1}) equals swap({1}, {0})")
    @CsvSource({"0, 1", "1, 3", "0, 3", "2, 0"})
    void type1IsSymmetricInItsIndicesOnAVector(
        final int i,
        final int j
    ) {
        var v = vector(1, 2, 3, 4);
        assertVectorEquals(new RowSwap(j, i).apply(v).data(), new RowSwap(i, j).apply(v));
    }

    @Test
    void type1AppliedTwiceRestoresTheVector() {
        var swap = new RowSwap(0, 2);
        Vector result = swap.apply(swap.apply(vector(5, -6, 7)));
        assertVectorEquals(new double[]{5, -6, 7}, result);
    }

    @Test
    void type1WithEqualIndicesLeavesTheVectorUnchanged() {
        assertVectorEquals(new double[]{1, 2, 3}, new RowSwap(1, 1).apply(vector(1, 2, 3)));
    }

    @Test
    void type1WorksOnASingleEntryVector() {
        assertVectorEquals(new double[]{9}, new RowSwap(0, 0).apply(vector(9)));
    }

    @Test
    void type1MatchesAReferenceSwapOnRandomVectors() {
        Random random = new Random(3); // fixed seed keeps failures reproducible
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            double[] entries = randomEntries(random, size);
            int i = random.nextInt(size);
            int j = random.nextInt(size);

            double[] expected = entries.clone();
            expected[i] = entries[j];
            expected[j] = entries[i];

            assertVectorEquals(expected, new RowSwap(i, j).apply(new Vector(entries)));
        }
    }

    // ---------- RowSwap: operand fit ----------

    @ParameterizedTest(name = "RowSwap({0}, {1}) does not fit three rows")
    @CsvSource({"3, 0", "0, 3", "3, 3", "100, 0", "0, 2147483647"})
    void type1RejectsIndicesBeyondTheOperand(
        final int source,
        final int target
    ) {
        assertDoesNotFitThreeRows(new RowSwap(source, target));
    }

    // ---------- RowScale on a matrix ----------

    @Test
    void type2ScalesEveryEntryOfTheTargetRowOnly() {
        Matrix result = new RowScale(1, 5).apply(new Matrix(sampleData()));
        assertMatrixEquals(new double[][]{{1, 2}, {15, 20}, {5, 6}}, result);
    }

    @ParameterizedTest(name = "multiply by {0}")
    @ValueSource(doubles = {-3, -0.5, 0.25, 1, 2, 1e6})
    void type2HandlesNegativeFractionalIdentityAndLargeScalarsOnAMatrix(
        final double scalar
    ) {
        Matrix result = new RowScale(1, scalar).apply(new Matrix(sampleData()));
        assertMatrixEquals(new double[][]{{1, 2}, {3 * scalar, 4 * scalar}, {5, 6}}, result);
    }

    @Test
    void type2ByTheReciprocalRestoresTheMatrix() {
        Matrix scaledUp = new RowScale(1, 49).apply(new Matrix(sampleData()));
        assertMatrixEquals(sampleData(), new RowScale(1, 1.0 / 49).apply(scaledUp));
    }

    // ---------- RowScale on a vector ----------

    @Test
    void type2ScalesOnlyTheTargetEntry() {
        Vector result = new RowScale(1, 5).apply(vector(1, 2, 3));
        assertVectorEquals(new double[]{1, 10, 3}, result);
    }

    @ParameterizedTest(name = "multiply by {0}")
    @ValueSource(doubles = {-3, -0.5, 0.25, 1, 2, 1e6})
    void type2HandlesNegativeFractionalIdentityAndLargeScalarsOnAVector(
        final double scalar
    ) {
        Vector result = new RowScale(2, scalar).apply(vector(4, 8, 12));
        assertVectorEquals(new double[]{4, 8, 12 * scalar}, result);
    }

    @Test
    void type2ByTheReciprocalRestoresTheVector() {
        Vector scaledUp = new RowScale(1, 49).apply(vector(3, 7, -2));
        assertVectorEquals(new double[]{3, 7, -2}, new RowScale(1, 1.0 / 49).apply(scaledUp));
    }

    @Test
    void type2AppliedTwiceEqualsApplyingTheProduct() {
        Random random = new Random(5);
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            double[] entries = randomEntries(random, size);
            int index = random.nextInt(size);
            double first = randomScalar(random);
            double second = randomScalar(random);

            Vector once = new RowScale(index, first).apply(new Vector(entries));
            Vector sequential = new RowScale(index, second).apply(once);
            Vector combined = new RowScale(index, first * second).apply(new Vector(entries));
            assertVectorEquals(combined.data(), sequential);
        }
    }

    // ---------- RowScale: operand fit and overflow ----------

    @ParameterizedTest(name = "row {0} does not fit three rows")
    @ValueSource(ints = {3, 100, Integer.MAX_VALUE})
    void type2RejectsARowBeyondTheOperand(
        final int row
    ) {
        assertDoesNotFitThreeRows(new RowScale(row, 2));
    }

    @Test
    void type2RejectsAResultThatOverflows() {
        var scale = new RowScale(0, 10);
        assertThrows(IllegalStateException.class, () -> scale.apply(vector(1e308)));
        assertThrows(IllegalStateException.class, () -> scale.apply(new Matrix(new double[][]{{1e308}})));
    }

    // ---------- RowCombine on a matrix ----------

    @Test
    void type3AddsAMultipleOfTheSourceRowToTheTargetRow() {
        Matrix result = new RowCombine(0, 2, 3).apply(new Matrix(sampleData())); // row 2 += 3 * row 0
        assertMatrixEquals(new double[][]{{1, 2}, {3, 4}, {8, 12}}, result);
    }

    @Test
    void type3WithANegativeScalarEliminatesTheTargetRow() {
        Matrix result = new RowCombine(0, 1, -2).apply(new Matrix(new double[][]{{2, 4}, {4, 8}}));
        assertMatrixEquals(new double[][]{{2, 4}, {0, 0}}, result);
    }

    @Test
    void type3WithAFractionalScalarOnAMatrix() {
        Matrix result = new RowCombine(0, 1, -0.25).apply(new Matrix(new double[][]{{4, 8}, {1, 2}}));
        assertMatrixEquals(new double[][]{{4, 8}, {0, 0}}, result);
    }

    @Test
    void type3NeverChangesTheSourceOrOtherRows() {
        var matrix = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8}});
        Matrix result = new RowCombine(1, 3, 10).apply(matrix);
        assertMatrixEquals(new double[][]{{1, 2}, {3, 4}, {5, 6}, {37, 48}}, result);
    }

    @Test
    void type3AppliesToEveryColumnOfAWideMatrix() {
        Matrix result = new RowCombine(0, 1, -4).apply(new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}}));
        assertMatrixEquals(new double[][]{{1, 2, 3}, {0, -3, -6}}, result);
    }

    @Test
    void type3IsUndoneByTheNegatedScalarOnAMatrix() {
        Matrix combined = new RowCombine(0, 2, 4.5).apply(new Matrix(sampleData()));
        assertMatrixEquals(sampleData(), new RowCombine(0, 2, -4.5).apply(combined));
    }

    @Test
    void type3WithAZeroSourceRowLeavesTheTargetUnchanged() {
        Matrix result = new RowCombine(0, 1, 7).apply(new Matrix(new double[][]{{0, 0}, {3, 4}}));
        assertMatrixEquals(new double[][]{{0, 0}, {3, 4}}, result);
    }

    // ---------- RowCombine on a vector ----------

    @Test
    void type3AddsAMultipleOfTheSourceEntryToTheTargetEntry() {
        Vector result = new RowCombine(0, 2, 3).apply(vector(2, 5, 7)); // v[2] += 3 * v[0]
        assertVectorEquals(new double[]{2, 5, 13}, result);
    }

    @Test
    void type3WithANegativeScalarEliminatesTheTargetEntry() {
        Vector result = new RowCombine(0, 1, -2).apply(vector(2, 4)); // v[1] += -2 * v[0]
        assertVectorEquals(new double[]{2, 0}, result);
    }

    @Test
    void type3WithAFractionalScalarOnAVector() {
        Vector result = new RowCombine(0, 1, -0.25).apply(vector(4, 1));
        assertVectorEquals(new double[]{4, 0}, result);
    }

    @Test
    void type3NeverChangesTheSourceOrOtherEntries() {
        Vector result = new RowCombine(1, 3, 10).apply(vector(1, 2, 3, 4, 5));
        assertVectorEquals(new double[]{1, 2, 3, 24, 5}, result);
    }

    @Test
    void type3IsUndoneByTheNegatedScalarOnAVector() {
        Vector combined = new RowCombine(0, 2, 4.5).apply(vector(3, -1, 8));
        assertVectorEquals(new double[]{3, -1, 8}, new RowCombine(0, 2, -4.5).apply(combined));
    }

    @Test
    void type3WithAZeroSourceEntryLeavesTheTargetUnchanged() {
        Vector result = new RowCombine(0, 2, 3).apply(vector(0, 5, 7));
        assertVectorEquals(new double[]{0, 5, 7}, result);
    }

    @Test
    void reducingByMinusTargetOverSourceEliminatesTheTarget() {
        Random random = new Random(13);
        for (int trial = 0; trial < 200; trial++) {
            int size = 2 + random.nextInt(7);
            double[] entries = randomEntries(random, size);
            List<Integer> picks = shuffledIndices(random, size);
            int source = picks.get(0);
            int target = picks.get(1);
            if (entries[source] == 0 || entries[target] == 0) {
                continue;
            }
            var reduction = new RowCombine(source, target, -entries[target] / entries[source]);
            Vector result = reduction.apply(new Vector(entries));
            assertEquals(0.0, result.get(target), TOL, "trial " + trial);
        }
    }

    @Test
    void twoReductionsEqualOneReductionByTheSumOfTheScalars() {
        Random random = new Random(9);
        for (int trial = 0; trial < 200; trial++) {
            int size = 2 + random.nextInt(7);
            double[] entries = randomEntries(random, size);
            List<Integer> picks = shuffledIndices(random, size);
            int source = picks.get(0);
            int target = picks.get(1);
            double first = randomScalar(random);
            double second = randomScalar(random);
            if (first + second == 0) {
                continue;
            }

            Vector once = new RowCombine(source, target, first).apply(new Vector(entries));
            Vector sequential = new RowCombine(source, target, second).apply(once);
            Vector combined = new RowCombine(source, target, first + second).apply(new Vector(entries));
            assertVectorEquals(combined.data(), sequential);
        }
    }

    @Test
    void reductionsFromTheSameSourceCommute() {
        Random random = new Random(11);
        for (int trial = 0; trial < 200; trial++) {
            int size = 3 + random.nextInt(5);
            double[] entries = randomEntries(random, size);
            List<Integer> picks = shuffledIndices(random, size);
            int source = picks.get(0);
            var first = new RowCombine(source, picks.get(1), randomScalar(random));
            var second = new RowCombine(source, picks.get(2), randomScalar(random));

            Vector forward = second.apply(first.apply(new Vector(entries)));
            Vector backward = first.apply(second.apply(new Vector(entries)));
            assertVectorEquals(backward.data(), forward);
        }
    }

    // ---------- RowCombine: operand fit and overflow ----------

    @ParameterizedTest(name = "RowCombine({0}, {1}) does not fit three rows")
    @CsvSource({"3, 0", "0, 3", "3, 4", "100, 0", "0, 2147483647"})
    void type3RejectsIndicesBeyondTheOperand(
        final int source,
        final int target
    ) {
        assertDoesNotFitThreeRows(new RowCombine(source, target, 2));
    }

    @Test
    void type3RejectsAResultThatOverflows() {
        var combine = new RowCombine(0, 1, 10);
        assertThrows(IllegalStateException.class, () -> combine.apply(vector(1e308, 1e308)));
        assertThrows(IllegalStateException.class,
            () -> combine.apply(new Matrix(new double[][]{{1e308}, {1e308}})));
    }

    // ---------- operations used together ----------

    // The operation log RREF produces for A = [[2,1],[1,3]]. Replaying it on b must give x with Ax = b,
    // which is the whole reason the operations can be applied to vectors.
    @Test
    void replayingTheRrefOperationsOnBSolvesAx_equals_b() {
        Vector b = vector(3, 5);
        b = new RowScale(0, 0.5).apply(b);
        b = new RowCombine(0, 1, -1).apply(b);
        b = new RowScale(1, 0.4).apply(b);
        b = new RowCombine(1, 0, -0.5).apply(b);

        assertVectorEquals(new double[]{0.8, 1.4}, b);
        assertEquals(3, 2 * b.get(0) + 1 * b.get(1), 1e-9); // row 1 of A * x
        assertEquals(5, 1 * b.get(0) + 3 * b.get(1), 1e-9); // row 2 of A * x
    }

    // Every elementary operation has an inverse, and applying both to a vector must give the vector back.
    @Test
    void everyOperationIsUndoneByItsInverse() {
        Random random = new Random(17);
        for (int trial = 0; trial < 200; trial++) {
            int size = 2 + random.nextInt(5);
            double[] entries = randomEntries(random, size);
            RowOperation operation = randomOperation(random, size);

            Vector changed = operation.apply(new Vector(entries));
            assertVectorEquals(entries, inverseOf(operation).apply(changed));
        }
    }

    // ---------- helpers ----------

    // One operation of each type, all valid for a matrix with three rows and a vector with three entries.
    private static List<RowOperation> operations() {
        return List.of(new RowSwap(0, 2), new RowScale(1, 3), new RowCombine(0, 2, -2));
    }

    // Asserts that both a three-row matrix and a three-entry vector reject the operation and stay unchanged.
    private static void assertDoesNotFitThreeRows(
        final RowOperation operation
    ) {
        var matrix = new Matrix(sampleData());
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> operation.apply(matrix));
        assertThrows(IllegalArgumentException.class, () -> operation.apply(v));
        assertMatrixEquals(sampleData(), matrix);
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }

    private static RowOperation randomOperation(
        final Random random,
        final int n
    ) {
        double scalar = randomScalar(random);
        int i = random.nextInt(n);
        int j = (i + 1 + random.nextInt(n - 1)) % n; // always different from i
        return switch (random.nextInt(3)) {
            case 0 -> new RowSwap(i, j);
            case 1 -> new RowScale(i, scalar);
            default -> new RowCombine(i, j, scalar);
        };
    }

    private static RowOperation inverseOf(
        final RowOperation operation
    ) {
        if (operation instanceof RowSwap swap) {
            return swap; // a swap is its own inverse
        } else if (operation instanceof RowScale scale) {
            return new RowScale(scale.targetRow(), 1.0 / scale.scalar());
        }
        RowCombine reduce = (RowCombine) operation;
        return new RowCombine(reduce.sourceRow(), reduce.targetRow(), -reduce.scalar());
    }
}
