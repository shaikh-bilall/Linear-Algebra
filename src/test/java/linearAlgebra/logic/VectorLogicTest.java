package linearAlgebra.logic;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.RREF.OperationRREF;
import linearAlgebra.domain.RREF.Type1;
import linearAlgebra.domain.RREF.Type2;
import linearAlgebra.domain.RREF.Type3;
import linearAlgebra.domain.Vector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorLogicTest {

    private static final double TOL = 1e-12;
    private static final double[] SCALARS = {-3, -2, -1, -0.5, 0.5, 1, 2, 3};

    // ---------- swapElements ----------

    @Test
    void swapElementsExchangesTwoEntriesAndLeavesTheRest() {
        var v = vector(1, 2, 3, 4);
        VectorLogic.swapElements(v, 0, 3);
        assertVectorEquals(new double[]{4, 2, 3, 1}, v);
    }
    @ParameterizedTest(name = "swap({0}, {1}) equals swap({1}, {0})")
    @CsvSource({"0, 1", "1, 3", "0, 3", "2, 0"})
    void swapElementsIsSymmetricInItsArguments(
            int i,
            int j
    ) {
        var forward = vector(1, 2, 3, 4);
        var backward = vector(1, 2, 3, 4);
        VectorLogic.swapElements(forward, i, j);
        VectorLogic.swapElements(backward, j, i);
        assertVectorEquals(backward.data(), forward);
    }
    @Test
    void swapElementsTwiceRestoresTheOriginal() {
        var v = vector(5, -6, 7);
        VectorLogic.swapElements(v, 0, 2);
        VectorLogic.swapElements(v, 0, 2);
        assertVectorEquals(new double[]{5, -6, 7}, v);
    }
    @Test
    void swapElementsWithTheSameIndexIsANoOp() {
        var v = vector(1, 2, 3);
        VectorLogic.swapElements(v, 1, 1);
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @Test
    void swapElementsWorksOnASingleElementVector() {
        var v = vector(9);
        VectorLogic.swapElements(v, 0, 0);
        assertVectorEquals(new double[]{9}, v);
    }
    @Test
    void swapElementsMatchesAReferenceSwapOnRandomVectors() {
        Random random = new Random(3); // fixed seed keeps failures reproducible
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            double[] entries = randomEntries(random, size);
            int i = random.nextInt(size);
            int j = random.nextInt(size);

            double[] expected = entries.clone();
            expected[i] = entries[j];
            expected[j] = entries[i];

            var v = new Vector(entries);
            VectorLogic.swapElements(v, i, j);
            assertVectorEquals(expected, v);
        }
    }
    @ParameterizedTest(name = "swap({0}, {1}) is rejected and leaves the vector unchanged")
    @CsvSource({"-1, 0", "0, -1", "4, 0", "0, 4", "-1, -1", "4, 4"})
    void swapElementsRejectsOutOfBoundsIndices(
            int source,
            int target
    ) {
        var v = vector(1, 2, 3, 4);
        assertThrows(IllegalArgumentException.class, () -> VectorLogic.swapElements(v, source, target));
        assertVectorEquals(new double[]{1, 2, 3, 4}, v);
    }
    @Test
    void swapElementsRejectsANullVector() {
        assertThrows(NullPointerException.class, () -> VectorLogic.swapElements(null, 0, 1));
    }

    // ---------- multiplyElement ----------

    @Test
    void multiplyElementScalesOnlyTheTargetEntry() {
        var v = vector(1, 2, 3);
        VectorLogic.multiplyElement(v, 1, 5);
        assertVectorEquals(new double[]{1, 10, 3}, v);
    }
    @ParameterizedTest(name = "multiply by {0}")
    @ValueSource(doubles = {-3, -0.5, 0.25, 1, 2, 1e6})
    void multiplyElementHandlesNegativeFractionalIdentityAndLargeScalars(
            double scalar
    ) {
        var v = vector(4, 8, 12);
        VectorLogic.multiplyElement(v, 2, scalar);
        assertVectorEquals(new double[]{4, 8, 12 * scalar}, v);
    }
    @Test
    void multiplyElementByTheReciprocalRestoresTheOriginal() {
        var v = vector(3, 7, -2);
        VectorLogic.multiplyElement(v, 1, 49);
        VectorLogic.multiplyElement(v, 1, 1.0 / 49);
        assertVectorEquals(new double[]{3, 7, -2}, v);
    }
    @Test
    void multiplyElementTwiceEqualsMultiplyingByTheProduct() {
        Random random = new Random(5);
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + random.nextInt(8);
            double[] entries = randomEntries(random, size);
            int index = random.nextInt(size);
            double first = randomScalar(random);
            double second = randomScalar(random);

            var sequential = new Vector(entries);
            VectorLogic.multiplyElement(sequential, index, first);
            VectorLogic.multiplyElement(sequential, index, second);

            var combined = new Vector(entries);
            VectorLogic.multiplyElement(combined, index, first * second);
            assertVectorEquals(combined.data(), sequential);
        }
    }
    @Test
    void multiplyElementRejectsAZeroScalarAndLeavesTheVectorUnchanged() {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> VectorLogic.multiplyElement(v, 1, 0));
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @ParameterizedTest(name = "scalar {0} is rejected")
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void multiplyElementRejectsNonFiniteScalars(
            double scalar
    ) {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> VectorLogic.multiplyElement(v, 1, scalar));
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @ParameterizedTest(name = "index {0} is rejected")
    @ValueSource(ints = {-1, 3, 100, Integer.MIN_VALUE})
    void multiplyElementRejectsOutOfBoundsIndex(
            int index
    ) {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> VectorLogic.multiplyElement(v, index, 2));
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @Test
    void multiplyElementRejectsANullVector() {
        assertThrows(NullPointerException.class, () -> VectorLogic.multiplyElement(null, 0, 2));
    }

    // ---------- reduceElement ----------

    @Test
    void reduceElementAddsAMultipleOfTheSourceToTheTarget() {
        var v = vector(2, 5, 7);
        VectorLogic.reduceElement(v, 0, 2, 3); // v[2] += 3 * v[0]
        assertVectorEquals(new double[]{2, 5, 13}, v);
    }
    @Test
    void reduceElementWithANegativeScalarEliminatesTheTarget() {
        var v = vector(2, 4);
        VectorLogic.reduceElement(v, 0, 1, -2); // v[1] += -2 * v[0]
        assertVectorEquals(new double[]{2, 0}, v);
    }
    @Test
    void reduceElementWithAFractionalScalar() {
        var v = vector(4, 1);
        VectorLogic.reduceElement(v, 0, 1, -0.25);
        assertVectorEquals(new double[]{4, 0}, v);
    }
    @Test
    void reduceElementNeverChangesTheSourceOrOtherEntries() {
        var v = vector(1, 2, 3, 4, 5);
        VectorLogic.reduceElement(v, 1, 3, 10);
        assertVectorEquals(new double[]{1, 2, 3, 24, 5}, v);
    }
    @Test
    void reduceElementIsUndoneByTheNegatedScalar() {
        var v = vector(3, -1, 8);
        VectorLogic.reduceElement(v, 0, 2, 4.5);
        VectorLogic.reduceElement(v, 0, 2, -4.5);
        assertVectorEquals(new double[]{3, -1, 8}, v);
    }
    @Test
    void reduceElementWithAZeroSourceEntryLeavesTheTargetUnchanged() {
        var v = vector(0, 5, 7);
        VectorLogic.reduceElement(v, 0, 2, 3);
        assertVectorEquals(new double[]{0, 5, 7}, v);
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
            var v = new Vector(entries);
            VectorLogic.reduceElement(v, source, target, -entries[target] / entries[source]);
            assertEquals(0.0, v.get(target), TOL, "trial " + trial);
        }
    }
    @Test
    void twoReductionsEqualOneReductionByTheSumOfTheScalars() {
        Random random = new Random(9);
        for (int trial = 0; trial < 200; trial++) {
            int size = 2 + random.nextInt(7);
            double[] entries = randomEntries(random, size);
            List<Integer> picks = shuffledIndices(random, size);
            double first = randomScalar(random);
            double second = randomScalar(random);
            if (first + second == 0) {
                continue;
            }

            var sequential = new Vector(entries);
            VectorLogic.reduceElement(sequential, picks.get(0), picks.get(1), first);
            VectorLogic.reduceElement(sequential, picks.get(0), picks.get(1), second);

            var combined = new Vector(entries);
            VectorLogic.reduceElement(combined, picks.get(0), picks.get(1), first + second);
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
            int firstTarget = picks.get(1);
            int secondTarget = picks.get(2);
            double first = randomScalar(random);
            double second = randomScalar(random);

            var forward = new Vector(entries);
            VectorLogic.reduceElement(forward, source, firstTarget, first);
            VectorLogic.reduceElement(forward, source, secondTarget, second);

            var backward = new Vector(entries);
            VectorLogic.reduceElement(backward, source, secondTarget, second);
            VectorLogic.reduceElement(backward, source, firstTarget, first);

            assertVectorEquals(backward.data(), forward);
        }
    }
    @Test
    void reduceElementRejectsAZeroScalarAndLeavesTheVectorUnchanged() {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> VectorLogic.reduceElement(v, 0, 1, 0));
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @ParameterizedTest(name = "scalar {0} is rejected")
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void reduceElementRejectsNonFiniteScalars(
            double scalar
    ) {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> VectorLogic.reduceElement(v, 0, 1, scalar));
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @ParameterizedTest(name = "source {0}, target {1} is rejected")
    @CsvSource({"-1, 0", "3, 0", "0, -1", "0, 3", "-1, -1", "3, 3"})
    void reduceElementRejectsOutOfBoundsIndices(
            int source,
            int target
    ) {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> VectorLogic.reduceElement(v, source, target, 2));
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @Test
    void reduceElementRejectsTheSameIndexAsSourceAndTarget() {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> VectorLogic.reduceElement(v, 1, 1, 2));
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }
    @Test
    void reduceElementRejectsANullVector() {
        assertThrows(NullPointerException.class, () -> VectorLogic.reduceElement(null, 0, 1, 2));
    }

    // ---------- the RREF operations that call into VectorLogic ----------

    @Test
    void type1AppliedToAVectorSwapsTheTwoEntries() {
        var v = vector(1, 2, 3);
        new Type1(0, 2).apply(v);
        assertVectorEquals(new double[]{3, 2, 1}, v);
    }
    @Test
    void type2AppliedToAVectorScalesTheTargetEntry() {
        var v = vector(1, 2, 3);
        new Type2(1, 0.5).apply(v);
        assertVectorEquals(new double[]{1, 1, 3}, v);
    }
    @Test
    void type3AppliedToAVectorAddsAMultipleOfTheSourceToTheTarget() {
        var v = vector(1, 2, 3);
        new Type3(0, 2, -1).apply(v); // v[2] += -1 * v[0]
        assertVectorEquals(new double[]{1, 2, 2}, v);
    }
    // The operation log RREF produces for A = [[2,1],[1,3]]. Replaying it on b must give x with Ax = b,
    // which is the whole reason the operations can be applied to vectors.
    @Test
    void replayingTheRrefOperationsOnBSolvesAx_equals_b() {
        var b = vector(3, 5);
        new Type2(0, 0.5).apply(b);
        new Type3(0, 1, -1).apply(b);
        new Type2(1, 0.4).apply(b);
        new Type3(1, 0, -0.5).apply(b);

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
            OperationRREF operation = randomOperation(random, size);

            var v = new Vector(entries);
            operation.apply(v);
            inverseOf(operation).apply(v);
            assertVectorEquals(entries, v);
        }
    }
    // Applying any operation to a vector must give the same numbers as applying it to a one-column
    // matrix, because the two paths (VectorLogic vs MatrixLogic) are written separately.
    @Test
    void vectorOperationsAgreeWithMatrixOperationsOnRandomSequences() {
        Random random = new Random(7);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + random.nextInt(5);
            double[] start = randomEntries(random, n);
            double[][] column = new double[n][1];
            for (int i = 0; i < n; i++) {
                column[i][0] = start[i];
            }
            var vector = new Vector(start);
            var matrix = new Matrix(column);

            for (int step = 0; step < 15; step++) {
                OperationRREF operation = randomOperation(random, n);
                operation.apply(vector);
                operation.apply(matrix);
                for (int i = 0; i < n; i++) {
                    // Both paths do identical arithmetic in identical order, so demand exact equality.
                    assertEquals(matrix.get(i, 0), vector.get(i), 0.0,
                            "trial " + trial + ", step " + step + " (" + operation + "), entry " + i);
                }
            }
        }
    }

    // ---------- helpers ----------

    private static OperationRREF randomOperation(
            Random random,
            int n
    ) {
        double scalar = randomScalar(random);
        int i = random.nextInt(n);
        int j = (i + 1 + random.nextInt(n - 1)) % n; // always different from i
        return switch (random.nextInt(3)) {
            case 0 -> new Type1(i, j);
            case 1 -> new Type2(i, scalar);
            default -> new Type3(i, j, scalar);
        };
    }
    private static OperationRREF inverseOf(
            OperationRREF operation
    ) {
        if (operation instanceof Type1 swap) {
            return swap; // a swap is its own inverse
        } else if (operation instanceof Type2 scale) {
            return new Type2(scale.targetRow(), 1.0 / scale.scalar());
        }
        Type3 reduce = (Type3) operation;
        return new Type3(reduce.sourceRow(), reduce.targetRow(), -reduce.scalar());
    }
    private static double randomScalar(
            Random random
    ) {
        return SCALARS[random.nextInt(SCALARS.length)];
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
}
