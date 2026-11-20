package linearAlgebra.domain.rref;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static linearAlgebra.TestUtils.assertMatrixEquals;
import static linearAlgebra.TestUtils.assertVectorEquals;
import static linearAlgebra.TestUtils.identity;
import static linearAlgebra.TestUtils.product;
import static linearAlgebra.TestUtils.randomData;
import static linearAlgebra.TestUtils.randomEntries;
import static linearAlgebra.TestUtils.randomScalar;
import static linearAlgebra.TestUtils.sampleData;
import static linearAlgebra.TestUtils.shuffledIndices;
import static linearAlgebra.TestUtils.vector;
import static org.junit.jupiter.api.Assertions.*;

class RowReducerTest {

    private static final double TOL = 1e-9;

    // ---------- reduce: known results ----------

    @Test
    void reduceGivesTheIdentityAndTheRecordedOperationsForAnInvertibleMatrix() {
        var reduction = RowReducer.reduce(new Matrix(new double[][]{{2, 4}, {1, 3}}));
        assertMatrixEquals(identity(2), reduction.reduced(), TOL);
        assertEquals(
            List.of(new RowScale(0, 0.5), new RowCombine(0, 1, -1.0), new RowCombine(1, 0, -2.0)),
            reduction.operations()
        );
    }

    @Test
    void reduceSwapsRowsWhenThePivotIsBelow() {
        var reduction = RowReducer.reduce(new Matrix(new double[][]{{0, 1}, {1, 0}}));
        assertMatrixEquals(identity(2), reduction.reduced(), TOL);
        assertEquals(List.of(new RowSwap(0, 1)), reduction.operations());
    }

    @Test
    void reduceChoosesTheLargestMagnitudePivot() {
        var reduction = RowReducer.reduce(new Matrix(new double[][]{{1, 2}, {-4, 1}, {2, 5}}));
        assertEquals(new RowSwap(0, 1), reduction.operations().getFirst());
    }

    @Test
    void reduceReturnsTheOriginalMatrixWhenNoOperationIsNeeded() {
        var matrix = identity(3);
        var reduction = RowReducer.reduce(matrix);
        assertSame(matrix, reduction.reduced());
        assertTrue(reduction.operations().isEmpty());
    }

    @Test
    void reduceLeavesAZeroMatrixAloneWithoutOperations() {
        var zero = new Matrix(new double[3][2]);
        var reduction = RowReducer.reduce(zero);
        assertSame(zero, reduction.reduced());
        assertTrue(reduction.operations().isEmpty());
    }

    @Test
    void reduceHandlesWideAndTallMatrices() {
        assertMatrixEquals(
            new double[][]{{1, 0, -1}, {0, 1, 2}},
            RowReducer.reduce(new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}})).reduced(),
            TOL
        );
        assertMatrixEquals(
            new double[][]{{1, 0}, {0, 1}, {0, 0}},
            RowReducer.reduce(new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}})).reduced(),
            TOL
        );
    }

    @Test
    void reduceSkipsAColumnWithoutAUsablePivotAndStaysOnTheSameRow() {
        var reduced = RowReducer.reduce(new Matrix(new double[][]{{0, 1, 2}, {0, 3, 4}})).reduced();
        assertMatrixEquals(new double[][]{{0, 1, 0}, {0, 0, 1}}, reduced, TOL);
    }

    @Test
    void reduceStopsWhenEveryRowHasAPivot() {
        var reduced = RowReducer.reduce(new Matrix(new double[][]{{1, 2, 3, 4}})).reduced();
        assertMatrixEquals(new double[][]{{1, 2, 3, 4}}, reduced, TOL);
    }

    @Test
    void reduceTreatsEntriesAtOrBelowTheRelativeToleranceAsZero() {
        var reduction = RowReducer.reduce(new Matrix(new double[][]{{1, 1}, {1, 1 + 1e-12}}));
        assertEquals(1.0, reduction.reduced().get(0, 0), TOL);
        assertEquals(0.0, reduction.reduced().get(1, 0), 0.0);
        assertEquals(1e-12, reduction.reduced().get(1, 1), 1e-13);
        for (RowOperation operation : reduction.operations()) {
            assertFalse(operation instanceof RowScale, "no pivot may be taken from a noise-level entry");
        }
    }

    @ParameterizedTest(name = "scaling by {0} does not change the operations")
    @CsvSource({"1e-6", "1e6", "-3"})
    void reduceDependsOnlyOnTheRelativeSizeOfTheEntries(
        final double factor
    ) {
        var base = new Matrix(new double[][]{{1, 2, 0}, {3, 4, 1}, {5, 6, 7}});
        var scaled = new Matrix(new double[][]{
            {factor, 2 * factor, 0}, {3 * factor, 4 * factor, factor}, {5 * factor, 6 * factor, 7 * factor}
        });
        assertMatrixEquals(
            RowReducer.reduce(base).reduced(),
            RowReducer.reduce(scaled).reduced(),
            1e-6
        );
    }

    // ---------- reduce: properties ----------

    @Test
    void reduceLeavesItsInputUnchanged() {
        var matrix = new Matrix(sampleData());
        RowReducer.reduce(matrix);
        assertMatrixEquals(sampleData(), matrix, 0.0);
    }

    @Test
    void replayingTheRecordedOperationsOnTheInputGivesExactlyTheReducedMatrix() {
        Random random = new Random(201); // fixed seed keeps failures reproducible
        for (int trial = 0; trial < 300; trial++) {
            var matrix = new Matrix(randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5));
            var reduction = RowReducer.reduce(matrix);
            assertMatrixEquals(
                reduction.reduced(),
                RowReducer.applyAll(reduction.operations(), matrix),
                0.0
            );
        }
    }

    @Test
    void replayingTheOperationsOnTheIdentityGivesTheInverseOfAnInvertibleMatrix() {
        Random random = new Random(202);
        int checked = 0;
        for (int trial = 0; trial < 300; trial++) {
            int n = 1 + random.nextInt(4);
            double[][] data = randomData(random, n, n, 5);
            var reduction = RowReducer.reduce(new Matrix(data));
            if (isNotIdentity(reduction.reduced())) {
                continue;
            }
            var inverse = RowReducer.applyAll(reduction.operations(), Matrix.identity(n));
            assertMatrixEquals(identity(n), new Matrix(product(data, inverse.data())), 1e-6);
            checked++;
        }
        assertTrue(checked > 0, "the seed must produce at least one invertible matrix");
    }

    @Test
    void theRecordedOperationsAreAlwaysValidForTheInputShape() {
        Random random = new Random(203);
        for (int trial = 0; trial < 300; trial++) {
            var matrix = new Matrix(randomData(random, 1 + random.nextInt(5), 1 + random.nextInt(5), 5));
            for (RowOperation operation : RowReducer.reduce(matrix).operations()) {
                operation.apply(matrix);
            }
        }
    }

    @Test
    void theOperationListOfAReductionCannotBeModified() {
        var operations = RowReducer.reduce(new Matrix(new double[][]{{2, 4}, {1, 3}})).operations();
        assertThrows(UnsupportedOperationException.class, () -> operations.add(new RowSwap(0, 1)));
        assertThrows(UnsupportedOperationException.class, operations::removeFirst);
    }

    // ---------- reduce: exceptions ----------

    @Test
    void reduceRejectsANullMatrix() {
        assertThrows(NullPointerException.class, () -> RowReducer.reduce(null));
    }

    // ---------- Reduction record ----------

    @Test
    void reductionReportsItsComponents() {
        var matrix = new Matrix(sampleData());
        List<RowOperation> operations = List.of(new RowSwap(0, 1));
        var reduction = new Reduction(matrix, operations);
        assertSame(matrix, reduction.reduced());
        assertEquals(operations, reduction.operations());
    }

    @Test
    void reductionCopiesTheOperationList() {
        List<RowOperation> operations = new ArrayList<>(List.of(new RowSwap(0, 1)));
        var reduction = new Reduction(new Matrix(sampleData()), operations);
        operations.add(new RowScale(0, 2));
        assertEquals(List.of(new RowSwap(0, 1)), reduction.operations());
        assertNotSame(operations, reduction.operations());
    }

    @Test
    void reductionRejectsANullMatrixAndNullOperations() {
        List<RowOperation> none = List.of();
        assertThrows(NullPointerException.class, () -> new Reduction(null, none));
        assertThrows(NullPointerException.class,
            () -> new Reduction(new Matrix(sampleData()), null));
    }

    @Test
    void reductionRejectsAnOperationListWithANullElement() {
        List<RowOperation> withNull = Arrays.asList(new RowSwap(0, 1), null);
        assertThrows(NullPointerException.class,
            () -> new Reduction(new Matrix(sampleData()), withNull));
    }

    // ---------- apply to a matrix ----------

    @Test
    void applyOfEachTypeMatchesTheResultOfTheOperationItself() {
        var matrix = new Matrix(sampleData());
        for (RowOperation operation : operations()) {
            assertMatrixEquals(operation.apply(matrix), RowReducer.apply(operation, matrix), 0.0);
        }
    }

    @Test
    void applyToAMatrixPerformsTheElementaryOperation() {
        var matrix = new Matrix(sampleData());
        assertMatrixEquals(
            new double[][]{{5, 6}, {3, 4}, {1, 2}}, RowReducer.apply(new RowSwap(0, 2), matrix), 0.0
        );
        assertMatrixEquals(
            new double[][]{{1, 2}, {6, 8}, {5, 6}}, RowReducer.apply(new RowScale(1, 2), matrix), 0.0
        );
        assertMatrixEquals(
            new double[][]{{1, 2}, {3, 4}, {8, 12}}, RowReducer.apply(new RowCombine(0, 2, 3), matrix), 0.0
        );
    }

    @Test
    void applyToAMatrixLeavesTheOperandUnchangedAndReturnsANewMatrix() {
        var matrix = new Matrix(sampleData());
        for (RowOperation operation : operations()) {
            Matrix result = RowReducer.apply(operation, matrix);
            assertNotSame(matrix, result);
            assertMatrixEquals(sampleData(), matrix, 0.0);
        }
    }

    @Test
    void applyToAMatrixRejectsANullOperationAndANullMatrix() {
        var matrix = new Matrix(sampleData());
        assertThrows(NullPointerException.class, () -> RowReducer.apply(null, matrix));
        assertThrows(NullPointerException.class, () -> RowReducer.apply(new RowSwap(0, 1), (Matrix) null));
    }

    @ParameterizedTest(name = "row {0} is beyond a three-row matrix")
    @CsvSource({"3", "4", "100", "2147483647"})
    void applyToAMatrixRejectsRowsBeyondTheOperand(
        final int row
    ) {
        var matrix = new Matrix(sampleData());
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowSwap(row, 0), matrix));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowSwap(0, row), matrix));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowScale(row, 2), matrix));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowCombine(row, 0, 2), matrix));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowCombine(0, row, 2), matrix));
    }

    @Test
    void applyToAMatrixRejectsAResultThatOverflows() {
        var big = new Matrix(new double[][]{{1e308, 1}, {1e308, 1}});
        assertThrows(IllegalStateException.class, () -> RowReducer.apply(new RowScale(0, 10), big));
        assertThrows(IllegalStateException.class, () -> RowReducer.apply(new RowCombine(0, 1, 10), big));
    }

    @Test
    void applyToAMatrixRejectsAnUnsupportedOperation() {
        var matrix = new Matrix(sampleData());
        assertThrows(
            NonExistentRowOperationException.class,
            () -> RowReducer.apply(unsupportedOperation(), matrix)
        );
    }

    // ---------- apply to a vector ----------

    @Test
    void applyToAVectorPerformsTheElementaryOperation() {
        var v = vector(1, 2, 3);
        assertVectorEquals(new double[]{3, 2, 1}, RowReducer.apply(new RowSwap(0, 2), v));
        assertVectorEquals(new double[]{1, 4, 3}, RowReducer.apply(new RowScale(1, 2), v));
        assertVectorEquals(new double[]{1, 2, 6}, RowReducer.apply(new RowCombine(0, 2, 3), v));
    }

    @Test
    void applyToAVectorMatchesTheResultOfTheOperationItself() {
        var v = vector(1, 2, 3);
        for (RowOperation operation : operations()) {
            assertVectorEquals(operation.apply(v).data(), RowReducer.apply(operation, v));
        }
    }

    @Test
    void applyToAVectorLeavesTheOperandUnchangedAndReturnsANewVector() {
        var v = vector(1, 2, 3);
        for (RowOperation operation : operations()) {
            Vector result = RowReducer.apply(operation, v);
            assertNotSame(v, result);
            assertVectorEquals(new double[]{1, 2, 3}, v);
        }
    }

    @Test
    void applyToAVectorRejectsANullOperationAndANullVector() {
        var v = vector(1, 2, 3);
        assertThrows(NullPointerException.class, () -> RowReducer.apply(null, v));
        assertThrows(NullPointerException.class, () -> RowReducer.apply(new RowSwap(0, 1), (Vector) null));
    }

    @ParameterizedTest(name = "index {0} is beyond a three-entry vector")
    @CsvSource({"3", "4", "100", "2147483647"})
    void applyToAVectorRejectsIndicesBeyondTheOperand(
        final int index
    ) {
        var v = vector(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowSwap(index, 0), v));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowSwap(0, index), v));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowScale(index, 2), v));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowCombine(index, 0, 2), v));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.apply(new RowCombine(0, index, 2), v));
    }

    @Test
    void applyToAVectorRejectsAResultThatOverflows() {
        var big = vector(1e308, 1e308);
        assertThrows(IllegalStateException.class, () -> RowReducer.apply(new RowScale(0, 10), big));
        assertThrows(IllegalStateException.class, () -> RowReducer.apply(new RowCombine(0, 1, 10), big));
    }

    @Test
    void applyToAVectorRejectsAnUnsupportedOperation() {
        var v = vector(1, 2, 3);
        assertThrows(
            NonExistentRowOperationException.class,
            () -> RowReducer.apply(unsupportedOperation(), v)
        );
    }

    // ---------- applyAll to a matrix ----------

    @Test
    void applyAllToAMatrixAppliesTheOperationsInOrder() {
        var matrix = new Matrix(sampleData());
        List<RowOperation> operations = List.of(new RowSwap(0, 2), new RowScale(1, 2), new RowCombine(0, 1, -1));
        Matrix expected = matrix;
        for (RowOperation operation : operations) {
            expected = operation.apply(expected);
        }
        assertMatrixEquals(expected, RowReducer.applyAll(operations, matrix), 0.0);
    }

    @Test
    void applyAllToAMatrixDependsOnTheOrderOfTheOperations() {
        var matrix = new Matrix(sampleData());
        var first = RowReducer.applyAll(List.of(new RowSwap(0, 2), new RowScale(0, 2)), matrix);
        var second = RowReducer.applyAll(List.of(new RowScale(0, 2), new RowSwap(0, 2)), matrix);
        assertMatrixEquals(new double[][]{{10, 12}, {3, 4}, {1, 2}}, first, 0.0);
        assertMatrixEquals(new double[][]{{5, 6}, {3, 4}, {2, 4}}, second, 0.0);
    }

    @Test
    void applyAllToAMatrixReturnsAnEqualMatrixForAnEmptyList() {
        var matrix = new Matrix(sampleData());
        assertEquals(matrix, RowReducer.applyAll(List.of(), matrix));
    }

    @Test
    void applyAllToAMatrixLeavesTheOperandUnchanged() {
        var matrix = new Matrix(sampleData());
        Matrix result = RowReducer.applyAll(operations(), matrix);
        assertNotSame(matrix, result);
        assertMatrixEquals(sampleData(), matrix, 0.0);
    }

    @Test
    void applyAllToAMatrixRejectsNullArguments() {
        var matrix = new Matrix(sampleData());
        List<RowOperation> operations = List.of(new RowSwap(0, 1));
        assertThrows(NullPointerException.class, () -> RowReducer.applyAll(null, matrix));
        assertThrows(NullPointerException.class, () -> RowReducer.applyAll(operations, (Matrix) null));
        assertThrows(NullPointerException.class, () -> RowReducer.applyAll(List.of(), (Matrix) null));
    }

    @Test
    void applyAllToAMatrixRejectsAListWithANullOperation() {
        var matrix = new Matrix(sampleData());
        List<RowOperation> withNull = Arrays.asList(new RowSwap(0, 1), null);
        assertThrows(NullPointerException.class, () -> RowReducer.applyAll(withNull, matrix));
    }

    @Test
    void applyAllToAMatrixRejectsAnOperationThatDoesNotFit() {
        var matrix = new Matrix(sampleData());
        List<RowOperation> operations = List.of(new RowSwap(0, 1), new RowScale(3, 2));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.applyAll(operations, matrix));
        assertMatrixEquals(sampleData(), matrix, 0.0);
    }

    @Test
    void applyAllToAMatrixRejectsAResultThatOverflowsPartWayThrough() {
        var big = new Matrix(new double[][]{{1e308}, {1}});
        List<RowOperation> operations = List.of(new RowSwap(0, 1), new RowCombine(1, 0, 10));
        assertThrows(IllegalStateException.class, () -> RowReducer.applyAll(operations, big));
    }

    @Test
    void applyAllToAMatrixRejectsAnUnsupportedOperation() {
        var matrix = new Matrix(sampleData());
        List<RowOperation> operations = List.of(unsupportedOperation());
        assertThrows(
            NonExistentRowOperationException.class,
            () -> RowReducer.applyAll(operations, matrix)
        );
    }

    // ---------- applyAll to a vector ----------

    @Test
    void applyAllToAVectorAppliesTheOperationsInOrder() {
        var v = vector(1, 2, 3);
        List<RowOperation> operations = List.of(new RowSwap(0, 2), new RowScale(1, 2), new RowCombine(0, 1, -1));
        Vector expected = v;
        for (RowOperation operation : operations) {
            expected = operation.apply(expected);
        }
        assertVectorEquals(expected.data(), RowReducer.applyAll(operations, v));
    }

    @Test
    void applyAllToAVectorReturnsAnEqualVectorForAnEmptyList() {
        var v = vector(1, 2, 3);
        assertEquals(v, RowReducer.applyAll(List.of(), v));
    }

    @Test
    void applyAllToAVectorLeavesTheOperandUnchanged() {
        var v = vector(1, 2, 3);
        Vector result = RowReducer.applyAll(operations(), v);
        assertNotSame(v, result);
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }

    @Test
    void applyAllToAVectorRejectsNullArguments() {
        var v = vector(1, 2, 3);
        List<RowOperation> operations = List.of(new RowSwap(0, 1));
        assertThrows(NullPointerException.class, () -> RowReducer.applyAll(null, v));
        assertThrows(NullPointerException.class, () -> RowReducer.applyAll(operations, (Vector) null));
        assertThrows(NullPointerException.class, () -> RowReducer.applyAll(List.of(), (Vector) null));
    }

    @Test
    void applyAllToAVectorRejectsAListWithANullOperation() {
        var v = vector(1, 2, 3);
        List<RowOperation> withNull = Arrays.asList(new RowSwap(0, 1), null);
        assertThrows(NullPointerException.class, () -> RowReducer.applyAll(withNull, v));
    }

    @Test
    void applyAllToAVectorRejectsAnOperationThatDoesNotFit() {
        var v = vector(1, 2, 3);
        List<RowOperation> operations = List.of(new RowSwap(0, 1), new RowCombine(0, 3, 2));
        assertThrows(IllegalArgumentException.class, () -> RowReducer.applyAll(operations, v));
        assertVectorEquals(new double[]{1, 2, 3}, v);
    }

    @Test
    void applyAllToAVectorRejectsAResultThatOverflowsPartWayThrough() {
        var big = vector(1e308, 1);
        List<RowOperation> operations = List.of(new RowSwap(0, 1), new RowCombine(1, 0, 10));
        assertThrows(IllegalStateException.class, () -> RowReducer.applyAll(operations, big));
    }

    @Test
    void applyAllToAVectorRejectsAnUnsupportedOperation() {
        var v = vector(1, 2, 3);
        List<RowOperation> operations = List.of(unsupportedOperation());
        assertThrows(
            NonExistentRowOperationException.class,
            () -> RowReducer.applyAll(operations, v)
        );
    }

    // ---------- matrix and vector paths together ----------

    // Solving A x = b: replaying the reduction of A on b must agree with the reduction of [A | b].
    @Test
    void replayingAReductionOnAVectorAgreesWithReducingTheAugmentedMatrix() {
        Random random = new Random(204);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + random.nextInt(4);
            var a = new Matrix(randomData(random, n, n, 5));
            var reduction = RowReducer.reduce(a);
            if (isNotIdentity(reduction.reduced())) {
                continue;
            }
            var b = new Vector(randomEntries(random, n));
            Vector x = RowReducer.applyAll(reduction.operations(), b);
            double[][] column = new double[n][1];
            for (int i = 0; i < n; i++) {
                column[i][0] = b.get(i);
            }
            Matrix augmented = RowReducer.reduce(a.merge(new Matrix(column))).reduced();
            for (int i = 0; i < n; i++) {
                assertEquals(augmented.get(i, n), x.get(i), 1e-6, "entry " + i);
            }
        }
    }

    @Test
    void shuffledOperationSequencesAgreeOnMatricesAndOneColumnMatrices() {
        Random random = new Random(205);
        for (int trial = 0; trial < 100; trial++) {
            int n = 2 + random.nextInt(4);
            List<RowOperation> operations = new ArrayList<>();
            List<Integer> picks = shuffledIndices(random, n);
            operations.add(new RowSwap(picks.get(0), picks.get(1)));
            operations.add(new RowScale(picks.get(1), randomScalar(random)));
            operations.add(new RowCombine(picks.get(0), picks.get(1), randomScalar(random)));
            double[] entries = randomEntries(random, n);
            double[][] column = new double[n][1];
            for (int i = 0; i < n; i++) {
                column[i][0] = entries[i];
            }
            Vector viaVector = RowReducer.applyAll(operations, new Vector(entries));
            Matrix viaMatrix = RowReducer.applyAll(operations, new Matrix(column));
            for (int i = 0; i < n; i++) {
                assertEquals(viaMatrix.get(i, 0), viaVector.get(i), 0.0, "entry " + i);
            }
        }
    }

    // ---------- helpers ----------

    // One operation of each type, all valid for a matrix with three rows and a vector with three entries.
    private static List<RowOperation> operations() {
        return List.of(new RowSwap(0, 2), new RowScale(1, 3), new RowCombine(0, 2, -2));
    }

    private static boolean isNotIdentity(
        final Matrix matrix
    ) {
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                if (Math.abs(matrix.get(i, j) - ((i == j) ? 1 : 0)) > TOL) {
                    return true;
                }
            }
        }
        return false;
    }
}
