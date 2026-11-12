package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * Row reduction and elementary row operations. It finds the reduced row echelon form of a matrix
 * together with the operations that lead to it, and applies operations, singly or as a recorded
 * sequence, to matrices and vectors. Operands are never modified.
 */
public final class RowReducer {

    // Relative to the largest absolute entry of the input; anything at or below this value counts as zero.
    private static final double RELATIVE_TOLERANCE = 1e-10;

    /**
     * The result of row reducing a matrix.
     *
     * @param reduced    the reduced row echelon form
     * @param operations the elementary operations that turn the original matrix into {@code reduced}, in order
     */
    public record Reduction(Matrix reduced, List<OperationRREF> operations) {

        /**
         * Creates a reduction from its reduced matrix and the operations that produced it.
         *
         * @param reduced    the reduced row echelon form
         * @param operations the elementary operations that turn the original matrix into
         *                   {@code reduced}, in order
         */
        public Reduction {
            Preconditions.checkNotNull(
                reduced,
                "Reduction: reduced matrix must not be null"
            );
            operations = List.copyOf(operations);
        }
    }

    // Prevents instantiation.
    private RowReducer() {
    }

    /**
     * Reduces a matrix to reduced row echelon form.
     *
     * @param A the matrix to reduce
     * @return the reduced matrix and the operations that produced it; the reduced matrix is {@code A} itself when no
     *         operation was needed
     */
    public static Reduction reduce(
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            A,
            "RowReducer.reduce: matrix must not be null"
        );

        final List<OperationRREF> operations = new ArrayList<>();
        final double[][] working = A.data();
        final double tolerance = RELATIVE_TOLERANCE * A.maxAbs();
        final int rows = working.length;
        final int cols = working[0].length;

        int pivotRow = 0;
        for (int pivotCol = 0; pivotCol < cols && pivotRow < rows; pivotCol++) {
            final int bestRow = findPivotRow(working, pivotRow, pivotCol, tolerance);
            if (bestRow == -1) {
                continue; // no usable pivot in this column: move right, stay on the same row
            }

            if (bestRow != pivotRow) {
                applyAndRecord(working, operations, new Type1(pivotRow, bestRow));
            }

            final double pivot = working[pivotRow][pivotCol];
            if (pivot != 1.0) {
                applyAndRecord(working, operations, new Type2(pivotRow, 1.0 / pivot));
            }

            eliminateColumn(working, operations, pivotRow, pivotCol, tolerance);
            pivotRow++;
        }

        return new Reduction(operations.isEmpty() ? A : new Matrix(working), operations);
    }
    /**
     * Applies one elementary operation to the rows of a matrix.
     *
     * @param operation the operation to apply
     * @param A         the matrix to operate on
     * @return a new matrix with the operation applied; {@code A} is unchanged
     */
    public static Matrix apply(
        final OperationRREF operation,
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            operation,
            "RowReducer.apply: operation must not be null"
        );
        Preconditions.checkNotNull(
            A,
            "RowReducer.apply: matrix must not be null"
        );

        final double[][] data = A.data();
        applyInPlace(data, operation);
        return new Matrix(data);
    }
    /**
     * Applies one elementary operation to the entries of a vector.
     *
     * @param operation the operation to apply
     * @param v         the vector to operate on
     * @return a new vector with the operation applied; {@code v} is unchanged
     */
    public static Vector apply(
        final OperationRREF operation,
        final Vector v
    ) {
        Preconditions.checkNotNull(
            operation,
            "RowReducer.apply: operation must not be null"
        );
        Preconditions.checkNotNull(
            v,
            "RowReducer.apply: vector must not be null"
        );

        final double[][] column = toColumn(v);
        applyInPlace(column, operation);
        return fromColumn(column);
    }
    /**
     * Applies a sequence of elementary operations, in order, to the rows of a matrix.
     *
     * @param operations the operations to apply
     * @param A          the matrix to operate on
     * @return a new matrix with every operation applied; {@code A} itself when the sequence is empty
     */
    public static Matrix applyAll(
        final List<OperationRREF> operations,
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            operations,
            "RowReducer.applyAll: operations must not be null"
        );
        Preconditions.checkNotNull(
            A,
            "RowReducer.applyAll: matrix must not be null"
        );
        if (operations.isEmpty()) {
            return A;
        }

        final double[][] data = A.data();
        for (var operation : operations) {
            Preconditions.checkNotNull(
                operation,
                "RowReducer.applyAll: operations must not contain null"
            );
            applyInPlace(data, operation);
        }
        return new Matrix(data);
    }
    /**
     * Applies a sequence of elementary operations, in order, to the entries of a vector.
     *
     * @param operations the operations to apply
     * @param v          the vector to operate on
     * @return a new vector with every operation applied; {@code v} itself when the sequence is empty
     */
    public static Vector applyAll(
        final List<OperationRREF> operations,
        final Vector v
    ) {
        Preconditions.checkNotNull(
            operations,
            "RowReducer.applyAll: operations must not be null"
        );
        Preconditions.checkNotNull(
            v,
            "RowReducer.applyAll: vector must not be null"
        );
        if (operations.isEmpty()) {
            return v;
        }

        final double[][] column = toColumn(v);
        for (var operation : operations) {
            Preconditions.checkNotNull(
                operation,
                "RowReducer.applyAll: operations must not contain null"
            );
            applyInPlace(column, operation);
        }
        return fromColumn(column);
    }

    // Holds the only copy of the elementary row-operation arithmetic on a private working array.
    private static void applyInPlace(
        final double[][] data,
        final OperationRREF operation
    ) {
        switch (operation) {
            case Type1 swap -> {
                checkRowFits(data.length, swap.sourceRow());
                checkRowFits(data.length, swap.targetRow());
                final double[] temp = data[swap.sourceRow()];
                data[swap.sourceRow()] = data[swap.targetRow()];
                data[swap.targetRow()] = temp;
            }
            case Type2 scale -> {
                final int target = scale.targetRow();
                final double factor = scale.scalar();
                checkRowFits(data.length, target);
                for (int j = 0; j < data[target].length; j++) {
                    data[target][j] *= factor;
                    Preconditions.checkState(
                        Double.isFinite(data[target][j]),
                        "RowReducer: scaled entry must be finite: column %s",
                        j
                    );
                }
            }
            case Type3 combine -> {
                final int source = combine.sourceRow();
                final int target = combine.targetRow();
                final double factor = combine.scalar();
                checkRowFits(data.length, source);
                checkRowFits(data.length, target);
                for (int j = 0; j < data[target].length; j++) {
                    data[target][j] += factor * data[source][j];
                    Preconditions.checkState(
                        Double.isFinite(data[target][j]),
                        "RowReducer: combined entry must be finite: column %s",
                        j
                    );
                }
            }
            default -> throw new IllegalStateException("RowReducer: unsupported elementary operation");
        }
    }
    // Checks that a row index exists in an operand with the given number of rows.
    private static void checkRowFits(
        final int rows,
        final int row
    ) {
        Preconditions.checkArgument(
            row >= 0 &&
                row < rows,
            "RowReducer: index out of bounds for an operand with %s rows: %s",
            rows,
            row
        );
    }
    // Applies an operation to the working array and records it for replay.
    private static void applyAndRecord(
        final double[][] working,
        final List<OperationRREF> operations,
        final OperationRREF operation
    ) {
        applyInPlace(working, operation);
        operations.add(operation);
    }
    // Returns the largest-magnitude row at or below startRow in a column, or -1 if none is significant.
    private static int findPivotRow(
        final double[][] data,
        final int startRow,
        final int col,
        final double tolerance
    ) {
        int bestRow = -1;
        double bestMagnitude = tolerance;
        for (int row = startRow; row < data.length; row++) {
            final double magnitude = Math.abs(data[row][col]);
            if (magnitude > bestMagnitude) {
                bestMagnitude = magnitude;
                bestRow = row;
            }
        }
        return bestRow;
    }
    // Clears every other entry in the pivot column.
    private static void eliminateColumn(
        final double[][] working,
        final List<OperationRREF> operations,
        final int pivotRow,
        final int pivotCol,
        final double tolerance
    ) {
        // Read the actual pivot rather than assuming 1: after Type2 it can be 1 +/- an ulp.
        final double pivot = working[pivotRow][pivotCol];
        for (int row = 0; row < working.length; row++) {
            if (row == pivotRow) {
                continue;
            }
            final double entry = working[row][pivotCol];
            if (Math.abs(entry) > tolerance) {
                applyAndRecord(working, operations, new Type3(pivotRow, row, -entry / pivot));
            }
        }
    }
    // A vector is operated on as a matrix with one column, so vectors and matrices share one implementation.
    private static double[][] toColumn(
        final Vector v
    ) {
        final double[] entries = v.data();
        final double[][] column = new double[entries.length][1];
        for (int i = 0; i < entries.length; i++) {
            column[i][0] = entries[i];
        }
        return column;
    }
    // Turns a one-column working array back into a vector.
    private static Vector fromColumn(
        final double[][] column
    ) {
        final double[] entries = new double[column.length];
        for (int i = 0; i < column.length; i++) {
            entries[i] = column[i][0];
        }
        return new Vector(entries);
    }

}