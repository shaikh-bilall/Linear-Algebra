package linearAlgebra.logic;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;
import linearAlgebra.domain.rref.RowOperation;
import linearAlgebra.domain.rref.RowSwap;
import linearAlgebra.domain.rref.RowScale;
import linearAlgebra.domain.rref.RowCombine;
import linearAlgebra.domain.rref.RowOperations;
import linearAlgebra.domain.rref.Reduction;
import linearAlgebra.logic.exceptions.DeterminantDoesNotExistException;
import linearAlgebra.logic.exceptions.InverseNotAvailableException;

import java.util.ArrayList;
import java.util.List;

/**
 * Defines basic operations involving {@link RowOperation} on {@link Matrix} objects.
 */
public final class RowOperationLogic {

    // Relative to the largest absolute entry of the input;
    // anything at or below this value counts as zero.
    private static final double RELATIVE_TOLERANCE = 1e-10;
    private static final double UNIT_PIVOT_TOLERANCE = 1e-9;
    private static final int NO_PIVOT_ROW = -1;

    // Prevents instantiation.
    private RowOperationLogic() {
    }

    /**
     * Reduces a {@link Matrix} to {@code RREF} and records the {@link RowOperations}.
     *
     * @param A the {@link Matrix} to reduce
     * @return the reduced {@link Matrix} and the {@link RowOperations} that produced it
     */
    public static Reduction reduce(
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            A,
            "RowOperationLogic.reduce: matrix must not be null"
        );

        final List<RowOperation> operations = new ArrayList<>();
        final double[][] working = A.data();
        final double tolerance = RELATIVE_TOLERANCE * MatrixLogic.maxAbs(A);
        final int rows = working.length;
        final int cols = working[0].length;

        int pivotRow = 0;
        int pivotCol = 0;

        while (
            pivotRow < rows &&
            pivotCol < cols
        ) {
            final int bestRow = findPivotRow(working, pivotRow, pivotCol, tolerance);
            if (bestRow != NO_PIVOT_ROW) {
                if (bestRow != pivotRow) {
                    applyAndRecord(working, operations, new RowSwap(pivotRow, bestRow));
                }

                final double pivot = working[pivotRow][pivotCol];
                if (pivot != 1.0) {
                    applyAndRecord(working, operations, new RowScale(pivotRow, 1.0 / pivot));
                }

                eliminateColumn(working, operations, pivotRow, pivotCol, tolerance);
                pivotRow++;
            } // otherwise, no usable pivot in this column: move right, stay on the same row.
            pivotCol++;
        }

        return new Reduction(
            new Matrix(working),
            new RowOperations(operations)
        );
    }

    /**
     * Applies one {@link RowOperation} to the rows of a {@link Matrix}.
     *
     * @param operation the {@link RowOperation} to apply
     * @param A         the {@link Matrix} to operate on
     * @return a {@link Matrix} with the {@link RowOperation} applied;
     *         {@code A} is unchanged
     */
    public static Matrix apply(
        final RowOperation operation,
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            operation,
            "RowOperationLogic.apply: operation must not be null"
        );
        Preconditions.checkNotNull(
            A,
            "RowOperationLogic.apply: matrix must not be null"
        );

        final double[][] data = A.data();
        applyInPlace(data, operation);
        return new Matrix(data);
    }

    /**
     * Applies one {@link RowOperation} to the entries of a {@link Vector}.
     *
     * @param operation the {@link RowOperation} to apply
     * @param v         the {@link Vector} to operate on
     * @return a {@link Vector} with the {@link RowOperation} applied;
     *         {@code v} is unchanged
     */
    public static Vector apply(
        final RowOperation operation,
        final Vector v
    ) {
        Preconditions.checkNotNull(
            operation,
            "RowOperationLogic.apply: operation must not be null"
        );
        Preconditions.checkNotNull(
            v,
            "RowOperationLogic.apply: vector must not be null"
        );

        final double[][] column = toColumn(v);
        applyInPlace(column, operation);
        return fromColumn(column);
    }

    /**
     * Applies a sequence of {@link RowOperations}, in order,
     * to the rows of a {@link Matrix}.
     *
     * @param operations the {@link RowOperations} to apply
     * @param A          the {@link Matrix} to operate on
     * @return a {@link Matrix} after all {@link RowOperations} are applied;
     *         a {@link Matrix} equal to {@code A} when the sequence is empty
     */
    public static Matrix applyAll(
        final RowOperations operations,
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            operations,
            "RowOperationLogic.applyAll: operations must not be null"
        );
        Preconditions.checkNotNull(
            A,
            "RowOperationLogic.applyAll: matrix must not be null"
        );

        return new Matrix(
            applyAll(operations, A.data())
        );
    }

    /**
     * Applies a sequence of {@link RowOperations}, in order,
     * to the entries of a {@link Vector}.
     *
     * @param operations the {@link RowOperations} to apply
     * @param v          the {@link Vector} to operate on
     * @return a {@link Vector} after all {@link RowOperations} are applied;
     *         a {@link Vector} equal to {@code v} when the sequence is empty
     */
    public static Vector applyAll(
        final RowOperations operations,
        final Vector v
    ) {
        Preconditions.checkNotNull(
            operations,
            "RowOperationLogic.applyAll: operations must not be null"
        );
        Preconditions.checkNotNull(
            v,
            "RowOperationLogic.applyAll: vector must not be null"
        );

        return fromColumn(
            applyAll(operations, toColumn(v))
        );
    }

    /**
     * Computes the reduced row echelon form of the provided {@link Matrix}.
     *
     * @param A the {@link Matrix} to reduce
     * @return a {@link Matrix} in reduced row echelon form
     */
    public static Matrix rref(
        final Matrix A
    ) {
        Preconditions.checkNotNull(
            A,
            "RowOperationLogic.rref: matrix must not be null"
        );

        return rref(reduce(A));
    }

    /**
     * Gives the reduced row echelon form from the provided {@link Reduction}.
     * The {@link Reduction} must be the one produced from the {@link Matrix} whose
     * reduced form is wanted; a {@link Reduction} does not record the
     * {@link Matrix} it came from, so this method cannot verify it.
     *
     * @param reduction the {@link Reduction} with the {@code RREF} {@link Matrix}
     * @return a {@link Matrix} in reduced row echelon form
     */
    public static Matrix rref(
        final Reduction reduction
    ) {
        Preconditions.checkNotNull(
            reduction,
            "RowOperationLogic.rref: reduction must not be null"
        );

        return reduction.reduced();
    }

    /**
     * Computes the inverse of the provided {@link Matrix}.
     *
     * @param A the {@link Matrix} to invert
     * @return the inverse, as a new {@link Matrix}
     * @throws InverseNotAvailableException if the {@link Matrix} is not square or is singular
     */
    public static Matrix inverse(
        final Matrix A
    )
        throws InverseNotAvailableException
    {
        Preconditions.checkNotNull(
            A,
            "RowOperationLogic.inverse: matrix must not be null"
        );

        return inverse(reduce(A));
    }

    /**
     * Computes the inverse from the provided {@link Reduction}.
     * The {@link Reduction} must be the one produced from the {@link Matrix} whose
     * inverse is wanted; a {@link Reduction} does not record the
     * {@link Matrix} it came from, so this method cannot verify it.
     *
     * @param reduction the {@link Reduction} with the {@code RREF} of the {@link Matrix} to invert
     * @return the inverse, as a new {@link Matrix}
     * @throws InverseNotAvailableException if the reduced {@link Matrix} is not square or is singular
     */
    public static Matrix inverse(
        final Reduction reduction
    )
        throws InverseNotAvailableException
    {
        Preconditions.checkNotNull(
            reduction,
            "RowOperationLogic.inverse: reduction must not be null"
        );

        final int rows = reduction.reduced().rows();
        final int cols = reduction.reduced().cols();

        if (rows != cols) {
            throw new InverseNotAvailableException();
        }

        if (getDeterminant(reduction) == 0) {
            throw new InverseNotAvailableException();
        }
        return applyAll(reduction.operations(), Matrix.identity(rows));
    }

    /**
     * Computes the determinant of the provided {@link Matrix}.
     *
     * @param A the {@link Matrix} to compute the determinant of
     * @return the determinant, which is 0 for a singular {@link Matrix}
     * @throws DeterminantDoesNotExistException if the {@link Matrix} is not square
     */
    public static double determinant(
        final Matrix A
    )
        throws DeterminantDoesNotExistException
    {
        Preconditions.checkNotNull(
            A,
            "RowOperationLogic.determinant: matrix must not be null"
        );

        return determinant(reduce(A));
    }

    /**
     * Computes the determinant from the provided {@link Reduction}.
     * The {@link Reduction} must be the one produced from the {@link Matrix} whose
     * determinant is wanted; a {@link Reduction} does not record the
     * {@link Matrix} it came from, so this method cannot verify it.
     *
     * @param reduction the {@link Reduction} with the {@code RREF} of the {@link Matrix}
     *                  to compute the determinant of
     * @return the determinant, which is 0 for a singular {@link Matrix}
     * @throws DeterminantDoesNotExistException if the reduced {@link Matrix} is not square
     */
    public static double determinant(
        final Reduction reduction
    )
        throws DeterminantDoesNotExistException
    {
        Preconditions.checkNotNull(
            reduction,
            "RowOperationLogic.determinant: reduction must not be null"
        );

        if (reduction.reduced().rows() != reduction.reduced().cols()) {
            throw new DeterminantDoesNotExistException();
        }

        return getDeterminant(reduction);
    }

    // Applies a sequence of elementary row-operations, in order, to a working array.
    private static double[][] applyAll(
        final RowOperations operations,
        final double[][] data
    ) {
        for (var operation : operations) {
            applyInPlace(data, operation);
        }
        return data;
    }

    // Holds the only copy of the elementary row-operation arithmetic on a private working array.
    private static double[][] applyInPlace(
        final double[][] data,
        final RowOperation operation
    ) {
        return switch (operation) {
            case RowSwap swap -> {
                final int source = swap.sourceRow();
                final int target = swap.targetRow();

                checkRowFits(data.length, source);
                checkRowFits(data.length, target);

                final double[] temp = data[source];

                data[source] = data[target];
                data[target] = temp;

                yield data;
            }
            case RowScale scale -> {
                final int target = scale.targetRow();
                final double factor = scale.scalar();

                checkRowFits(data.length, target);

                for (int j = 0; j < data[target].length; j++) {
                    data[target][j] *= factor;
                    Preconditions.checkState(
                        Double.isFinite(data[target][j]),
                        "RowOperationLogic: scaled entry must be finite: column %s",
                        j
                    );
                }

                yield data;
            }
            case RowCombine combine -> {
                final int source = combine.sourceRow();
                final int target = combine.targetRow();
                final double factor = combine.scalar();

                checkRowFits(data.length, source);
                checkRowFits(data.length, target);

                for (int j = 0; j < data[target].length; j++) {
                    data[target][j] += factor * data[source][j];
                    Preconditions.checkState(
                        Double.isFinite(data[target][j]),
                        "RowOperationLogic: combined entry must be finite: column %s",
                        j
                    );
                }

                yield data;
            }
        };
    }

    // Checks that a row index exists in an operand with the given number of rows.
    private static void checkRowFits(
        final int rows,
        final int row
    ) {
        Preconditions.checkArgument(
            row >= 0 &&
                row < rows,
            "RowOperationLogic: index out of bounds for an operand with %s rows: %s",
            rows,
            row
        );
    }

    // Applies an operation to the working array and records it for replay.
    private static void applyAndRecord(
        final double[][] working,
        final List<RowOperation> operations,
        final RowOperation operation
    ) {
        applyInPlace(working, operation);
        operations.add(operation);
    }

    // Returns the largest-magnitude row at or below startRow
    // in a column, or NO_PIVOT_ROW if none is significant.
    private static int findPivotRow(
        final double[][] data,
        final int startRow,
        final int col,
        final double tolerance
    ) {
        int bestRow = NO_PIVOT_ROW;
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
        final List<RowOperation> operations,
        final int pivotRow,
        final int pivotCol,
        final double tolerance
    ) {
        // Read the actual pivot rather than assuming 1: after RowScale it can be 1 +/- an ulp.
        final double pivot = working[pivotRow][pivotCol];
        for (int row = 0; row < working.length; row++) {
            if (row != pivotRow) {
                final double entry = working[row][pivotCol];
                if (Math.abs(entry) > tolerance) {
                    applyAndRecord(working, operations, new RowCombine(pivotRow, row, -entry / pivot));
                }
            }
        }
    }

    // A vector is operated on as a matrix with one column,
    // so vectors and matrices share one implementation.
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

    // Returns the determinant from the operations recorded while reducing a square matrix.
    private static double getDeterminant(
        final Reduction reduction
    ) {
        final Matrix reduced = reduction.reduced();
        final int size = reduced.rows();
        int unitPivots = 0;

        for (int i = 0; i < size; i++) {
            if (Math.abs(reduced.get(i, i) - 1) <= UNIT_PIVOT_TOLERANCE) {
                unitPivots++;
            }
        }

        double determinant = 0;
        if (unitPivots == size) {
            determinant = 1;
            for (var operation : reduction.operations()) {
                determinant = operation.updatedDeterminant(determinant);
            }
            determinant = 1 / determinant;
        }
        return determinant;
    }
}
