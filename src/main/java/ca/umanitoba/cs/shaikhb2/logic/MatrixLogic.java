package ca.umanitoba.cs.shaikhb2.logic;

import ca.umanitoba.cs.shaikhb2.domain.Exceptions.*;
import ca.umanitoba.cs.shaikhb2.domain.Matrix;
import ca.umanitoba.cs.shaikhb2.domain.RREF.OperationRREF;
import ca.umanitoba.cs.shaikhb2.domain.RREF.Type1;
import ca.umanitoba.cs.shaikhb2.domain.RREF.Type2;
import ca.umanitoba.cs.shaikhb2.domain.RREF.Type3;
import com.google.common.base.Preconditions;

import java.util.ArrayList;
import java.util.List;

public final class MatrixLogic {

    private final Matrix A;
    private final List<OperationRREF> operationsRREF;
    private double determinant;

    public MatrixLogic(
            final Matrix A
    ) {
        this.A = A;
        this.operationsRREF = getOperationsRREF(A.data());

        try {
            this.determinant = getDeterminant(A, operationsRREF);
        } catch (DeterminantDoesNotExistException e) {
            this.determinant = Double.NaN;
        }
    }

    public Matrix multiply(
            final Matrix other
    )
            throws MatrixMultiplyDimMismatchException
    {
        Preconditions.checkNotNull(other, "Other matrix must not be null");

        if (A.cols() != other.rows()) {
            throw new MatrixMultiplyDimMismatchException();
        }

        var resultData = new double[A.rows()][other.cols()];

        for (int i = 0; i < A.rows(); i++) {
            for (int j = 0; j < other.cols(); j++) {
                double sum = 0;
                for (int k = 0; k < A.cols(); k++) {
                    sum += A.get(i,k) * other.get(k,j);
                }
                resultData[i][j] = sum;
            }
        }


        return new Matrix(resultData);
    }
    public Matrix rref() {
        var rrefMatrix = A.copy();
        for (var operation : operationsRREF) {
            operation.apply(rrefMatrix);
        }
        return rrefMatrix;
    }
    public Matrix inverse()
            throws InverseNotAvailableException
    {
        if ((A.rows() != A.cols()) || (determinant == 0)) {
            throw new InverseNotAvailableException();
        }
        var inverse = getIdentity(A.rows());
        for (var operation : operationsRREF) {
            operation.apply(inverse);
        }
        return inverse;
    }
    public double determinant() {
        return determinant;
    }

    // Relative to the largest absolute entry of the input; anything at or below this value counts as zero.
    private static final double RELATIVE_TOLERANCE = 1e-10;

    public static void swapRows(
            final Matrix A, final int sourceRow, final int targetRow
    ) {
        Preconditions.checkNotNull(A, "Matrix must not be null");
        Preconditions.checkArgument(sourceRow >= 0, "Source row index must be non-negative");
        Preconditions.checkArgument(sourceRow < A.rows(), "Source row index out of bounds");
        Preconditions.checkArgument(targetRow >= 0, "Target row index must be non-negative");
        Preconditions.checkArgument(targetRow < A.rows(), "Target row index out of bounds");

        double[] temp = A.data()[sourceRow];
        A.data()[sourceRow] = A.data()[targetRow];
        A.data()[targetRow] = temp;
    }
    public static void multiplyRow(
            final Matrix A, final int targetRow, final double scalar
    ) {
        Preconditions.checkNotNull(A, "Matrix must not be null");
        Preconditions.checkArgument(targetRow >= 0, "Target row index must be non-negative");
        Preconditions.checkArgument(targetRow < A.rows(), "Target row index out of bounds");
        Preconditions.checkArgument(scalar != 0, "Scalar must be non-zero");
        Preconditions.checkArgument(Double.isFinite(scalar), "Scalar must be finite");

        for (int j = 0; j < A.cols(); j++) {
            A.data()[targetRow][j] *= scalar;
        }
    }
    public static void reduceRow(
            final Matrix A, final int sourceRow, final int targetRow, final double scalar
    ) {
        Preconditions.checkNotNull(A, "Matrix must not be null");
        Preconditions.checkArgument(sourceRow < A.rows(), "Source row index out of bounds");
        Preconditions.checkArgument(sourceRow >= 0, "Source row index must be non-negative");
        Preconditions.checkArgument(targetRow < A.rows(), "Target row index out of bounds");
        Preconditions.checkArgument(targetRow >= 0, "Target row index must be non-negative");
        Preconditions.checkArgument(sourceRow != targetRow, "Source and target rows must differ");
        Preconditions.checkArgument(scalar != 0, "Scalar must be non-zero");
        Preconditions.checkArgument(Double.isFinite(scalar), "Scalar must be finite");

        for (int j = 0; j < A.cols(); j++) {
            A.data()[targetRow][j] += scalar * A.data()[sourceRow][j];
        }
    }
    private static List<OperationRREF> getOperationsRREF(
            final double[][] data
    ) {
        final List<OperationRREF> operations = new ArrayList<>();

        // Matrix's constructor deep-copies, so the caller's array (A.data()) is never modified.
        final Matrix working = new Matrix(data);
        final double tolerance = getTolerance(working);

        int pivotRow = 0;
        for (int pivotCol = 0; pivotCol < working.cols() && pivotRow < working.rows(); pivotCol++) {
            final int bestRow = findPivotRow(working, pivotRow, pivotCol, tolerance);
            if (bestRow == -1) {
                continue; // no usable pivot in this column: move right, stay on the same row
            }

            if (bestRow != pivotRow) {
                applyAndRecord(working, operations, new Type1(pivotRow, bestRow));
            }

            final double pivot = working.get(pivotRow, pivotCol);
            if (pivot != 1.0) {
                applyAndRecord(working, operations, new Type2(pivotRow, 1.0 / pivot));
            }

            eliminateColumn(working, operations, pivotRow, pivotCol, tolerance);
            pivotRow++;
        }

        return operations;
    }
    // Applies the operation to the working copy and logs it. This is the same apply() path that
    // rref() and inverse() replay later, so the simulation and the log can't drift apart.
    private static void applyAndRecord(
            final Matrix working,
            final List<OperationRREF> operations,
            final OperationRREF operation
    ) {
        operation.apply(working);
        operations.add(operation);
    }
    // Partial pivoting: largest magnitude in the column at or below startRow.
    // Ties go to the upper row (fewer swaps). Returns -1 if nothing exceeds the tolerance.
    private static int findPivotRow(
            final Matrix A,
            final int startRow,
            final int col,
            final double tolerance
    ) {
        int bestRow = -1;
        double bestMagnitude = tolerance;
        for (int row = startRow; row < A.rows(); row++) {
            final double magnitude = Math.abs(A.get(row, col));
            if (magnitude > bestMagnitude) {
                bestMagnitude = magnitude;
                bestRow = row;
            }
        }
        return bestRow;
    }
    // Clears every other entry in the pivot column, above and below (Gauss-Jordan).
    private static void eliminateColumn(
            final Matrix working,
            final List<OperationRREF> operations,
            final int pivotRow,
            final int pivotCol,
            final double tolerance
    ) {
        // Read the actual pivot rather than assuming 1: after Type2 it can be 1 +/- an ulp.
        final double pivot = working.get(pivotRow, pivotCol);
        for (int row = 0; row < working.rows(); row++) {
            if (row == pivotRow) {
                continue;
            }
            final double entry = working.get(row, pivotCol);
            if (Math.abs(entry) > tolerance) {
                applyAndRecord(working, operations, new Type3(pivotRow, row, -entry / pivot));
            }
        }
    }
    private static double getDeterminant(
            final Matrix A,
            final List<OperationRREF> operationsRREF
    )
            throws DeterminantDoesNotExistException
    {
        if (A.rows() != A.cols()) {
            throw new DeterminantDoesNotExistException();
        }

        // A square matrix is singular iff its RREF is not the identity (some column got no pivot).
        // In that case the first pivotless column leaves a 0 on the diagonal.
        var reduced = A.copy();
        for (var operation : operationsRREF) {
            operation.apply(reduced);
        }
        for (int i = 0; i < reduced.rows(); i++) {
            if (Math.abs(reduced.get(i, i)) < 0.5) {
                return 0;
            }
        }

        // RREF(A) = I and det(I) = 1. Type1 flips the sign, Type2 scales the determinant
        // by its scalar, Type3 leaves it unchanged. So det(A) = (-1)^swaps / product(scalars).
        double determinant = 1;
        for (var operation : operationsRREF) {
            if (operation instanceof Type1) {
                determinant = -determinant;
            } else if (operation instanceof Type2 scale) {
                determinant /= scale.scalar();
            }
        }
        return determinant;
    }
    private static Matrix getIdentity(
            final int dimension
    ) {
        double[][] identityData = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            identityData[i][i] = 1;
        }
        return new Matrix(identityData);
    }
    private static double getTolerance(
            final Matrix A
    ) {
        double maxAbs = 0;
        for (int i = 0; i < A.rows(); i++) {
            for (int j = 0; j < A.cols(); j++) {
                maxAbs = Math.max(maxAbs, Math.abs(A.get(i, j)));
            }
        }
        return RELATIVE_TOLERANCE * maxAbs;
    }
}
