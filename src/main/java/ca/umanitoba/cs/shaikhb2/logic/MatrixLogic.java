package ca.umanitoba.cs.shaikhb2.logic;

import ca.umanitoba.cs.shaikhb2.domain.Exceptions.InverseNotAvailableException;
import ca.umanitoba.cs.shaikhb2.domain.Exceptions.RREFNotCalculatedException;
import ca.umanitoba.cs.shaikhb2.domain.Matrix;
import ca.umanitoba.cs.shaikhb2.domain.RREF.OperationRREF;
import com.google.common.base.Preconditions;

import java.util.ArrayList;
import java.util.List;

public final class MatrixLogic {

    private final Matrix A;
    private final double determinant;
    private final List<OperationRREF> operationsRREF;

    public MatrixLogic(final Matrix A) {
        this.A = A;
        this.operationsRREF = getOperationsRREF(A.data());
        this.determinant = getDeterminant(operationsRREF);
    }

    public Matrix multiply(final Matrix other) {
        Preconditions.checkNotNull(other, "Other matrix must not be null");
        Preconditions.checkArgument(A.cols() == other.rows(), "Incompatible matrix dimensions for matrix multiplication");

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
    public Matrix rref() throws RREFNotCalculatedException {
        if (operationsRREF.isEmpty()) {
            throw new RREFNotCalculatedException();
        } else {
            var rrefMatrix = A.copy();
            for (var operation : operationsRREF) {
                operation.apply(rrefMatrix);
            }
            return rrefMatrix;
        }
    }
    public Matrix inverse() throws InverseNotAvailableException, RREFNotCalculatedException {
        if (operationsRREF.isEmpty()) {
            throw new RREFNotCalculatedException();
        } else if ((A.rows() != A.cols()) || (determinant == 0)) {
            throw new InverseNotAvailableException();
        } else {
            var inverse = getIdentity(A.rows());
            for (var operation : operationsRREF) {
                operation.apply(inverse);
            }
            return inverse;
        }
    }

    public static void swapRows(final Matrix A, final int sourceRow, final int targetRow) {
        Preconditions.checkState(sourceRow < A.rows(), "Row 1 index out of bounds");
        Preconditions.checkState(targetRow < A.rows(), "Row 2 index out of bounds");
        double[] temp = A.data()[sourceRow];
        A.data()[sourceRow] = A.data()[targetRow];
        A.data()[targetRow] = temp;
    }
    public static void multiplyRow(final Matrix A, final int targetRow, final double scalar) {
        Preconditions.checkState(targetRow < A.rows(), "Row index out of bounds");
        Preconditions.checkState(scalar != 0, "Scalar must be non-zero");
        for (int j = 0; j < A.cols(); j++) {
            A.data()[targetRow][j] *= scalar;
        }
    }
    public static void reduceRow(final Matrix A, final int sourceRow, final int targetRow, final double scalar) {
        Preconditions.checkState(sourceRow < A.rows(), "Source row index out of bounds");
        Preconditions.checkState(sourceRow >= 0, "Source row index must be non-negative");
        Preconditions.checkState(targetRow < A.rows(), "Target row index out of bounds");
        Preconditions.checkState(targetRow >= 0, "Target row index must be non-negative");
        Preconditions.checkState(scalar != 0, "Scalar must be non-zero");
        for (int j = 0; j < A.cols(); j++) {
            A.data()[targetRow][j] += scalar * A.data()[sourceRow][j];
        }
    }

    //!fixme: implement RREF and determinant calculation
    private static List<OperationRREF> getOperationsRREF(final double[][] data) {
        List<OperationRREF> operationsRREF = new ArrayList<>();
        // Implementation of RREF operations to be added here
        return operationsRREF;
    }
    private static double getDeterminant(final List<OperationRREF> operationsRREF) {
        double determinant = 0;
        // Implementation of determinant calculation based on RREF operations to be added here
        return determinant;
    }
    private static Matrix getIdentity(final int dimension) {
        double[][] identityData = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            identityData[i][i] = 1;
        }
        return new Matrix(identityData);
    }

}
