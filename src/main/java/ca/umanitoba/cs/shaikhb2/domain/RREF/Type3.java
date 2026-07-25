package ca.umanitoba.cs.shaikhb2.domain.RREF;

import ca.umanitoba.cs.shaikhb2.domain.Matrix;
import ca.umanitoba.cs.shaikhb2.domain.Vector;
import ca.umanitoba.cs.shaikhb2.logic.MatrixLogic;
import ca.umanitoba.cs.shaikhb2.logic.VectorLogic;

public record Type3(int sourceRow, int targetRow, double scalar) implements OperationRREF {
    public void apply(Matrix A) {
        // Implementation for applying Type 1 operation on Matrix A
        MatrixLogic.reduceRow(A, sourceRow, targetRow, scalar);
    }
    public void apply(Vector v) {
        // Implementation for applying Type 1 operation on Vector v
        VectorLogic.reduceElement(v, sourceRow, targetRow, scalar);
    }
}
