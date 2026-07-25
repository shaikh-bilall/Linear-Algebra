package ca.umanitoba.cs.shaikhb2.domain.RREF;

import ca.umanitoba.cs.shaikhb2.domain.Matrix;
import ca.umanitoba.cs.shaikhb2.domain.Vector;
import ca.umanitoba.cs.shaikhb2.logic.MatrixLogic;
import ca.umanitoba.cs.shaikhb2.logic.VectorLogic;

public record Type1(int sourceRow, int targetRow) implements OperationRREF {

    public void apply(Matrix A) {
        // Implementation for applying Type 1 operation on Matrix A
        MatrixLogic.swapRows(A, sourceRow, targetRow);

    }
    public void apply(Vector v) {
        // Implementation for applying Type 1 operation on Vector v
        VectorLogic.swapElements(v, sourceRow, targetRow);
    }
}
