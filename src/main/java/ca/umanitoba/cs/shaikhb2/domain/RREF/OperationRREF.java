package ca.umanitoba.cs.shaikhb2.domain.RREF;

import ca.umanitoba.cs.shaikhb2.domain.Matrix;
import ca.umanitoba.cs.shaikhb2.domain.Vector;

public interface OperationRREF {
    void apply(Matrix A);
    void apply(Vector v);
}
