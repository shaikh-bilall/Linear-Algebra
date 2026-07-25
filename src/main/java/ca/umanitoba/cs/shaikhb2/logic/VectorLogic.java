package ca.umanitoba.cs.shaikhb2.logic;

import ca.umanitoba.cs.shaikhb2.domain.Vector;
import com.google.common.base.Preconditions;

public final class VectorLogic {
    public static void swapElements(final Vector v, final int sourceIndex, final int targetIndex) {
        double temp = v.data()[sourceIndex];
        v.data()[sourceIndex] = v.data()[targetIndex];
        v.data()[targetIndex] = temp;
    }
    public static void multiplyElement(final Vector v, final int targetIndex, final double scalar) {
        Preconditions.checkState(targetIndex >= 0, "Index must be non-negative");
        Preconditions.checkState(targetIndex < v.data().length, "Index out of bounds");
        Preconditions.checkState(scalar != 0, "Scalar must be non-zero");
        v.data()[targetIndex] *= scalar;
    }
    public static void reduceElement(final Vector v, final int sourceIndex, final int targetIndex, final double scalar) {
        Preconditions.checkState(sourceIndex >= 0, "Source index must be non-negative");
        Preconditions.checkState(sourceIndex < v.data().length, "Source index out of bounds");
        Preconditions.checkState(targetIndex >= 0, "Target index must be non-negative");
        Preconditions.checkState(targetIndex < v.data().length, "Target index out of bounds");
        Preconditions.checkState(scalar != 0, "Scalar must be non-zero");
        v.data()[targetIndex] += scalar * v.data()[sourceIndex];
    }
}
