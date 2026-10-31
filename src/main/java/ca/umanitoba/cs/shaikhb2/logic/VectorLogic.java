package ca.umanitoba.cs.shaikhb2.logic;

import ca.umanitoba.cs.shaikhb2.domain.Vector;
import com.google.common.base.Preconditions;

public final class VectorLogic {
    public static void swapElements(
            final Vector v,
            final int sourceIndex,
            final int targetIndex
    ) {
        Preconditions.checkArgument(targetIndex >= 0, "Index must be non-negative");
        Preconditions.checkArgument(targetIndex < v.data().length, "Index out of bounds");
        Preconditions.checkArgument(sourceIndex >= 0, "Source index must be non-negative");
        Preconditions.checkArgument(sourceIndex < v.data().length, "Source index out of bounds");

        double temp = v.data()[sourceIndex];
        v.data()[sourceIndex] = v.data()[targetIndex];
        v.data()[targetIndex] = temp;
    }
    public static void multiplyElement(
            final Vector v,
            final int targetIndex,
            final double scalar
    ) {
        Preconditions.checkArgument(targetIndex >= 0, "Index must be non-negative");
        Preconditions.checkArgument(targetIndex < v.data().length, "Index out of bounds");
        Preconditions.checkArgument(scalar != 0, "Scalar must be non-zero");
        Preconditions.checkArgument(Double.isFinite(scalar) && !Double.isNaN(scalar), "Scalar must be finite");

        v.data()[targetIndex] *= scalar;
    }
    public static void reduceElement(
            final Vector v,
            final int sourceIndex,
            final int targetIndex,
            final double scalar
    ) {
        Preconditions.checkArgument(sourceIndex >= 0, "Source index must be non-negative");
        Preconditions.checkArgument(sourceIndex < v.data().length, "Source index out of bounds");
        Preconditions.checkArgument(targetIndex >= 0, "Target index must be non-negative");
        Preconditions.checkArgument(targetIndex < v.data().length, "Target index out of bounds");
        Preconditions.checkArgument(sourceIndex != targetIndex, "Source and target indices must be different");
        Preconditions.checkArgument(scalar != 0, "Scalar must be non-zero");
        Preconditions.checkArgument(Double.isFinite(scalar) && !Double.isNaN(scalar), "Scalar must be finite");

        v.data()[targetIndex] += scalar * v.data()[sourceIndex];
    }
}
