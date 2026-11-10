package linearAlgebra.logic;

import linearAlgebra.domain.Vector;
import com.google.common.base.Preconditions;

/**
 * Elementary operations on the entries of a vector, each applied in place: swapping two entries, scaling one entry,
 * and adding a multiple of one entry to another.
 */
public final class VectorLogic {
    /**
     * Swaps two entries of a vector, in place.
     *
     * @param v           the vector to modify
     * @param sourceIndex the zero-based index of one of the entries
     * @param targetIndex the zero-based index of the other entry
     * @throws NullPointerException     if {@code v} is {@code null}
     * @throws IllegalArgumentException if either index is out of range
     */
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
    /**
     * Multiplies one entry of a vector by a scalar, in place.
     *
     * @param v           the vector to modify
     * @param targetIndex the zero-based index of the entry to scale
     * @param scalar      the factor, which must be nonzero and finite
     * @throws NullPointerException     if {@code v} is {@code null}
     * @throws IllegalArgumentException if {@code targetIndex} is out of range, or {@code scalar} is zero or not finite
     */
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
    /**
     * Adds a multiple of one entry of a vector to another entry, in place.
     *
     * @param v           the vector to modify
     * @param sourceIndex the zero-based index of the entry whose multiple is added; it is left unchanged
     * @param targetIndex the zero-based index of the entry that receives the multiple
     * @param scalar      the multiple, which must be nonzero and finite
     * @throws NullPointerException     if {@code v} is {@code null}
     * @throws IllegalArgumentException if either index is out of range, the two indices are the same, or
     *                                  {@code scalar} is zero or not finite
     */
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
