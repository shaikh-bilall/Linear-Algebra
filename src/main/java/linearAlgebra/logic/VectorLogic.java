package linearAlgebra.logic;

import com.google.common.base.Preconditions;
import linearAlgebra.domain.Permutations;
import linearAlgebra.domain.Vector;

import java.util.List;

/**
 * Defines basic mathematical operations for {@link Vector}s.
 */
public final class VectorLogic {

    //Prevents instantiation.
    private VectorLogic() {
    }

    /**
     * Computes the entry-wise sum of two {@link Vector}s.
     *
     * @param left the {@link Vector} to add, on the left of the operation
     * @param right the {@link Vector} to add, on the right of the operation
     * @return a {@link Vector} holding the sum
     */
    public static Vector add(
        final Vector left,
        final Vector right
    ) {
        Preconditions.checkNotNull(
            left,
            "Vector.add: left vector must not be null"
        );
        Preconditions.checkNotNull(
            right,
            "Vector.add: right vector must not be null"
        );
        Preconditions.checkArgument(
            left.dimension() == right.dimension(),
            "Vector.add: dimensions must match for both vectors"
        );

        final double[] rightData = right.data();
        final double[] result = left.data();

        for (int i = 0; i < result.length; i++) {
            result[i] += rightData[i];

            Preconditions.checkState(
                Double.isFinite(result[i]),
                "Vector.add: sum entry must be finite"
            );
        }

        return new Vector(result);
    }

    /**
     * Scales the given {@link Vector} by the given scalar.
     *
     * @param vector the {@link Vector} to scale
     * @param scalar the finite factor to multiply every entry by
     * @return a {@link Vector} holding the product
     */
    public static Vector scale(
        final Vector vector,
        final double scalar
    ) {
        Preconditions.checkNotNull(
            vector,
            "Vector.scale: vector must not be null"
        );
        Preconditions.checkArgument(
            Double.isFinite(scalar),
            "Vector.scale: scalar must be finite"
        );

        double[] result = vector.data();
        for (int i = 0; i < result.length; i++) {
            result[i] *= scalar;

            Preconditions.checkState(
                Double.isFinite(result[i]),
                "Vector.scale: product entry must be finite"
            );
        }

        return new Vector(result);
    }

    /**
     * Computes the dot product of the given {@link Vector}s.
     *
     * @param left the {@link Vector} to take the dot product with, on the left of the operation
     * @param right the {@link Vector} to take the dot product with, on the right of the operation
     * @return the dot product
     */
    public static double dot(
        final Vector left,
        final Vector right
    ) {
        Preconditions.checkNotNull(
            left,
            "Vector.dot: left vector must not be null"
        );
        Preconditions.checkNotNull(
            right,
            "Vector.dot: right vector must not be null"
        );
        Preconditions.checkArgument(
            left.dimension() == right.dimension(),
            "Vector.dot: dimensions must match for both vectors"
        );

        final double[] leftData = left.data();
        final double[] rightData = right.data();

        double result = 0;
        for (int i = 0; i < left.dimension(); i++) {
            result += leftData[i] * rightData[i];
        }

        Preconditions.checkState(
            Double.isFinite(result),
            "Vector.dot: dot product must be finite"
        );

        return result;
    }

    /**
     * Calculates the Euclidean length of the given {@link Vector}.
     *
     * @param vector the {@link Vector} to calculate the Euclidean length of
     * @return the Euclidean length
     */
    public static double norm(
        final Vector vector
    ) {
        Preconditions.checkNotNull(
            vector,
            "Vector.norm: vector must not be null"
        );

        final double[] data = vector.data();

        double norm = 0;
        for (double v : data) {
            norm = Math.hypot(norm, v);
        }

        Preconditions.checkState(
            Double.isFinite(norm),
            "Vector.norm: norm must be finite"
        );

        return norm;
    }

    /**
     * Reorders the {@link Vector} entries according to the given specification.
     *
     * @param vector the {@link Vector} to reorder
     * @param shuffledIndices a permutation of the indices
     * @return a {@link Vector} with the reordered entries
     */
    public Vector permute(
        final Vector vector,
        final List<Integer> shuffledIndices
    ) {
        Preconditions.checkNotNull(
            vector,
            "Vector.permute: vector must not be null"
        );
        Permutations.checkPermutation(
            shuffledIndices,
            vector.dimension()
        );

        final double[] shuffledData = vector.data();
        for (int i = 0; i < vector.dimension(); i++) {
            double temp = shuffledData[i];
            shuffledData[i] = shuffledData[shuffledIndices.get(i)];
            shuffledData[shuffledIndices.get(i)] = temp;

        }

        return new Vector(shuffledData);
    }
}
