package linearAlgebra.domain;

import com.google.common.base.Preconditions;

import java.util.List;

/**
 * Validation of index permutations shared by {@link Matrix} and {@link Vector}.
 */
public final class Permutations {

    // Prevents instantiation.
    private Permutations() {
    }

    /**
     * Checks that a list holds every index from {@code 0} to {@code size - 1} exactly once.
     *
     * @param indices the list to be checked
     * @param size    the number of indices the list must hold
     */
    public static void checkPermutation(
        final List<Integer> indices,
        final int size
    ) {
        Preconditions.checkNotNull(
            indices,
            "Permutations.checkPermutation: shuffled indices list must not be null"
        );
        Preconditions.checkArgument(
            indices.size() == size,
            "Permutations.checkPermutation: number of shuffled indices and size must match"
        );

        boolean[] seen = new boolean[size];
        for (Integer index : indices) {
            Preconditions.checkNotNull(
                index,
                "Permutations.checkPermutation: shuffled indices must not contain null"
            );
            Preconditions.checkElementIndex(
                index,
                size,
                "Permutations.checkPermutation: shuffled index out of bounds"
            );
            Preconditions.checkArgument(
                !seen[index],
                "Permutations.checkPermutation: shuffled indices must not repeat"
            );

            seen[index] = true;
        }
    }
}
