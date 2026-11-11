package linearAlgebra.domain;

import com.google.common.base.Preconditions;

import java.util.List;

/**
 * Validation of index permutations shared by {@link Matrix} and {@link Vector}.
 */
final class Permutations {

    private Permutations() {
    }

    /**
     * Checks that a list holds every index from {@code 0} to {@code size - 1} exactly once.
     *
     * @param indices the list to check
     * @param size    the number of indices the list must hold
     * @param owner   the name of the calling method, used as a prefix in failure messages
     */
    static void checkPermutation(
        final List<Integer> indices,
        final int size,
        final String owner
    ) {
        Preconditions.checkNotNull(
            indices,
            "%s: shuffled indices list must not be null",
            owner
        );
        Preconditions.checkArgument(
            indices.size() == size,
            "%s: shuffled indices list size must match: %s != %s",
            owner,
            indices.size(),
            size
        );

        boolean[] seen = new boolean[size];
        for (Integer index : indices) {
            Preconditions.checkNotNull(
                index,
                "%s: shuffled indices must not contain null",
                owner
            );
            Preconditions.checkArgument(
                index >= 0 &&
                    index < size,
                "%s: shuffled index out of bounds: %s",
                owner,
                index
            );
            Preconditions.checkArgument(
                !seen[index],
                "%s: shuffled indices must not repeat: %s",
                owner,
                index
            );
            seen[index] = true;
        }
    }

}