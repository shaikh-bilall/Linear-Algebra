package linearAlgebra.domain;

import com.google.common.base.Preconditions;
import org.jspecify.annotations.NonNull;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * A vector of finite real numbers.
 * This class stores an immutable copy of the data and does not allow any modifications.
 *
 * @param data the entries in order
 */
public record Vector(
    double[] data
) implements Iterable<Double> {

    /**
     * Stores a vector from the given data.
     *
     * @param data the entries
     */
    public Vector(
        final double[] data
    ) {
        checkVector(data);

        this.data = data.clone();
    }

    /**
     * Creates a vector of ones.
     *
     * @param size the number of entries
     */
    public Vector(
        final int size
    ) {
        this(ones(size));
    }

    /**
     * @return the entries of this vector
     */
    @Override
    public double[] data() {
        return data.clone();
    }

    /**
     * Returns the entry at the specified index.
     *
     * @param index the zero-based index
     * @return the entry at that index
     */
    public double get(
        final int index
    ) {
        Preconditions.checkElementIndex(
            index,
            data.length,
            "Vector.get: index"
        );

        return data[index];
    }

    /**
     * @return the number of entries.
     */
    public int dimension() {
        return data.length;
    }

    /**
     * @return an {@link Iterator} over the entries
     */
    @Override
    @NonNull
    public Iterator<Double> iterator() {
        return new Iterator<>() {
            private int currentIndex = 0;

            /**
             * Tells whether another entry remains.
             *
             * @return {@code true} if {@link #next} would return an entry
             */
            @Override
            public boolean hasNext() {
                return currentIndex < data.length;
            }

            /**
             * Returns the next entry.
             *
             * @return the next entry
             */
            @Override
            public Double next()
                throws NoSuchElementException
            {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }

                return data[currentIndex++];
            }
        };
    }

    /**
     * @return a new vector with the same entries
     */
    public Vector copy() {
        return new Vector(data);
    }

    /**
     * Compares this vector to another object for equality, entry by entry.
     * Two vectors are equal when they have the same length and
     * every pair of matching entries is numerically equal.
     *
     * @param otherObject the object to compare with
     * @return {@code true} if {@code otherObject} is a vector with the same length and entries
     */
    @Override
    public boolean equals(
        final Object otherObject
    ) {
        boolean equals;
        if (!(otherObject instanceof Vector otherVector)) {
            equals = false;
        } else {
            equals = entriesEqual(
                data,
                otherVector.data
            );
        }
        return equals;
    }

    /**
     * @return the hash code that agrees with {@link #equals}
     */
    @Override
    public int hashCode() {
        return entriesHashCode(data);
    }

    /**
     * Tells whether two arrays have the same length and numerically equal entries.
     *
     * @param first  the first array
     * @param second the second array
     * @return {@code true} if the arrays have the same length and
     * all pairs of matching entries are equal
     */
    public static boolean entriesEqual(
        final double[] first,
        final double[] second
    ) {
        Preconditions.checkNotNull(
            first,
            "Vector.entriesEqual: first array must not be null"
        );
        Preconditions.checkNotNull(
            second,
            "Vector.entriesEqual: second array must not be null"
        );

        boolean entriesEqual;
        if (first.length != second.length) {
            entriesEqual = false;
        } else {
            entriesEqual = Arrays.equals(
                normalizeZeros(first),
                normalizeZeros(second)
            );
        }
        return entriesEqual;
    }

    /**
     * Returns a hash code of the entries of an array that agrees with {@link #entriesEqual}.
     *
     * @param entries the array to hash
     * @return the hash code
     */
    public static int entriesHashCode(
        final double[] entries
    ) {
        Preconditions.checkNotNull(
            entries,
            "Vector.entriesHashCode: entries must not be null"
        );

        return Arrays.hashCode(normalizeZeros(entries));
    }

    // Returns a copy with every signed zero represented as positive zero.
    private static double[] normalizeZeros(
        final double[] entries
    ) {
        double[] normalizedEntries = entries.clone();
        for (int i = 0; i < normalizedEntries.length; i++) {
            if (normalizedEntries[i] == 0.0) {
                normalizedEntries[i] = 0.0;
            }
        }
        return normalizedEntries;
    }

    // Checks the cheap structural invariants: the data exists and has at least one entry.
    private static void checkVector(
        final double[] data
    ) {
        Preconditions.checkNotNull(
            data,
            "Vector: data must not be null"
        );
        Preconditions.checkArgument(
            data.length > 0,
            "Vector: vector must have at least one element"
        );
        Preconditions.checkArgument(
            Arrays.stream(data).allMatch(Double::isFinite),
            "Vector: all entries must be finite numbers"
        );
    }

    // Returns a new array of the given size, filled with ones.
    private static double[] ones(
        final int size
    ) {
        Preconditions.checkArgument(
            size > 0,
            "Vector: size must be positive"
        );

        final double[] ones = new double[size];
        Arrays.fill(ones, 1);

        return ones;
    }
}
