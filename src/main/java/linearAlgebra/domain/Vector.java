package linearAlgebra.domain;

import com.google.common.base.Preconditions;
import org.jspecify.annotations.NonNull;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * A vector of finite real numbers, with basic arithmetic: addition, scaling, dot product and length. It can be
 * iterated over entry by entry.
 *
 * @param data the entries in order; {@code data()} returns a copy of this array
 */
public record Vector(double[] data) implements Iterable<Double> {

    /**
     * Creates a vector from the given entries.
     *
     * @param data the entries in order
     */
    public Vector(
        final double[] data
    ) {
        this.data = data.clone();

        checkVector();
    }
    /**
     * Creates a zero vector.
     *
     * @param size the number of entries, which must be positive
     */
    public Vector(
        final int size
    ) {
        this(zeros(size));
    }

    /**
     * Returns the entries in order.
     *
     * @return a copy of the entries; changing it does not affect this vector
     */
    @Override
    public double[] data() {
        checkVector();

        return data.clone();
    }
    /**
     * Returns the entrywise sum of this vector and another of the same length.
     *
     * @param other the vector to add
     * @return a new vector holding the sum
     */
    public Vector add(
        final Vector other
    ) {
        checkVector();
        checkOtherVector(
            other,
            "Vector.add"
        );

        double[] result = data.clone();
        for (int i = 0; i < data.length; i++) {
            result[i] += other.data[i];
            Preconditions.checkState(
                Double.isFinite(result[i]),
                "Vector.add: sum entry must be finite: index %s",
                i
            );
        }
        return new Vector(result);
    }
    /**
     * Returns this vector multiplied by a scalar.
     *
     * @param scalar the finite factor to multiply every entry by
     * @return a new vector holding the product
     */
    public Vector scale(
        final double scalar
    ) {
        checkVector();
        Preconditions.checkArgument(
            Double.isFinite(scalar),
            "Vector.scale: scalar must be finite: %s",
            scalar
        );

        double[] result = data.clone();
        for (int i = 0; i < data.length; i++) {
            result[i] *= scalar;
            Preconditions.checkState(
                Double.isFinite(result[i]),
                "Vector.scale: product entry must be finite: index %s",
                i
            );
        }
        return new Vector(result);
    }
    /**
     * Returns the dot product of this vector and another of the same length.
     *
     * @param other the vector to take the dot product with
     * @return the dot product
     */
    public double dot(
        final Vector other
    ) {
        checkVector();
        checkOtherVector(
            other,
            "Vector.dot"
        );

        double result = 0;
        for (int i = 0; i < data.length; i++) {
            result += this.data[i] * other.data[i];
        }
        Preconditions.checkState(
            Double.isFinite(result),
            "Vector.dot: dot product must be finite"
        );
        return result;
    }
    /**
     * Returns the Euclidean length of this vector.
     *
     * @return the length, never negative
     */
    public double norm() {
        checkVector();

        double norm = 0;
        for (double v : data) {
            norm = Math.hypot(norm, v); // hypot avoids overflowing on the intermediate squares
        }
        Preconditions.checkState(
            Double.isFinite(norm),
            "Vector.norm: norm must be finite"
        );
        return norm;
    }
    /**
     * Reorders the entries: entry {@code i} of the result is entry {@code shuffledIndices.get(i)} of this vector.
     *
     * @param shuffledIndices each index from {@code 0} to {@code length() - 1}, exactly once
     * @return a new vector with the entries reordered
     */
    public Vector permute(
        final List<Integer> shuffledIndices
    ) {
        checkVector();
        Permutations.checkPermutation(
            shuffledIndices,
            data.length,
            "Vector.permute"
        );

        double[] shuffledData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            shuffledData[i] = data[shuffledIndices.get(i)];
        }
        return new Vector(shuffledData);
    }
    /**
     * Returns a single entry.
     *
     * @param index the zero-based position
     * @return the entry at that position
     */
    public double get(
        final int index
    ) {
        checkVector();
        Preconditions.checkArgument(
            index >= 0 &&
            index < data.length,
            "Vector.get: index out of bounds: %s",
            index
        );

        return data[index];
    }
    /**
     * Returns the number of entries.
     *
     * @return the length
     */
    public int length() {
        checkVector();

        return data.length;
    }
    /**
     * Iterates over the entries in order.
     *
     * @return an iterator over the entries
     */
    @Override
    @NonNull
    public Iterator<Double> iterator() {
        checkVector();

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
     * Returns a new vector that is a copy of this one.
     *
     * @return a new vector with the same entries
     */
    public Vector copy() {
        checkVector();

        return new Vector(data);
    }
    /**
     * Compares this vector to another object for equality, entry by entry. Two vectors are equal when they have the
     * same length and every pair of matching entries is numerically equal, so {@code 0.0} and {@code -0.0} count as
     * the same entry.
     *
     * @param otherObject the object to compare with
     * @return {@code true} if {@code otherObject} is a vector with the same length and entries
     */
    @Override
    public boolean equals(
        final Object otherObject
    ) {
        checkVector();

        boolean equals;
        if (!(otherObject instanceof Vector otherVector)) {
            equals = false;
        } else {
            otherVector.checkVector();
            equals = entriesEqual(data, otherVector.data);
        }
        return equals;
    }
    /**
     * Returns a hash code that agrees with {@link #equals}: vectors that are equal have the same hash code.
     *
     * @return the hash code of the entries
     */
    @Override
    public int hashCode() {
        checkVector();

        return entriesHashCode(data);
    }
    /**
     * Tells whether two arrays have the same length and numerically equal entries, so {@code 0.0} equals {@code -0.0}.
     *
     * @param first  the first array
     * @param second the second array
     * @return {@code true} if the arrays have the same length and every pair of matching entries is equal
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
     * Returns a hash code of the entries of an array that agrees with {@link #entriesEqual}, so {@code 0.0} and
     * {@code -0.0} give the same hash code.
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

    // Checks that the other vector exists and has the same length as this one.
    private void checkOtherVector(
        final Vector other,
        final String owner
    ) {
        Preconditions.checkNotNull(
            other,
            "%s: other vector must not be null",
            owner
        );
        Preconditions.checkArgument(
            other.length() == this.length(),
            "%s: vectors must have the same length: %s != %s",
            owner,
            other.length(),
            this.length()
        );
        Preconditions.checkArgument(
            Arrays.stream(data).allMatch(Double::isFinite),
            "%s: all entries must be finite numbers",
            owner
        );
    }
    // Checks the cheap structural invariants: the data exists and has at least one entry.
    private void checkVector() {
        Preconditions.checkNotNull(
            data,
            "Vector: data must not be null"
        );
        Preconditions.checkState(
            data.length > 0,
            "Vector: vector must have at least one element"
        );
        Preconditions.checkState(
            Arrays.stream(data).allMatch(Double::isFinite),
            "Vector: all entries must be finite numbers"
        );
    }
    // Returns a new array of the given size, filled with zeros.
    private static double[] zeros(
        final int size
    ) {
        Preconditions.checkState(
            size > 0,
            "Vector: size must be positive: %s",
            size
        );
        return new double[size];
    }

}
