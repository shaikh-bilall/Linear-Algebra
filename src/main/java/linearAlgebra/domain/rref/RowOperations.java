package linearAlgebra.domain.rref;

import com.google.common.base.Preconditions;
import org.jspecify.annotations.NonNull;

import java.util.Iterator;
import java.util.List;
import java.util.function.Consumer;

/**
 * An ordered sequence of {@link RowOperation}s.
 *
 * @param operations the {@link RowOperation}s, in order
 */
public record RowOperations(
    List<RowOperation> operations
) implements Iterable<RowOperation> {

    /**
     * Creates an ordered sequence of {@link RowOperation}s.
     *
     * @param operations the {@link RowOperation}s, in order
     */
    public RowOperations(
        List<RowOperation> operations
    ) {
        checkRowOperations(operations);

        this.operations = List.copyOf(operations);
    }

    /**
     * Returns the {@link RowOperation} at the given index.
     *
     * @param index the zero-based index of the operation
     * @return the {@link RowOperation} at {@code index}
     */
    public RowOperation get(
        int index
    ) {
        Preconditions.checkElementIndex(
            index,
            operations.size(),
            "RowOperations: index"
        );
        return operations.get(index);
    }

    /**
     * @return the number of {@link RowOperation}s
     */
    public int size() {
        return operations.size();
    }

    /**
     * @return {@code true} if there are no {@link RowOperation}s
     */
    public boolean isEmpty() {
        return operations.isEmpty();
    }

    /**
     * @return the {@link RowOperation}s
     */
    @Override
    public List<RowOperation> operations() {
        return List.copyOf(operations);
    }

    /**
     * @return an {@link Iterator} over the {@link RowOperation}s
     */
    @Override
    @NonNull
    public Iterator<RowOperation> iterator() {
        return operations.iterator();
    }

    /**
     * Performs the given {@code action} for each {@link RowOperation}, in order.
     *
     * @param action the action to perform on each {@link RowOperation}
     */
    @Override
    public void forEach(
        Consumer<? super RowOperation> action
    ) {
        Preconditions.checkNotNull(
            action,
            "RowOperations: action must not be null"
        );
        operations.forEach(action);
    }

    // Checks that the operations list and it's elements are not empty
    private static void checkRowOperations(
        List<RowOperation> operations
    ) {
        Preconditions.checkNotNull(
            operations,
            "RowOperations: operations must not be null"
        );
        for (var operation : operations) {
            Preconditions.checkNotNull(
                operation,
                "RowOperations: operations list must not contain null elements"
            );
        }
    }
}