# Linear-Algebra

A small, reusable linear algebra library written in Java. It provides matrices and vectors, row reduction (RREF), determinants, inverses, matrix products, and diagonalization of real matrices, all built on plain `double` arithmetic with no numerical dependencies.

The project is organized as a library: data types live in `domain`, the algorithms that work on them live in `logic`, and `Main` is a short program that runs the library on sample problems. Everything is covered by a JUnit 5 test suite.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Project structure](#project-structure)
- [Running the sample problems](#running-the-sample-problems)
- [Running the tests](#running-the-tests)
- [Using the library](#using-the-library)
- [Possible future additions](#possible-future-additions)

## Features

### Data types (`domain`)

| Type | What it is |
| --- | --- |
| `Matrix` | A rectangular matrix of finite real numbers (at least 1x1). Supports `get`, `getRow`, `rows`, `cols`, `copy`, `transpose`, `multiply` by a vector, `merge` (placing another matrix to the right, as in an augmented matrix `[A \| b]`), and `permuteRows`. |
| `Vector` | A vector of finite real numbers. Supports `add`, `scale`, `dot`, `norm`, `permute`, `get`, `length`, and iteration over its entries. It can be created from an array or as a zero vector of a given size. |
| `Diagonalization` | The result of diagonalizing `A`: the matrices `P`, `D` and `P^-1` in `A = P D P^-1`. |
| `OperationRREF`, `Type1`, `Type2`, `Type3` | The elementary row operations as objects: swap two rows (`Type1`), multiply a row by a nonzero scalar (`Type2`), and add a multiple of one row to another (`Type3`). Each can be applied to a matrix or, entry by entry, to a vector. |

`Matrix` and `Vector` reject empty shapes, ragged rows and non-finite entries (`NaN`, infinities), and their operations reject results that overflow.

### Algorithms (`logic`)

**`MatrixLogic`** wraps one matrix and provides:

- `rref()`: the reduced row echelon form of a matrix of any shape.
- `determinant()`: the determinant of a square matrix. A singular matrix gives `0`, and a non-square matrix gives `NaN`.
- `inverse()`: the inverse of a square, invertible matrix.
- `multiply(Matrix)`: the matrix product.
- Static elementary row operations, applied in place: `swapRows`, `multiplyRow`, `reduceRow`.

The row reduction is worked out once, when the `MatrixLogic` is created, and reused by `rref`, `determinant` and `inverse`, so asking for several of them does not repeat the work.

**`VectorLogic`** provides the elementary operations on the entries of a vector, applied in place: `swapElements`, `multiplyElement`, `reduceElement`.

**`DiagonalizationLogic`** factors a real square matrix as `A = P D P^-1`:

- The columns of `P` are unit eigenvectors, and `D` holds the matching eigenvalues on its diagonal. Eigenvalues are ordered by decreasing magnitude, and the larger value comes first when two have the same magnitude. This puts the eigenvalue 1 of a Markov chain first, even when -1 is also an eigenvalue.
- Works on non-symmetric matrices, and handles repeated eigenvalues as long as there are enough independent eigenvectors.
- Works entirely in real numbers. A matrix with complex eigenvalues, or a defective matrix (an eigenvalue with fewer independent eigenvectors than its multiplicity), has no real diagonalization and is rejected with a `NotDiagonalizableException`.
- `P^-1` is returned with the result, so powers `A^k = P D^k P^-1` do not need a second inverse.

### Exceptions (`domain.Exceptions`)

| Exception | When it is thrown |
| --- | --- |
| `InverseNotAvailableException` | `MatrixLogic.inverse()` on a matrix that is not square or is singular. |
| `MatrixMultiplyDimMismatchException` | `MatrixLogic.multiply` when the left matrix's column count differs from the right matrix's row count. |
| `NotDiagonalizableException` | `DiagonalizationLogic.diagonalize()` on a matrix with complex eigenvalues or a defective matrix. The message says which. |
| `DeterminantDoesNotExistException` | Used internally for non-square matrices. `MatrixLogic.determinant()` reports `NaN` instead. |

Invalid arguments (null values, indices out of range, mismatched sizes) raise the standard `NullPointerException`, `IllegalArgumentException` and `IllegalStateException`, as documented in the Javadoc of each method.

### Numerical behavior

All arithmetic is in double precision, so results are accurate to rounding error and not exact. Decisions about "zero" are made relative to the largest entry of the matrix, so the library behaves the same on a matrix and on a copy scaled by a large or tiny factor.

- RREF treats entries at or below `1e-10` times the largest entry as zero.
- Diagonalization treats eigenvalues closer than `1e-6` times the largest entry of `A` as one repeated eigenvalue, and an imaginary part that small as zero. Two genuinely distinct eigenvalues that close are therefore reported as a defective matrix.

## Requirements

- JDK 25
- Maven 3.9 or newer (developed with 3.9.16)
- Optional: IntelliJ IDEA 2025.2.1 or newer, which supports Java 25

Maven downloads the dependencies on the first build: Guava (`33.5.0-jre`), and for the tests only, JUnit Jupiter (`5.11.3`) and the JUnit Platform Launcher (`1.11.3`).

## Project structure

```
src
├── main
│   └── java
│       ├── Main.java                         Sample problems
│       ├── domain
│       │   ├── Diagonalization.java
│       │   ├── Matrix.java
│       │   ├── Vector.java
│       │   ├── Exceptions
│       │   │   ├── DeterminantDoesNotExistException.java
│       │   │   ├── InverseNotAvailableException.java
│       │   │   ├── MatrixMultiplyDimMismatchException.java
│       │   │   └── NotDiagonalizableException.java
│       │   └── RREF
│       │       ├── OperationRREF.java
│       │       ├── Type1.java                Swap two rows
│       │       ├── Type2.java                Scale a row
│       │       └── Type3.java                Add a multiple of a row to another
│       └── logic
│           ├── DiagonalizationLogic.java
│           ├── MatrixLogic.java
│           └── VectorLogic.java
└── test
    └── java
        ├── RunAllTests.java                  Runs every test and prints a summary
        ├── domain
        │   ├── DiagonalizationTest.java
        │   ├── MatrixTest.java
        │   └── VectorTest.java
        └── logic
            ├── DiagonalizationLogicTest.java
            ├── MatrixLogicTest.java
            └── VectorLogicTest.java
```

## Running the sample problems

`Main` runs the library on nine sample problems and prints the results to the console:

1. Solving a linear system `A x = b` by row reducing the augmented matrix.
2. The determinant and inverse of an invertible matrix, with `A * A^-1` shown to be the identity.
3. A singular matrix, which has no inverse.
4. A matrix product.
5. A product whose dimensions do not match.
6. Diagonalizing a matrix with distinct eigenvalues.
7. Diagonalizing a matrix with a repeated eigenvalue.
8. A rotation, which has complex eigenvalues and is rejected.
9. A Jordan block, which is defective and is rejected.

### From the terminal

Open a terminal in the project root (the folder that contains `pom.xml`) and run:

```powershell
mvn compile exec:java "-Dexec.mainClass=Main"
```

The quotes around the `-D` option are needed in PowerShell, which otherwise splits the argument at the dot. They are harmless in other shells. The first run downloads the dependencies and the `exec` plugin, so it needs an internet connection.

### From IntelliJ IDEA

1. Choose **File | Open**, select the project's `pom.xml`, and open it as a project. Wait for the Maven import to finish.
2. Check **File | Project Structure | Project** and make sure the SDK is a JDK 25.
3. Open `src/main/java/Main.java`.
4. Click the green arrow in the gutter next to `public static void main` and choose **Run 'Main.main()'**. The output appears in the **Run** tool window.

## Running the tests

The tests use JUnit 5. They check the library against hand-computed cases and against randomized inputs built from a known answer (fixed random seeds keep any failure reproducible).

### From the terminal

The standard way is Maven's test phase. In the project root:

```powershell
mvn test
```

Maven prints a summary per test class and ends with `BUILD SUCCESS` when everything passes.

The project also contains `RunAllTests`, a small program that finds every test class under the project's package, runs them, and prints one summary with the failures listed first. It exits with code 0 when every test passes and 1 otherwise:

```powershell
mvn test-compile exec:java "-Dexec.mainClass=RunAllTests" "-Dexec.classpathScope=test"
```

`RunAllTests` ends the program itself, so Maven stops as soon as the summary is printed and shows no `BUILD SUCCESS` line. In PowerShell, `$LASTEXITCODE` holds the exit code afterward.

### From IntelliJ IDEA

Any of these work once the project is open (see the steps above):

- **All tests:** In the **Project** tool window, right-click `src/test/java` and choose **Run 'All Tests'**. Results appear as a tree in the **Run** tool window, where you can click a failing test to jump to it.
- **One class or one test:** Open a test class and click the green arrow in the gutter next to the class name, or next to a single test method.
- **Through Maven:** Open the **Maven** tool window, expand **LinearAlgebra | Lifecycle**, and double-click **test**. This runs the same thing as `mvn test`.
- **With `RunAllTests`:** Open `src/test/java/RunAllTests.java` and click the green arrow next to `main`. The summary is printed in the **Run** tool window.

## Using the library

The pattern is the same throughout: wrap your numbers in a `Matrix` or `Vector`, hand the matrix to the logic class that does the work, and read the result from the returned object. `MatrixLogic` takes a `Matrix` and answers questions about it, `DiagonalizationLogic` takes a square `Matrix` and returns a `Diagonalization`, and `VectorLogic` and the static methods of `MatrixLogic` modify a vector or matrix in place. The checked exceptions listed above tell you when a request has no answer, for example an inverse that does not exist. `Main` shows each of these end to end.

Every public class and method has Javadoc. In IntelliJ, press **Ctrl+Q** on a name (or hover over it) to read it.

## Possible future additions

Nothing here exists yet. These are ideas for extending the library:

- Matrix powers `A^k`, computed through diagonalization.
- Matrix addition, subtraction and multiplication by a scalar, and an identity-matrix constructor.
- Solving `A x = b` directly, including the cases with no solution or infinitely many.
- Rank, pivot columns, and bases for the null space and column space.
- Eigenvalues on their own, and a helper for the dominant eigenvector (steady-state vector) of a matrix.
- Other factorizations: LU, QR, Cholesky.
- Support for complex eigenvalues.