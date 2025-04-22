# COMP 7640 Database Systems & Administration Written Assignment #3 
# GongFan 24439169
## Question 1: Extendible Hashing Index
### (1) Query with key=11 (I/O Cost)
For extendible hashing, the typical process for a query involves:
1. **Directory Look-up**: The first step is to access the directory to determine which bucket the key might belong to. This requires checking the directory entry, and typically, it would require 1 I/O operation. 
2. **Bucket Access**: Once the correct bucket is found, the second I/O operation is needed to access the bucket and locate the data.

Since the directory is typically one level deep (assuming the default case here, as no specific details are provided), the I/O cost would be 2 operations: one for the directory and one for the bucket.

The I/O cost is 2 operations in this typical scenario.

### (2) Insertion of 2* and 8*
Let's assume a bucket capacity of 4 entries and an initial global depth (directory depth) of 1, meaning the directory has 2^1 = 2 entries (e.g., pointing to buckets for hash prefixes 0 and 1).

- **Hash Function**: Assume the hash function `h(k)` produces a binary string. We use the first `d` bits (where `d` is the global depth) to find the directory entry, and each bucket has a local depth `d'` indicating how many bits its entries actually share.

- **Initial State**: Global depth `d=1`. Directory entries for '0' and '1' point to Bucket A and Bucket B respectively. Both buckets have local depth `d'=1`.

  ```
  Initial State (d=1):
  Directory:
  [0] -> Bucket A (d'=1)
  [1] -> Bucket B (d'=1)

  Bucket A: [Assume 4 entries matching hash prefix '0']
  Bucket B: [Assume 4 entries matching hash prefix '1']
  ```

- **Inserting 2***: 
  1. Calculate `h(2*)`. Let's assume the first bit is '0'.
  2. Look up directory entry '0', which points to Bucket A.
  3. Attempt to insert 2* into Bucket A. Assume Bucket A is already full (contains 4 entries).
  4. **Bucket Split**: Since Bucket A is full, it needs to split.
     a. Create a new bucket, Bucket C.
     b. Increase the local depth of Bucket A and Bucket C to `d'=2`.
     c. Rehash all entries originally in Bucket A (including the new key 2*) using the first 2 bits. Distribute them between Bucket A (e.g., for '00') and Bucket C (e.g., for '01').
     d. **Directory Update**: Check if the global depth `d` needs to increase. Since the local depth of the split bucket (now 2) is greater than the global depth (1), the directory must double.
     e. Increase global depth `d` to 2. The directory now has 2^2 = 4 entries ('00', '01', '10', '11').
     f. Update directory pointers: Entry '00' points to Bucket A, entry '01' points to Bucket C. Since Bucket B did not split, its local depth is still 1. Both new directory entries '10' and '11' will point to Bucket B.

  ```
  State after inserting 2* (d=2):
  Directory:
  [00] -> Bucket A (d'=2)  <- Split from original Bucket A
  [01] -> Bucket C (d'=2)  <- New bucket from split
  [10] -> Bucket B (d'=1)
  [11] -> Bucket B (d'=1)

  Bucket A: [Entries matching '00', including potentially 2* if h(2*) starts '00']
  Bucket C: [Entries matching '01', including potentially 2* if h(2*) starts '01']
  Bucket B: [Original 4 entries matching '1']
  ```

- **Inserting 8***:
  1. Calculate `h(8*)`. Let's assume the first two bits are '10'.
  2. Look up directory entry '10', which points to Bucket B.
  3. Attempt to insert 8* into Bucket B. Assume Bucket B is also full.
  4. **Bucket Split**: Split Bucket B similarly.
     a. Create a new bucket, Bucket D.
     b. Increase local depth of Bucket B and Bucket D to `d'=2`.
     c. Rehash entries from Bucket B (including 8*) using the first 2 bits and distribute them between Bucket B (e.g., '10') and Bucket D (e.g., '11').
     d. **Directory Update**: The global depth `d` is already 2, which is equal to the new local depth of the split bucket. No directory doubling is needed.
     e. Update directory pointers: Entry '10' now points to Bucket B, and entry '11' now points to Bucket D. The pointers for '00' (Bucket A) and '01' (Bucket C) remain unchanged.

  ```
  State after inserting 8* (d=2):
  Directory:
  [00] -> Bucket A (d'=2)
  [01] -> Bucket C (d'=2)
  [10] -> Bucket B (d'=2)  <- Split from original Bucket B
  [11] -> Bucket D (d'=2)  <- New bucket from split

  Bucket A: [Entries matching '00']
  Bucket C: [Entries matching '01']
  Bucket B: [Entries matching '10', including potentially 8* if h(8*) starts '10']
  Bucket D: [Entries matching '11', including potentially 8* if h(8*) starts '11']
  ```

After these insertions, the directory has a global depth of 2 with 4 entries ('00', '01', '10', '11'), pointing to 4 different buckets (A, C, B, D respectively), each now having a local depth of 2.

## Question 2: Estimating I/O Costs for Join Operations
### (a) Page-Oriented Nested Loop Join (PLNLJ)
For Page-Oriented Nested Loop Join, we calculate the I/O cost as follows:

- **Pages in S**: 10,000 records divided by 100 records per page equals 100 pages.
- **Pages in E**: 1,000 records divided by 5 records per page equals 200 pages.

The I/O cost formula is:
Cost = Pages in S + (Pages in S × Pages in E)
Cost = 100 + (100 × 200) = 20,100 I/O operations.

The calculation is correct, and the I/O cost is 20,100 operations.

### (b) Simple-Nested Loop Join (SNKJ)
For Simple-Nested Loop Join, the correct I/O cost formula is:
- **Tuples in S**: 10,000 tuples.
- **Pages in E**: 200 pages.

The I/O cost is:
Cost = Tuples in S × Pages in E
Cost = 10,000 × 200 = 2,000,000 I/O operations.

The key error in the initial answer was using page numbers instead of the number of tuples in S. The correct approach uses the number of tuples in S (10,000) multiplied by the pages in E (200), which results in 2,000,000 I/O operations.

## Question 3: Estimating I/O Cost for a Query with Join
### (1) Page-Oriented Nested Loop Join (On-the-fly)
For this query, we have:
- **Relation R**: 10 pages, each containing approximately 34 records (since each record is 300 bytes).
- **Relation S**: 100 pages, each containing approximately 205 records (since each record is 500 bytes).

The I/O cost for the page-oriented nested loop join is:
Cost = Pages in R + (Pages in R × Pages in S)
Cost = 10 + (10 × 100) = 1,010 I/O operations.

### (2) Plan with Temporary Files (Sorting - Merge Join)
This plan involves sorting both relations on the join attribute and then merging them.

**Assumptions**: We use a standard 2-pass external merge sort algorithm. Let N be the number of pages in a relation. The cost of sorting is typically estimated as 4 * N I/Os (2 reads and 2 writes for the initial run creation pass, and 2 reads and 2 writes for the merge pass, simplified). The merge join phase requires reading each sorted relation once.

**Step 1: Sorting Phase**
- **Sort R**: Relation R has 10 pages. Cost = 4 * Pages(R) = 4 * 10 = 40 I/Os.
- **Sort S**: Relation S has 100 pages. Cost = 4 * Pages(S) = 4 * 100 = 400 I/Os.
- **Total Sorting Cost**: 40 + 400 = **440 I/Os**.

**Step 2: Merge Phase**
- After sorting, the relations are read once to perform the merge join.
- **Read Sorted R**: 10 pages.
- **Read Sorted S**: 100 pages.
- **Total Merge Cost**: 10 + 100 = **110 I/Os**.

**Total I/O Cost**:
- Total Cost = Sorting Cost + Merge Cost
- Total Cost = 440 + 110 = **550 I/O operations**.

## Question 4: Query Evaluation Plans
For the SQL query:
```sql
SELECT S.Name, S.GPA
FROM S, E
WHERE S.GPA > 3.5 AND S.SID = E.SID
```
Three possible query evaluation plans:

1. **Filter S first, then join**: 
   - First, apply the filter S.GPA > 3.5 to reduce the size of S.
   - Perform a nested loop join between the filtered S and E on S.SID = E.SID.
   - The I/O cost involves scanning and filtering S, and then for each tuple in filtered S, scanning E.
   - **Pros**: Simple to implement. Can be efficient if the filtered outer relation (S) is very small.
   - **Cons**: Can have very high I/O cost if the filtered outer relation is large or the inner relation (E) is large and not indexed. Scans the inner relation multiple times.
   - **Scenarios**: Best suited when the outer relation (after filtering) is extremely small, and the inner relation fits comfortably in memory, or when no suitable indexes are available.

   **Flowchart:**
   ```
   [Scan S] -> [Filter S (GPA > 3.5)] -> [Filtered S (S')]
                                          |
                                          v
                                       [Loop: For each s' in S'] -> [Scan E] -> [Join (s'.SID = E.SID)] -> [Output (s'.Name, s'.GPA)]
   ```

2. **Index on E’s SID**: 
   - Create or utilize an existing index on E.SID to speed up the join operation.
   - First, filter S based on GPA > 3.5.
   - For each tuple in the filtered S, use the index on E.SID to quickly find matching tuples in E.
   - This reduces the number of scans of E by leveraging the index, leading to a potentially lower I/O cost compared to a simple nested loop join.
   - **Pros**: Significantly reduces I/O cost for accessing the inner relation (E) if a suitable index exists and the join condition is selective. Avoids repeated full scans of E.
   - **Cons**: Requires an index on the join attribute of the inner relation. Index lookup cost can add up if the outer relation (filtered S) is still large. Creating an index incurs overhead if it doesn't already exist.
   - **Scenarios**: Effective when an index exists (or can be created) on the inner table's join column and the outer table (after filtering) is not excessively large. Particularly good for selective joins where only a few tuples match.

   **Flowchart:**
   ```
   [Scan S] -> [Filter S (GPA > 3.5)] -> [Filtered S (S')]
                                          |
                                          v
                                       [Loop: For each s' in S'] -> [Use Index on E.SID with s'.SID] -> [Fetch matching E tuple(s)] -> [Join] -> [Output (s'.Name, s'.GPA)]
   ```

3. **Sort-Merge Join**:
   - First, sort both the filtered S (after applying GPA > 3.5) and E based on the join attribute SID, if they are not already sorted.
   - Then, use a merge join technique to join the sorted relations by scanning both relations concurrently.
   - This method could be more efficient when both relations are large, as sorting and merging might be cheaper than a nested loop join.
   - **Pros**: Efficient when both relations are large and pre-sorted or can be sorted efficiently using available memory buffers. Reads each relation only once during the merge phase. Good for equi-joins and can produce sorted output, which might benefit subsequent operations (e.g., GROUP BY, ORDER BY).
   - **Cons**: The initial sorting step can be expensive in terms of I/O and CPU if relations are large and do not fit in memory. Not efficient if one relation is very small compared to the other (an index join might be better). Requires relations to be sorted on the join attribute.
   - **Scenarios**: Suitable for joining large relations, especially if they are already sorted on the join attributes or if the output needs to be sorted anyway. Often outperforms nested loop joins when indexes are not available and relations are large.

   **Flowchart:**
   ```
   +--------------------------------+      +-----------------------+
   | [Scan S] -> [Filter S (GPA>3.5)] | ---> | [Sort Filtered S on SID] | -> [Sorted S']
   +--------------------------------+      +-----------------------+      |
                                                                         v
                                                                      [Merge Join]
                                                                         ^
   +--------------------------------+      +-----------------------+      |
   | [Scan E]                       | ---> | [Sort E on SID]         | -> [Sorted E]
   +--------------------------------+      +-----------------------+
                                                                         |
                                                                         v
                                                                      [Output (S'.Name, S'.GPA)]
   ```

Each of these plans has a different execution cost depending on the available indexes, the filter size, and the number of tuples in S and E.
