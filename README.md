# Group 6 - Integrating Project
## VIDEO LINK: https://drive.google.com/file/d/19yyaQhGoQeou_CAgr178Mi-IT0y9Hr60/view?usp=sharing

- This project aimts to implement the MAGNET DNA pre-alignment filter in C and utilize CUDA programming to parallelize certain functions in the source code. The CUDA implementation uses shared memory concept, memory management and atomic operations. This is tested through using 1000 genomic sequences with a length of 100 characters. The filter was tested with edit distance threshold of 0, 3, and 8.

## Members

* Alamay, Carl Justine
* Ang, Czarina Damienne
* Esteban, Janina Angela
* Herrera, Diego Martin

## MAGNET pre-alignment filter algorithm
- The MAGNET pre-alignment filter would first create a hamming mask of the query and reference sequence through checking the nucleobase of the query and reference sequence in the same location. It would mark as 0 if the sequences matches, and 1 if it does not match. It would also create a deletion and insertion mask, having a total of 2E+1 hamming masks, where E is the edit distance threshold. This parameter could be set by the user. To create a deletion mask, it would shift the query sequence to the right, and to create an insertion mask, it would shift the query to the left. The number of shifts are dependent on the edit distance threshold. This is done in the MAGNET function. Next, it would find the longest consecutive zeroes in each masks and its information such as number of zeroes, and start and end index of the zeroes would be stored. The longest zeroes for each mask would be padded with 1s. This is done in Consecutive Zeroes function. It would iteratively use divide and conquer to find the longest zeroes among all the hamming masks and copy these zeroes to the final bit-vector. This is done in Extraction Encapsulation function. Lastly, it will count the number of 1s in the final bit-vector, if the number of 1s are greater than the edit distance threshold, the sequence will be rejected, but if the number of 1s are less than or equal to the edit distance threshold, the sequence will be accepted. This is done in the MAGNET function. 
  
## Report
### I. Screenshot of the program output with execution time and correctness check (C)
   **E = 0**
   ![Untitled2](https://github.com/user-attachments/assets/4cc1a30a-6636-45b5-ae70-cf3aa7238228)

   **E = 3**
   ![Untitled](https://github.com/user-attachments/assets/9f1a1adf-c473-459b-b710-2ebc124df5f4)

   **E = 8**
   ![image](https://github.com/user-attachments/assets/ff44b3a5-b5d4-47be-8d56-6e2adaddbbea)

   
<br/>

### II. Screenshot of the program output with execution time and correctness check (CUDA) <br/>

**E = 0** <br/>
  **256 Blocks** <br/>
    
  ![image](https://github.com/user-attachments/assets/3c21103a-259d-443a-94a3-8307fe1a75da)

  **E = 3**<br/>
    **256 Blocks**
    ![image](https://github.com/user-attachments/assets/dd030d92-e0e4-4f6b-b7a6-c75fbfc68249)

  **E = 8**<br/>
    **256 Blocks**
    ![image](https://github.com/user-attachments/assets/3b47e7e3-9847-4c19-b5d8-d3feac83b1bc)

<br/>

**E = 0**<br/>
  **1024 Blocks**<br/>
    
  ![image](https://github.com/user-attachments/assets/97860dca-c34e-4b92-a0ab-31bb5ebd662a)
    
**E = 3**<br/>
    **1024 Blocks**
    ![image](https://github.com/user-attachments/assets/2aeb2a45-f111-4c25-acca-d11738f84998)

**E = 8**<br/>
    **1024 Blocks**
    ![image](https://github.com/user-attachments/assets/27980cfb-5389-4e40-94f8-ae6dab330132)

  
<br/>

### III. Screenshot of the program implementation

**1. C Implementation**

   - Consecutive Zeros Info function<br/>
    ![image](https://github.com/user-attachments/assets/8f735bb9-5c3d-49e0-9508-0b503eadbd89)

The ConsecutiveZeros() function scans and finds groups of consecutive zeros within in array, more specifically in the specified sequence. It creates three variables to store the starting and ending positions, and the lengths of the zero groups. When a "1" appears as it's going through the array, it marks the end of the zero sequence and saves the details before looking for the next zeros group. It will then return all the collected information from these zero groups.

   - Extraction Encapsulation function<br/>
    ![image](https://github.com/user-attachments/assets/6db3d59f-20a3-4b74-9ffb-f103b71ffe96)

  ![image](https://github.com/user-attachments/assets/7bc4072a-a5e6-47ed-a251-718ba0394ec4)

The Extraction_Encapsulation() function finds long sequences of zeros in a given range and marks in the MagnetMask. It starts by storing the range in a stack and processes it until all sections are checked. For each range, it looks at different versions of the sequence and finds the longest group of consecutive zeros. If a second-longest group exists, it keeps track of that too. Once the longest zero sequence is found, it marks those positions as zero in MagnetMask. The function will then split the remaining unprocessed parts and add them back to the stack to continue searching.
 
   - MAGNET function<br/>
    ![image](https://github.com/user-attachments/assets/326bf040-8377-4b99-888f-31c1da944611)

  ![image](https://github.com/user-attachments/assets/1e60b933-3afb-4a2c-99dc-09a8ea446b9d)

  ![image](https://github.com/user-attachments/assets/43fac034-12d8-47de-845f-fee05d5bbff8)

The MAGNET() function compares the reference and read sequence to assess whether they are within the predefined error threshold. It first creates hamming masks that mark differences between the sequences. If the number of differences is equal or less than the error threshold, then the function immediately accepts the sequence. Otherwise, it creates extra hamming masks by shifting the read sequence left and right, accounting for possible insertions or deletions.

The function will then call Extraction_Encapsulation() to identify the longest consecutive streaks of zeros present within the insertion and deletion masks. These zero streaks represent regions where the read sequence aligns well with the reference. The function updates the MagnetMask to mark these high confidence regions while filtering out misaligned ones. The remaining mismatches will then be counted, and if it's more than the predefined edit threshold, then it will be rejected; Otherwise, it will be accepted.
    
   - Correctness checker<br/>
  ![image](https://github.com/user-attachments/assets/954d3b87-2b06-4825-81f8-f0c1fb61939f)

The snippet of the code showing the correctness checker compares two bit vectors, those being the finalBitVector from the intended output and another from a comparison variant. It iterates over the entire sequence being compared, and for every index, it checks if the corresponding bit in both the bit vectors match or not. If they are a mismatch at that specific index, then the error counter will be incremented by 1. This is to keep track of the number of matches between the two bit vectors.   

  
**2. CUDA Implementation**

This CUDA implementation of the MAGNET algorithm allocates memory on the GPU (cudaMalloc), transfers data between CPU and GPU (cudaMemcpy), and optimizes memory transfers using pinned memory (cudaMallocHost). It ensures proper cleanup with cudaFree and cudaFreeHost and optimizes execution using cudaDeviceScheduleYield.  The computeHammingAndErrorsKernel calculates mismatches across multiple error thresholds and stores results in shared memory. 

   - Compute Hamming and Errors Kernel<br/>
  ![image](https://github.com/user-attachments/assets/bf950964-8479-45cb-8c84-6af45d1ac1be)
  ![image](https://github.com/user-attachments/assets/9369f273-1d30-437c-80d0-d3851fb5ddde)

The Hamming distance is computed between the reference (RefSeq) and read sequence (ReadSeq) while identifying mismatches and generating error masks. Each thread processes a specific character pair in parallel, storing mismatch results in shared memory to reduce global memory access latency. By using thread parallelism, the computation is distributed efficiently across multiple threads. 

   - Find Consecutive Zeroes Kernel<br/>
   ![image](https://github.com/user-attachments/assets/aeefe70a-bbfc-4467-a0e5-27628f94bfac)
   ![image](https://github.com/user-attachments/assets/bd3271c5-b0b5-46f4-9d48-90e66185054b)

   - Find Consecutive Zeroes Function<br/>
    ![image](https://github.com/user-attachments/assets/c15b952c-08e8-43c5-9b7a-1a68b557b666)

It uses parallel reduction to efficiently scan the bit vector and locate zero sequences. Each thread works on a segment of the error mask, and shared memory is used to store intermediate results, improving performance by reducing redundant global memory accesses. Thread synchronization ensures that threads complete their computations before modifying shared memory. 

   - Extract Magnet Mask Function<br/>
   ![image](https://github.com/user-attachments/assets/b929b9eb-1666-4814-b702-70652c088179)
    ![image](https://github.com/user-attachments/assets/77e27dbd-18af-473d-a05b-8eff3225a804)

CUDA does not natively support recursion efficiently due to stack limitations so for this function, it uses an iterative approach with shared memory caching to refine the alignment zones. It applies bitwise operations to filter out misaligned segments and continuously updates the mask until an optimal alignment is achieved.

   - Magnet CUDA Function<br/>
   ![image](https://github.com/user-attachments/assets/70bf328a-cd0a-47d3-802f-e7de76847935)
    ![image](https://github.com/user-attachments/assets/8cde2d75-a218-4d6a-9a56-5c5217533b31)

 It first allocates GPU memory using cudaMalloc and copies sequences from host to device using cudaMemcpy. The function then launches the necessary CUDA kernels (computeHammingAndErrorsKernel and findConsecutiveZerosKernel), ensuring proper execution order with cudaDeviceSynchronize. Once kernel execution is complete, results are copied back to the host for analysis. It also applies asynchronous execution to overlap computation with memory transfers, maximizing GPU utilization. Finally, cudaFree is used to release allocated memory, preventing memory leaks.

 - Correctness checker<br/>
  ![image](https://github.com/user-attachments/assets/5c22772d-6d6e-43c1-bba1-249d0c35f0f1)

This checker tracks sequence alignment outcomes by counting accepted and rejected alignments while detecting errors. If accepted is true, acceptedCount increments; otherwise, rejectedCount increases. If result differs from check, errorCount logs mismatches.

### IV. Comparative table of execution time and Analysis
Average execution time of C Program
| Dataset size | E = 0      | E = 3 | E = 8 |
| ------------ | ---------- | ----- | ------ |
| 1000         | 6.33 ms | 31.05 ms | 67.53 ms |


Average execution times of CUDA
| Dataset size = 1000 | E = 0 | E = 3| E = 8 |
| ------------------- | ----- |------|--------|
| Blocks = 256        | 492.51 ms | 395.81 ms | 561.447 ms |
| Blocks = 1024       | 280.63 ms | 397.46 ms | 573.018 ms |

Speedup of CUDA compared to C
| Edit distance threshold | Speedup compared to C |
| ----------------------- | --------------------- |
| E = 0 | 0.013x |
| E = 3 | 0.078x |
| E = 8 | 0.120x |


Both implementations were timed with a dataset size of 1000 sequences and their average execution times after 10 repeats were compared. The C implementation had an average execution time of 6.33ms if E = 0, 31.05ms if E = 3, and 67.53ms if E = 8. It is observed that as the edit distance threshold increases, the execution time also increases. This is because as the edit distance threshold increase, there would be more hamming masks that would be generated and has to sequentially find the longest zeroes in each hamming mask. 

Meanwhile, for CUDA implementation with 256 blocks, it has an average execution time of 492.51ms, if E = 0, 395.81ms if E = 3, and 561.447ms if E = 8. With 1024 blocks, it has an average execution time of 280.63ms if E = 0, 397.46ms if E = 3, and 573.018ms if E = 8. This resulted to a speedup of 0.013x for E = 0, 0.078x for E = 3, and 0.120x for E = 8. Similar to the sequential implementation, the higher the edit distance threshold, the execution time also increases. Moreover, it is observed that at 1024 blocks, the execution time is higher compared to its original implementation and 256 blocks. This could be due too many threads executing for a small dataset size. With this, it is suggested to use 256 blocks for this dataset size.

One of the main observation is that the CUDA implementation is slower than the C, which could be due to a small dataset size and CUDA works more efficiently with larger datasets. However, current CUDA implementation causes a segmentation fault at larger dataset sizes.

<br/>

### V. Discussion

- **Parallel Algorithms** <br/>
The functionalities that were converted from sequential to parallel processing were the hamming distance computation, extraction and encapsulation function, and the locating of the longest set of consecutive zeros in a mask.

  The hamming mask calculation is used to track the number of errors between the reference and query genome. The hamming mask calculation for the sequential processing implementation was originally placed into the MAGNET function. One corresponding character from the reference and query genome were compared one at a time during execution. This process was converted into a parallel algorithm through the creation of the computeHammingAndErrorsKernel function. Instead of comparing sequentially, multiple threads are deployed and each one is assigned to compare one corresponding character from the reference and query genome which allows multiple comparisons to be made in one cycle. Global memory is then used to store the results from these comparisons with the additional use of atomic operations to prevent race conditions from occurring. 

  The extraction and encapsulation function in the original sequential implementation uses a stack-based approach to modify the segment runs of zeros in the hamming masks. This process is done by processing one sequence region at a time. In the parallelized version, multiple threads are assigned to process more sequence regions at a time and update these hamming masks in parallel. 

  The searching for consecutive zeros is done sequentially by iteratively looking over each character in the hamming mask. It marks the starting index of the zero sequence and marks the end index when looking at a non-zero number. This does it one sequence at a time and loops until all sequences have been checked. For the parallel implementation, multiple threads process different portions of the arrays with the additional use of global memory for storage of results while using atomic operations to avoid race conditions.

- **Problems encountered** <br/>
One problem encountered during the creation of the project was the conversion of the MATLAB source code into the C code. The original implementation had the functions separated so unifying them into one source code proved difficult. Another problem encountered was the encountering of segmentation faults at higher dataset sizes for the CUDA implementation. It is also observed as the Edit distance threshold increases, there would be a difference of 2 between the number of accepted sequences in CUDA and C. 

- **AHA moments** <br/>
The processes that were best suited for parallelization in the original sequential implementation were the processes that utilized iterative statements to process their data. Since sequential processing processes one piece of data per cycle, these areas proved the best suited for parallelization. 
